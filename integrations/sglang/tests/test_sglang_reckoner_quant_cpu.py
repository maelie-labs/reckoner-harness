"""CPU unit tests for quantized and fused linear layers (`sglang_reckoner/modeling_reckoner.py`).

Covers:
1. Golden bit-equality (`torch.equal`) for prefill and 4 decode steps against
   `tests/fixtures/quant_tiny_golden.pt`.
2. Module-level bit-equality (`torch.equal`) between every `ReplicatedLinear` in the
   tiny model and `F.linear(x, module.weight, module.bias)`.
3. Quant plumbing on the full 1.4B `ReckonerConfig` on the `meta` device with
   `Fp8Config(is_checkpoint_fp8_serialized=False)`:
   - 196 projections have `Fp8LinearMethod`
   - 6 `attn_gate` projections have `UnquantizedLinearMethod`
   - `embed_tokens` has no `quant_method`
   - every projection's `prefix` + `.weight` is in the `model.safetensors` header of
     `$RECKONER_TEST_MODEL_DIR/model.safetensors` (when `RECKONER_TEST_MODEL_DIR` is set)
   - quantized weight parameter count is 1,375,469,568
   - optional `keep_gdn_gates_bf16=True` leaves `g_proj`, `b_proj`, `w_proj` unquantized
     (154 `Fp8LinearMethod` + 48 `UnquantizedLinearMethod`).
4. Compressed-tensors `ignore` path on CPU (`CompressedTensorsConfig.from_config` W4A16 g128
   with `ignore=["lm_head", "re:.*attn_gate$"]` gives `UnquantizedLinearMethod` on `attn_gate`).
5. Weight round-trip via `load_weights` on the tiny model (`len(loaded) == len(named_tensors)`).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import struct
import sys

import pytest
import torch
import torch.nn.functional as F

_PLUGIN_ROOT = Path(__file__).resolve().parents[1]
if str(_PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_ROOT))

from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

from sglang_reckoner import (
    ReckonerConfig,
    ReckonerForCausalLM,
    ReckonerGDN2AttnBackend,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
GOLDEN_FIXTURE_PATH = FIXTURES_DIR / "quant_tiny_golden.pt"
TINY_PARITY_FIXTURE_PATH = FIXTURES_DIR / "tiny_parity_fixtures.pt"


def _get_test_model_dir() -> Path | None:
    raw = os.environ.get("RECKONER_TEST_MODEL_DIR", "").strip()
    if not raw:
        return None
    p = Path(raw)
    return p if p.is_dir() else None


def _read_safetensors_header_keys(safetensors_path: Path) -> set[str]:
    with safetensors_path.open("rb") as f:
        header_len_bytes = f.read(8)
        assert len(header_len_bytes) == 8
        (header_len,) = struct.unpack("<Q", header_len_bytes)
        header_raw = f.read(header_len)
    header_obj = json.loads(header_raw.decode("utf-8"))
    return {k for k in header_obj.keys() if k != "__metadata__"}


def _build_tiny_sgl_model() -> tuple[ReckonerForCausalLM, list[tuple[str, torch.Tensor]], ReckonerGDN2AttnBackend]:
    assert TINY_PARITY_FIXTURE_PATH.is_file(), f"Missing fixture {TINY_PARITY_FIXTURE_PATH}"
    fix = torch.load(TINY_PARITY_FIXTURE_PATH, map_location="cpu", weights_only=True)
    named_tensors = [(k, v.clone().to(dtype=torch.float32)) for k, v in fix["reckoner"]["named_tensors"]]

    rcfg = ReckonerConfig(
        vocab_size=256,
        d_model=64,
        intermediate_size=128,
        num_hidden_layers=4,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        rotary_dim=8,
        layer_layout="A G A G",
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        qk_norm=True,
        use_gated_attn_output=True,
        gated_attn_position="pre_oproj",
        gdn_v_heads=8,
        conv_kernel=4,
        tie_word_embeddings=True,
    )
    backend = ReckonerGDN2AttnBackend.create_standalone(
        rcfg, max_reqs=8, max_context_len=128, device="cpu", dtype=torch.float32
    )
    sgl_model = ReckonerForCausalLM(rcfg, attn_backend=backend).to(dtype=torch.float32).eval()
    return sgl_model, named_tensors, backend


def _run_tiny_sgl_model_for_golden() -> dict:
    sgl_model, named_tensors, backend = _build_tiny_sgl_model()
    loaded = sgl_model.load_weights(named_tensors)
    assert len(loaded) == len(named_tensors)

    prompt_ids = [2, 11, 25, 42, 99, 108, 3]
    decode_tokens = [9, 18, 54, 14]
    seq_len = len(prompt_ids)
    inp_sgl = torch.tensor(prompt_ids, dtype=torch.long)
    pos_sgl = torch.arange(0, seq_len, dtype=torch.long)
    batch_ext = backend.make_forward_batch(slot_indices=[0], seq_lens=[seq_len], is_decode=False)
    with torch.no_grad():
        prefill_logits = sgl_model(inp_sgl, pos_sgl, batch_ext).detach().clone()

    decode_logits: list[torch.Tensor] = []
    curr_pos = seq_len
    for tok in decode_tokens:
        batch_dec = backend.make_forward_batch(slot_indices=[0], seq_lens=[1], is_decode=True)
        with torch.no_grad():
            step_logits = sgl_model(
                torch.tensor([tok], dtype=torch.long),
                torch.tensor([curr_pos], dtype=torch.long),
                batch_dec,
            )
        decode_logits.append(step_logits.detach().clone())
        curr_pos += 1
    return {
        "prefill_logits": prefill_logits,
        "decode_logits": decode_logits,
    }


def test_quant_golden_equality() -> None:
    assert GOLDEN_FIXTURE_PATH.is_file(), f"Missing golden fixture {GOLDEN_FIXTURE_PATH}"
    golden = torch.load(GOLDEN_FIXTURE_PATH, map_location="cpu", weights_only=True)
    current = _run_tiny_sgl_model_for_golden()

    assert torch.equal(current["prefill_logits"], golden["prefill_logits"]), (
        f"Prefill logits not bit-equal; max abs diff = "
        f"{(current['prefill_logits'] - golden['prefill_logits']).abs().max().item()}"
    )
    assert len(current["decode_logits"]) == 4
    assert len(golden["decode_logits"]) == 4
    for step_idx, (cur_step, gold_step) in enumerate(
        zip(current["decode_logits"], golden["decode_logits"])
    ):
        assert torch.equal(cur_step, gold_step), (
            f"Decode step {step_idx} logits not bit-equal; max abs diff = "
            f"{(cur_step - gold_step).abs().max().item()}"
        )


def test_quant_module_level_equality() -> None:
    sgl_model, named_tensors, _ = _build_tiny_sgl_model()
    sgl_model.load_weights(named_tensors)

    torch.manual_seed(9876)
    replicated_modules = [
        (name, mod)
        for name, mod in sgl_model.named_modules()
        if isinstance(mod, ReplicatedLinear)
    ]
    # 2 'A' layers * (4 + 1 attn_gate) + 2 'G' layers * 8 + 4 MLP * 3 = 10 + 16 + 12 = 38
    assert len(replicated_modules) == 38

    for name, mod in replicated_modules:
        x = torch.randn(5, mod.input_size, dtype=torch.float32)
        with torch.no_grad():
            out, out_bias = mod(x)
            expected = F.linear(x, mod.weight, mod.bias)
        assert out_bias is None, f"{name} returned non-None output_bias"
        assert torch.equal(out, expected), f"{name} output not bit-equal to F.linear"


def test_quant_plumbing_full_1p4b_config() -> None:
    export_dir = _get_test_model_dir()
    header_keys: set[str] | None = None
    if export_dir is not None and (export_dir / "model.safetensors").is_file():
        header_keys = _read_safetensors_header_keys(export_dir / "model.safetensors")
        rcfg = ReckonerConfig.from_pretrained(str(export_dir))
    else:
        rcfg = ReckonerConfig()
    fp8_cfg = Fp8Config(is_checkpoint_fp8_serialized=False)

    with torch.device("meta"):
        model = ReckonerForCausalLM(rcfg, quant_config=fp8_cfg)

    assert not hasattr(model.embed_tokens, "quant_method") or getattr(
        model.embed_tokens, "quant_method", None
    ) is None

    fp8_modules: list[tuple[str, ReplicatedLinear]] = []
    unquant_modules: list[tuple[str, ReplicatedLinear]] = []
    quantized_param_count = 0

    for name, mod in model.named_modules():
        if isinstance(mod, ReplicatedLinear):
            assert getattr(mod, "prefix", None) == name, (
                f"Module {name} has mismatched prefix {getattr(mod, 'prefix', None)}"
            )
            if header_keys is not None:
                assert f"{mod.prefix}.weight" in header_keys, (
                    f"Prefix {mod.prefix}.weight not found in model.safetensors header"
                )
                if mod.bias is not None:
                    assert f"{mod.prefix}.bias" in header_keys, (
                        f"Prefix {mod.prefix}.bias not found in model.safetensors header"
                    )
            if isinstance(mod.quant_method, Fp8LinearMethod):
                fp8_modules.append((name, mod))
                quantized_param_count += int(mod.weight.numel())
            elif isinstance(mod.quant_method, UnquantizedLinearMethod):
                unquant_modules.append((name, mod))
            else:
                raise AssertionError(
                    f"Unexpected quant_method {type(mod.quant_method)} on {name}"
                )

    assert len(fp8_modules) == 196, f"Expected 196 Fp8LinearMethod modules, got {len(fp8_modules)}"
    assert len(unquant_modules) == 6, (
        f"Expected 6 UnquantizedLinearMethod modules, got {len(unquant_modules)}"
    )
    assert all(name.endswith(".mixer.attn_gate") for name, _ in unquant_modules)
    assert quantized_param_count == 1_375_469_568, (
        f"Expected 1,375,469_568 quantized params, got {quantized_param_count}"
    )

    # Also verify constructor/config option `keep_gdn_gates_bf16=True`
    rcfg_keep_gates = (
        ReckonerConfig.from_pretrained(str(export_dir))
        if export_dir is not None
        else ReckonerConfig()
    )
    rcfg_keep_gates.keep_gdn_gates_bf16 = True
    with torch.device("meta"):
        model_keep_gates = ReckonerForCausalLM(rcfg_keep_gates, quant_config=fp8_cfg)
    fp8_keep_count = sum(
        1
        for m in model_keep_gates.modules()
        if isinstance(m, ReplicatedLinear) and isinstance(m.quant_method, Fp8LinearMethod)
    )
    unquant_keep_count = sum(
        1
        for m in model_keep_gates.modules()
        if isinstance(m, ReplicatedLinear) and isinstance(m.quant_method, UnquantizedLinearMethod)
    )
    assert fp8_keep_count == 154
    assert unquant_keep_count == 48


def test_quant_compressed_tensors_ignore_path_cpu() -> None:
    ct_cfg = CompressedTensorsConfig.from_config(
        {
            "format": "pack-quantized",
            "config_groups": {
                "group_0": {
                    "format": "pack-quantized",
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 4,
                        "type": "int",
                        "symmetric": True,
                        "strategy": "group",
                        "group_size": 128,
                    },
                    "input_activations": None,
                }
            },
            "ignore": ["lm_head", "re:.*attn_gate$"],
        }
    )
    attn_gate_mod = ReplicatedLinear(
        1536,
        1536,
        bias=False,
        quant_config=ct_cfg,
        prefix="layers.0.mixer.attn_gate",
    )
    assert isinstance(attn_gate_mod.quant_method, UnquantizedLinearMethod)


def test_quant_weight_round_trip() -> None:
    sgl_model, named_tensors, _ = _build_tiny_sgl_model()
    loaded = sgl_model.load_weights(named_tensors)
    assert len(loaded) == len(named_tensors)
    for name, ref_tensor in named_tensors:
        norm_name = (
            name.replace("_orig_mod.", "").replace("trunk.", "").removeprefix("model.")
        )
        if norm_name == "lm_head.weight":
            norm_name = "embed_tokens.weight"
        param = dict(sgl_model.named_parameters(remove_duplicate=False))[norm_name]
        assert torch.equal(param.data, ref_tensor), f"Mismatch after load_weights for {norm_name}"


def test_quant_partial_merge_and_quant_scope_guard_cpu(monkeypatch) -> None:
    """Verifies atomic layer-level f2 merge fallback on mixed groups, FP8 bail-out, and zero-point/g_idx bail-out."""
    from sglang_reckoner.gdn2_backend import ReckonerGDN2AttnBackend
    from sglang_reckoner.modeling_reckoner import ReckonerGDN2Layer, _merge_replicated_linears

    monkeypatch.setenv("SGLANG_RECKONER_FUSED_GATES", "1")
    monkeypatch.setenv("SGLANG_RECKONER_FUSION_MASK", "all")

    rcfg = ReckonerConfig(
        vocab_size=256,
        d_model=64,
        n_layers=2,
        num_hidden_layers=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        rotary_dim=8,
        intermediate_size=128,
        layer_layout="A G",
        gdn_v_heads=8,
        conv_kernel=4,
        tie_word_embeddings=True,
    )

    backend = ReckonerGDN2AttnBackend.create_standalone(rcfg, max_reqs=2, max_context_len=32, dtype=torch.bfloat16)
    gdn_layer = ReckonerGDN2Layer(rcfg, layer_id=1, quant_config=None, prefix="layers.1.mixer").bfloat16().eval()

    x = torch.randn(4, 64, dtype=torch.bfloat16)
    fb_ref = backend.make_forward_batch([0], [4], is_decode=False)
    monkeypatch.setenv("SGLANG_RECKONER_FUSED_GATES", "0")
    out_ref = gdn_layer(x, fb_ref).clone()

    monkeypatch.setenv("SGLANG_RECKONER_FUSED_GATES", "1")
    gdn_layer.q_proj.register_buffer("weight_zero_point", torch.ones((4, 1), dtype=torch.int32), persistent=False)
    ok = gdn_layer.build_fused_projections(free_unmerged=True)
    assert ok is False, "Expected build_fused_projections to return False when one group in the layer fails"
    assert gdn_layer._fused_qkvz_proj is None
    assert gdn_layer._fused_gbw_proj is None
    assert gdn_layer._f2_merge_attempted is True
    for sub in (
        gdn_layer.q_proj,
        gdn_layer.k_proj,
        gdn_layer.v_proj,
        gdn_layer.z_proj,
        gdn_layer.g_proj,
        gdn_layer.b_proj,
        gdn_layer.w_proj,
    ):
        assert sub.weight.numel() > 0, "Unmerged weight was freed despite partial merge failure!"

    fb_test = backend.make_forward_batch([1], [4], is_decode=False)
    out_test = gdn_layer(x, fb_test)
    assert torch.allclose(out_ref.float(), out_test.float(), atol=1e-2)

    lin_a = ReplicatedLinear(64, 32, bias=False).bfloat16()
    lin_b = ReplicatedLinear(64, 32, bias=False).bfloat16()
    lin_a.register_buffer("weight_g_idx", torch.zeros(64, dtype=torch.int32), persistent=False)
    assert _merge_replicated_linears([lin_a, lin_b], quant_config=None, prefix="test") is None
    assert lin_a.weight.numel() == 64 * 32

    rcfg_fp8 = ReckonerConfig(
        vocab_size=256,
        d_model=64,
        n_layers=2,
        num_hidden_layers=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        rotary_dim=8,
        intermediate_size=128,
        layer_layout="A G",
        gdn_v_heads=8,
        conv_kernel=4,
        tie_word_embeddings=True,
        keep_gdn_gates_bf16=True,
    )
    fp8_cfg = Fp8Config(is_checkpoint_fp8_serialized=False)
    gdn_fp8 = ReckonerGDN2Layer(rcfg_fp8, layer_id=1, quant_config=fp8_cfg, prefix="layers.1.mixer")
    ok_fp8 = gdn_fp8.build_fused_projections(free_unmerged=True)
    assert ok_fp8 is False
    assert gdn_fp8._fused_qkvz_proj is None
    assert gdn_fp8._fused_gbw_proj is None
    assert gdn_fp8.g_proj.weight.numel() > 0
    assert gdn_fp8.b_proj.weight.numel() > 0
    assert gdn_fp8.w_proj.weight.numel() > 0


def test_quant_keep_unmerged_reload_refused_cpu(monkeypatch) -> None:
    """Verifies that SGLANG_RECKONER_KEEP_UNMERGED_WEIGHTS=1 rebuilds are restricted to unquantized bf16 and refused after a quantized f2 merge."""
    import pytest
    from sglang_reckoner.modeling_reckoner import _merge_replicated_linears

    monkeypatch.setenv("SGLANG_RECKONER_FUSED_GATES", "1")
    monkeypatch.setenv("SGLANG_RECKONER_FUSION_MASK", "all")
    monkeypatch.setenv("SGLANG_RECKONER_KEEP_UNMERGED_WEIGHTS", "1")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **kw: (8, 9))

    rcfg = ReckonerConfig(
        vocab_size=256,
        d_model=128,
        n_layers=2,
        num_hidden_layers=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=32,
        rotary_dim=16,
        intermediate_size=256,
        layer_layout="A G",
        gdn_v_heads=8,
        conv_kernel=4,
        tie_word_embeddings=True,
    )
    ct_cfg = CompressedTensorsConfig.from_config(
        {
            "config_groups": {
                "group_0": {
                    "format": "pack-quantized",
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 4,
                        "type": "int",
                        "symmetric": True,
                        "strategy": "group",
                        "group_size": 128,
                    },
                    "input_activations": None,
                }
            },
            "ignore": ["lm_head", "re:.*attn_gate$"],
        }
    )
    model = ReckonerForCausalLM(rcfg, quant_config=ct_cfg).bfloat16().eval()

    model.load_weights([])
    assert model._quantized_projections_merged is True
    assert model._has_quantized_merged_projections() is True
    assert model.layers[1].mixer._fused_qkvz_proj is not None
    assert model.layers[1].mixer._fused_qkvz_proj.quant_mod is not None

    with pytest.raises(
        RuntimeError,
        match="SGLANG_RECKONER_KEEP_UNMERGED_WEIGHTS=1 rebuilds are restricted to unquantized bfloat16 weights",
    ):
        model.load_weights([])

    with pytest.raises(RuntimeError, match="Cannot rebuild merged quantized projections in-place"):
        _merge_replicated_linears(
            [model.layers[1].mlp.gate_proj, model.layers[1].mlp.up_proj],
            quant_config=ct_cfg,
            prefix="layers.1.mlp._fused_gate_up_proj",
            existing_runner=model.layers[1].mlp._fused_gate_up_proj,
        )
