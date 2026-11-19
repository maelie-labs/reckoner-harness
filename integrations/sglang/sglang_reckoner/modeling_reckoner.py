# Copyright 2026 Maelie Labs
# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
#
# Notice of Modification (Apache-2.0 Section 4(b)):
# Modified by Maelie Labs (2026): adapted from SGLang v0.5.22 model layer interfaces
# (`sglang.srt.models`, `RadixAttention`, `RadixLinearAttention`, `ReplicatedLinear`)
# to implement `ReckonerForCausalLM` hybrid GDN-2 + GQA layers.
"""
Out-of-tree SGLang model definition (`ReckonerForCausalLM`) for Reckoner-1.4B (`Trunk@S3`).

Architecture (`AAGGGGGGGGGGGGGAGAGAGA`, 20 layers):
- 6 Full Attention layers (`op_type == 'A'`, layers `[0, 2, 4, 15, 17, 19]`):
  GQA (12 Q heads, 3 KV heads, head_dim=128), per-head QK RMSNorm (eps=1e-5),
  partial RoPE (rotary_dim=64, rope_theta=500,000), SGLang `RadixAttention`,
  and `pre_oproj` gated attention output (`2 * sigmoid(attn_gate(x))`).
- 14 GDN-2 Linear Attention layers (`op_type == 'G'`, layers `[1, 3, 5..14, 16, 18]`):
  12 Q/K heads (head_dim=128), 24 V heads (head_dim=128, v_dim=3072),
  causal depthwise 1D convolution (`kernel_size=4`, `activation=None`, `bias=False`),
  channel-wise log-decay (`g = -exp(clamp(g_proj(x), max=5.0))`), erase gate (`b = sigmoid(b_proj(x))`),
  write gate (`w = sigmoid(w_proj(x))`), SGLang `RadixLinearAttention` dispatched to
  `ReckonerGDN2AttnBackend`, and `FusedRMSNormGated(hidden_size=v_dim, eps=1e-6)` before `out_proj`.
- SwiGLU MLP (`d_model=1536 -> intermediate_size=10240 -> 1536`) on all 20 layers.
- Tied input/output token embeddings (`vocab_size=36864, d_model=1536`).
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.radix_attention import AttentionType, RadixAttention
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


def _build_replicated_linear(
    input_size: int,
    output_size: int,
    bias: bool,
    quant_config: Optional[QuantizationConfig],
    prefix: str,
) -> ReplicatedLinear:
    layer = ReplicatedLinear(
        input_size,
        output_size,
        bias=bias,
        quant_config=quant_config,
        prefix=prefix,
    )
    layer.prefix = prefix
    if (
        isinstance(getattr(layer, "quant_method", None), UnquantizedLinearMethod)
        and getattr(layer, "weight", None) is not None
        and not layer.weight.is_meta
    ):
        nn.Linear.reset_parameters(layer)
    return layer

try:
    from sgl_kernel import rmsnorm as sgl_rmsnorm
except ImportError:
    sgl_rmsnorm = None

try:
    from fla.modules.fused_norm_gate import FusedRMSNormGated
except ImportError:
    FusedRMSNormGated = None

from .gdn2_backend import (
    is_fusion_step_enabled,
    is_weight_sync_mode,
    reckoner_fused_partial_rope_inplace_fwd,
    reckoner_fused_sigmoid_mul_fwd,
    reckoner_fused_silu_mul_fwd,
)


class _FusedLinearRunner:
    """Lightweight non-nn.Module callable for merged projections (keeps named_modules() unchanged)."""

    __slots__ = ("weight", "bias", "quant_mod")

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        quant_mod: Optional[ReplicatedLinear] = None,
    ) -> None:
        self.weight = weight
        self.bias = bias
        self.quant_mod = quant_mod

    def matches_tensor(self, x: torch.Tensor) -> bool:
        if x.dtype != torch.bfloat16:
            return False
        if self.weight is not None:
            return self.weight.device == x.device and self.weight.dtype == x.dtype
        if self.quant_mod is not None:
            wp = getattr(self.quant_mod, "weight_packed", None)
            ws = getattr(self.quant_mod, "weight_scale", None)
            if wp is not None and ws is not None:
                return wp.device == x.device and ws.dtype == x.dtype
            return True
        return False

    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        if self.quant_mod is not None:
            out, _ = self.quant_mod(x)
            return out, None
        return F.linear(x, self.weight, self.bias), None


def _has_non_empty_attr(layer: Any, attr: str) -> bool:
    val = getattr(layer, attr, None)
    if val is None:
        return False
    if isinstance(val, torch.Tensor):
        return val.numel() > 0
    return True


def _is_fp8_linear(
    layer: ReplicatedLinear, quant_config: Optional[QuantizationConfig]
) -> bool:
    if quant_config is not None and "fp8" in type(quant_config).__name__.lower():
        return True
    qm = getattr(layer, "quant_method", None)
    if qm is not None and "fp8" in type(qm).__name__.lower():
        return True
    w = getattr(layer, "weight", None)
    if isinstance(w, torch.Tensor) and "float8" in str(w.dtype).lower():
        return True
    return False


def _free_replicated_linears(layers: List[ReplicatedLinear]) -> None:
    """Release original unmerged projection tensors only after all merges in a layer succeed."""
    for l in layers:
        w = getattr(l, "weight", None)
        if isinstance(w, torch.Tensor) and w.numel() > 0 and not w.is_meta:
            l.weight = nn.Parameter(
                torch.empty(0, dtype=w.dtype, device=w.device),
                requires_grad=False,
            )
        b = getattr(l, "bias", None)
        if isinstance(b, torch.Tensor) and b.numel() > 0 and not b.is_meta:
            l.bias = nn.Parameter(
                torch.empty(0, dtype=b.dtype, device=b.device),
                requires_grad=False,
            )
        wp = getattr(l, "weight_packed", None)
        if isinstance(wp, torch.Tensor) and wp.numel() > 0 and not wp.is_meta:
            l.weight_packed = nn.Parameter(
                torch.empty(0, dtype=wp.dtype, device=wp.device),
                requires_grad=False,
            )
        ws = getattr(l, "weight_scale", None)
        if isinstance(ws, torch.Tensor) and ws.numel() > 0 and not ws.is_meta:
            l.weight_scale = nn.Parameter(
                torch.empty(0, dtype=ws.dtype, device=ws.device),
                requires_grad=False,
            )
        if wp is not None or ws is not None:
            l.quant_method = UnquantizedLinearMethod()


def _merge_replicated_linears(
    layers: List[ReplicatedLinear],
    quant_config: Optional[QuantizationConfig],
    prefix: str,
    existing_runner: Optional[_FusedLinearRunner] = None,
) -> Optional[_FusedLinearRunner]:
    """Merge a group of `ReplicatedLinear` layers sharing the same input_size and quant_method.

    Supported formats (in `bfloat16` only):
    - Unquantized BF16 (`UnquantizedLinearMethod`).
    - Symmetric packed W8A16 / W4A16 (`CompressedTensorsWNA16` with `weight_packed` and
      `weight_scale`, and no `weight_zero_point` or `weight_g_idx`).
    Explicitly skips FP8, asymmetric (`weight_zero_point` / `qzeros`), activation-reordered
    (`weight_g_idx` / `g_idx`), mixed-method groups, and non-BF16 models without freeing
    any original weights.
    """
    if not layers:
        return None
    first = layers[0]
    in_dim = int(first.input_size)
    out_dim = sum(int(l.output_size) for l in layers)
    has_bias = getattr(first, "bias", None) is not None
    qm = getattr(first, "quant_method", None)
    if qm is None:
        return None

    for l in layers:
        if int(l.input_size) != in_dim:
            return None
        if (getattr(l, "bias", None) is not None) != has_bias:
            return None
        if type(getattr(l, "quant_method", None)) is not type(qm):
            return None
        if _is_fp8_linear(l, quant_config):
            return None
        if any(
            _has_non_empty_attr(l, attr)
            for attr in ("weight_zero_point", "weight_g_idx", "qzeros", "g_idx")
        ):
            return None

    if isinstance(qm, UnquantizedLinearMethod):
        for l in layers:
            w = getattr(l, "weight", None)
            if (
                w is None
                or w.numel() == 0
                or w.is_meta
                or w.dtype != torch.bfloat16
            ):
                return None
            if has_bias:
                b = getattr(l, "bias", None)
                if (
                    b is None
                    or b.numel() == 0
                    or b.is_meta
                    or b.dtype != torch.bfloat16
                ):
                    return None
        w_cat = torch.cat([l.weight.data for l in layers], dim=0).contiguous()
        b_cat = (
            torch.cat([l.bias.data for l in layers], dim=0).contiguous()
            if has_bias
            else None
        )
        if (
            existing_runner is not None
            and existing_runner.weight is not None
            and existing_runner.weight.shape == w_cat.shape
            and existing_runner.weight.dtype == w_cat.dtype
            and existing_runner.weight.device == w_cat.device
        ):
            existing_runner.weight.copy_(w_cat)
            if b_cat is not None and existing_runner.bias is not None:
                existing_runner.bias.copy_(b_cat)
            return existing_runner
        return _FusedLinearRunner(weight=w_cat, bias=b_cat)

    if hasattr(first, "weight_packed") and hasattr(first, "weight_scale"):
        if existing_runner is not None:
            raise RuntimeError(
                "Cannot rebuild merged quantized projections in-place: "
                "SGLANG_RECKONER_KEEP_UNMERGED_WEIGHTS=1 rebuilds are supported only for "
                "unquantized bfloat16 weights."
            )
        for l in layers:
            if not (hasattr(l, "weight_packed") and hasattr(l, "weight_scale")):
                return None
            wp = l.weight_packed
            ws = l.weight_scale
            if (
                wp is None
                or ws is None
                or wp.numel() == 0
                or ws.numel() == 0
                or wp.is_meta
                or ws.is_meta
                or ws.dtype != torch.bfloat16
                or wp.dtype != first.weight_packed.dtype
                or wp.shape[1:] != first.weight_packed.shape[1:]
                or ws.shape[1:] != first.weight_scale.shape[1:]
            ):
                return None
            if has_bias:
                b = getattr(l, "bias", None)
                if (
                    b is None
                    or b.numel() == 0
                    or b.is_meta
                    or b.dtype != torch.bfloat16
                ):
                    return None
        wp_cat = torch.cat([l.weight_packed.data for l in layers], dim=0).contiguous()
        ws_cat = torch.cat([l.weight_scale.data for l in layers], dim=0).contiguous()
        merged = _build_replicated_linear(
            in_dim, out_dim, bias=has_bias, quant_config=quant_config, prefix=prefix
        ).to(device=wp_cat.device)
        if not (hasattr(merged, "weight_packed") and hasattr(merged, "weight_scale")):
            return None
        if any(
            _has_non_empty_attr(merged, attr)
            for attr in ("weight_zero_point", "weight_g_idx", "qzeros", "g_idx")
        ):
            return None
        merged.weight_packed.data = wp_cat
        merged.weight_scale.data = ws_cat
        if hasattr(merged, "weight_shape") and hasattr(first, "weight_shape"):
            merged.weight_shape.data = torch.tensor(
                [out_dim, in_dim],
                dtype=first.weight_shape.dtype,
                device=first.weight_shape.device,
            )
        if has_bias and merged.bias is not None:
            b_cat = torch.cat([l.bias.data for l in layers], dim=0).contiguous()
            merged.bias.data = b_cat
        if wp_cat.is_cuda and getattr(merged, "quant_method", None) is not None:
            merged.quant_method.process_weights_after_loading(merged)
        return _FusedLinearRunner(quant_mod=merged)

    return None


class ReckonerRMSNorm(nn.Module):
    """RMSNorm matching `model/ladder_model.py:RMSNorm` on both CUDA (`sgl_kernel.rmsnorm`) and CPU."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda and sgl_rmsnorm is not None and x.numel() > 0:
            orig_shape = x.shape
            x_2d = x.contiguous().view(-1, self.dim)
            return sgl_rmsnorm(x_2d, self.weight.data, self.eps).view(orig_shape)
        orig_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        var = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_fp32 * torch.rsqrt(var + self.eps)
        return (self.weight * x_norm).to(orig_dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


class ReckonerRotaryEmbedding(nn.Module):
    """CUDA-graph-safe partial RoPE matching `model/ladder_model.py:RotaryEmbedding` bit-for-bit."""

    def __init__(
        self,
        rotary_dim: int = 64,
        max_position_embeddings: int = 32768,
        base: float = 500000.0,
    ):
        super().__init__()
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max(int(max_position_embeddings), 163840)
        self.base = float(base)

        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim)
        )
        t = torch.arange(self.max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # q: [total_tokens, n_heads, head_dim], k: [total_tokens, n_kv_heads, head_dim]
        cos = self.cos_cached[positions].to(dtype=q.dtype).unsqueeze(1)
        sin = self.sin_cached[positions].to(dtype=q.dtype).unsqueeze(1)

        if self.rotary_dim < q.shape[-1]:
            q_rot, q_pass = q[..., : self.rotary_dim], q[..., self.rotary_dim :]
            k_rot, k_pass = k[..., : self.rotary_dim], k[..., self.rotary_dim :]
            q_rot = (q_rot * cos) + (_rotate_half(q_rot) * sin)
            k_rot = (k_rot * cos) + (_rotate_half(k_rot) * sin)
            q = torch.cat((q_rot, q_pass), dim=-1)
            k = torch.cat((k_rot, k_pass), dim=-1)
        else:
            q = (q * cos) + (_rotate_half(q) * sin)
            k = (k * cos) + (_rotate_half(k) * sin)
        return q, k


class ReckonerMLP(nn.Module):
    """SwiGLU MLP matching `model/ladder_model.py:SwiGLUMLP`."""

    def __init__(
        self,
        d_model: int,
        intermediate_size: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.d_model = d_model
        self.intermediate_size = intermediate_size
        self.quant_config = quant_config
        self.prefix = prefix
        self.gate_proj = _build_replicated_linear(
            d_model,
            intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_proj", prefix),
        )
        self.up_proj = _build_replicated_linear(
            d_model,
            intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("up_proj", prefix),
        )
        self.down_proj = _build_replicated_linear(
            intermediate_size,
            d_model,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        self._fused_gate_up_proj: Optional[_FusedLinearRunner] = None
        self._f2_merge_attempted: bool = False
        self._unmerged_freed: bool = False
        self._disable_f2: bool = False

    def build_fused_projections(self, free_unmerged: bool = False) -> bool:
        self._f2_merge_attempted = True
        if self._disable_f2:
            self._fused_gate_up_proj = None
            return False
        proj_list = [self.gate_proj, self.up_proj]
        merged = _merge_replicated_linears(
            proj_list,
            quant_config=self.quant_config,
            prefix=add_prefix("_fused_gate_up_proj", self.prefix),
            existing_runner=self._fused_gate_up_proj,
        )
        if merged is None:
            self._fused_gate_up_proj = None
            return False
        self._fused_gate_up_proj = merged
        if free_unmerged:
            _free_replicated_linears(proj_list)
            self._unmerged_freed = True
        return True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.bfloat16 and is_fusion_step_enabled("f2") and not self._disable_f2:
            if self._fused_gate_up_proj is None and not self._f2_merge_attempted:
                self.build_fused_projections(free_unmerged=False)
            elif (
                self._fused_gate_up_proj is not None
                and not self._fused_gate_up_proj.matches_tensor(x)
                and not self._unmerged_freed
            ):
                self.build_fused_projections(free_unmerged=False)
            if self._fused_gate_up_proj is not None and self._fused_gate_up_proj.matches_tensor(x):
                gate_up, _ = self._fused_gate_up_proj(x)
                gate = gate_up[..., : self.intermediate_size]
                up = gate_up[..., self.intermediate_size :]
            else:
                gate, _ = self.gate_proj(x)
                up, _ = self.up_proj(x)
        else:
            gate, _ = self.gate_proj(x)
            up, _ = self.up_proj(x)

        if (
            x.is_cuda
            and x.dtype == torch.bfloat16
            and x.numel() > 0
            and is_fusion_step_enabled("f3")
        ):
            act = reckoner_fused_silu_mul_fwd(gate, up)
        else:
            act = F.silu(gate) * up
        down, _ = self.down_proj(act)
        return down


class ReckonerFullAttentionLayer(nn.Module):
    """Full or sliding-window GQA attention layer for SGLang (`RadixAttention`)."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        sliding_window: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_id = layer_id
        self.d_model = getattr(config, "d_model", config.hidden_size)
        self.n_heads = getattr(config, "n_heads", config.num_attention_heads)
        self.n_kv_heads = getattr(config, "n_kv_heads", config.num_key_value_heads)
        self.head_dim = getattr(config, "head_dim", self.d_model // self.n_heads)
        self.q_dim = self.n_heads * self.head_dim
        self.kv_dim = self.n_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.quant_config = quant_config
        self.prefix = prefix

        self.q_proj = _build_replicated_linear(
            self.d_model,
            self.q_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("q_proj", prefix),
        )
        self.k_proj = _build_replicated_linear(
            self.d_model,
            self.kv_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("k_proj", prefix),
        )
        self.v_proj = _build_replicated_linear(
            self.d_model,
            self.kv_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("v_proj", prefix),
        )
        self.o_proj = _build_replicated_linear(
            self.q_dim,
            self.d_model,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

        self.qk_norm = bool(getattr(config, "qk_norm", True))
        if self.qk_norm:
            self.q_norm = ReckonerRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = ReckonerRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.use_gated_output = bool(getattr(config, "use_gated_attn_output", True))
        self.gate_pre_oproj = (
            getattr(config, "gated_attn_position", "pre_oproj") == "pre_oproj"
        )
        if self.use_gated_output:
            gate_width = self.q_dim if self.gate_pre_oproj else self.d_model
            self.attn_gate = _build_replicated_linear(
                self.d_model,
                gate_width,
                bias=False,
                quant_config=None,
                prefix=add_prefix("attn_gate", prefix),
            )
        else:
            self.attn_gate = None

        self._fused_qkv_proj: Optional[_FusedLinearRunner] = None
        self._fused_includes_gate: bool = False
        self._f2_merge_attempted: bool = False
        self._unmerged_freed: bool = False
        self._disable_f2: bool = False

        swa_size = sliding_window if sliding_window is not None else -1
        self.attn = RadixAttention(
            num_heads=self.n_heads,
            head_dim=self.head_dim,
            scaling=self.scaling,
            num_kv_heads=self.n_kv_heads,
            layer_id=layer_id,
            sliding_window_size=swa_size,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def build_fused_projections(self, free_unmerged: bool = False) -> bool:
        self._f2_merge_attempted = True
        if self._disable_f2:
            self._fused_qkv_proj = None
            self._fused_includes_gate = False
            return False
        can_include_gate = (
            self.attn_gate is not None
            and isinstance(getattr(self.q_proj, "quant_method", None), UnquantizedLinearMethod)
            and isinstance(getattr(self.attn_gate, "quant_method", None), UnquantizedLinearMethod)
        )
        proj_list = [self.q_proj, self.k_proj, self.v_proj]
        if can_include_gate:
            proj_list.append(self.attn_gate)
        merged = _merge_replicated_linears(
            proj_list,
            quant_config=self.quant_config,
            prefix=add_prefix("_fused_qkv_proj", self.prefix),
            existing_runner=self._fused_qkv_proj,
        )
        if merged is None:
            self._fused_qkv_proj = None
            self._fused_includes_gate = False
            return False
        self._fused_qkv_proj = merged
        self._fused_includes_gate = bool(can_include_gate)
        if free_unmerged:
            _free_replicated_linears(proj_list)
            self._unmerged_freed = True
        return True

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        rotary_emb: ReckonerRotaryEmbedding,
    ) -> torch.Tensor:
        gate_out: Optional[torch.Tensor] = None
        if (
            hidden_states.dtype == torch.bfloat16
            and is_fusion_step_enabled("f2")
            and not self._disable_f2
        ):
            if self._fused_qkv_proj is None and not self._f2_merge_attempted:
                self.build_fused_projections(free_unmerged=False)
            elif (
                self._fused_qkv_proj is not None
                and not self._fused_qkv_proj.matches_tensor(hidden_states)
                and not self._unmerged_freed
            ):
                self.build_fused_projections(free_unmerged=False)
            if self._fused_qkv_proj is not None and self._fused_qkv_proj.matches_tensor(hidden_states):
                qkv, _ = self._fused_qkv_proj(hidden_states)
                o0 = self.q_dim
                o1 = o0 + self.kv_dim
                o2 = o1 + self.kv_dim
                q = qkv[:, :o0].view(-1, self.n_heads, self.head_dim)
                k = qkv[:, o0:o1].view(-1, self.n_kv_heads, self.head_dim)
                v = qkv[:, o1:o2].contiguous().view(-1, self.n_kv_heads, self.head_dim)
                if self._fused_includes_gate:
                    gate_out = qkv[:, o2:]
            else:
                q_raw, _ = self.q_proj(hidden_states)
                k_raw, _ = self.k_proj(hidden_states)
                v_raw, _ = self.v_proj(hidden_states)
                q = q_raw.view(-1, self.n_heads, self.head_dim)
                k = k_raw.view(-1, self.n_kv_heads, self.head_dim)
                v = v_raw.view(-1, self.n_kv_heads, self.head_dim)
        else:
            q_raw, _ = self.q_proj(hidden_states)
            k_raw, _ = self.k_proj(hidden_states)
            v_raw, _ = self.v_proj(hidden_states)
            q = q_raw.view(-1, self.n_heads, self.head_dim)
            k = k_raw.view(-1, self.n_kv_heads, self.head_dim)
            v = v_raw.view(-1, self.n_kv_heads, self.head_dim)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if (
            hidden_states.is_cuda
            and q.dtype == torch.bfloat16
            and k.dtype == torch.bfloat16
            and hidden_states.numel() > 0
            and is_fusion_step_enabled("f4")
        ):
            if rotary_emb.cos_cached.device != hidden_states.device:
                rotary_emb.cos_cached = rotary_emb.cos_cached.to(hidden_states.device)
                rotary_emb.sin_cached = rotary_emb.sin_cached.to(hidden_states.device)
            q, k = reckoner_fused_partial_rope_inplace_fwd(
                q,
                k,
                positions,
                rotary_dim=rotary_emb.rotary_dim,
                cos_cached=rotary_emb.cos_cached,
                sin_cached=rotary_emb.sin_cached,
            )
        else:
            q, k = rotary_emb(positions, q, k)

        q_flat = q.reshape(-1, self.q_dim)

        if not getattr(forward_batch, "_standalone_cpu", False):
            try:
                has_ctx = get_attn_backend() is not None
            except Exception:
                has_ctx = False
            if has_ctx:
                attn_out = self.attn(q_flat, k, v, forward_batch)
            else:
                attn_out = forward_batch.attn_backend.forward(
                    q_flat, k, v, self.attn, forward_batch
                )
        else:
            attn_out = forward_batch.attn_backend.forward(
                q_flat, k, v, self.attn, forward_batch
            )

        attn_out = attn_out.reshape(-1, self.q_dim)
        if self.attn_gate is not None and self.gate_pre_oproj:
            if gate_out is None:
                gate_out, _ = self.attn_gate(hidden_states)
            if (
                attn_out.is_cuda
                and attn_out.dtype == torch.bfloat16
                and gate_out.dtype == torch.bfloat16
                and attn_out.numel() > 0
                and is_fusion_step_enabled("f3")
            ):
                attn_out = reckoner_fused_sigmoid_mul_fwd(attn_out, gate_out, scale=2.0)
            else:
                attn_out = attn_out * (2.0 * torch.sigmoid(gate_out))
        out, _ = self.o_proj(attn_out)
        if self.attn_gate is not None and not self.gate_pre_oproj:
            if gate_out is None:
                gate_out, _ = self.attn_gate(hidden_states)
            if (
                out.is_cuda
                and out.dtype == torch.bfloat16
                and gate_out.dtype == torch.bfloat16
                and out.numel() > 0
                and is_fusion_step_enabled("f3")
            ):
                out = reckoner_fused_sigmoid_mul_fwd(out, gate_out, scale=2.0)
            else:
                out = out * (2.0 * torch.sigmoid(gate_out))
        return out


class ReckonerGDN2Layer(nn.Module):
    """GDN-2 linear attention layer for SGLang (`ReckonerGDN2AttnBackend`)."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_id = layer_id
        self.d_model = getattr(config, "d_model", config.hidden_size)
        self.n_heads = getattr(config, "n_heads", config.num_attention_heads)
        self.head_dim = getattr(config, "head_dim", self.d_model // self.n_heads)
        self.n_v_heads = getattr(config, "gdn_v_heads", None) or self.n_heads
        self.conv_kernel_size = int(getattr(config, "conv_kernel", 4))
        self.eps = float(getattr(config, "rms_norm_eps", 1e-6))
        self.quant_config = quant_config
        self.prefix = prefix

        self.q_dim = self.n_heads * self.head_dim
        self.k_dim = self.n_heads * self.head_dim
        self.v_dim = self.n_v_heads * self.head_dim
        self.qkv_dim = self.q_dim + self.k_dim + self.v_dim

        gate_quant_config = (
            None
            if bool(getattr(config, "keep_gdn_gates_bf16", False))
            else quant_config
        )
        self.gate_quant_config = gate_quant_config

        self.q_proj = _build_replicated_linear(
            self.d_model,
            self.q_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("q_proj", prefix),
        )
        self.k_proj = _build_replicated_linear(
            self.d_model,
            self.k_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("k_proj", prefix),
        )
        self.v_proj = _build_replicated_linear(
            self.d_model,
            self.v_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("v_proj", prefix),
        )

        self.g_proj = _build_replicated_linear(
            self.d_model,
            self.k_dim,
            bias=True,
            quant_config=gate_quant_config,
            prefix=add_prefix("g_proj", prefix),
        )
        self.b_proj = _build_replicated_linear(
            self.d_model,
            self.k_dim,
            bias=True,
            quant_config=gate_quant_config,
            prefix=add_prefix("b_proj", prefix),
        )
        self.w_proj = _build_replicated_linear(
            self.d_model,
            self.v_dim,
            bias=True,
            quant_config=gate_quant_config,
            prefix=add_prefix("w_proj", prefix),
        )

        self.z_proj = _build_replicated_linear(
            self.d_model,
            self.v_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("z_proj", prefix),
        )
        self.out_proj = _build_replicated_linear(
            self.v_dim,
            self.d_model,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("out_proj", prefix),
        )

        self._fused_qkvz_proj: Optional[_FusedLinearRunner] = None
        self._fused_gbw_proj: Optional[_FusedLinearRunner] = None
        self._f2_merge_attempted: bool = False
        self._unmerged_freed: bool = False
        self._disable_f2: bool = False

        # Matches fla ShortConvolution parameter layout: qkv_conv.weight [qkv_dim, 1, conv_kernel_size]
        self.qkv_conv = nn.Conv1d(
            in_channels=self.qkv_dim,
            out_channels=self.qkv_dim,
            kernel_size=self.conv_kernel_size,
            groups=self.qkv_dim,
            padding=self.conv_kernel_size - 1,
            bias=False,
        )

        if FusedRMSNormGated is not None:
            self.norm_gate = FusedRMSNormGated(hidden_size=self.v_dim, eps=self.eps)
        else:
            self.norm_gate = ReckonerRMSNorm(self.v_dim, eps=self.eps)

        self.attn = RadixLinearAttention(
            layer_id=layer_id,
            num_q_heads=self.n_heads,
            num_k_heads=self.n_heads,
            num_v_heads=self.n_v_heads,
            head_q_dim=self.head_dim,
            head_k_dim=self.head_dim,
            head_v_dim=self.head_dim,
            conv_weights=self.qkv_conv.weight.squeeze(1),
            bias=self.qkv_conv.bias,
            activation=None,
        )
        # Set AttentionType.LINEAR_ATTENTION if present for compatibility
        if hasattr(AttentionType, "LINEAR_ATTENTION"):
            self.attn.attn_type = AttentionType.LINEAR_ATTENTION

    def build_fused_projections(self, free_unmerged: bool = False) -> bool:
        """Build both `_fused_qkvz_proj` and `_fused_gbw_proj` atomically.

        Originals are freed ONLY after both merges in the layer succeed; if either merge fails
        (e.g., FP8 + `keep_gdn_gates_bf16=True`, or an ignore list that splits a group), both
        fused runners are cleared to `None` and the whole layer falls back to unfused projections.
        """
        self._f2_merge_attempted = True
        if self._disable_f2:
            self._fused_qkvz_proj = None
            self._fused_gbw_proj = None
            return False
        qkvz_list = [self.q_proj, self.k_proj, self.v_proj, self.z_proj]
        gbw_list = [self.g_proj, self.b_proj, self.w_proj]
        fused_qkvz = _merge_replicated_linears(
            qkvz_list,
            quant_config=self.quant_config,
            prefix=add_prefix("_fused_qkvz_proj", self.prefix),
            existing_runner=self._fused_qkvz_proj,
        )
        fused_gbw = _merge_replicated_linears(
            gbw_list,
            quant_config=self.gate_quant_config,
            prefix=add_prefix("_fused_gbw_proj", self.prefix),
            existing_runner=self._fused_gbw_proj,
        )
        if fused_qkvz is None or fused_gbw is None:
            self._fused_qkvz_proj = None
            self._fused_gbw_proj = None
            return False
        self._fused_qkvz_proj = fused_qkvz
        self._fused_gbw_proj = fused_gbw
        if free_unmerged:
            _free_replicated_linears(qkvz_list + gbw_list)
            self._unmerged_freed = True
        return True

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if (
            hidden_states.dtype == torch.bfloat16
            and is_fusion_step_enabled("f2")
            and not self._disable_f2
        ):
            if (
                (self._fused_qkvz_proj is None or self._fused_gbw_proj is None)
                and not self._f2_merge_attempted
            ):
                self.build_fused_projections(free_unmerged=False)
            elif (
                self._fused_qkvz_proj is not None
                and self._fused_gbw_proj is not None
                and (
                    not self._fused_qkvz_proj.matches_tensor(hidden_states)
                    or not self._fused_gbw_proj.matches_tensor(hidden_states)
                )
                and not self._unmerged_freed
            ):
                self.build_fused_projections(free_unmerged=False)
            if (
                self._fused_qkvz_proj is not None
                and self._fused_gbw_proj is not None
                and self._fused_qkvz_proj.matches_tensor(hidden_states)
                and self._fused_gbw_proj.matches_tensor(hidden_states)
            ):
                qkvz, _ = self._fused_qkvz_proj(hidden_states)
                mixed_qkv = qkvz[:, : self.qkv_dim]
                z = qkvz[:, self.qkv_dim :]
                gbw, _ = self._fused_gbw_proj(hidden_states)
                g_raw = gbw[:, : self.k_dim]
                b_raw = gbw[:, self.k_dim : 2 * self.k_dim]
                w_raw = gbw[:, 2 * self.k_dim :]
            else:
                q, _ = self.q_proj(hidden_states)
                k, _ = self.k_proj(hidden_states)
                v, _ = self.v_proj(hidden_states)
                mixed_qkv = torch.cat((q, k, v), dim=-1)
                g_raw, _ = self.g_proj(hidden_states)
                b_raw, _ = self.b_proj(hidden_states)
                w_raw, _ = self.w_proj(hidden_states)
                z, _ = self.z_proj(hidden_states)
        else:
            q, _ = self.q_proj(hidden_states)
            k, _ = self.k_proj(hidden_states)
            v, _ = self.v_proj(hidden_states)
            mixed_qkv = torch.cat((q, k, v), dim=-1)
            g_raw, _ = self.g_proj(hidden_states)
            b_raw, _ = self.b_proj(hidden_states)
            w_raw, _ = self.w_proj(hidden_states)
            z, _ = self.z_proj(hidden_states)

        if hidden_states.dtype == torch.bfloat16 and is_fusion_step_enabled("f1"):
            mamba_input = {
                "mixed_qkv": mixed_qkv,
                "conv_weights": self.qkv_conv.weight.squeeze(1),
                "bias": self.qkv_conv.bias,
                "activation": None,
                "key_dim": self.k_dim,
                "value_dim": self.v_dim,
                "num_k_heads": self.n_heads,
                "num_v_heads": self.n_v_heads,
                "head_k_dim": self.head_dim,
                "head_v_dim": self.head_dim,
                "g": g_raw,
                "b": b_raw,
                "w": w_raw,
                "layer_id": self.layer_id,
                "gate_mode": "1.4b",
            }
        else:
            g = -torch.exp(g_raw.float().clamp(max=5.0)).to(
                hidden_states.dtype
            )
            b = torch.sigmoid(b_raw).to(hidden_states.dtype)
            w = torch.sigmoid(w_raw).to(hidden_states.dtype)
            mamba_input = {
                "mixed_qkv": mixed_qkv,
                "conv_weights": self.qkv_conv.weight.squeeze(1),
                "bias": self.qkv_conv.bias,
                "activation": None,
                "key_dim": self.k_dim,
                "value_dim": self.v_dim,
                "num_k_heads": self.n_heads,
                "num_v_heads": self.n_v_heads,
                "head_k_dim": self.head_dim,
                "head_v_dim": self.head_dim,
                "g": g,
                "b": b,
                "w": w,
                "layer_id": self.layer_id,
            }

        try:
            backend = get_attn_backend() or forward_batch.attn_backend
        except Exception:
            backend = forward_batch.attn_backend
        core_out = backend.forward(
            q=None,
            k=None,
            v=None,
            layer=self.attn,
            forward_batch=forward_batch,
            save_kv_cache=False,
            mixed_qkv=mixed_qkv,
            mamba_Input=mamba_input,
        )
        out = core_out.reshape(-1, self.v_dim)

        if hidden_states.is_cuda and FusedRMSNormGated is not None:
            out = self.norm_gate(out, z)
        else:
            eps = getattr(self.norm_gate, "eps", self.eps)
            w_norm = getattr(self.norm_gate, "weight", None)
            out_norm = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + eps)
            if w_norm is not None:
                out_norm = out_norm * w_norm
            out = out_norm * F.silu(z)

        out_proj, _ = self.out_proj(out)
        return out_proj


class ReckonerDecoderLayer(nn.Module):
    """Single hybrid decoder block (`op_type` in `{'A', 'S', 'G'}`)."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx: int,
        op_type: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.op_type = op_type
        d_model = getattr(config, "d_model", config.hidden_size)
        self.input_layernorm = ReckonerRMSNorm(d_model, eps=config.rms_norm_eps)

        if op_type == "A":
            self.mixer = ReckonerFullAttentionLayer(
                config=config,
                layer_id=layer_idx,
                sliding_window=None,
                quant_config=quant_config,
                prefix=add_prefix("mixer", prefix),
            )
        elif op_type == "S":
            self.mixer = ReckonerFullAttentionLayer(
                config=config,
                layer_id=layer_idx,
                sliding_window=getattr(config, "sliding_window", 512),
                quant_config=quant_config,
                prefix=add_prefix("mixer", prefix),
            )
        elif op_type == "G":
            self.mixer = ReckonerGDN2Layer(
                config=config,
                layer_id=layer_idx,
                quant_config=quant_config,
                prefix=add_prefix("mixer", prefix),
            )
        else:
            raise ValueError(f"Unsupported layer operator '{op_type}'")

        self.post_attention_layernorm = ReckonerRMSNorm(
            d_model, eps=config.rms_norm_eps
        )
        self.mlp = ReckonerMLP(
            d_model,
            config.intermediate_size,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        rotary_emb: ReckonerRotaryEmbedding,
    ) -> torch.Tensor:
        normed = self.input_layernorm(hidden_states)
        if self.op_type in ("A", "S"):
            mixer_out = self.mixer(
                positions=positions,
                hidden_states=normed,
                forward_batch=forward_batch,
                rotary_emb=rotary_emb,
            )
        else:
            mixer_out = self.mixer(
                hidden_states=normed,
                forward_batch=forward_batch,
            )
        hidden_states = hidden_states + mixer_out
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        return hidden_states


class ReckonerForCausalLM(nn.Module):
    """Out-of-tree SGLang model class for Reckoner-1.4B (`Trunk@S3`)."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        attn_backend: Optional[Any] = None,
    ):
        super().__init__()
        try:
            from . import set_active_model_arch

            set_active_model_arch("ReckonerForCausalLM")
        except Exception:
            pass
        self.config = config
        self.quant_config = quant_config
        self.default_attn_backend = attn_backend
        self.d_model = getattr(config, "d_model", config.hidden_size)
        self.vocab_size = config.vocab_size
        self._unmerged_weights_freed = False
        self._quantized_projections_merged = False

        try:
            self.embed_tokens: nn.Module = VocabParallelEmbedding(
                config.vocab_size,
                self.d_model,
                prefix=add_prefix("embed_tokens", prefix),
            )
            self._sglang_parallel_active = True
        except Exception:
            self.embed_tokens = nn.Embedding(config.vocab_size, self.d_model)
            self._sglang_parallel_active = False

        rotary_dim = getattr(config, "rotary_dim", None) or getattr(
            config, "head_dim", 128
        )
        self.rotary_emb = ReckonerRotaryEmbedding(
            rotary_dim=rotary_dim,
            max_position_embeddings=max(
                int(getattr(config, "max_position_embeddings", 163840)), 163840
            ),
            base=getattr(config, "rope_theta", 500000.0),
        )

        raw_layout = getattr(config, "layer_layout", "AAGGGGGGGGGGGGGAGAGAGA")
        if isinstance(raw_layout, str) and "," not in raw_layout:
            layout = [ch for ch in raw_layout.strip().upper() if not ch.isspace()]
        elif isinstance(raw_layout, str):
            layout = [tok.strip().upper() for tok in raw_layout.split(",") if tok.strip()]
        else:
            layout = list(raw_layout)

        self.layers = nn.ModuleList(
            [
                ReckonerDecoderLayer(
                    config=config,
                    layer_idx=idx,
                    op_type=op,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{idx}", prefix),
                )
                for idx, op in enumerate(layout)
            ]
        )
        self.norm = ReckonerRMSNorm(self.d_model, eps=config.rms_norm_eps)

        if getattr(config, "tie_word_embeddings", True):
            self.lm_head: nn.Module = self.embed_tokens
        elif self._sglang_parallel_active:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                self.d_model,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )
        else:
            self.lm_head = nn.Linear(self.d_model, config.vocab_size, bias=False)

        self.logits_processor = LogitsProcessor(config) if self._sglang_parallel_active else None
        self._record_extend_timing = (
            os.environ.get("RECKONER_RECORD_EXTEND_TIMING", "0").strip() == "1"
        )
        self._extend_trace_path = os.environ.get(
            "RECKONER_EXTEND_TRACE_PATH", "/dev/shm/reckoner_sgl_extend_trace.jsonl"
        )

        quantized_count = 0
        total_linear_count = 0
        method_names: set[str] = set()
        scheme_names: set[str] = set()
        for mod in self.modules():
            if isinstance(mod, ReplicatedLinear):
                total_linear_count += 1
                qm = getattr(mod, "quant_method", None)
                if qm is not None and not isinstance(qm, UnquantizedLinearMethod):
                    quantized_count += 1
                    method_names.add(type(qm).__name__)
                    scheme = getattr(mod, "scheme", None)
                    if scheme is not None:
                        scheme_names.add(type(scheme).__name__)
        qc_name = type(quant_config).__name__ if quant_config is not None else "None"
        methods_str = (
            ",".join(sorted(method_names))
            if method_names
            else "UnquantizedLinearMethod"
        )
        schemes_str = ",".join(sorted(scheme_names)) if scheme_names else "None"
        logger.info(
            "ReckonerForCausalLM init: file=%s quant_config=%s quant_method=%s scheme=%s quantized_projections=%d/%d",
            __file__,
            qc_name,
            methods_str,
            schemes_str,
            quantized_count,
            total_linear_count,
        )

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if getattr(forward_batch, "attn_backend", None) is None and self.default_attn_backend is not None:
            forward_batch.attn_backend = self.default_attn_backend

        is_Standalone_cpu = bool(getattr(forward_batch, "_standalone_cpu", False))
        do_trace_extend = (
            self._record_extend_timing
            and not is_Standalone_cpu
            and input_ids.is_cuda
            and hasattr(forward_batch, "forward_mode")
            and forward_batch.forward_mode.is_extend()
        )
        if do_trace_extend:
            torch.cuda.synchronize(input_ids.device)
            t_ext_start = time.perf_counter()

        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds

        for layer in self.layers:
            hidden_states = layer(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                rotary_emb=self.rotary_emb,
            )

        hidden_states = self.norm(hidden_states)
        if self.logits_processor is None or is_Standalone_cpu:
            return F.linear(hidden_states, self.lm_head.weight)
        out = self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )
        if do_trace_extend:
            torch.cuda.synchronize(input_ids.device)
            extend_ms = (time.perf_counter() - t_ext_start) * 1000.0
            try:
                prefix_lens = [
                    int(x) for x in (getattr(forward_batch, "extend_prefix_lens_cpu", None) or [])
                ]
                extend_lens = [
                    int(x) for x in (getattr(forward_batch, "extend_seq_lens_cpu", None) or [])
                ]
                rec = {
                    "ts_end": time.time(),
                    "extend_ms": round(extend_ms, 4),
                    "batch_size": int(getattr(forward_batch, "batch_size", len(extend_lens) or 1)),
                    "extend_num_tokens": int(input_ids.shape[0]),
                    "prefix_lens": prefix_lens,
                    "extend_lens": extend_lens,
                }
                with open(self._extend_trace_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
            except OSError:
                pass
        return out

    def _has_quantized_merged_projections(self) -> bool:
        if self._quantized_projections_merged:
            return True
        for layer in self.layers:
            for obj in (layer.mixer, layer.mlp):
                for attr in (
                    "_fused_qkvz_proj",
                    "_fused_gbw_proj",
                    "_fused_qkv_proj",
                    "_fused_gate_up_proj",
                ):
                    runner = getattr(obj, attr, None)
                    if runner is not None and getattr(runner, "quant_mod", None) is not None:
                        return True
        return False

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set[str]:
        if self._unmerged_weights_freed:
            raise RuntimeError(
                "Cannot call load_weights / update_weights_from_tensor after unmerged projection "
                "weights were freed by f2 fusion. Set SGLANG_RECKONER_RL_WEIGHT_SYNC=1 (or "
                "SGLANG_RECKONER_KEEP_UNMERGED_WEIGHTS=1 on unquantized bfloat16 models) when starting "
                "the engine for live RL weight sync."
            )
        if self._has_quantized_merged_projections():
            raise RuntimeError(
                "Cannot call load_weights / update_weights_from_tensor after quantized f2 projection "
                "merging: SGLANG_RECKONER_KEEP_UNMERGED_WEIGHTS=1 rebuilds are restricted to unquantized "
                "bfloat16 weights (reloading quantized weights would re-concatenate Marlin-repacked "
                "weight_packed tensors and invalidate captured CUDA graphs). Set "
                "SGLANG_RECKONER_RL_WEIGHT_SYNC=1 or disable f2 before initial weight loading."
            )
        params_dict: Dict[str, nn.Parameter] = dict(
            self.named_parameters(remove_duplicate=False)
        )
        tie_word_embeddings = bool(getattr(self.config, "tie_word_embeddings", True))
        loaded_params: set[str] = set()

        for raw_name, loaded_weight in weights:
            name = (
                raw_name.replace("_orig_mod.", "")
                .replace("trunk.", "")
                .removeprefix("model.")
            )
            if "rotary_emb." in name:
                continue
            if tie_word_embeddings and name == "lm_head.weight":
                if "embed_tokens.weight" in params_dict:
                    param = params_dict["embed_tokens.weight"]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add("embed_tokens.weight")
                continue

            if name not in params_dict:
                raise KeyError(
                    f"Unexpected weight key '{raw_name}' (normalized '{name}') not in ReckonerForCausalLM"
                )
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

        weight_sync = is_weight_sync_mode(self.config)
        keep_unmerged = (
            weight_sync
            or os.environ.get("SGLANG_RECKONER_KEEP_UNMERGED_WEIGHTS", "0").strip() == "1"
            or bool(getattr(self.config, "keep_unmerged_weights", False))
        )
        for layer in self.layers:
            layer.mixer._disable_f2 = weight_sync
            layer.mlp._disable_f2 = weight_sync
            layer.mixer._f2_merge_attempted = False
            layer.mlp._f2_merge_attempted = False

        if is_fusion_step_enabled("f2") and not weight_sync:
            self.build_fused_projections(
                free_unmerged=self._sglang_parallel_active and not keep_unmerged
            )
        else:
            for layer in self.layers:
                if hasattr(layer.mixer, "_fused_qkvz_proj"):
                    layer.mixer._fused_qkvz_proj = None
                    layer.mixer._fused_gbw_proj = None
                if hasattr(layer.mixer, "_fused_qkv_proj"):
                    layer.mixer._fused_qkv_proj = None
                    layer.mixer._fused_includes_gate = False
                layer.mlp._fused_gate_up_proj = None

        return loaded_params

    def build_fused_projections(self, free_unmerged: bool = False) -> None:
        if is_weight_sync_mode(self.config):
            return
        keep_unmerged = (
            os.environ.get("SGLANG_RECKONER_KEEP_UNMERGED_WEIGHTS", "0").strip() == "1"
            or bool(getattr(self.config, "keep_unmerged_weights", False))
        )
        do_free = bool(free_unmerged and not keep_unmerged)
        any_freed = False
        for layer in self.layers:
            if hasattr(layer.mixer, "build_fused_projections"):
                if layer.mixer.build_fused_projections(free_unmerged=do_free) and do_free:
                    any_freed = True
            if layer.mlp.build_fused_projections(free_unmerged=do_free) and do_free:
                any_freed = True
        if any_freed:
            self._unmerged_weights_freed = True
        if self._has_quantized_merged_projections():
            self._quantized_projections_merged = True


EntryClass = ReckonerForCausalLM
