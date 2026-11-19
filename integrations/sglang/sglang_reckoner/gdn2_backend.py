# Copyright 2026 Maelie Labs
# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
#
# Notice of Modification (Apache-2.0 Section 4(b)):
# Modified by Maelie Labs (2026): adapted from SGLang v0.5.22
# `sglang.srt.layers.attention.hybrid_linear_attn_backend.MambaAttnBackendBase`
# to dispatch GDN-2 prefill/extend and decode steps through `fla.ops.gdn2` and custom Triton kernels.
"""
Out-of-tree SGLang Mamba/Linear-Attention backend for Reckoner GDN-2 (Gated DeltaNet 2) on SGLang 0.5.22.

Subclasses `MambaAttnBackendBase` from `sglang.srt.layers.attention.hybrid_linear_attn_backend`
and uses `MambaPool` state management (`conv_states`, `ssm_states`) while dispatching:
- `forward_extend` (prefill/extend) to `causal_conv1d_fn` + `fla.ops.gdn2.chunk_gdn2`
- `forward_decode` (1-token decode) to `causal_conv1d_update` + `fla.ops.gdn2.fused_recurrent.fused_recurrent_gdn2_fwd`
- `_forward_cpu` (CPU unit-test path) to `F.conv1d` + `fla.ops.gdn2.naive.naive_recurrent_gdn2`.
"""

from __future__ import annotations

import logging
import os
from typing import Any, List, Optional, Tuple, Union
import warnings

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase
from sglang.srt.layers.attention.mamba.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

logger = logging.getLogger(__name__)

try:
    from sglang.kernels.ops.mamba.causal_conv1d_triton import (
        CAUSAL_CONV1D_FWD_BLOCK_M,
        causal_conv1d_fn as _causal_conv1d_fn_triton,
    )
except ImportError:
    CAUSAL_CONV1D_FWD_BLOCK_M = 8
    _causal_conv1d_fn_triton = causal_conv1d_fn

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

try:
    from triton.language.extra import libdevice as tl_libdevice
except ImportError:
    tl_libdevice = None

try:
    from fla.ops.utils.op import exp as fla_exp
except ImportError:
    fla_exp = None

try:
    from fla.ops.gdn2 import chunk_gdn2
    from fla.ops.gdn2.fused_recurrent import fused_recurrent_gdn2_fwd
except ImportError:
    chunk_gdn2 = None
    fused_recurrent_gdn2_fwd = None

from fla.ops.gdn2.naive import naive_recurrent_gdn2

VALID_FUSION_MASK_TOKENS: frozenset[str] = frozenset(
    {"all", "default", "no_f2", "bit_exact", "exact", "f1", "f2", "f3", "f4", "none"}
)
_ALL_FUSION_STEPS: frozenset[str] = frozenset({"f1", "f2", "f3", "f4"})
_NO_F2_FUSION_STEPS: frozenset[str] = frozenset({"f1", "f3", "f4"})
_BIT_EXACT_FUSION_STEPS: frozenset[str] = frozenset({"f1", "f3"})
_FUSION_ENV_CACHE: Optional[Tuple[str, str, str, bool, frozenset[str]]] = None
_WARNED_UNKNOWN_MASK_TOKENS: set[str] = set()


def is_weight_sync_mode(config: Optional[Any] = None) -> bool:
    """Return True when the engine is configured for live RL weight synchronization (`update_weights_from_tensor`)."""
    if os.environ.get("SGLANG_RECKONER_RL_WEIGHT_SYNC", "0").strip() == "1":
        return True
    if config is not None and (
        bool(getattr(config, "enable_weight_sync", False))
        or bool(getattr(config, "rl_weight_sync", False))
    ):
        return True
    return False


def _warn_deprecated_exact_mask_once() -> None:
    if "exact" not in _WARNED_UNKNOWN_MASK_TOKENS:
        _WARNED_UNKNOWN_MASK_TOKENS.add("exact")
        msg = (
            "SGLANG_RECKONER_FUSION_MASK='exact' is a deprecated alias for 'no_f2' (f1,f3,f4) "
            "because f4 changes per-head RMSNorm reduction order on Reckoner-Q2B and is not "
            "bit-exact end-to-end; use 'no_f2' for f1,f3,f4, 'bit_exact' for f1,f3, or 'default'/'all' for f1,f2,f3,f4."
        )
        logger.warning(msg)
        warnings.warn(msg, UserWarning, stacklevel=4)


def _get_fusion_env_state() -> Tuple[bool, frozenset[str]]:
    global _FUSION_ENV_CACHE
    raw_gates = os.environ.get("SGLANG_RECKONER_FUSED_GATES", "0")
    raw_mask = os.environ.get("SGLANG_RECKONER_FUSION_MASK", "all")
    raw_rl = os.environ.get("SGLANG_RECKONER_RL_WEIGHT_SYNC", "0")
    if (
        _FUSION_ENV_CACHE is not None
        and _FUSION_ENV_CACHE[0] == raw_gates
        and _FUSION_ENV_CACHE[1] == raw_mask
        and _FUSION_ENV_CACHE[2] == raw_rl
    ):
        return _FUSION_ENV_CACHE[3], _FUSION_ENV_CACHE[4]

    gates_on = raw_gates.strip() == "1"
    mask_clean = raw_mask.strip().lower()
    if not mask_clean or mask_clean in ("all", "default"):
        active_steps: set[str] = set(_ALL_FUSION_STEPS)
    elif mask_clean == "no_f2":
        active_steps = set(_NO_F2_FUSION_STEPS)
    elif mask_clean == "bit_exact":
        active_steps = set(_BIT_EXACT_FUSION_STEPS)
    elif mask_clean == "exact":
        _warn_deprecated_exact_mask_once()
        active_steps = set(_NO_F2_FUSION_STEPS)
    elif mask_clean == "none":
        active_steps = set()
    else:
        tokens = [tok.strip() for tok in mask_clean.split(",") if tok.strip()]
        active_steps = set()
        for tok in tokens:
            if tok not in VALID_FUSION_MASK_TOKENS:
                if tok not in _WARNED_UNKNOWN_MASK_TOKENS:
                    _WARNED_UNKNOWN_MASK_TOKENS.add(tok)
                    msg = (
                        f"Unknown SGLANG_RECKONER_FUSION_MASK token {tok!r}; "
                        f"valid tokens are {sorted(VALID_FUSION_MASK_TOKENS)}"
                    )
                    logger.warning(msg)
                    warnings.warn(msg, UserWarning, stacklevel=3)
            elif tok in ("all", "default"):
                active_steps.update(_ALL_FUSION_STEPS)
            elif tok == "no_f2":
                active_steps.update(_NO_F2_FUSION_STEPS)
            elif tok == "bit_exact":
                active_steps.update(_BIT_EXACT_FUSION_STEPS)
            elif tok == "exact":
                _warn_deprecated_exact_mask_once()
                active_steps.update(_NO_F2_FUSION_STEPS)
            elif tok != "none":
                active_steps.add(tok)

    if raw_rl.strip() == "1":
        active_steps.discard("f2")

    frozen_steps = frozenset(active_steps) if gates_on else frozenset()
    _FUSION_ENV_CACHE = (raw_gates, raw_mask, raw_rl, gates_on, frozen_steps)
    return gates_on, frozen_steps


def is_fused_gates_enabled() -> bool:
    """Return True when `SGLANG_RECKONER_FUSED_GATES=1` is set (default off: `'0'`)."""
    gates_on, _ = _get_fusion_env_state()
    return gates_on


def is_fusion_step_enabled(step_tag: str) -> bool:
    """Return True when `SGLANG_RECKONER_FUSED_GATES=1` and `step_tag` (`'f1'..'f4'`) is active.

    When `SGLANG_RECKONER_FUSION_MASK` is unset or `'all'`, all fusion steps (`f1..f4`) are
    enabled whenever `SGLANG_RECKONER_FUSED_GATES=1` (except `f2` when `SGLANG_RECKONER_RL_WEIGHT_SYNC=1`).
    Setting `SGLANG_RECKONER_FUSION_MASK=exact` enables only `f1,f3,f4`, and setting a
    comma-separated list (e.g. `f1,f2`) enables only those steps for ablation.
    """
    gates_on, active_steps = _get_fusion_env_state()
    if not gates_on:
        return False
    return step_tag.lower() in active_steps


if triton is not None:

    @triton.jit
    def _round_bf16_in_fp32(x_fp32):
        return tl.inline_asm_elementwise(
            "{\n.reg .b16 tmp;\ncvt.rn.bf16.f32 tmp, $1;\ncvt.f32.bf16 $0, tmp;\n}",
            "=f,f",
            [x_fp32],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit(do_not_specialize=["N", "T"])
    def reckoner_fused_recurrent_gdn2_kernel(
        mixed_qkv_conv,
        g,
        b,
        w,
        a_raw,
        b_raw,
        A_log,
        dt_bias,
        o,
        ssm_states,
        cu_seqlens,
        ssm_state_indices,
        has_initial_states,
        scale: tl.constexpr,
        N: tl.int64,
        T: tl.int64,
        H: tl.constexpr,
        HV: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BK: tl.constexpr,
        BV: tl.constexpr,
        stride_qkv_tok: tl.constexpr,
        stride_g_tok: tl.constexpr,
        stride_b_tok: tl.constexpr,
        stride_w_tok: tl.constexpr,
        stride_a_tok: tl.constexpr,
        stride_b_in_tok: tl.constexpr,
        stride_state_slot: tl.constexpr,
        stride_indices_seq: tl.constexpr,
        USE_INITIAL_STATE: tl.constexpr,
        HAS_INIT_STATES_MASK: tl.constexpr,
        IS_VARLEN: tl.constexpr,
        GATE_MODE: tl.constexpr,
        APPLY_QKV_SILU: tl.constexpr,
        num_stages: tl.constexpr,
    ):
        pid = tl.program_id(0)
        NV = tl.cdiv(V, BV)
        NK = tl.cdiv(K, BK)
        i_k = pid % NK
        pid_rest = pid // NK

        i_v = pid_rest % NV
        i_nh = pid_rest // NV
        i_n, i_hv = i_nh // HV, i_nh % HV
        i_h = i_hv // (HV // H)

        if IS_VARLEN:
            bos = tl.load(cu_seqlens + i_n).to(tl.int64)
            eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
            T = eos - bos
        else:
            bos = i_n * T
            eos = bos + T

        if T <= 0:
            return

        o_k = i_k * BK + tl.arange(0, BK)
        o_v = i_v * BV + tl.arange(0, BV)
        mask_k = o_k < K
        mask_v = o_v < V
        mask_h = mask_k[:, None] & mask_v[None, :]

        raw_slot = tl.load(ssm_state_indices + i_n * stride_indices_seq).to(tl.int64)
        safe_slot = tl.maximum(raw_slot, 0)

        b_h = tl.zeros([BK, BV], dtype=tl.float32)
        if USE_INITIAL_STATE:
            load_init = raw_slot >= 0
            if HAS_INIT_STATES_MASK:
                has_init = tl.load(has_initial_states + i_n).to(tl.int1)
                load_init = load_init & has_init
            if load_init:
                p_h0 = (
                    ssm_states
                    + safe_slot * stride_state_slot
                    + i_hv * K * V
                    + o_k[:, None] * V
                    + o_v[None, :]
                )
                b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

        p_qkv_base = mixed_qkv_conv + bos * stride_qkv_tok
        p_q = p_qkv_base + i_h * K + o_k
        p_k = p_qkv_base + (H * K) + i_h * K + o_k
        p_v = p_qkv_base + (2 * H * K) + i_hv * V + o_v

        if GATE_MODE == 2:
            p_g = g + bos * stride_g_tok + i_hv * K + o_k
            p_b = b + bos * stride_b_tok + i_hv * K + o_k
            p_a_in = a_raw + bos * stride_a_tok + i_hv
            p_b_in = b_raw + bos * stride_b_in_tok + i_hv
            a_log_val = tl.load(A_log + i_hv).to(tl.float32)
            neg_A_val = -tl_libdevice.exp(a_log_val)
            dt_bias_val = tl.load(dt_bias + i_hv).to(tl.float32)
        else:
            p_g = g + bos * stride_g_tok + i_h * K + o_k
            p_b = b + bos * stride_b_tok + i_h * K + o_k
            p_a_in = g
            p_b_in = b
            neg_A_val = 0.0
            dt_bias_val = 0.0

        p_w = w + bos * stride_w_tok + i_hv * V + o_v
        p_o = o + (bos * HV + i_hv) * V + o_v

        for _ in tl.range(0, T, num_stages=num_stages):
            b_q = tl.load(p_q, mask=mask_k, other=0, eviction_policy="evict_last").to(tl.float32)
            b_k = tl.load(p_k, mask=mask_k, other=0, eviction_policy="evict_last").to(tl.float32)
            b_v = tl.load(p_v, mask=mask_v, other=0, eviction_policy="evict_first").to(tl.float32)

            if APPLY_QKV_SILU:
                b_q = _round_bf16_in_fp32(b_q / (1.0 + tl_libdevice.exp(-b_q)))
                b_k = _round_bf16_in_fp32(b_k / (1.0 + tl_libdevice.exp(-b_k)))
                b_v = _round_bf16_in_fp32(b_v / (1.0 + tl_libdevice.exp(-b_v)))

            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
            b_q = b_q * scale

            if GATE_MODE == 0:
                b_gk = tl.load(p_g, mask=mask_k, other=0, eviction_policy="evict_last").to(tl.float32)
                b_b = tl.load(p_b, mask=mask_k, other=0, eviction_policy="evict_last").to(tl.float32)
                b_w = tl.load(p_w, mask=mask_v, other=0, eviction_policy="evict_first").to(tl.float32)
            elif GATE_MODE == 1:
                b_g_raw = tl.load(p_g, mask=mask_k, other=0, eviction_policy="evict_last").to(tl.float32)
                b_gk = _round_bf16_in_fp32(-tl_libdevice.exp(tl.minimum(b_g_raw, 5.0)))
                b_b_raw = tl.load(p_b, mask=mask_k, other=0, eviction_policy="evict_last").to(tl.float32)
                b_b = _round_bf16_in_fp32(1.0 / (1.0 + tl_libdevice.exp(-b_b_raw)))
                b_w_raw = tl.load(p_w, mask=mask_v, other=0, eviction_policy="evict_first").to(tl.float32)
                b_w = _round_bf16_in_fp32(1.0 / (1.0 + tl_libdevice.exp(-b_w_raw)))
            else:
                a_s = tl.load(p_a_in).to(tl.float32)
                b_s = tl.load(p_b_in).to(tl.float32)
                base_a = a_s + dt_bias_val
                b_dg = tl.load(p_g, mask=mask_k, other=0, eviction_policy="evict_last").to(tl.float32)
                sp_in = base_a + b_dg
                sp = tl.where(sp_in > 20.0, sp_in, tl_libdevice.log1p(tl_libdevice.exp(sp_in)))
                b_gk = neg_A_val * sp
                b_db = tl.load(p_b, mask=mask_k, other=0, eviction_policy="evict_last").to(tl.float32)
                b_b = _round_bf16_in_fp32(1.0 / (1.0 + tl_libdevice.exp(-(b_s + b_db))))
                b_dw = tl.load(p_w, mask=mask_v, other=0, eviction_policy="evict_first").to(tl.float32)
                b_w = _round_bf16_in_fp32(1.0 / (1.0 + tl_libdevice.exp(-(b_s + b_dw))))
                p_a_in += stride_a_tok
                p_b_in += stride_b_in_tok

            b_h *= fla_exp(b_gk[:, None])

            b_bk = b_b * b_k
            erase_d = tl.sum(b_h * b_bk[:, None], 0)

            b_v_new = b_w * b_v - erase_d

            b_h += b_k[:, None] * b_v_new[None, :]
            b_o = tl.sum(b_h * b_q[:, None], 0)
            tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v, eviction_policy="evict_first")

            p_q += stride_qkv_tok
            p_k += stride_qkv_tok
            p_v += stride_qkv_tok
            p_g += stride_g_tok
            p_b += stride_b_tok
            p_w += stride_w_tok
            p_o += HV * V

        if raw_slot >= 0:
            p_ht = (
                ssm_states
                + safe_slot * stride_state_slot
                + i_hv * K * V
                + o_k[:, None] * V
                + o_v[None, :]
            )
            tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)

    @triton.jit
    def _reckoner_gate_prep_1p4b_kernel(
        g_raw_ptr,
        b_raw_ptr,
        w_raw_ptr,
        g_out_ptr,
        b_out_ptr,
        w_out_ptr,
        stride_g_tok: tl.constexpr,
        stride_b_tok: tl.constexpr,
        stride_w_tok: tl.constexpr,
        key_dim: tl.constexpr,
        value_dim: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        i_tok = tl.program_id(0).to(tl.int64)
        i_blk = tl.program_id(1).to(tl.int64)
        offs = i_blk * BLOCK + tl.arange(0, BLOCK)

        mask_k = offs < key_dim
        if i_blk * BLOCK < key_dim:
            g_in = tl.load(g_raw_ptr + i_tok * stride_g_tok + offs, mask=mask_k, other=0.0).to(tl.float32)
            b_in = tl.load(b_raw_ptr + i_tok * stride_b_tok + offs, mask=mask_k, other=0.0).to(tl.float32)
            g_val = -tl_libdevice.exp(tl.minimum(g_in, 5.0))
            b_val = 1.0 / (1.0 + tl_libdevice.exp(-b_in))
            tl.store(g_out_ptr + i_tok * key_dim + offs, g_val.to(g_out_ptr.dtype.element_ty), mask=mask_k)
            tl.store(b_out_ptr + i_tok * key_dim + offs, b_val.to(b_out_ptr.dtype.element_ty), mask=mask_k)

        mask_v = offs < value_dim
        w_in = tl.load(w_raw_ptr + i_tok * stride_w_tok + offs, mask=mask_v, other=0.0).to(tl.float32)
        w_val = 1.0 / (1.0 + tl_libdevice.exp(-w_in))
        tl.store(w_out_ptr + i_tok * value_dim + offs, w_val.to(w_out_ptr.dtype.element_ty), mask=mask_v)

    @triton.jit
    def _reckoner_gate_prep_q2b_kernel(
        dg_ptr,
        db_ptr,
        dw_ptr,
        a_raw_ptr,
        b_raw_ptr,
        A_log_ptr,
        dt_bias_ptr,
        g_out_ptr,
        b_out_ptr,
        w_out_ptr,
        stride_dg_tok: tl.constexpr,
        stride_db_tok: tl.constexpr,
        stride_dw_tok: tl.constexpr,
        stride_a_tok: tl.constexpr,
        stride_b_tok: tl.constexpr,
        HV: tl.constexpr,
        V: tl.constexpr,
        BV: tl.constexpr,
    ):
        i_tok = tl.program_id(0).to(tl.int64)
        i_hv = tl.program_id(1).to(tl.int64)
        i_v = tl.program_id(2).to(tl.int64)
        offs_v = i_v * BV + tl.arange(0, BV)
        mask_v = offs_v < V
        col = i_hv * V + offs_v

        a_s = tl.load(a_raw_ptr + i_tok * stride_a_tok + i_hv).to(tl.float32)
        b_s = tl.load(b_raw_ptr + i_tok * stride_b_tok + i_hv).to(tl.float32)
        a_log_val = tl.load(A_log_ptr + i_hv).to(tl.float32)
        dt_bias_val = tl.load(dt_bias_ptr + i_hv).to(tl.float32)
        neg_A = -tl_libdevice.exp(a_log_val)
        base_a = a_s + dt_bias_val

        dg = tl.load(dg_ptr + i_tok * stride_dg_tok + col, mask=mask_v, other=0.0).to(tl.float32)
        db = tl.load(db_ptr + i_tok * stride_db_tok + col, mask=mask_v, other=0.0).to(tl.float32)
        dw = tl.load(dw_ptr + i_tok * stride_dw_tok + col, mask=mask_v, other=0.0).to(tl.float32)

        sp_in = base_a + dg
        sp = tl.where(sp_in > 20.0, sp_in, tl_libdevice.log1p(tl_libdevice.exp(sp_in)))
        g_val = neg_A * sp
        b_val = 1.0 / (1.0 + tl_libdevice.exp(-(b_s + db)))
        w_val = 1.0 / (1.0 + tl_libdevice.exp(-(b_s + dw)))

        out_offset = i_tok * (HV * V) + col
        tl.store(g_out_ptr + out_offset, g_val.to(g_out_ptr.dtype.element_ty), mask=mask_v)
        tl.store(b_out_ptr + out_offset, b_val.to(b_out_ptr.dtype.element_ty), mask=mask_v)
        tl.store(w_out_ptr + out_offset, w_val.to(w_out_ptr.dtype.element_ty), mask=mask_v)


@torch.compiler.disable
def reckoner_fused_gate_prep_fwd(
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    *,
    gate_mode: str,
    out_dtype: torch.dtype,
    key_dim: int,
    value_dim: int,
    num_v_heads: int,
    head_v_dim: int,
    a_raw: Optional[torch.Tensor] = None,
    b_raw: Optional[torch.Tensor] = None,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if gate_mode == "raw" or not gate_mode:
        return g, b, w

    total_tokens = g.shape[0]
    if gate_mode == "1.4b":
        if (
            g.is_cuda
            and triton is not None
            and tl_libdevice is not None
            and out_dtype == torch.bfloat16
            and total_tokens > 0
        ):
            g_out = torch.empty((total_tokens, key_dim), dtype=out_dtype, device=g.device)
            b_out = torch.empty((total_tokens, key_dim), dtype=out_dtype, device=g.device)
            w_out = torch.empty((total_tokens, value_dim), dtype=out_dtype, device=g.device)
            BLOCK = 256
            max_dim = max(key_dim, value_dim)
            grid = (total_tokens, triton.cdiv(max_dim, BLOCK))
            _reckoner_gate_prep_1p4b_kernel[grid](
                g_raw_ptr=g,
                b_raw_ptr=b,
                w_raw_ptr=w,
                g_out_ptr=g_out,
                b_out_ptr=b_out,
                w_out_ptr=w_out,
                stride_g_tok=g.stride(0),
                stride_b_tok=b.stride(0),
                stride_w_tok=w.stride(0),
                key_dim=key_dim,
                value_dim=value_dim,
                BLOCK=BLOCK,
                num_warps=4,
            )
            return g_out, b_out, w_out
        g_out = -torch.exp(g.float().clamp(max=5.0)).to(out_dtype)
        b_out = torch.sigmoid(b).to(out_dtype)
        w_out = torch.sigmoid(w).to(out_dtype)
        return g_out, b_out, w_out

    if gate_mode == "q2b":
        assert a_raw is not None and b_raw is not None and A_log is not None and dt_bias is not None
        if (
            g.is_cuda
            and triton is not None
            and tl_libdevice is not None
            and out_dtype == torch.bfloat16
            and total_tokens > 0
        ):
            g_out = torch.empty((total_tokens, value_dim), dtype=torch.float32, device=g.device)
            b_out = torch.empty((total_tokens, value_dim), dtype=out_dtype, device=g.device)
            w_out = torch.empty((total_tokens, value_dim), dtype=out_dtype, device=g.device)
            BV = min(128, triton.next_power_of_2(head_v_dim))
            grid = (total_tokens, num_v_heads, triton.cdiv(head_v_dim, BV))
            _reckoner_gate_prep_q2b_kernel[grid](
                dg_ptr=g,
                db_ptr=b,
                dw_ptr=w,
                a_raw_ptr=a_raw,
                b_raw_ptr=b_raw,
                A_log_ptr=A_log,
                dt_bias_ptr=dt_bias,
                g_out_ptr=g_out,
                b_out_ptr=b_out,
                w_out_ptr=w_out,
                stride_dg_tok=g.stride(0),
                stride_db_tok=b.stride(0),
                stride_dw_tok=w.stride(0),
                stride_a_tok=a_raw.stride(0),
                stride_b_tok=b_raw.stride(0),
                HV=num_v_heads,
                V=head_v_dim,
                BV=BV,
                num_warps=4,
            )
            return g_out, b_out, w_out
        base_a = (a_raw.float() + dt_bias.float()).unsqueeze(-1)
        base_b = b_raw.float().unsqueeze(-1)
        A = A_log.float().exp().view(1, num_v_heads, 1)
        dg = g.view(-1, num_v_heads, head_v_dim).float()
        db = b.view(-1, num_v_heads, head_v_dim).float()
        dw = w.view(-1, num_v_heads, head_v_dim).float()
        g_out = (-A * F.softplus(base_a + dg)).reshape(-1, value_dim)
        b_out = torch.sigmoid(base_b + db).to(out_dtype).reshape(-1, value_dim)
        w_out = torch.sigmoid(base_b + dw).to(out_dtype).reshape(-1, value_dim)
        return g_out, b_out, w_out

    raise ValueError(f"Unsupported gate_mode: {gate_mode!r}")


if triton is not None:

    @triton.jit
    def _fused_qknorm_partial_rope_kernel(
        q_ptr,
        k_ptr,
        q_out_ptr,
        k_out_ptr,
        q_weight_ptr,
        k_weight_ptr,
        positions_ptr,
        cos_or_inv_ptr,
        sin_ptr,
        eps,
        stride_q_tok,
        stride_q_head,
        stride_k_tok,
        stride_k_head,
        H_Q: tl.constexpr,
        H_KV: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        ROTARY_HALF: tl.constexpr,
        WEIGHT_OFFSET: tl.constexpr,
        USE_INV_FREQ: tl.constexpr,
    ):
        i_tok = tl.program_id(0).to(tl.int64)
        i_h = tl.program_id(1).to(tl.int64)
        o_d = tl.arange(0, HEAD_DIM)
        is_q = i_h < H_Q
        if is_q:
            base_in = q_ptr + i_tok * stride_q_tok + i_h * stride_q_head
            base_out = q_out_ptr + (i_tok * H_Q + i_h) * HEAD_DIM
            w_all = tl.load(q_weight_ptr + o_d).to(tl.float32) + WEIGHT_OFFSET
        else:
            ik = i_h - H_Q
            base_in = k_ptr + i_tok * stride_k_tok + ik * stride_k_head
            base_out = k_out_ptr + (i_tok * H_KV + ik) * HEAD_DIM
            w_all = tl.load(k_weight_ptr + o_d).to(tl.float32) + WEIGHT_OFFSET

        x_fp32 = tl.load(base_in + o_d).to(tl.float32)
        var = tl.sum(x_fp32 * x_fp32, axis=0) / HEAD_DIM
        rstd = tl_libdevice.rsqrt(var + eps)
        x_norm_bf16 = ((x_fp32 * rstd) * w_all).to(q_out_ptr.dtype.element_ty)
        tl.store(base_out + o_d, x_norm_bf16, mask=o_d >= (2 * ROTARY_HALF))

        o_r = tl.arange(0, ROTARY_HALF)
        if is_q:
            w0 = tl.load(q_weight_ptr + o_r).to(tl.float32) + WEIGHT_OFFSET
            w1 = tl.load(q_weight_ptr + o_r + ROTARY_HALF).to(tl.float32) + WEIGHT_OFFSET
        else:
            w0 = tl.load(k_weight_ptr + o_r).to(tl.float32) + WEIGHT_OFFSET
            w1 = tl.load(k_weight_ptr + o_r + ROTARY_HALF).to(tl.float32) + WEIGHT_OFFSET

        x0_fp32 = tl.load(base_in + o_r).to(tl.float32)
        x1_fp32 = tl.load(base_in + o_r + ROTARY_HALF).to(tl.float32)
        q0 = _round_bf16_in_fp32((x0_fp32 * rstd) * w0)
        q1 = _round_bf16_in_fp32((x1_fp32 * rstd) * w1)

        pos = tl.load(positions_ptr + i_tok).to(tl.int64)
        if USE_INV_FREQ:
            inv_f = tl.load(cos_or_inv_ptr + o_r).to(tl.float32)
            freq = pos.to(tl.float32) * inv_f
            c = _round_bf16_in_fp32(tl_libdevice.cos(freq))
            s = _round_bf16_in_fp32(tl_libdevice.sin(freq))
        else:
            rot_dim = 2 * ROTARY_HALF
            c = _round_bf16_in_fp32(
                tl.load(cos_or_inv_ptr + pos * rot_dim + o_r).to(tl.float32)
            )
            s = _round_bf16_in_fp32(
                tl.load(sin_ptr + pos * rot_dim + o_r).to(tl.float32)
            )

        rot0 = (_round_bf16_in_fp32(q0 * c) + _round_bf16_in_fp32(-q1 * s)).to(
            q_out_ptr.dtype.element_ty
        )
        rot1 = (_round_bf16_in_fp32(q1 * c) + _round_bf16_in_fp32(q0 * s)).to(
            q_out_ptr.dtype.element_ty
        )
        tl.store(base_out + o_r, rot0)
        tl.store(base_out + o_r + ROTARY_HALF, rot1)

    @triton.jit
    def _fused_partial_rope_inplace_kernel(
        q_ptr,
        k_ptr,
        positions_ptr,
        cos_ptr,
        sin_ptr,
        stride_q_tok,
        stride_q_head,
        stride_k_tok,
        stride_k_head,
        H_Q: tl.constexpr,
        ROTARY_HALF: tl.constexpr,
    ):
        i_tok = tl.program_id(0).to(tl.int64)
        i_h = tl.program_id(1).to(tl.int64)
        is_q = i_h < H_Q
        if is_q:
            base_ptr = q_ptr + i_tok * stride_q_tok + i_h * stride_q_head
        else:
            ik = i_h - H_Q
            base_ptr = k_ptr + i_tok * stride_k_tok + ik * stride_k_head

        o_r = tl.arange(0, ROTARY_HALF)
        q0 = tl.load(base_ptr + o_r).to(tl.float32)
        q1 = tl.load(base_ptr + ROTARY_HALF + o_r).to(tl.float32)

        pos = tl.load(positions_ptr + i_tok).to(tl.int64)
        rot_dim = 2 * ROTARY_HALF
        c0 = _round_bf16_in_fp32(tl.load(cos_ptr + pos * rot_dim + o_r).to(tl.float32))
        c1 = _round_bf16_in_fp32(
            tl.load(cos_ptr + pos * rot_dim + ROTARY_HALF + o_r).to(tl.float32)
        )
        s0 = _round_bf16_in_fp32(tl.load(sin_ptr + pos * rot_dim + o_r).to(tl.float32))
        s1 = _round_bf16_in_fp32(
            tl.load(sin_ptr + pos * rot_dim + ROTARY_HALF + o_r).to(tl.float32)
        )

        rot0 = (_round_bf16_in_fp32(q0 * c0) + _round_bf16_in_fp32(-q1 * s0)).to(
            q_ptr.dtype.element_ty
        )
        rot1 = (_round_bf16_in_fp32(q1 * c1) + _round_bf16_in_fp32(q0 * s1)).to(
            q_ptr.dtype.element_ty
        )
        tl.store(base_ptr + o_r, rot0)
        tl.store(base_ptr + ROTARY_HALF + o_r, rot1)


@torch.compiler.disable
def reckoner_fused_qknorm_rope_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    positions: torch.Tensor,
    *,
    eps: float,
    rotary_dim: int = 64,
    weight_offset: float = 0.0,
    cos_cached: Optional[torch.Tensor] = None,
    sin_cached: Optional[torch.Tensor] = None,
    inv_freq: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fuse per-head Q/K RMSNorm and partial RoPE into a single Triton kernel (`bfloat16` only)."""
    assert q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16, (
        f"reckoner_fused_qknorm_rope_fwd requires bfloat16, got {q.dtype} and {k.dtype}"
    )
    total_tokens, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    q_out = torch.empty(
        (total_tokens, num_q_heads, head_dim), dtype=q.dtype, device=q.device
    )
    k_out = torch.empty(
        (total_tokens, num_kv_heads, head_dim), dtype=k.dtype, device=k.device
    )
    if total_tokens == 0:
        return q_out, k_out

    use_inv_freq = inv_freq is not None
    if use_inv_freq:
        cos_or_inv = inv_freq
        sin_buf = inv_freq
    else:
        assert cos_cached is not None and sin_cached is not None
        cos_or_inv = cos_cached
        sin_buf = sin_cached

    grid = (total_tokens, num_q_heads + num_kv_heads)
    _fused_qknorm_partial_rope_kernel[grid](
        q_ptr=q,
        k_ptr=k,
        q_out_ptr=q_out,
        k_out_ptr=k_out,
        q_weight_ptr=q_weight,
        k_weight_ptr=k_weight,
        positions_ptr=positions,
        cos_or_inv_ptr=cos_or_inv,
        sin_ptr=sin_buf,
        eps=eps,
        stride_q_tok=q.stride(0),
        stride_q_head=q.stride(1),
        stride_k_tok=k.stride(0),
        stride_k_head=k.stride(1),
        H_Q=num_q_heads,
        H_KV=num_kv_heads,
        HEAD_DIM=head_dim,
        ROTARY_HALF=rotary_dim // 2,
        WEIGHT_OFFSET=float(weight_offset),
        USE_INV_FREQ=use_inv_freq,
        num_warps=2,
    )
    return q_out, k_out


@torch.compiler.disable
def reckoner_fused_partial_rope_inplace_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    *,
    rotary_dim: int = 32,
    cos_cached: torch.Tensor,
    sin_cached: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply partial RoPE in-place on already-normalized Q and K tensors in a single Triton kernel (`bfloat16` only)."""
    assert q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16, (
        f"reckoner_fused_partial_rope_inplace_fwd requires bfloat16, got {q.dtype} and {k.dtype}"
    )
    if not q.is_contiguous():
        q = q.contiguous()
    if not k.is_contiguous():
        k = k.contiguous()
    total_tokens, num_q_heads, _ = q.shape
    num_kv_heads = k.shape[1]
    if total_tokens == 0:
        return q, k
    grid = (total_tokens, num_q_heads + num_kv_heads)
    _fused_partial_rope_inplace_kernel[grid](
        q_ptr=q,
        k_ptr=k,
        positions_ptr=positions,
        cos_ptr=cos_cached,
        sin_ptr=sin_cached,
        stride_q_tok=q.stride(0),
        stride_q_head=q.stride(1),
        stride_k_tok=k.stride(0),
        stride_k_head=k.stride(1),
        H_Q=num_q_heads,
        ROTARY_HALF=rotary_dim // 2,
        num_warps=1,
    )
    return q, k


if triton is not None:

    @triton.jit
    def _qwen35_rmsnorm_kernel(
        x_ptr,
        w_ptr,
        out_ptr,
        eps,
        stride_x_row,
        DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        i_row = tl.program_id(0).to(tl.int64)
        o_d = tl.arange(0, BLOCK_D)
        mask = o_d < DIM
        x_fp32 = tl.load(x_ptr + i_row * stride_x_row + o_d, mask=mask, other=0.0).to(tl.float32)
        w_fp32 = tl.load(w_ptr + o_d, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(x_fp32 * x_fp32, axis=0) / DIM
        rstd = tl_libdevice.rsqrt(var + eps)
        out = ((x_fp32 * rstd) * (1.0 + w_fp32)).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + i_row * DIM + o_d, out, mask=mask)

    @triton.jit
    def _qwen35_rmsnorm_gated_kernel(
        h_ptr,
        g_ptr,
        w_ptr,
        out_ptr,
        eps,
        stride_h_tok,
        stride_h_head,
        stride_g_tok,
        stride_g_head,
        H: tl.constexpr,
        DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        i_tok = tl.program_id(0).to(tl.int64)
        i_h = tl.program_id(1).to(tl.int64)
        o_d = tl.arange(0, BLOCK_D)
        mask = o_d < DIM
        h_fp32 = tl.load(
            h_ptr + i_tok * stride_h_tok + i_h * stride_h_head + o_d, mask=mask, other=0.0
        ).to(tl.float32)
        g_fp32 = tl.load(
            g_ptr + i_tok * stride_g_tok + i_h * stride_g_head + o_d, mask=mask, other=0.0
        ).to(tl.float32)
        w_bf16_fp32 = _round_bf16_in_fp32(
            tl.load(w_ptr + o_d, mask=mask, other=0.0).to(tl.float32)
        )
        var = tl.sum(h_fp32 * h_fp32, axis=0) / DIM
        rstd = tl_libdevice.rsqrt(var + eps)
        h_norm_bf16 = _round_bf16_in_fp32(h_fp32 * rstd)
        h_weighted_bf16 = _round_bf16_in_fp32(w_bf16_fp32 * h_norm_bf16)
        silu_g = g_fp32 / (1.0 + tl_libdevice.exp(-g_fp32))
        out = (h_weighted_bf16 * silu_g).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + (i_tok * H + i_h) * DIM + o_d, out, mask=mask)

    @triton.jit
    def _fused_silu_mul_kernel(
        gate_ptr,
        up_ptr,
        out_ptr,
        stride_gate_row,
        stride_up_row,
        INTERMEDIATE_SIZE: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        i_row = tl.program_id(0).to(tl.int64)
        i_blk = tl.program_id(1).to(tl.int64)
        o_d = i_blk * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = o_d < INTERMEDIATE_SIZE
        g_fp32 = tl.load(
            gate_ptr + i_row * stride_gate_row + o_d, mask=mask, other=0.0
        ).to(tl.float32)
        u_fp32 = tl.load(
            up_ptr + i_row * stride_up_row + o_d, mask=mask, other=0.0
        ).to(tl.float32)
        silu_bf16 = _round_bf16_in_fp32(
            g_fp32 / (1.0 + tl_libdevice.exp(-g_fp32))
        )
        out = (silu_bf16 * u_fp32).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + i_row * INTERMEDIATE_SIZE + o_d, out, mask=mask)

    @triton.jit
    def _fused_sigmoid_mul_kernel(
        x_ptr,
        gate_ptr,
        out_ptr,
        stride_x_row,
        stride_gate_row,
        SCALE: tl.constexpr,
        DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        i_row = tl.program_id(0).to(tl.int64)
        i_blk = tl.program_id(1).to(tl.int64)
        o_d = i_blk * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = o_d < DIM
        x_fp32 = tl.load(x_ptr + i_row * stride_x_row + o_d, mask=mask, other=0.0).to(tl.float32)
        g_fp32 = tl.load(
            gate_ptr + i_row * stride_gate_row + o_d, mask=mask, other=0.0
        ).to(tl.float32)
        sig_bf16 = _round_bf16_in_fp32(1.0 / (1.0 + tl_libdevice.exp(-g_fp32)))
        if SCALE != 1.0:
            sig_bf16 = _round_bf16_in_fp32(SCALE * sig_bf16)
        out = (x_fp32 * sig_bf16).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + i_row * DIM + o_d, out, mask=mask)


@torch.compiler.disable
def reckoner_qwen35_rmsnorm_fwd(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    assert x.dtype == torch.bfloat16, (
        f"reckoner_qwen35_rmsnorm_fwd requires bfloat16, got {x.dtype}"
    )
    orig_shape = x.shape
    dim = orig_shape[-1]
    x_2d = x.view(-1, dim) if x.stride(-1) == 1 else x.contiguous().view(-1, dim)
    n_rows = x_2d.shape[0]
    out = torch.empty((n_rows, dim), dtype=x.dtype, device=x.device)
    if n_rows == 0:
        return out.view(orig_shape)
    block_d = triton.next_power_of_2(dim)
    num_warps = 4 if block_d <= 512 else 8
    _qwen35_rmsnorm_kernel[(n_rows,)](
        x_ptr=x_2d,
        w_ptr=weight,
        out_ptr=out,
        eps=eps,
        stride_x_row=x_2d.stride(0),
        DIM=dim,
        BLOCK_D=block_d,
        num_warps=num_warps,
    )
    return out.view(orig_shape)


@torch.compiler.disable
def reckoner_qwen35_rmsnorm_gated_fwd(
    hidden_states: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    assert hidden_states.dtype == torch.bfloat16 and gate.dtype == torch.bfloat16, (
        f"reckoner_qwen35_rmsnorm_gated_fwd requires bfloat16, got {hidden_states.dtype} and {gate.dtype}"
    )
    orig_shape = hidden_states.shape
    dim = orig_shape[-1]
    if hidden_states.ndim == 3 and gate.ndim == 3:
        h_3d = hidden_states if hidden_states.stride(-1) == 1 else hidden_states.contiguous()
        g_3d = gate if gate.stride(-1) == 1 else gate.contiguous()
    else:
        h_2d = (
            hidden_states.view(-1, dim)
            if hidden_states.stride(-1) == 1
            else hidden_states.contiguous().view(-1, dim)
        )
        g_2d = (
            gate.view(-1, dim)
            if gate.stride(-1) == 1
            else gate.contiguous().view(-1, dim)
        )
        h_3d = h_2d.unsqueeze(1)
        g_3d = g_2d.unsqueeze(1)
    n_tok, n_heads, _ = h_3d.shape
    out = torch.empty(
        (n_tok, n_heads, dim), dtype=hidden_states.dtype, device=hidden_states.device
    )
    if n_tok == 0 or n_heads == 0:
        return out.view(orig_shape)
    block_d = triton.next_power_of_2(dim)
    _qwen35_rmsnorm_gated_kernel[(n_tok, n_heads)](
        h_ptr=h_3d,
        g_ptr=g_3d,
        w_ptr=weight,
        out_ptr=out,
        eps=eps,
        stride_h_tok=h_3d.stride(0),
        stride_h_head=h_3d.stride(1),
        stride_g_tok=g_3d.stride(0),
        stride_g_head=g_3d.stride(1),
        H=n_heads,
        DIM=dim,
        BLOCK_D=block_d,
        num_warps=2 if block_d <= 128 else 4,
    )
    return out.view(orig_shape)


@torch.compiler.disable
def reckoner_fused_silu_mul_fwd(
    gate: torch.Tensor, up: torch.Tensor
) -> torch.Tensor:
    assert gate.dtype == torch.bfloat16 and up.dtype == torch.bfloat16, (
        f"reckoner_fused_silu_mul_fwd requires bfloat16, got {gate.dtype} and {up.dtype}"
    )
    orig_shape = gate.shape
    inter_size = orig_shape[-1]
    g_2d = gate.view(-1, inter_size) if gate.stride(-1) == 1 else gate.contiguous().view(-1, inter_size)
    u_2d = up.view(-1, inter_size) if up.stride(-1) == 1 else up.contiguous().view(-1, inter_size)
    n_rows = g_2d.shape[0]
    out = torch.empty((n_rows, inter_size), dtype=gate.dtype, device=gate.device)
    if n_rows == 0:
        return out.view(orig_shape)
    block_d = 256
    grid = (n_rows, triton.cdiv(inter_size, block_d))
    _fused_silu_mul_kernel[grid](
        gate_ptr=g_2d,
        up_ptr=u_2d,
        out_ptr=out,
        stride_gate_row=g_2d.stride(0),
        stride_up_row=u_2d.stride(0),
        INTERMEDIATE_SIZE=inter_size,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return out.view(orig_shape)


@torch.compiler.disable
def reckoner_fused_sigmoid_mul_fwd(
    x: torch.Tensor, gate: torch.Tensor, scale: float = 1.0
) -> torch.Tensor:
    assert x.dtype == torch.bfloat16 and gate.dtype == torch.bfloat16, (
        f"reckoner_fused_sigmoid_mul_fwd requires bfloat16, got {x.dtype} and {gate.dtype}"
    )
    orig_shape = x.shape
    dim = orig_shape[-1]
    x_2d = x.view(-1, dim) if x.stride(-1) == 1 else x.contiguous().view(-1, dim)
    g_2d = gate.view(-1, dim) if gate.stride(-1) == 1 else gate.contiguous().view(-1, dim)
    n_rows = x_2d.shape[0]
    out = torch.empty((n_rows, dim), dtype=x.dtype, device=x.device)
    if n_rows == 0:
        return out.view(orig_shape)
    block_d = 256
    grid = (n_rows, triton.cdiv(dim, block_d))
    _fused_sigmoid_mul_kernel[grid](
        x_ptr=x_2d,
        gate_ptr=g_2d,
        out_ptr=out,
        stride_x_row=x_2d.stride(0),
        stride_gate_row=g_2d.stride(0),
        SCALE=float(scale),
        DIM=dim,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return out.view(orig_shape)


@torch.compiler.disable
def reckoner_fused_recurrent_gdn2_fwd(
    mixed_qkv_conv: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    ssm_states: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    *,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    scale: float,
    cu_seqlens: Optional[torch.Tensor] = None,
    has_initial_states: Optional[torch.Tensor] = None,
    use_initial_state: bool = True,
    gate_mode: str = "raw",
    a_raw: Optional[torch.Tensor] = None,
    b_raw: Optional[torch.Tensor] = None,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    apply_qkv_silu: bool = False,
) -> torch.Tensor:
    if (gate_mode not in ("raw", "")) or apply_qkv_silu:
        assert mixed_qkv_conv.dtype == torch.bfloat16, (
            f"Fused recurrent gate/silu mode requires bfloat16, got {mixed_qkv_conv.dtype}"
        )
    if mixed_qkv_conv.stride(-1) != 1:
        mixed_qkv_conv = mixed_qkv_conv.contiguous()
    total_tokens = mixed_qkv_conv.shape[0]
    out = torch.empty(
        (total_tokens, num_v_heads, head_v_dim),
        dtype=mixed_qkv_conv.dtype,
        device=mixed_qkv_conv.device,
    )
    if total_tokens == 0:
        return out

    is_varlen = cu_seqlens is not None
    N = (cu_seqlens.numel() - 1) if is_varlen else total_tokens
    if N <= 0:
        return out
    T = total_tokens if is_varlen else 1

    if gate_mode == "raw" or not gate_mode:
        gate_mode_id = 0
    elif gate_mode == "1.4b":
        gate_mode_id = 1
    elif gate_mode == "q2b":
        gate_mode_id = 2
        assert a_raw is not None and b_raw is not None and A_log is not None and dt_bias is not None
    else:
        raise ValueError(f"Unsupported gate_mode: {gate_mode!r}")

    BK = triton.next_power_of_2(head_k_dim)
    BV = 32
    grid = (triton.cdiv(head_v_dim, BV) * N * num_v_heads,)

    reckoner_fused_recurrent_gdn2_kernel[grid](
        mixed_qkv_conv=mixed_qkv_conv,
        g=g,
        b=b,
        w=w,
        a_raw=a_raw if gate_mode_id == 2 else g,
        b_raw=b_raw if gate_mode_id == 2 else b,
        A_log=A_log if gate_mode_id == 2 else g,
        dt_bias=dt_bias if gate_mode_id == 2 else b,
        o=out,
        ssm_states=ssm_states,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        has_initial_states=has_initial_states,
        scale=scale,
        N=N,
        T=T,
        H=num_k_heads,
        HV=num_v_heads,
        K=head_k_dim,
        V=head_v_dim,
        BK=BK,
        BV=BV,
        stride_qkv_tok=mixed_qkv_conv.stride(0),
        stride_g_tok=g.stride(0),
        stride_b_tok=b.stride(0),
        stride_w_tok=w.stride(0),
        stride_a_tok=a_raw.stride(0) if (gate_mode_id == 2 and a_raw is not None) else 0,
        stride_b_in_tok=b_raw.stride(0) if (gate_mode_id == 2 and b_raw is not None) else 0,
        stride_state_slot=ssm_states.stride(0),
        stride_indices_seq=ssm_state_indices.stride(0),
        USE_INITIAL_STATE=use_initial_state,
        HAS_INIT_STATES_MASK=has_initial_states is not None,
        IS_VARLEN=is_varlen,
        GATE_MODE=gate_mode_id,
        APPLY_QKV_SILU=bool(apply_qkv_silu),
        num_warps=4,
        num_stages=2,
    )
    return out


class ReckonerGDN2AttnBackend(MambaAttnBackendBase):
    """Mamba attention backend for Reckoner's 14 GDN-2 linear attention layers on SGLang 0.5.22."""

    needs_cpu_seq_lens: bool = True
    supports_mis: bool = False
    requires_contiguous_prefill_state: bool = False
    SHORT_EXTEND_RECURRENT_MAX_LEN: int = 64

    def __init__(self, model_runner):
        super().__init__(model_runner)
        try:
            self.conv_states_shape = (
                model_runner.req_to_token_pool.mamba_pool.mamba_cache.conv[0].shape
            )
        except Exception:
            self.conv_states_shape = (1, 6144, 3)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        super().init_forward_metadata(forward_batch)
        meta = self.forward_metadata
        meta.conv_block_table = None
        meta.cu_seqlens_long = None
        meta.cu_seqlens_cpu = None
        meta.max_extend_len = None
        meta.any_has_init = None
        meta.all_has_init = None
        meta.seq_lens_cpu = None

        if not forward_batch.forward_mode.is_decode():
            seq_lens_cpu = getattr(forward_batch, "extend_seq_lens_cpu", None)
            if seq_lens_cpu is None:
                seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
            if isinstance(seq_lens_cpu, torch.Tensor):
                seq_lens_cpu = seq_lens_cpu.tolist()
            meta.seq_lens_cpu = seq_lens_cpu

            prefix_lens_cpu = getattr(forward_batch, "extend_prefix_lens_cpu", None)
            if isinstance(prefix_lens_cpu, torch.Tensor):
                prefix_lens_cpu = prefix_lens_cpu.tolist()
            if prefix_lens_cpu is not None and len(prefix_lens_cpu) > 0:
                meta.any_has_init = any(int(p) > 0 for p in prefix_lens_cpu)
                meta.all_has_init = all(int(p) > 0 for p in prefix_lens_cpu)

            if seq_lens_cpu:
                max_ext = int(max(seq_lens_cpu))
                meta.max_extend_len = max_ext
                if max_ext > self.SHORT_EXTEND_RECURRENT_MAX_LEN and meta.query_start_loc is not None:
                    meta.cu_seqlens_long = meta.query_start_loc.to(dtype=torch.long)
                    cu_cpu = [0]
                    for s in seq_lens_cpu:
                        cu_cpu.append(cu_cpu[-1] + int(s))
                    meta.cu_seqlens_cpu = torch.tensor(cu_cpu, dtype=torch.long, device="cpu")

                if (
                    meta.query_start_loc is not None
                    and len(seq_lens_cpu) == meta.query_start_loc.numel() - 1
                ):
                    block_counts = [
                        (int(length) + CAUSAL_CONV1D_FWD_BLOCK_M - 1) // CAUSAL_CONV1D_FWD_BLOCK_M
                        for length in seq_lens_cpu
                    ]
                    if block_counts and len(block_counts) * max(block_counts) > 4 * sum(block_counts):
                        pairs = [
                            (seq, block)
                            for seq, count in enumerate(block_counts)
                            for block in range(count)
                        ]
                        meta.conv_block_table = torch.tensor(
                            pairs, dtype=torch.int32, device="cpu"
                        ).to(self.device, non_blocking=True)

    def forward(
        self,
        q: Optional[torch.Tensor] = None,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
        layer=None,
        forward_batch: ForwardBatch = None,
        save_kv_cache: bool = True,
        mixed_qkv: Optional[torch.Tensor] = None,
        a: Optional[torch.Tensor] = None,
        b: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if forward_batch.forward_mode.is_decode():
            return self.forward_decode(
                q=q,
                k=k,
                v=v,
                layer=layer,
                forward_batch=forward_batch,
                save_kv_cache=save_kv_cache,
                mixed_qkv=mixed_qkv,
                a=a,
                b=b,
                **kwargs,
            )
        return self.forward_extend(
            q=q,
            k=k,
            v=v,
            layer=layer,
            forward_batch=forward_batch,
            save_kv_cache=save_kv_cache,
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            **kwargs,
        )

    def _unpack_inputs(
        self,
        layer,
        mixed_qkv: Optional[torch.Tensor],
        kwargs: dict,
    ):
        mamba_input = kwargs.get("mamba_Input") or kwargs
        if mixed_qkv is None:
            mixed_qkv = mamba_input["mixed_qkv"]
        conv_weights = mamba_input.get("conv_weights", getattr(layer, "conv_weights", None))
        bias = mamba_input.get("bias", getattr(layer, "bias", None))
        activation = mamba_input.get("activation", getattr(layer, "activation", None))
        key_dim = mamba_input.get("key_dim", getattr(layer, "k_dim", None))
        value_dim = mamba_input.get("value_dim", getattr(layer, "v_dim", None))
        num_k_heads = mamba_input.get("num_k_heads", getattr(layer, "num_k_heads", None))
        num_v_heads = mamba_input.get("num_v_heads", getattr(layer, "num_v_heads", None))
        head_k_dim = mamba_input.get("head_k_dim", getattr(layer, "head_k_dim", None))
        head_v_dim = mamba_input.get("head_v_dim", getattr(layer, "head_v_dim", None))
        g = mamba_input["g"]
        b_gate = mamba_input["b"]
        w = mamba_input["w"]
        layer_id = mamba_input.get("layer_id", getattr(layer, "layer_id", None))
        gate_mode = mamba_input.get("gate_mode", "raw")
        a_raw = mamba_input.get("a_raw", None)
        b_raw = mamba_input.get("b_raw", None)
        A_log = mamba_input.get("A_log", None)
        dt_bias = mamba_input.get("dt_bias", None)
        return (
            mixed_qkv,
            conv_weights,
            bias,
            activation,
            key_dim,
            value_dim,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            g,
            b_gate,
            w,
            layer_id,
            gate_mode,
            a_raw,
            b_raw,
            A_log,
            dt_bias,
        )

    def _get_cache_indices(self) -> torch.Tensor:
        meta = self.forward_metadata
        cache_indices = getattr(meta, "mamba_cache_indices", None)
        if cache_indices is None:
            cache_indices = getattr(meta, "cache_indices", None)
        assert cache_indices is not None, "Missing mamba_cache_indices in forward_metadata"
        return cache_indices

    def _get_has_initial_states(self, forward_batch: ForwardBatch) -> Optional[torch.Tensor]:
        meta = self.forward_metadata
        if getattr(meta, "extend_has_initial_states", None) is not None:
            return meta.extend_has_initial_states
        if getattr(forward_batch, "extend_prefix_lens", None) is not None:
            return forward_batch.extend_prefix_lens > 0
        if getattr(meta, "mixed_metadata", None) is not None:
            return meta.mixed_metadata.has_initial_states
        return None

    def forward_extend(
        self,
        q: Optional[torch.Tensor] = None,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
        layer=None,
        forward_batch: ForwardBatch = None,
        save_kv_cache: bool = True,
        mixed_qkv: Optional[torch.Tensor] = None,
        a: Optional[torch.Tensor] = None,
        b: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        (
            mixed_qkv,
            conv_weights,
            bias,
            activation,
            key_dim,
            value_dim,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            g,
            b_gate,
            w,
            layer_id,
            gate_mode,
            a_raw,
            b_raw,
            A_log,
            dt_bias,
        ) = self._unpack_inputs(layer, mixed_qkv, kwargs)

        mamba_cache: MambaPool.State = self.req_to_token_pool.mamba2_layer_cache(layer_id)
        conv_states, ssm_states = mamba_cache.conv[0], mamba_cache.temporal
        cache_indices = self._get_cache_indices()
        meta = self.forward_metadata
        query_start_loc = meta.query_start_loc
        has_initial_states = self._get_has_initial_states(forward_batch)
        seq_lens_cpu = getattr(meta, "seq_lens_cpu", None)
        if seq_lens_cpu is None:
            seq_lens_cpu = (
                getattr(forward_batch, "extend_seq_lens_cpu", None)
                or getattr(forward_batch, "seq_lens_cpu", None)
            )
            if isinstance(seq_lens_cpu, torch.Tensor):
                seq_lens_cpu = seq_lens_cpu.tolist()

        group = num_v_heads // num_k_heads if num_v_heads > num_k_heads else 1
        scale = head_k_dim ** -0.5

        if not mixed_qkv.is_cuda:
            return self._forward_cpu(
                mixed_qkv=mixed_qkv,
                conv_weights=conv_weights,
                bias=bias,
                activation=activation,
                key_dim=key_dim,
                value_dim=value_dim,
                num_k_heads=num_k_heads,
                num_v_heads=num_v_heads,
                head_k_dim=head_k_dim,
                head_v_dim=head_v_dim,
                g=g,
                b=b_gate,
                w=w,
                conv_states=conv_states,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                forward_mode=forward_batch.forward_mode,
                has_initial_states=has_initial_states,
                query_start_loc=query_start_loc,
                group=group,
                scale=scale,
                gate_mode=gate_mode,
                a_raw=a_raw,
                b_raw=b_raw,
                A_log=A_log,
                dt_bias=dt_bias,
            )

        assert not forward_batch.forward_mode.is_target_verify(), (
            "Speculative verification is not enabled for ReckonerGDN2AttnBackend"
        )

        num_seqs = query_start_loc.numel() - 1
        active_indices = cache_indices[:num_seqs]
        if active_indices.numel() == 0 or mixed_qkv.shape[0] == 0:
            return torch.zeros(
                (mixed_qkv.shape[0], num_v_heads, head_v_dim),
                dtype=mixed_qkv.dtype,
                device=mixed_qkv.device,
            )

        has_init_active = (
            has_initial_states[:num_seqs] if has_initial_states is not None else None
        )
        post_silu = activation in ("silu", "swish")
        kernel_act = None if post_silu else activation
        legacy_extend = os.environ.get("SGLANG_RECKONER_LEGACY_EXTEND", "0") == "1"
        if legacy_extend:
            valid_mask = active_indices >= 0
            all_valid = bool(valid_mask.all())
            safe_active_indices = active_indices.clamp(min=0)
            active_conv_states = conv_states[safe_active_indices].contiguous()
            mixed_qkv_conv = causal_conv1d_fn(
                mixed_qkv.transpose(0, 1),
                conv_weights,
                bias,
                activation=kernel_act,
                conv_states=active_conv_states,
                has_initial_state=has_init_active,
                cache_indices=torch.arange(
                    num_seqs, dtype=active_indices.dtype, device=active_indices.device
                ),
                query_start_loc=query_start_loc,
                seq_lens_cpu=seq_lens_cpu,
            ).transpose(0, 1)[: mixed_qkv.shape[0]]
            if all_valid:
                conv_states[active_indices] = active_conv_states
            else:
                conv_states[active_indices[valid_mask]] = active_conv_states[valid_mask]
        else:
            block_table = getattr(meta, "conv_block_table", None)
            conv_kwargs = {"block_table": block_table} if block_table is not None else {}
            mixed_qkv_conv = _causal_conv1d_fn_triton(
                mixed_qkv.transpose(0, 1),
                conv_weights,
                bias,
                activation=kernel_act,
                conv_states=conv_states,
                has_initial_state=has_init_active,
                cache_indices=active_indices,
                query_start_loc=query_start_loc,
                seq_lens_cpu=seq_lens_cpu,
                **conv_kwargs,
            ).transpose(0, 1)[: mixed_qkv.shape[0]]

        max_extend_len = getattr(meta, "max_extend_len", None)
        if max_extend_len is None and seq_lens_cpu:
            max_extend_len = int(max(seq_lens_cpu))
        any_has_init = None if legacy_extend else getattr(meta, "any_has_init", None)
        all_has_init = (
            bool(has_init_active.all())
            if (legacy_extend and has_init_active is not None)
            else getattr(meta, "all_has_init", None)
        )

        use_short_recurrent = (
            not legacy_extend
            and triton is not None
            and max_extend_len is not None
            and max_extend_len <= self.SHORT_EXTEND_RECURRENT_MAX_LEN
            and bool(any_has_init)
            and not getattr(meta, "has_mamba_track_mask", False)
        )
        fuse_qkv_silu = (
            post_silu
            and use_short_recurrent
            and mixed_qkv.dtype == torch.bfloat16
            and is_fusion_step_enabled("f1")
        )
        if post_silu and not fuse_qkv_silu:
            mixed_qkv_conv = F.silu(mixed_qkv_conv)

        if use_short_recurrent:
            use_init = (has_init_active is not None) and (any_has_init is not False)
            return reckoner_fused_recurrent_gdn2_fwd(
                mixed_qkv_conv=mixed_qkv_conv,
                g=g,
                b=b_gate,
                w=w,
                ssm_states=ssm_states,
                ssm_state_indices=active_indices,
                num_k_heads=num_k_heads,
                num_v_heads=num_v_heads,
                head_k_dim=head_k_dim,
                head_v_dim=head_v_dim,
                scale=scale,
                cu_seqlens=query_start_loc,
                has_initial_states=has_init_active if (use_init and not all_has_init) else None,
                use_initial_state=use_init,
                gate_mode=gate_mode,
                a_raw=a_raw,
                b_raw=b_raw,
                A_log=A_log,
                dt_bias=dt_bias,
                apply_qkv_silu=fuse_qkv_silu,
            )

        if gate_mode != "raw":
            g, b_gate, w = reckoner_fused_gate_prep_fwd(
                g,
                b_gate,
                w,
                gate_mode=gate_mode,
                out_dtype=mixed_qkv.dtype,
                key_dim=key_dim,
                value_dim=value_dim,
                num_v_heads=num_v_heads,
                head_v_dim=head_v_dim,
                a_raw=a_raw,
                b_raw=b_raw,
                A_log=A_log,
                dt_bias=dt_bias,
            )

        query, key, value = torch.split(
            mixed_qkv_conv, [key_dim, key_dim, value_dim], dim=-1
        )
        query = query.view(1, -1, num_k_heads, head_k_dim)
        key = key.view(1, -1, num_k_heads, head_k_dim)
        value = value.view(1, -1, num_v_heads, head_v_dim).contiguous()

        g_4d = g.view(1, -1, num_k_heads, head_k_dim)
        b_4d = b_gate.view(1, -1, num_k_heads, head_k_dim)
        w_4d = w.view(1, -1, num_v_heads, head_v_dim).contiguous()

        if group > 1:
            query = query.unsqueeze(3).expand(-1, -1, -1, group, -1).reshape(1, -1, num_v_heads, head_k_dim).contiguous()
            key = key.unsqueeze(3).expand(-1, -1, -1, group, -1).reshape(1, -1, num_v_heads, head_k_dim).contiguous()
            g_4d = g_4d.unsqueeze(3).expand(-1, -1, -1, group, -1).reshape(1, -1, num_v_heads, head_k_dim).contiguous()
            b_4d = b_4d.unsqueeze(3).expand(-1, -1, -1, group, -1).reshape(1, -1, num_v_heads, head_k_dim).contiguous()
        else:
            query = query.contiguous()
            key = key.contiguous()
            g_4d = g_4d.contiguous()
            b_4d = b_4d.contiguous()

        safe_active_indices = active_indices.clamp(min=0)
        if has_init_active is None or any_has_init is False:
            init_state = None
        elif all_has_init is True:
            init_state = ssm_states[safe_active_indices]
        else:
            init_state = torch.where(
                has_init_active.view(-1, 1, 1, 1),
                ssm_states[safe_active_indices],
                0.0,
            )

        cu_seqlens_long = getattr(meta, "cu_seqlens_long", None)
        if cu_seqlens_long is None:
            cu_seqlens_long = query_start_loc.to(dtype=torch.long)
            meta.cu_seqlens_long = cu_seqlens_long
        cu_seqlens_cpu = None if legacy_extend else getattr(meta, "cu_seqlens_cpu", None)

        core_attn_out, final_state = chunk_gdn2(
            q=query,
            k=key,
            v=value,
            g=g_4d,
            b=b_4d,
            w=w_4d,
            scale=scale,
            initial_state=init_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens_long,
            cu_seqlens_cpu=cu_seqlens_cpu,
        )
        ssm_states[safe_active_indices] = final_state.to(dtype=ssm_states.dtype)
        return core_attn_out.squeeze(0)

    def forward_decode(
        self,
        q: Optional[torch.Tensor] = None,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
        layer=None,
        forward_batch: ForwardBatch = None,
        save_kv_cache: bool = True,
        mixed_qkv: Optional[torch.Tensor] = None,
        a: Optional[torch.Tensor] = None,
        b: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        (
            mixed_qkv,
            conv_weights,
            bias,
            activation,
            key_dim,
            value_dim,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            g,
            b_gate,
            w,
            layer_id,
            gate_mode,
            a_raw,
            b_raw,
            A_log,
            dt_bias,
        ) = self._unpack_inputs(layer, mixed_qkv, kwargs)

        mamba_cache: MambaPool.State = self.req_to_token_pool.mamba2_layer_cache(layer_id)
        conv_states, ssm_states = mamba_cache.conv[0], mamba_cache.temporal
        cache_indices = self._get_cache_indices()
        query_start_loc = self.forward_metadata.query_start_loc

        group = num_v_heads // num_k_heads if num_v_heads > num_k_heads else 1
        scale = head_k_dim ** -0.5

        if not mixed_qkv.is_cuda:
            return self._forward_cpu(
                mixed_qkv=mixed_qkv,
                conv_weights=conv_weights,
                bias=bias,
                activation=activation,
                key_dim=key_dim,
                value_dim=value_dim,
                num_k_heads=num_k_heads,
                num_v_heads=num_v_heads,
                head_k_dim=head_k_dim,
                head_v_dim=head_v_dim,
                g=g,
                b=b_gate,
                w=w,
                conv_states=conv_states,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                forward_mode=forward_batch.forward_mode,
                has_initial_states=None,
                query_start_loc=query_start_loc,
                group=group,
                scale=scale,
                gate_mode=gate_mode,
                a_raw=a_raw,
                b_raw=b_raw,
                A_log=A_log,
                dt_bias=dt_bias,
            )

        post_silu = activation in ("silu", "swish")
        fuse_qkv_silu = (
            post_silu
            and mixed_qkv.dtype == torch.bfloat16
            and is_fusion_step_enabled("f1")
        )
        mixed_qkv_conv = causal_conv1d_update(
            mixed_qkv,
            conv_states,
            conv_weights,
            bias,
            None if post_silu else activation,
            conv_state_indices=cache_indices,
        )
        if post_silu and not fuse_qkv_silu:
            mixed_qkv_conv = F.silu(mixed_qkv_conv)
        return reckoner_fused_recurrent_gdn2_fwd(
            mixed_qkv_conv=mixed_qkv_conv,
            g=g,
            b=b_gate,
            w=w,
            ssm_states=ssm_states,
            ssm_state_indices=cache_indices,
            num_k_heads=num_k_heads,
            num_v_heads=num_v_heads,
            head_k_dim=head_k_dim,
            head_v_dim=head_v_dim,
            scale=scale,
            cu_seqlens=None,
            has_initial_states=None,
            use_initial_state=True,
            gate_mode=gate_mode,
            a_raw=a_raw,
            b_raw=b_raw,
            A_log=A_log,
            dt_bias=dt_bias,
            apply_qkv_silu=fuse_qkv_silu,
        )

    def _forward_cpu(
        self,
        *,
        mixed_qkv: torch.Tensor,
        conv_weights: torch.Tensor,
        bias: Optional[torch.Tensor],
        activation: Optional[str],
        key_dim: int,
        value_dim: int,
        num_k_heads: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        g: torch.Tensor,
        b: torch.Tensor,
        w: torch.Tensor,
        conv_states: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        forward_mode: ForwardMode,
        has_initial_states: Optional[torch.Tensor],
        query_start_loc: Optional[torch.Tensor],
        group: int,
        scale: float,
        gate_mode: str = "raw",
        a_raw: Optional[torch.Tensor] = None,
        b_raw: Optional[torch.Tensor] = None,
        A_log: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pure-PyTorch CPU execution path for unit testing ReckonerGDN2AttnBackend without GPU."""
        if gate_mode != "raw":
            g, b, w = reckoner_fused_gate_prep_fwd(
                g,
                b,
                w,
                gate_mode=gate_mode,
                out_dtype=mixed_qkv.dtype,
                key_dim=key_dim,
                value_dim=value_dim,
                num_v_heads=num_v_heads,
                head_v_dim=head_v_dim,
                a_raw=a_raw,
                b_raw=b_raw,
                A_log=A_log,
                dt_bias=dt_bias,
            )
        conv_dim, width = conv_weights.shape
        win_len = width - 1
        w_conv_3d = conv_weights.unsqueeze(1)  # [conv_dim, 1, width]

        if forward_mode.is_extend():
            assert query_start_loc is not None
            num_seqs = query_start_loc.numel() - 1
            out_chunks: List[torch.Tensor] = []

            for i in range(num_seqs):
                s0 = int(query_start_loc[i].item())
                s1 = int(query_start_loc[i + 1].item())
                if s1 <= s0:
                    continue
                slot = int(cache_indices[i].item())
                has_init = (
                    bool(has_initial_states[i].item())
                    if has_initial_states is not None
                    else False
                )

                seg_qkv = mixed_qkv[s0:s1].transpose(0, 1)  # [conv_dim, L]
                if has_init and slot >= 0:
                    init_conv = conv_states[slot]  # [conv_dim, width - 1]
                else:
                    init_conv = torch.zeros(
                        (conv_dim, win_len),
                        dtype=mixed_qkv.dtype,
                        device=mixed_qkv.device,
                    )
                pad = torch.cat([init_conv, seg_qkv], dim=-1)  # [conv_dim, win_len + L]
                if slot >= 0:
                    conv_states[slot] = pad[:, -win_len:].to(conv_states.dtype)

                conv_out = F.conv1d(
                    pad.unsqueeze(0).float(),
                    w_conv_3d.float(),
                    bias=bias.float() if bias is not None else None,
                    groups=conv_dim,
                ).squeeze(0).transpose(0, 1).to(mixed_qkv.dtype)  # [L, conv_dim]
                if activation in ("silu", "swish"):
                    conv_out = F.silu(conv_out)

                L = s1 - s0
                q_seg, k_seg, v_seg = torch.split(
                    conv_out, [key_dim, key_dim, value_dim], dim=-1
                )
                q_seg = q_seg.view(1, L, num_k_heads, head_k_dim)
                k_seg = k_seg.view(1, L, num_k_heads, head_k_dim)
                v_seg = v_seg.view(1, L, num_v_heads, head_v_dim)
                g_seg = g[s0:s1].view(1, L, num_k_heads, head_k_dim)
                b_seg = b[s0:s1].view(1, L, num_k_heads, head_k_dim)
                w_seg = w[s0:s1].view(1, L, num_v_heads, head_v_dim)

                if group > 1:
                    q_seg = q_seg.repeat_interleave(group, dim=2)
                    k_seg = k_seg.repeat_interleave(group, dim=2)
                    g_seg = g_seg.repeat_interleave(group, dim=2)
                    b_seg = b_seg.repeat_interleave(group, dim=2)

                q_n = F.normalize(q_seg.float(), p=2, dim=-1, eps=1e-6)
                k_n = F.normalize(k_seg.float(), p=2, dim=-1, eps=1e-6)
                init_ssm = (
                    ssm_states[slot : slot + 1].float()
                    if (has_init and slot >= 0)
                    else None
                )
                o_seg, final_ssm = naive_recurrent_gdn2(
                    q_n,
                    k_n,
                    v_seg.float(),
                    g_seg.float(),
                    b_seg.float(),
                    w_seg.float(),
                    scale=scale,
                    initial_state=init_ssm,
                    output_final_state=True,
                )
                if slot >= 0 and final_ssm is not None:
                    ssm_states[slot] = final_ssm.squeeze(0).to(ssm_states.dtype)
                out_chunks.append(o_seg.squeeze(0).to(mixed_qkv.dtype))

            if not out_chunks:
                return torch.zeros(
                    (mixed_qkv.shape[0], num_v_heads, head_v_dim),
                    dtype=mixed_qkv.dtype,
                    device=mixed_qkv.device,
                )
            return torch.cat(out_chunks, dim=0)

        else:
            bs = mixed_qkv.shape[0]
            out_rows: List[torch.Tensor] = []
            for i in range(bs):
                slot = int(cache_indices[i].item())
                safe_slot = max(slot, 0)
                x_row = mixed_qkv[i]  # [conv_dim]
                window = torch.cat(
                    [conv_states[safe_slot].float(), x_row.float().unsqueeze(-1)],
                    dim=-1,
                )  # [conv_dim, width]
                if slot >= 0:
                    conv_states[slot] = window[:, 1:].to(conv_states.dtype)
                conv_row = (window * conv_weights.float()).sum(dim=-1)
                if bias is not None:
                    conv_row = conv_row + bias.float()
                if activation in ("silu", "swish"):
                    conv_row = F.silu(conv_row)
                conv_row = conv_row.to(mixed_qkv.dtype)

                q_row, k_row, v_row = torch.split(
                    conv_row, [key_dim, key_dim, value_dim], dim=-1
                )
                q_row = q_row.view(1, 1, num_k_heads, head_k_dim)
                k_row = k_row.view(1, 1, num_k_heads, head_k_dim)
                v_row = v_row.view(1, 1, num_v_heads, head_v_dim)
                g_row = g[i].view(1, 1, num_k_heads, head_k_dim)
                b_row = b[i].view(1, 1, num_k_heads, head_k_dim)
                w_row = w[i].view(1, 1, num_v_heads, head_v_dim)

                if group > 1:
                    q_row = q_row.repeat_interleave(group, dim=2)
                    k_row = k_row.repeat_interleave(group, dim=2)
                    g_row = g_row.repeat_interleave(group, dim=2)
                    b_row = b_row.repeat_interleave(group, dim=2)

                q_n = F.normalize(q_row.float(), p=2, dim=-1, eps=1e-6)
                k_n = F.normalize(k_row.float(), p=2, dim=-1, eps=1e-6)
                init_ssm = ssm_states[safe_slot : safe_slot + 1].float()
                o_row, final_ssm = naive_recurrent_gdn2(
                    q_n,
                    k_n,
                    v_row.float(),
                    g_row.float(),
                    b_row.float(),
                    w_row.float(),
                    scale=scale,
                    initial_state=init_ssm,
                    output_final_state=True,
                )
                if slot >= 0 and final_ssm is not None:
                    ssm_states[slot] = final_ssm.squeeze(0).to(ssm_states.dtype)
                out_rows.append(o_row.squeeze(0).to(mixed_qkv.dtype))

            return torch.cat(out_rows, dim=0)

    @classmethod
    def create_standalone(
        cls,
        config,
        max_reqs: int = 64,
        max_context_len: int = 2048,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "_StandaloneCPUHybridBackend":
        """Create a standalone in-process hybrid backend for CPU unit tests and dry runs."""
        return _StandaloneCPUHybridBackend(
            config=config,
            max_reqs=max_reqs,
            max_context_len=max_context_len,
            device=torch.device(device),
            dtype=dtype,
        )


class _StandaloneForwardBatch:
    """Lightweight ForwardBatch stand-in for standalone CPU testing."""

    _standalone_cpu: bool = True

    def __init__(
        self,
        attn_backend: "_StandaloneCPUHybridBackend",
        slot_indices: List[int],
        seq_lens: List[int],
        is_decode: bool,
        device: torch.device,
        has_initial_states: Optional[List[bool]] = None,
    ) -> None:
        self.attn_backend = attn_backend
        self.slot_indices = [int(s) for s in slot_indices]
        self.seq_lens = [int(l) for l in seq_lens]
        self.forward_mode = ForwardMode.DECODE if is_decode else ForwardMode.EXTEND
        self.device = device
        self.cache_indices = torch.tensor(self.slot_indices, dtype=torch.int64, device=device)
        if has_initial_states is not None:
            self.has_initial_states: Optional[torch.Tensor] = torch.tensor(
                [bool(x) for x in has_initial_states], dtype=torch.bool, device=device
            )
        else:
            self.has_initial_states = None
        locs = [0]
        for l in self.seq_lens:
            locs.append(locs[-1] + l)
        self.query_start_loc = torch.tensor(locs, dtype=torch.int32, device=device)


class _StandaloneCPUHybridBackend:
    """Standalone CPU hybrid backend exercising `ReckonerGDN2AttnBackend._forward_cpu` + GQA."""

    def __init__(
        self,
        config,
        max_reqs: int = 64,
        max_context_len: int = 2048,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.config = config
        self.max_reqs = int(max_reqs)
        self.max_context_len = int(max_context_len)
        self.device = device
        self.dtype = dtype
        self.gdn_helper = ReckonerGDN2AttnBackend.__new__(ReckonerGDN2AttnBackend)

        n_heads = int(getattr(config, "n_heads", config.num_attention_heads))
        n_kv_heads = int(getattr(config, "n_kv_heads", config.num_key_value_heads))
        head_dim = int(getattr(config, "head_dim", 128))
        lin_k_heads = int(getattr(config, "linear_num_key_heads", n_heads))
        lin_k_dim = int(getattr(config, "linear_key_head_dim", head_dim))
        n_v_heads = int(
            getattr(config, "linear_num_value_heads", None)
            or getattr(config, "gdn_v_heads", None)
            or n_heads
        )
        lin_v_dim = int(getattr(config, "linear_value_head_dim", head_dim))
        conv_kernel = int(
            getattr(config, "linear_conv_kernel_dim", None)
            or getattr(config, "conv_kernel", 4)
        )
        conv_dim = 2 * (lin_k_heads * lin_k_dim) + (n_v_heads * lin_v_dim)

        self.conv_states: dict[int, torch.Tensor] = {}
        self.ssm_states: dict[int, torch.Tensor] = {}
        self.k_cache: dict[int, torch.Tensor] = {}
        self.v_cache: dict[int, torch.Tensor] = {}
        self.kv_lens: dict[int, torch.Tensor] = {}

        layout = config.parsed_layout()
        for layer_id, op in enumerate(layout):
            if op == "G":
                self.conv_states[layer_id] = torch.zeros(
                    (self.max_reqs, conv_dim, conv_kernel - 1), dtype=dtype, device=device
                )
                self.ssm_states[layer_id] = torch.zeros(
                    (self.max_reqs, n_v_heads, lin_k_dim, lin_v_dim),
                    dtype=torch.float32,
                    device=device,
                )
            else:
                self.k_cache[layer_id] = torch.zeros(
                    (self.max_reqs, n_kv_heads, self.max_context_len, head_dim),
                    dtype=dtype,
                    device=device,
                )
                self.v_cache[layer_id] = torch.zeros(
                    (self.max_reqs, n_kv_heads, self.max_context_len, head_dim),
                    dtype=dtype,
                    device=device,
                )
                self.kv_lens[layer_id] = torch.zeros(
                    (self.max_reqs,), dtype=torch.int64, device=device
                )

    def reset_slot(self, slot_idx: int) -> None:
        s = int(slot_idx)
        for t in self.conv_states.values():
            t[s].zero_()
        for t in self.ssm_states.values():
            t[s].zero_()
        for t in self.k_cache.values():
            t[s].zero_()
        for t in self.v_cache.values():
            t[s].zero_()
        for t in self.kv_lens.values():
            t[s] = 0

    def reset_all_states(self) -> None:
        for s in range(self.max_reqs):
            self.reset_slot(s)

    def make_forward_batch(
        self,
        slot_indices: List[int],
        seq_lens: List[int],
        is_decode: bool = False,
        device: Union[str, torch.device] = "cpu",
        has_initial_states: Optional[List[bool]] = None,
    ) -> _StandaloneForwardBatch:
        dev = torch.device(device)
        if not is_decode:
            for idx, s in enumerate(slot_indices):
                if has_initial_states is None or not has_initial_states[idx]:
                    self.reset_slot(int(s))
        return _StandaloneForwardBatch(
            attn_backend=self,
            slot_indices=slot_indices,
            seq_lens=seq_lens,
            is_decode=is_decode,
            device=dev,
            has_initial_states=has_initial_states,
        )

    def init_forward_metadata(self, forward_batch: _StandaloneForwardBatch) -> None:
        any_init = False
        all_init = False
        if forward_batch.has_initial_states is not None and forward_batch.has_initial_states.numel() > 0:
            vals = [bool(x) for x in forward_batch.has_initial_states.tolist()]
            any_init = any(vals)
            all_init = all(vals)
        self._ext_meta = {
            "max_extend_len": max(forward_batch.seq_lens) if forward_batch.seq_lens else 0,
            "any_has_init": any_init,
            "all_has_init": all_init,
        }

    def forward(
        self,
        q: Optional[torch.Tensor] = None,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
        layer=None,
        forward_batch: _StandaloneForwardBatch = None,
        save_kv_cache: bool = True,
        mixed_qkv: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if mixed_qkv is not None or "mamba_Input" in kwargs:
            (
                mixed_qkv,
                conv_weights,
                bias,
                activation,
                key_dim,
                value_dim,
                num_k_heads,
                num_v_heads,
                head_k_dim,
                head_v_dim,
                g,
                b_gate,
                w,
                layer_id,
                gate_mode,
                a_raw,
                b_raw,
                A_log,
                dt_bias,
            ) = self.gdn_helper._unpack_inputs(layer, mixed_qkv, kwargs)
            group = num_v_heads // num_k_heads if num_v_heads > num_k_heads else 1
            scale = head_k_dim ** -0.5
            return self.gdn_helper._forward_cpu(
                mixed_qkv=mixed_qkv,
                conv_weights=conv_weights,
                bias=bias,
                activation=activation,
                key_dim=key_dim,
                value_dim=value_dim,
                num_k_heads=num_k_heads,
                num_v_heads=num_v_heads,
                head_k_dim=head_k_dim,
                head_v_dim=head_v_dim,
                g=g,
                b=b_gate,
                w=w,
                conv_states=self.conv_states[layer_id],
                ssm_states=self.ssm_states[layer_id],
                cache_indices=forward_batch.cache_indices,
                forward_mode=forward_batch.forward_mode,
                has_initial_states=forward_batch.has_initial_states,
                query_start_loc=forward_batch.query_start_loc,
                group=group,
                scale=scale,
                gate_mode=gate_mode,
                a_raw=a_raw,
                b_raw=b_raw,
                A_log=A_log,
                dt_bias=dt_bias,
            )

        # Full Attention layer (`RadixAttention`) on CPU
        layer_id = int(layer.layer_id)
        n_heads = int(layer.tp_q_head_num)
        n_kv_heads = int(layer.tp_k_head_num)
        head_dim = int(layer.head_dim)
        group = n_heads // n_kv_heads
        scale = float(layer.scaling)

        q_3d = q.view(-1, n_heads, head_dim)
        k_3d = k.view(-1, n_kv_heads, head_dim)
        v_3d = v.view(-1, n_kv_heads, head_dim)

        k_buf = self.k_cache[layer_id]
        v_buf = self.v_cache[layer_id]
        lens_buf = self.kv_lens[layer_id]

        out_chunks: List[torch.Tensor] = []
        offset = 0
        for slot, seq_len in zip(forward_batch.slot_indices, forward_batch.seq_lens):
            q_seg = q_3d[offset : offset + seq_len].transpose(0, 1)  # [H, L, D]
            k_seg = k_3d[offset : offset + seq_len].transpose(0, 1)  # [H_kv, L, D]
            v_seg = v_3d[offset : offset + seq_len].transpose(0, 1)  # [H_kv, L, D]
            offset += seq_len

            cur_len = int(lens_buf[slot].item())
            new_len = cur_len + seq_len
            k_buf[slot, :, cur_len:new_len, :] = k_seg
            v_buf[slot, :, cur_len:new_len, :] = v_seg
            lens_buf[slot] = new_len

            k_full = k_buf[slot, :, :new_len, :]
            v_full = v_buf[slot, :, :new_len, :]
            if group > 1:
                k_full = k_full.repeat_interleave(group, dim=0)
                v_full = v_full.repeat_interleave(group, dim=0)

            scores = torch.matmul(q_seg.float(), k_full.float().transpose(-2, -1)) * scale
            if seq_len > 1:
                q_pos = torch.arange(cur_len, new_len, device=q.device).unsqueeze(1)
                k_pos = torch.arange(0, new_len, device=q.device).unsqueeze(0)
                causal_mask = k_pos <= q_pos
                scores = scores.masked_fill(~causal_mask.unsqueeze(0), -float("inf"))

            probs = torch.softmax(scores, dim=-1).to(v_full.dtype)
            attn_seg = torch.matmul(probs, v_full).transpose(0, 1).reshape(seq_len, n_heads * head_dim)
            out_chunks.append(attn_seg)

        return torch.cat(out_chunks, dim=0)

