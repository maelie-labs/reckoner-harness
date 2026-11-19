# Copyright 2026 Maelie Labs
# Copyright 2025 The Qwen Team, Alibaba Group, and the HuggingFace Inc. team.
# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
#
# Notice of Modification (Apache-2.0 Section 4(b)):
# Modified by Maelie Labs (2026): adapted from Hugging Face Transformers
# `transformers.models.qwen3_5.modeling_qwen3_5` (Qwen3.5-2B by Alibaba Cloud / Qwen Team)
# and SGLang v0.5.22 (`sglang.srt.models`), replacing Gated DeltaNet layers with GDN-2
# channel-wise log-decay, erase, and write gates dispatched through `RadixLinearAttention`
# and `ReckonerGDN2AttnBackend`.
"""
Out-of-tree SGLang model definition (`Qwen35GDN2ForCausalLM`) for the Qwen3.5-2B GDN-2 retrofit.

Architecture (24 layers, `full_attention_interval = 4`):
- 18 GDN-2 Linear Attention layers (`op_type == 'G'`, layers `[0,1,2, 4,5,6, 8,9,10, 12,13,14, 16,17,18, 20,21,22]`):
  16 Q/K heads (`head_k_dim = 128`), 16 V heads (`head_v_dim = 128`, `v_dim = 2048`),
  causal depthwise 1D convolution (`kernel_size = 4`, `activation = "silu"`, `bias = False`),
  rank-64 low-rank gate deltas (`delta_{g,b,w}_{down,up}`):
    base_a = a + dt_bias,  base_b = b
    g      = -exp(A_log) * softplus(base_a + delta_g(x))
    b_gate = sigmoid(base_b + delta_b(x))
    w_gate = sigmoid(base_b + delta_w(x))
  routed through SGLang `RadixLinearAttention` -> `ReckonerGDN2AttnBackend`,
  followed by `Qwen35RMSNormGated(head_v_dim=128, eps=1e-6)` and `out_proj`.
- 6 Full Attention layers (`op_type == 'A'`, layers `[3, 7, 11, 15, 19, 23]`):
  GQA (`8` Q heads, `2` KV heads, `head_dim = 256`), interleaved Q + output gate in `q_proj`
  (`attn_out * sigmoid(gate)` before `o_proj`), zero-centered `Qwen35RMSNorm` on Q and K (`eps = 1e-6`),
  partial RoPE (`rotary_dim = 64`, `partial_rotary_factor = 0.25`, `rope_theta = 10,000,000`),
  routed through SGLang `RadixAttention`.
- SwiGLU MLP (`hidden_size = 2048 -> intermediate_size = 6144 -> 2048`) on all 24 layers.
- Zero-centered `Qwen35RMSNorm` (`x_norm * (1 + weight)`) on `input_layernorm`,
  `post_attention_layernorm`, and final `model.norm`.
- Tied input/output token embeddings (`vocab_size = 248320, hidden_size = 2048`).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
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

from .config import Qwen35GDN2Config
from .gdn2_backend import (
    is_fusion_step_enabled,
    is_weight_sync_mode,
    reckoner_fused_qknorm_rope_fwd,
    reckoner_fused_sigmoid_mul_fwd,
    reckoner_fused_silu_mul_fwd,
    reckoner_qwen35_rmsnorm_fwd,
    reckoner_qwen35_rmsnorm_gated_fwd,
)


class Qwen35RotaryEmbedding(nn.Module):
    """CUDA-graph-safe partial RoPE matching `Qwen3_5TextRotaryEmbedding` + `apply_rotary_pos_emb` on GPU and CPU."""

    def __init__(
        self,
        rotary_dim: int = 64,
        max_position_embeddings: int = 262144,
        base: float = 10000000.0,
    ):
        super().__init__()
        self.rotary_dim = int(rotary_dim)
        self.max_position_embeddings = int(max_position_embeddings)
        self.base = float(base)
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        freqs = positions.float().unsqueeze(-1) * self.inv_freq.to(
            device=q.device, dtype=torch.float32
        ).unsqueeze(0)
        cos = freqs.cos()
        sin = freqs.sin()
        emb_cos = torch.cat((cos, cos), dim=-1).to(dtype=q.dtype).unsqueeze(1)
        emb_sin = torch.cat((sin, sin), dim=-1).to(dtype=q.dtype).unsqueeze(1)
        q_rot, q_pass = q[..., : self.rotary_dim], q[..., self.rotary_dim :]
        k_rot, k_pass = k[..., : self.rotary_dim], k[..., self.rotary_dim :]
        half = self.rotary_dim // 2
        q_r = torch.cat((-q_rot[..., half:], q_rot[..., :half]), dim=-1)
        k_r = torch.cat((-k_rot[..., half:], k_rot[..., :half]), dim=-1)
        q_rot = (q_rot * emb_cos) + (q_r * emb_sin)
        k_rot = (k_rot * emb_cos) + (k_r * emb_sin)
        return torch.cat((q_rot, q_pass), dim=-1), torch.cat((k_rot, k_pass), dim=-1)


class Qwen35RMSNorm(nn.Module):
    """Zero-centered RMSNorm matching `transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5RMSNorm`."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            x.is_cuda
            and x.dtype == torch.bfloat16
            and x.numel() > 0
            and is_fusion_step_enabled("f3")
        ):
            return reckoner_qwen35_rmsnorm_fwd(x, self.weight, self.eps)
        x_fp32 = x.float()
        out = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        out = out * (1.0 + self.weight.float())
        return out.type_as(x)


class Qwen35RMSNormGated(nn.Module):
    """Per-head gated RMSNorm matching `transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5RMSNormGated`."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.activation = "silu"

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        if (
            hidden_states.is_cuda
            and hidden_states.dtype == torch.bfloat16
            and gate.dtype == torch.bfloat16
            and hidden_states.numel() > 0
            and is_fusion_step_enabled("f3")
        ):
            return reckoner_qwen35_rmsnorm_gated_fwd(
                hidden_states, gate, self.weight, self.variance_epsilon
            )
        input_dtype = hidden_states.dtype
        h_fp32 = hidden_states.to(torch.float32)
        variance = h_fp32.pow(2).mean(-1, keepdim=True)
        h_norm = h_fp32 * torch.rsqrt(variance + self.variance_epsilon)
        h_weighted = self.weight * h_norm.to(input_dtype)
        out = h_weighted * F.silu(gate.to(torch.float32))
        return out.to(input_dtype)


class Qwen35MLP(nn.Module):
    """SwiGLU MLP matching `Qwen3_5MLP` (`2048 -> 6144 -> 2048`)."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self._fused_gate_up_weight: Optional[torch.Tensor] = None
        self._f2_merge_attempted: bool = False
        self._unmerged_freed: bool = False
        self._disable_f2: bool = False

    def build_fused_projections(self) -> bool:
        self._f2_merge_attempted = True
        if self._disable_f2:
            self._fused_gate_up_weight = None
            return False
        for mod in (self.gate_proj, self.up_proj):
            w = getattr(mod, "weight", None)
            if w is None or w.numel() == 0 or w.is_meta or w.dtype != torch.bfloat16:
                self._fused_gate_up_weight = None
                return False
        w_cat = torch.cat(
            [self.gate_proj.weight.data, self.up_proj.weight.data], dim=0
        ).contiguous()
        if (
            self._fused_gate_up_weight is not None
            and self._fused_gate_up_weight.shape == w_cat.shape
            and self._fused_gate_up_weight.dtype == w_cat.dtype
            and self._fused_gate_up_weight.device == w_cat.device
        ):
            self._fused_gate_up_weight.copy_(w_cat)
        else:
            self._fused_gate_up_weight = w_cat
        return True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.bfloat16 and is_fusion_step_enabled("f2") and not self._disable_f2:
            if self._fused_gate_up_weight is None and not self._f2_merge_attempted:
                self.build_fused_projections()
            elif (
                self._fused_gate_up_weight is not None
                and (
                    self._fused_gate_up_weight.device != x.device
                    or self._fused_gate_up_weight.dtype != x.dtype
                )
                and not self._unmerged_freed
            ):
                self.build_fused_projections()
            if (
                self._fused_gate_up_weight is not None
                and self._fused_gate_up_weight.device == x.device
                and self._fused_gate_up_weight.dtype == x.dtype
            ):
                gate_up = F.linear(x, self._fused_gate_up_weight)
                gate = gate_up[..., : self.intermediate_size]
                up = gate_up[..., self.intermediate_size :]
            else:
                gate = self.gate_proj(x)
                up = self.up_proj(x)
        else:
            gate = self.gate_proj(x)
            up = self.up_proj(x)

        if (
            x.is_cuda
            and x.dtype == torch.bfloat16
            and x.numel() > 0
            and is_fusion_step_enabled("f3")
        ):
            act = reckoner_fused_silu_mul_fwd(gate, up)
        else:
            act = F.silu(gate) * up
        return self.down_proj(act)


class Qwen35FullAttentionLayer(nn.Module):
    """Full GQA attention layer for Qwen3.5 (`8` Q heads, `2` KV heads, `head_dim=256`, interleaved Q+gate)."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = int(getattr(config, "hidden_size", 2048))
        self.num_heads = int(getattr(config, "num_attention_heads", 8))
        self.num_kv_heads = int(getattr(config, "num_key_value_heads", 2))
        self.head_dim = int(getattr(config, "head_dim", 256))
        self.q_dim = self.num_heads * self.head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        attn_bias = bool(getattr(config, "attention_bias", False))
        eps = float(getattr(config, "rms_norm_eps", 1e-6))

        self.q_proj = nn.Linear(self.hidden_size, self.q_dim * 2, bias=attn_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=attn_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=attn_bias)
        self.o_proj = nn.Linear(self.q_dim, self.hidden_size, bias=attn_bias)

        self.q_norm = Qwen35RMSNorm(self.head_dim, eps=eps)
        self.k_norm = Qwen35RMSNorm(self.head_dim, eps=eps)
        self._fused_qkv_weight: Optional[torch.Tensor] = None
        self._f2_merge_attempted: bool = False
        self._unmerged_freed: bool = False
        self._disable_f2: bool = False

        self.attn = RadixAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            scaling=self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            sliding_window_size=-1,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def build_fused_projections(self) -> bool:
        self._f2_merge_attempted = True
        if (
            self._disable_f2
            or self.q_proj.bias is not None
            or self.k_proj.bias is not None
            or self.v_proj.bias is not None
        ):
            self._fused_qkv_weight = None
            return False
        for mod in (self.q_proj, self.k_proj, self.v_proj):
            w = getattr(mod, "weight", None)
            if w is None or w.numel() == 0 or w.is_meta or w.dtype != torch.bfloat16:
                self._fused_qkv_weight = None
                return False
        qw_view = self.q_proj.weight.data.view(
            self.num_heads, 2, self.head_dim, self.hidden_size
        )
        q_rows = qw_view[:, 0, :, :].reshape(self.q_dim, self.hidden_size)
        gate_rows = qw_view[:, 1, :, :].reshape(self.q_dim, self.hidden_size)
        w_cat = torch.cat(
            [q_rows, gate_rows, self.k_proj.weight.data, self.v_proj.weight.data], dim=0
        ).contiguous()
        if (
            self._fused_qkv_weight is not None
            and self._fused_qkv_weight.shape == w_cat.shape
            and self._fused_qkv_weight.dtype == w_cat.dtype
            and self._fused_qkv_weight.device == w_cat.device
        ):
            self._fused_qkv_weight.copy_(w_cat)
        else:
            self._fused_qkv_weight = w_cat
        return True

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        rotary_emb: Qwen35RotaryEmbedding,
    ) -> torch.Tensor:
        use_f2 = (
            hidden_states.dtype == torch.bfloat16
            and is_fusion_step_enabled("f2")
            and not self._disable_f2
            and self.q_proj.bias is None
        )
        if use_f2:
            if self._fused_qkv_weight is None and not self._f2_merge_attempted:
                self.build_fused_projections()
            elif (
                self._fused_qkv_weight is not None
                and (
                    self._fused_qkv_weight.device != hidden_states.device
                    or self._fused_qkv_weight.dtype != hidden_states.dtype
                )
                and not self._unmerged_freed
            ):
                self.build_fused_projections()
        if (
            use_f2
            and self._fused_qkv_weight is not None
            and self._fused_qkv_weight.device == hidden_states.device
            and self._fused_qkv_weight.dtype == hidden_states.dtype
        ):
            qkvg = F.linear(hidden_states, self._fused_qkv_weight)
            query_states = qkvg[:, : self.q_dim].view(
                -1, self.num_heads, self.head_dim
            )
            gate = qkvg[:, self.q_dim : 2 * self.q_dim]
            k_raw = qkvg[:, 2 * self.q_dim : 2 * self.q_dim + self.kv_dim].view(
                -1, self.num_kv_heads, self.head_dim
            )
            v = (
                qkvg[:, 2 * self.q_dim + self.kv_dim : 2 * self.q_dim + 2 * self.kv_dim]
                .contiguous()
                .view(-1, self.num_kv_heads, self.head_dim)
            )
        else:
            q_and_gate = self.q_proj(hidden_states).view(
                -1, self.num_heads, self.head_dim * 2
            )
            query_states, gate = torch.chunk(q_and_gate, 2, dim=-1)
            gate = gate.reshape(-1, self.q_dim)
            k_raw = self.k_proj(hidden_states).view(
                -1, self.num_kv_heads, self.head_dim
            )
            v = self.v_proj(hidden_states).view(-1, self.num_kv_heads, self.head_dim)

        if (
            hidden_states.is_cuda
            and query_states.dtype == torch.bfloat16
            and k_raw.dtype == torch.bfloat16
            and hidden_states.numel() > 0
            and is_fusion_step_enabled("f4")
        ):
            if rotary_emb.inv_freq.device != hidden_states.device:
                rotary_emb.inv_freq = rotary_emb.inv_freq.to(hidden_states.device)
            q, k = reckoner_fused_qknorm_rope_fwd(
                query_states,
                k_raw,
                self.q_norm.weight,
                self.k_norm.weight,
                positions,
                eps=self.q_norm.eps,
                rotary_dim=rotary_emb.rotary_dim,
                weight_offset=1.0,
                inv_freq=rotary_emb.inv_freq,
            )
        else:
            q = self.q_norm(query_states.contiguous())
            k = self.k_norm(k_raw.contiguous())
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
        if (
            attn_out.is_cuda
            and attn_out.dtype == torch.bfloat16
            and gate.dtype == torch.bfloat16
            and attn_out.numel() > 0
            and is_fusion_step_enabled("f3")
        ):
            attn_out = reckoner_fused_sigmoid_mul_fwd(attn_out, gate, scale=1.0)
        else:
            attn_out = attn_out * torch.sigmoid(gate)
        return self.o_proj(attn_out)


class Qwen35GDN2Layer(nn.Module):
    """GDN-2 linear attention layer with rank-64 low-rank gate deltas for SGLang (`ReckonerGDN2AttnBackend`)."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = int(getattr(config, "hidden_size", 2048))
        self.num_v_heads = int(getattr(config, "linear_num_value_heads", 16))
        self.num_k_heads = int(getattr(config, "linear_num_key_heads", 16))
        self.head_k_dim = int(getattr(config, "linear_key_head_dim", 128))
        self.head_v_dim = int(getattr(config, "linear_value_head_dim", 128))
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = int(getattr(config, "linear_conv_kernel_dim", 4))
        self.activation = str(getattr(config, "hidden_act", "silu"))
        self.layer_norm_epsilon = float(getattr(config, "rms_norm_eps", 1e-6))
        self.gate_rank = int(getattr(config, "gate_rank", 64))

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads))

        self.norm = Qwen35RMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        self.in_proj_qkv = nn.Linear(self.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

        self.delta_g_down = nn.Linear(self.hidden_size, self.gate_rank, bias=False)
        self.delta_g_up = nn.Linear(self.gate_rank, self.value_dim, bias=False)
        self.delta_b_down = nn.Linear(self.hidden_size, self.gate_rank, bias=False)
        self.delta_b_up = nn.Linear(self.gate_rank, self.value_dim, bias=False)
        self.delta_w_down = nn.Linear(self.hidden_size, self.gate_rank, bias=False)
        self.delta_w_up = nn.Linear(self.gate_rank, self.value_dim, bias=False)

        self._fused_in_proj_weight: Optional[torch.Tensor] = None
        self._fused_delta_up_weight: Optional[torch.Tensor] = None
        self._f2_merge_attempted: bool = False
        self._unmerged_freed: bool = False
        self._disable_f2: bool = False

        self.attn = RadixLinearAttention(
            layer_id=layer_id,
            num_q_heads=self.num_k_heads,
            num_k_heads=self.num_k_heads,
            num_v_heads=self.num_v_heads,
            head_q_dim=self.head_k_dim,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            conv_weights=self.conv1d.weight.squeeze(1),
            bias=self.conv1d.bias,
            activation=self.activation,
        )
        if hasattr(AttentionType, "LINEAR_ATTENTION"):
            self.attn.attn_type = AttentionType.LINEAR_ATTENTION

    def build_fused_projections(self) -> bool:
        self._f2_merge_attempted = True
        if self._disable_f2:
            self._fused_in_proj_weight = None
            self._fused_delta_up_weight = None
            return False
        all_mods = (
            self.in_proj_qkv,
            self.in_proj_z,
            self.in_proj_b,
            self.in_proj_a,
            self.delta_g_down,
            self.delta_b_down,
            self.delta_w_down,
            self.delta_g_up,
            self.delta_b_up,
            self.delta_w_up,
        )
        for mod in all_mods:
            w = getattr(mod, "weight", None)
            if w is None or w.numel() == 0 or w.is_meta or w.dtype != torch.bfloat16:
                self._fused_in_proj_weight = None
                self._fused_delta_up_weight = None
                return False
        w_in = torch.cat(
            [
                self.in_proj_qkv.weight.data,
                self.in_proj_z.weight.data,
                self.in_proj_b.weight.data,
                self.in_proj_a.weight.data,
                self.delta_g_down.weight.data,
                self.delta_b_down.weight.data,
                self.delta_w_down.weight.data,
            ],
            dim=0,
        ).contiguous()
        w_up = torch.stack(
            [
                self.delta_g_up.weight.data,
                self.delta_b_up.weight.data,
                self.delta_w_up.weight.data,
            ],
            dim=0,
        ).contiguous()
        if (
            self._fused_in_proj_weight is not None
            and self._fused_in_proj_weight.shape == w_in.shape
            and self._fused_in_proj_weight.dtype == w_in.dtype
            and self._fused_in_proj_weight.device == w_in.device
        ):
            self._fused_in_proj_weight.copy_(w_in)
        else:
            self._fused_in_proj_weight = w_in
        if (
            self._fused_delta_up_weight is not None
            and self._fused_delta_up_weight.shape == w_up.shape
            and self._fused_delta_up_weight.dtype == w_up.dtype
            and self._fused_delta_up_weight.device == w_up.device
        ):
            self._fused_delta_up_weight.copy_(w_up)
        else:
            self._fused_delta_up_weight = w_up
        return True

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        use_f2 = (
            hidden_states.dtype == torch.bfloat16
            and is_fusion_step_enabled("f2")
            and not self._disable_f2
        )
        if use_f2:
            if (
                (self._fused_in_proj_weight is None or self._fused_delta_up_weight is None)
                and not self._f2_merge_attempted
            ):
                self.build_fused_projections()
            elif (
                self._fused_in_proj_weight is not None
                and self._fused_delta_up_weight is not None
                and (
                    self._fused_in_proj_weight.device != hidden_states.device
                    or self._fused_in_proj_weight.dtype != hidden_states.dtype
                )
                and not self._unmerged_freed
            ):
                self.build_fused_projections()
        if (
            use_f2
            and self._fused_in_proj_weight is not None
            and self._fused_delta_up_weight is not None
            and self._fused_in_proj_weight.device == hidden_states.device
            and self._fused_in_proj_weight.dtype == hidden_states.dtype
        ):
            proj = F.linear(hidden_states, self._fused_in_proj_weight)
            o0 = self.conv_dim
            o1 = o0 + self.value_dim
            o2 = o1 + self.num_v_heads
            o3 = o2 + self.num_v_heads
            r = self.gate_rank
            mixed_qkv = proj[:, :o0]
            z = proj[:, o0:o1]
            b_scalar = proj[:, o1:o2]
            a_scalar = proj[:, o2:o3]
            d_low_all = proj[:, o3 : o3 + 3 * r]
            if hidden_states.shape[0] == 1:
                d_low_bmm = d_low_all.view(3, 1, r)
            else:
                d_low_bmm = d_low_all.view(-1, 3, r).transpose(0, 1).contiguous()
            d_up = torch.bmm(d_low_bmm, self._fused_delta_up_weight.transpose(1, 2))
            delta_g_2d = d_up[0]
            delta_b_2d = d_up[1]
            delta_w_2d = d_up[2]
        else:
            mixed_qkv = self.in_proj_qkv(hidden_states)
            z = self.in_proj_z(hidden_states)
            b_scalar = self.in_proj_b(hidden_states)
            a_scalar = self.in_proj_a(hidden_states)
            delta_g_2d = self.delta_g_up(self.delta_g_down(hidden_states))
            delta_b_2d = self.delta_b_up(self.delta_b_down(hidden_states))
            delta_w_2d = self.delta_w_up(self.delta_w_down(hidden_states))

        if hidden_states.dtype == torch.bfloat16 and is_fusion_step_enabled("f1"):
            mamba_input = {
                "mixed_qkv": mixed_qkv,
                "conv_weights": self.conv1d.weight.squeeze(1),
                "bias": self.conv1d.bias,
                "activation": self.activation,
                "key_dim": self.key_dim,
                "value_dim": self.value_dim,
                "num_k_heads": self.num_k_heads,
                "num_v_heads": self.num_v_heads,
                "head_k_dim": self.head_k_dim,
                "head_v_dim": self.head_v_dim,
                "g": delta_g_2d,
                "b": delta_b_2d,
                "w": delta_w_2d,
                "layer_id": self.layer_id,
                "gate_mode": "q2b",
                "a_raw": a_scalar,
                "b_raw": b_scalar,
                "A_log": self.A_log,
                "dt_bias": self.dt_bias,
            }
        else:
            delta_g = delta_g_2d.view(-1, self.num_v_heads, self.head_v_dim)
            delta_b = delta_b_2d.view(-1, self.num_v_heads, self.head_v_dim)
            delta_w = delta_w_2d.view(-1, self.num_v_heads, self.head_v_dim)

            base_a = (a_scalar.float() + self.dt_bias.float()).unsqueeze(-1)
            base_b = b_scalar.float().unsqueeze(-1)
            A = self.A_log.float().exp().view(1, self.num_v_heads, 1)

            g = (-A * F.softplus(base_a + delta_g.float())).reshape(-1, self.value_dim)
            b_gate = (
                torch.sigmoid(base_b + delta_b.float())
                .to(hidden_states.dtype)
                .reshape(-1, self.value_dim)
            )
            w_gate = (
                torch.sigmoid(base_b + delta_w.float())
                .to(hidden_states.dtype)
                .reshape(-1, self.value_dim)
            )

            mamba_input = {
                "mixed_qkv": mixed_qkv,
                "conv_weights": self.conv1d.weight.squeeze(1),
                "bias": self.conv1d.bias,
                "activation": self.activation,
                "key_dim": self.key_dim,
                "value_dim": self.value_dim,
                "num_k_heads": self.num_k_heads,
                "num_v_heads": self.num_v_heads,
                "head_k_dim": self.head_k_dim,
                "head_v_dim": self.head_v_dim,
                "g": g,
                "b": b_gate,
                "w": w_gate,
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
        if (
            core_out.is_cuda
            and core_out.dtype == torch.bfloat16
            and core_out.numel() > 0
            and is_fusion_step_enabled("f3")
        ):
            out_3d = core_out.view(-1, self.num_v_heads, self.head_v_dim)
            z_3d = z.view(-1, self.num_v_heads, self.head_v_dim)
            out = self.norm(out_3d, z_3d).view(-1, self.value_dim)
        else:
            out = core_out.reshape(-1, self.head_v_dim)
            z_flat = z.reshape(-1, self.head_v_dim)
            out = self.norm(out, z_flat).reshape(-1, self.value_dim)
        return self.out_proj(out)


class Qwen35GDN2DecoderLayer(nn.Module):
    """Single hybrid decoder block for Qwen3.5 + GDN-2 (`op_type` in `{'G', 'A'}`)."""

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
        hidden_size = int(getattr(config, "hidden_size", 2048))
        eps = float(getattr(config, "rms_norm_eps", 1e-6))

        self.input_layernorm = Qwen35RMSNorm(hidden_size, eps=eps)
        if op_type == "G":
            self.linear_attn = Qwen35GDN2Layer(
                config=config,
                layer_id=layer_idx,
                prefix=add_prefix("linear_attn", prefix),
            )
        elif op_type == "A":
            self.self_attn = Qwen35FullAttentionLayer(
                config=config,
                layer_id=layer_idx,
                quant_config=quant_config,
                prefix=add_prefix("self_attn", prefix),
            )
        else:
            raise ValueError(f"Unsupported layer operator '{op_type}'")

        self.post_attention_layernorm = Qwen35RMSNorm(hidden_size, eps=eps)
        self.mlp = Qwen35MLP(hidden_size, int(getattr(config, "intermediate_size", 6144)))

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        rotary_emb: Qwen35RotaryEmbedding,
    ) -> torch.Tensor:
        normed = self.input_layernorm(hidden_states)
        if self.op_type == "A":
            mixer_out = self.self_attn(
                positions=positions,
                hidden_states=normed,
                forward_batch=forward_batch,
                rotary_emb=rotary_emb,
            )
        else:
            mixer_out = self.linear_attn(
                hidden_states=normed,
                forward_batch=forward_batch,
            )
        hidden_states = hidden_states + mixer_out
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        return hidden_states


class Qwen35GDN2ForCausalLM(nn.Module):
    """Out-of-tree SGLang model class for Reckoner-Q2B (Qwen3.5-2B GDN-2 retrofit)."""

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

            set_active_model_arch("Qwen35GDN2ForCausalLM")
        except Exception:
            pass
        self.config = config
        self.quant_config = quant_config
        self.default_attn_backend = attn_backend
        self.hidden_size = int(getattr(config, "hidden_size", 2048))
        self.d_model = self.hidden_size
        self.vocab_size = int(getattr(config, "vocab_size", 248320))
        self._unmerged_weights_freed = False
        self.banned_tokens: List[str] = list(
            getattr(config, "banned_tokens", ["<think>", "</think>"])
        )
        self.banned_token_ids: List[int] = [
            int(x) for x in getattr(config, "banned_token_ids", [248068, 248069])
        ]

        try:
            self.embed_tokens: nn.Module = VocabParallelEmbedding(
                self.vocab_size,
                self.hidden_size,
                prefix=add_prefix("embed_tokens", prefix),
            )
            self._sglang_parallel_active = True
        except Exception:
            self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)
            self._sglang_parallel_active = False

        head_dim = int(getattr(config, "head_dim", 256))
        partial_rotary = float(getattr(config, "partial_rotary_factor", 0.25))
        rotary_dim = int(
            getattr(config, "rotary_dim", None) or round(head_dim * partial_rotary)
        )
        rope_theta = float(getattr(config, "rope_theta", 10000000.0))
        if hasattr(config, "rope_parameters") and isinstance(config.rope_parameters, dict):
            rope_theta = float(config.rope_parameters.get("rope_theta", rope_theta))

        self.rotary_emb = Qwen35RotaryEmbedding(
            rotary_dim=rotary_dim,
            max_position_embeddings=max(
                int(getattr(config, "max_position_embeddings", 262144)), 163840
            ),
            base=rope_theta,
        )

        if hasattr(config, "parsed_layout"):
            layout = config.parsed_layout()
        elif hasattr(config, "layer_types") and config.layer_types:
            layout = [
                "A" if lt == "full_attention" else "G" for lt in config.layer_types
            ]
        else:
            interval = int(getattr(config, "full_attention_interval", 4))
            n_layers = int(getattr(config, "num_hidden_layers", 24))
            layout = [
                "A" if (i + 1) % interval == 0 else "G" for i in range(n_layers)
            ]

        self.layers = nn.ModuleList(
            [
                Qwen35GDN2DecoderLayer(
                    config=config,
                    layer_idx=idx,
                    op_type=op,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{idx}", prefix),
                )
                for idx, op in enumerate(layout)
            ]
        )
        self.norm = Qwen35RMSNorm(
            self.hidden_size, eps=float(getattr(config, "rms_norm_eps", 1e-6))
        )

        if getattr(config, "tie_word_embeddings", True):
            self.lm_head: nn.Module = self.embed_tokens
        elif self._sglang_parallel_active:
            self.lm_head = ParallelLMHead(
                self.vocab_size,
                self.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )
        else:
            self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

        self.logits_processor = (
            LogitsProcessor(config) if self._sglang_parallel_active else None
        )

        pt_ckpt = getattr(config, "pt_checkpoint_path", None) or os.environ.get(
            "QWEN35_GDN2_PT_CKPT"
        )
        if pt_ckpt and os.path.isfile(str(pt_ckpt)) and not self._sglang_parallel_active:
            self.load_pt_checkpoint(str(pt_ckpt))

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
        if (
            getattr(forward_batch, "attn_backend", None) is None
            and self.default_attn_backend is not None
        ):
            forward_batch.attn_backend = self.default_attn_backend

        is_standalone_cpu = bool(getattr(forward_batch, "_standalone_cpu", False))
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
        if self.logits_processor is None or is_standalone_cpu:
            logits = F.linear(hidden_states, self.lm_head.weight)
            if (
                self.banned_token_ids
                and os.environ.get("QWEN35_MASK_BANNED_IN_FORWARD", "1") == "1"
            ):
                for tid in self.banned_token_ids:
                    logits[..., tid] = -float("inf")
            return logits

        out = self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )
        if (
            self.banned_token_ids
            and hasattr(out, "next_token_logits")
            and out.next_token_logits is not None
            and os.environ.get("QWEN35_MASK_BANNED_IN_FORWARD", "1") == "1"
        ):
            for tid in self.banned_token_ids:
                out.next_token_logits[:, tid] = -float("inf")
        return out

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set[str]:
        if self._unmerged_weights_freed:
            raise RuntimeError(
                "Cannot call load_weights / update_weights_from_tensor after unmerged projection "
                "weights were freed by f2 fusion. Set SGLANG_RECKONER_RL_WEIGHT_SYNC=1 (or "
                "SGLANG_RECKONER_KEEP_UNMERGED_WEIGHTS=1) when starting the engine for live RL weight sync."
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
                .removeprefix("model.language_model.")
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
                    f"Unexpected weight key '{raw_name}' (normalized '{name}') not in Qwen35GDN2ForCausalLM"
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
            if hasattr(layer, "linear_attn"):
                layer.linear_attn._disable_f2 = weight_sync
                layer.linear_attn._f2_merge_attempted = False
            if hasattr(layer, "self_attn"):
                layer.self_attn._disable_f2 = weight_sync
                layer.self_attn._f2_merge_attempted = False
            layer.mlp._disable_f2 = weight_sync
            layer.mlp._f2_merge_attempted = False

        if is_fusion_step_enabled("f2") and not weight_sync:
            self.build_fused_projections(
                free_unmerged=self._sglang_parallel_active and not keep_unmerged
            )
        else:
            for layer in self.layers:
                if hasattr(layer, "linear_attn"):
                    layer.linear_attn._fused_in_proj_weight = None
                    layer.linear_attn._fused_delta_up_weight = None
                if hasattr(layer, "self_attn"):
                    layer.self_attn._fused_qkv_weight = None
                layer.mlp._fused_gate_up_weight = None

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
            if hasattr(layer, "linear_attn"):
                la = layer.linear_attn
                if la.build_fused_projections() and do_free:
                    for mod in (
                        la.in_proj_qkv,
                        la.in_proj_z,
                        la.in_proj_b,
                        la.in_proj_a,
                        la.delta_g_down,
                        la.delta_b_down,
                        la.delta_w_down,
                        la.delta_g_up,
                        la.delta_b_up,
                        la.delta_w_up,
                    ):
                        mod.weight = nn.Parameter(
                            torch.empty(0, dtype=mod.weight.dtype, device=mod.weight.device),
                            requires_grad=False,
                        )
                    la._unmerged_freed = True
                    any_freed = True
            if hasattr(layer, "self_attn"):
                sa = layer.self_attn
                if sa.build_fused_projections() and do_free:
                    for mod in (sa.q_proj, sa.k_proj, sa.v_proj):
                        mod.weight = nn.Parameter(
                            torch.empty(0, dtype=mod.weight.dtype, device=mod.weight.device),
                            requires_grad=False,
                        )
                    sa._unmerged_freed = True
                    any_freed = True
            if layer.mlp.build_fused_projections() and do_free:
                for mod in (layer.mlp.gate_proj, layer.mlp.up_proj):
                    mod.weight = nn.Parameter(
                        torch.empty(0, dtype=mod.weight.dtype, device=mod.weight.device),
                        requires_grad=False,
                    )
                layer.mlp._unmerged_freed = True
                any_freed = True
        if any_freed:
            self._unmerged_weights_freed = True

    def load_pt_checkpoint(self, pt_path: str) -> set[str]:
        """Load weights directly from a `.pt` checkpoint (`model_state_dict` or raw state_dict)."""
        ckpt = torch.load(pt_path, map_location="cpu", weights_only=False, mmap=True)
        sd = (
            ckpt.get("model_state_dict", ckpt.get("model", ckpt))
            if isinstance(ckpt, dict)
            else ckpt
        )
        target_dtype = next(self.parameters()).dtype
        items = [
            (k, v.to(dtype=target_dtype) if v.is_floating_point() else v)
            for k, v in sd.items()
        ]
        return self.load_weights(items)

    @classmethod
    def from_pt_checkpoint(
        cls,
        pt_path: str,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.bfloat16,
        attn_backend: Optional[Any] = None,
    ) -> "Qwen35GDN2ForCausalLM":
        """Instantiate `Qwen35GDN2ForCausalLM` and load weights directly from a `.pt` checkpoint."""
        ckpt = torch.load(pt_path, map_location="cpu", weights_only=False, mmap=True)
        raw_cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
        gate_rank = int(raw_cfg.get("gate_rank", 64)) if isinstance(raw_cfg, dict) else 64
        qcfg = Qwen35GDN2Config(
            gate_rank=gate_rank,
            torch_dtype="bfloat16" if dtype == torch.bfloat16 else "float32",
        )
        if attn_backend is None and torch.device(device).type == "cpu":
            from .gdn2_backend import ReckonerGDN2AttnBackend

            attn_backend = ReckonerGDN2AttnBackend.create_standalone(
                qcfg, device=device, dtype=dtype
            )
        model = cls(qcfg, attn_backend=attn_backend).to(device=device, dtype=dtype)
        sd = (
            ckpt.get("model_state_dict", ckpt.get("model", ckpt))
            if isinstance(ckpt, dict)
            else ckpt
        )
        items = [
            (k, v.to(dtype=dtype) if v.is_floating_point() else v)
            for k, v in sd.items()
        ]
        model.load_weights(items)
        model.eval()
        return model


EntryClass = Qwen35GDN2ForCausalLM
