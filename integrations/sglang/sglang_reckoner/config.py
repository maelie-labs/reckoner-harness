# Copyright 2026 Maelie Labs
# Copyright 2025 The Qwen Team, Alibaba Group, and the HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
#
# Notice of Modification (Apache-2.0 Section 4(b)):
# Modified by Maelie Labs (2026): `Qwen35GDN2Config` adapted from Hugging Face Transformers
# `Qwen3_5TextConfig` (`transformers.models.qwen3_5.configuration_qwen3_5`) with GDN-2
# low-rank gate delta configuration fields.
"""
Hugging Face and SGLang configuration for Reckoner hybrid GDN-2 + GQA models.
Can be used both inside `serve.sglang_reckoner` and as `configuration_reckoner.py`
in exported Hugging Face model directories (`auto_map` for `AutoConfig`).
"""

from typing import List, Optional, Tuple
import torch
from transformers import PretrainedConfig


class ReckonerConfig(PretrainedConfig):
    model_type = "reckoner"

    def __init__(
        self,
        vocab_size: int = 36864,
        d_model: int = 1536,
        hidden_size: Optional[int] = None,
        intermediate_size: int = 10240,
        num_hidden_layers: int = 20,
        n_heads: int = 12,
        num_attention_heads: Optional[int] = None,
        n_kv_heads: int = 3,
        num_key_value_heads: Optional[int] = None,
        head_dim: int = 128,
        rotary_dim: int = 64,
        layer_layout: str = "A G A G A G G G G G G G G G G A G A G A",
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 800000.0,
        sliding_window: Optional[int] = None,
        qk_norm: bool = True,
        use_gated_attn_output: bool = True,
        gated_attn_position: str = "pre_oproj",
        gdn_v_heads: int = 24,
        conv_kernel: int = 4,
        num_thinking_loops: int = 1,
        loop_start_layer: Optional[int] = None,
        loop_end_layer: Optional[int] = None,
        tie_word_embeddings: bool = True,
        max_position_embeddings: int = 163840,
        bos_token_id: int = 2,
        eos_token_id: int = 4,
        pad_token_id: int = 1,
        torch_dtype: str = "bfloat16",
        **kwargs,
    ):
        resolved_hidden = int(hidden_size if hidden_size is not None else d_model)
        resolved_heads = int(num_attention_heads if num_attention_heads is not None else n_heads)
        resolved_kv_heads = int(num_key_value_heads if num_key_value_heads is not None else n_kv_heads)

        self.vocab_size = int(vocab_size)
        self.d_model = resolved_hidden
        self.hidden_size = resolved_hidden
        self.intermediate_size = int(intermediate_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.n_heads = resolved_heads
        self.num_attention_heads = resolved_heads
        self.n_kv_heads = resolved_kv_heads
        self.num_key_value_heads = resolved_kv_heads
        self.head_dim = int(head_dim)
        self.rotary_dim = int(rotary_dim) if rotary_dim is not None else int(head_dim)
        self.layer_layout = str(layer_layout)
        self.rms_norm_eps = float(rms_norm_eps)
        self.rope_theta = float(rope_theta)
        self.sliding_window = int(sliding_window) if sliding_window is not None else None
        self.qk_norm = bool(qk_norm)
        self.use_gated_attn_output = bool(use_gated_attn_output)
        self.gated_attn_position = str(gated_attn_position)
        self.gdn_v_heads = int(gdn_v_heads) if gdn_v_heads is not None else resolved_heads
        self.conv_kernel = int(conv_kernel)
        self.num_thinking_loops = int(num_thinking_loops)
        self.loop_start_layer = loop_start_layer
        self.loop_end_layer = loop_end_layer
        self.max_position_embeddings = int(max_position_embeddings)

        # Linear (GDN-2) head dimensions used by SGLang MambaPool and backend
        self.linear_num_key_heads = self.n_heads
        self.linear_key_head_dim = self.head_dim
        self.linear_num_value_heads = self.gdn_v_heads
        self.linear_value_head_dim = self.head_dim
        self.linear_conv_kernel_dim = self.conv_kernel

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            torch_dtype=torch_dtype,
            **kwargs,
        )

    def parsed_layout(self) -> List[str]:
        tokens = self.layer_layout.strip().split()
        if len(tokens) == 1 and len(tokens[0]) == self.num_hidden_layers:
            tokens = list(tokens[0])
        if len(tokens) != self.num_hidden_layers:
            raise ValueError(
                f"layer_layout has {len(tokens)} ops ({tokens}), expected {self.num_hidden_layers}"
            )
        return tokens

    @property
    def linear_layer_ids(self) -> List[int]:
        return [i for i, op in enumerate(self.parsed_layout()) if op == "G"]

    @property
    def full_attention_layer_ids(self) -> List[int]:
        return [i for i, op in enumerate(self.parsed_layout()) if op in ("A", "S")]

    @property
    def mamba2_cache_params(self):
        from sglang.srt.configs.mamba_utils import (
            Mamba2CacheParams,
            Mamba2StateShape,
            mamba2_state_dtype,
        )

        try:
            from sglang.srt.distributed import get_tensor_model_parallel_world_size

            world_size = get_tensor_model_parallel_world_size()
        except Exception:
            world_size = 1
        q_dim = self.n_heads * self.head_dim
        k_dim = self.n_heads * self.head_dim
        v_dim = self.gdn_v_heads * self.head_dim
        shape = Mamba2StateShape.create(
            tp_world_size=world_size,
            intermediate_size=v_dim,
            n_groups=self.n_heads,
            num_heads=self.gdn_v_heads,
            head_dim=self.head_dim,
            state_size=self.head_dim,
            conv_kernel=self.conv_kernel,
            conv_shard_groups=[q_dim, k_dim, v_dim],
        )
        return Mamba2CacheParams(
            shape=shape,
            dtype=mamba2_state_dtype(self),
            layers=self.linear_layer_ids,
        )


class Qwen35GDN2Config(PretrainedConfig):
    """Hugging Face and SGLang configuration for Qwen3.5-2B + GDN-2 (rank-64 low-rank gate deltas).

    Subclasses `PretrainedConfig` directly so SGLang's `_LINEAR_ATTN_MODEL_REGISTRY` routes
    linear-attention layers to `ReckonerGDN2AttnBackend` rather than scalar `GDNAttnBackend`.
    """

    model_type = "qwen35_gdn2"

    def __init__(
        self,
        vocab_size: int = 248320,
        hidden_size: int = 2048,
        d_model: Optional[int] = None,
        intermediate_size: int = 6144,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 8,
        n_heads: Optional[int] = None,
        num_key_value_heads: int = 2,
        n_kv_heads: Optional[int] = None,
        head_dim: int = 256,
        attention_bias: bool = False,
        hidden_act: str = "silu",
        rms_norm_eps: float = 1e-6,
        full_attention_interval: int = 4,
        layer_types: Optional[List[str]] = None,
        linear_num_key_heads: int = 16,
        linear_key_head_dim: int = 128,
        linear_num_value_heads: int = 16,
        linear_value_head_dim: int = 128,
        linear_conv_kernel_dim: int = 4,
        gate_rank: int = 64,
        gate_delta_rank: Optional[int] = None,
        attn_output_gate: bool = True,
        partial_rotary_factor: float = 0.25,
        rope_theta: float = 10000000.0,
        rope_parameters: Optional[dict] = None,
        rotary_dim: Optional[int] = None,
        max_position_embeddings: int = 262144,
        tie_word_embeddings: bool = True,
        bos_token_id: int = 248044,
        eos_token_id: int = 248073,
        pad_token_id: int = 248070,
        banned_tokens: Optional[List[str]] = None,
        banned_token_ids: Optional[List[int]] = None,
        torch_dtype: str = "bfloat16",
        text_config: Optional[dict] = None,
        **kwargs,
    ):
        if isinstance(text_config, dict):
            vocab_size = int(text_config.get("vocab_size", vocab_size))
            hidden_size = int(text_config.get("hidden_size", hidden_size))
            intermediate_size = int(text_config.get("intermediate_size", intermediate_size))
            num_hidden_layers = int(text_config.get("num_hidden_layers", num_hidden_layers))
            num_attention_heads = int(text_config.get("num_attention_heads", num_attention_heads))
            num_key_value_heads = int(text_config.get("num_key_value_heads", num_key_value_heads))
            head_dim = int(text_config.get("head_dim", head_dim))
            attention_bias = bool(text_config.get("attention_bias", attention_bias))
            hidden_act = str(text_config.get("hidden_act", hidden_act))
            rms_norm_eps = float(text_config.get("rms_norm_eps", rms_norm_eps))
            full_attention_interval = int(text_config.get("full_attention_interval", full_attention_interval))
            layer_types = text_config.get("layer_types", layer_types)
            linear_num_key_heads = int(text_config.get("linear_num_key_heads", linear_num_key_heads))
            linear_key_head_dim = int(text_config.get("linear_key_head_dim", linear_key_head_dim))
            linear_num_value_heads = int(text_config.get("linear_num_value_heads", linear_num_value_heads))
            linear_value_head_dim = int(text_config.get("linear_value_head_dim", linear_value_head_dim))
            linear_conv_kernel_dim = int(text_config.get("linear_conv_kernel_dim", linear_conv_kernel_dim))
            max_position_embeddings = int(text_config.get("max_position_embeddings", max_position_embeddings))
            rope_parameters = text_config.get("rope_parameters", rope_parameters)
            if "partial_rotary_factor" in text_config:
                partial_rotary_factor = float(text_config["partial_rotary_factor"])
            if "rope_theta" in text_config:
                rope_theta = float(text_config["rope_theta"])

        if isinstance(rope_parameters, dict):
            rope_theta = float(rope_parameters.get("rope_theta", rope_theta))
            partial_rotary_factor = float(rope_parameters.get("partial_rotary_factor", partial_rotary_factor))

        resolved_hidden = int(d_model if d_model is not None else hidden_size)
        resolved_heads = int(n_heads if n_heads is not None else num_attention_heads)
        resolved_kv_heads = int(n_kv_heads if n_kv_heads is not None else num_key_value_heads)
        resolved_rank = int(gate_delta_rank if gate_delta_rank is not None else gate_rank)

        self.vocab_size = int(vocab_size)
        self.hidden_size = resolved_hidden
        self.d_model = resolved_hidden
        self.intermediate_size = int(intermediate_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = resolved_heads
        self.n_heads = resolved_heads
        self.num_key_value_heads = resolved_kv_heads
        self.n_kv_heads = resolved_kv_heads
        self.head_dim = int(head_dim)
        self.attention_bias = bool(attention_bias)
        self.attn_output_gate = bool(attn_output_gate)
        self.hidden_act = str(hidden_act)
        self.rms_norm_eps = float(rms_norm_eps)
        self.full_attention_interval = int(full_attention_interval)
        if layer_types is None:
            self.layer_types = [
                "full_attention" if (i + 1) % self.full_attention_interval == 0 else "linear_attention"
                for i in range(self.num_hidden_layers)
            ]
        else:
            self.layer_types = list(layer_types)

        self.linear_num_key_heads = int(linear_num_key_heads)
        self.linear_key_head_dim = int(linear_key_head_dim)
        self.linear_num_value_heads = int(linear_num_value_heads)
        self.linear_value_head_dim = int(linear_value_head_dim)
        self.linear_conv_kernel_dim = int(linear_conv_kernel_dim)
        self.conv_kernel = self.linear_conv_kernel_dim
        self.gdn_v_heads = self.linear_num_value_heads
        self.gate_rank = resolved_rank
        self.gate_delta_rank = resolved_rank

        self.partial_rotary_factor = float(partial_rotary_factor)
        self.rope_theta = float(rope_theta)
        self.rotary_dim = (
            int(rotary_dim)
            if rotary_dim is not None
            else int(round(self.head_dim * self.partial_rotary_factor))
        )
        clean_rope_params = dict(
            rope_parameters
            or {
                "rope_type": "default",
                "rope_theta": self.rope_theta,
                "partial_rotary_factor": self.partial_rotary_factor,
            }
        )
        clean_rope_params.pop("mrope_section", None)
        clean_rope_params.pop("mrope_interleaved", None)
        self.rope_parameters = clean_rope_params
        self.max_position_embeddings = int(max_position_embeddings)
        self.banned_tokens = list(banned_tokens) if banned_tokens is not None else ["<think>", "</think>"]
        self.banned_token_ids = [int(x) for x in (banned_token_ids if banned_token_ids is not None else [248068, 248069])]
        self.layer_layout = " ".join(self.parsed_layout())

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            torch_dtype=torch_dtype,
            **kwargs,
        )

    def parsed_layout(self) -> List[str]:
        return ["A" if lt == "full_attention" else "G" for lt in self.layer_types]

    @property
    def linear_layer_ids(self) -> List[int]:
        return [i for i, op in enumerate(self.parsed_layout()) if op == "G"]

    @property
    def full_attention_layer_ids(self) -> List[int]:
        return [i for i, op in enumerate(self.parsed_layout()) if op == "A"]

    @property
    def mamba2_cache_params(self):
        from sglang.srt.configs.mamba_utils import (
            Mamba2CacheParams,
            Mamba2StateShape,
            mamba2_state_dtype,
        )

        try:
            from sglang.srt.distributed import get_tensor_model_parallel_world_size

            world_size = get_tensor_model_parallel_world_size()
        except Exception:
            world_size = 1
        q_dim = self.linear_num_key_heads * self.linear_key_head_dim
        k_dim = self.linear_num_key_heads * self.linear_key_head_dim
        v_dim = self.linear_num_value_heads * self.linear_value_head_dim
        shape = Mamba2StateShape.create(
            tp_world_size=world_size,
            intermediate_size=v_dim,
            n_groups=self.linear_num_key_heads,
            num_heads=self.linear_num_value_heads,
            head_dim=self.linear_value_head_dim,
            state_size=self.linear_key_head_dim,
            conv_kernel=self.linear_conv_kernel_dim,
            conv_shard_groups=[q_dim, k_dim, v_dim],
        )
        return Mamba2CacheParams(
            shape=shape,
            dtype=mamba2_state_dtype(self),
            layers=self.linear_layer_ids,
        )

