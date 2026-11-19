# Copyright 2026 Maelie Labs
# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
#
# Notice of Modification (Apache-2.0 Section 4(b)):
# Modified by Maelie Labs (2026): hooks SGLang v0.5.22 model/backend registries
# (`sglang.srt.models.registry`, `sglang.srt.configs.linear_attn_model_registry`)
# and `OpenAIServingChat._apply_jinja_template` for Reckoner front-door enforcement.
"""Out-of-tree SGLang 0.5.22 plugin for Project Reckoner (`Reckoner-1.4B` and `Reckoner-Q2B`).

Registers `ReckonerConfig`, `ReckonerForCausalLM`, `Qwen35GDN2Config`,
`Qwen35GDN2ForCausalLM`, and `ReckonerGDN2AttnBackend` using SGLang's
registration APIs without modifying SGLang source files, and enforces
front-door serving via `reckoner-harness serve` on `/v1/chat/completions`
unless `--allow-raw-chat` is explicitly enabled for development.
"""

from __future__ import annotations

import logging
import os
import sys
import types
from typing import Any, Optional

from . import config as _config_mod
from . import gdn2_backend as _gdn2_backend_mod
from . import harness_bridge as _harness_bridge_mod
from . import modeling_qwen35_gdn2 as _modeling_qwen35_mod
from . import modeling_reckoner as _modeling_reckoner_mod
from . import tool_processor as _tool_processor_mod
from .config import Qwen35GDN2Config, ReckonerConfig
from .gdn2_backend import ReckonerGDN2AttnBackend
from .harness_bridge import format_dialogue, format_prompt
from .modeling_qwen35_gdn2 import (
    Qwen35FullAttentionLayer,
    Qwen35GDN2DecoderLayer,
    Qwen35GDN2ForCausalLM,
    Qwen35GDN2Layer,
    Qwen35MLP,
    Qwen35RMSNorm,
    Qwen35RMSNormGated,
)
from .modeling_reckoner import (
    ReckonerDecoderLayer,
    ReckonerForCausalLM,
    ReckonerFullAttentionLayer,
    ReckonerGDN2Layer,
    ReckonerMLP,
    ReckonerRMSNorm,
)
from .tool_processor import (
    ReckonerToolLogitProcessor,
    supports_inflight_extend,
)

# Register module aliases so both `sglang_reckoner.*` (standalone harness repo)
# and `serve.sglang_reckoner.*` (research repo / SGLang dynamic class loader) resolve identically.
_self_mod = sys.modules[__name__]
sys.modules.setdefault("sglang_reckoner", _self_mod)
if "serve" not in sys.modules:
    _serve_pkg = types.ModuleType("serve")
    _serve_pkg.__path__ = []  # type: ignore[attr-defined]
    sys.modules["serve"] = _serve_pkg
sys.modules.setdefault("serve.sglang_reckoner", _self_mod)
for _sub_name, _sub_mod in (
    ("config", _config_mod),
    ("gdn2_backend", _gdn2_backend_mod),
    ("harness_bridge", _harness_bridge_mod),
    ("modeling_qwen35_gdn2", _modeling_qwen35_mod),
    ("modeling_reckoner", _modeling_reckoner_mod),
    ("tool_processor", _tool_processor_mod),
):
    sys.modules.setdefault(f"sglang_reckoner.{_sub_name}", _sub_mod)
    sys.modules.setdefault(f"serve.sglang_reckoner.{_sub_name}", _sub_mod)

logger = logging.getLogger(__name__)

RECKONER_VOCAB_SIZE = 36864
RECKONER_FRONT_DOOR_ERROR_MSG = (
    "Reckoner models run only through the Reckoner harness (reckoner-harness serve). "
    "See the model card for install and run commands."
)
RECKONER_ERRORING_CHAT_TEMPLATE = (
    '{{- raise_exception("Reckoner models run only through the Reckoner harness '
    '(reckoner-harness serve). See the model card for install and run commands.") -}}'
)

_PLUGIN_REGISTERED = False
_ACTIVE_MODEL_ARCH: Optional[str] = None


def set_active_model_arch(arch: Optional[str]) -> None:
    """Record the architecture name of the currently loaded model in this process."""
    global _ACTIVE_MODEL_ARCH
    _ACTIVE_MODEL_ARCH = arch


def get_active_model_arch() -> Optional[str]:
    """Return the architecture name of the currently loaded model in this process."""
    return _ACTIVE_MODEL_ARCH


def is_reckoner_model_active() -> bool:
    """Return True iff the currently loaded model in this process is `ReckonerForCausalLM`."""
    return _ACTIVE_MODEL_ARCH == "ReckonerForCausalLM"


def is_raw_chat_allowed() -> bool:
    """Return True iff the developer `--allow-raw-chat` flag (`SGLANG_RECKONER_ALLOW_RAW_CHAT=1`) is enabled."""
    return os.environ.get("SGLANG_RECKONER_ALLOW_RAW_CHAT", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def canonicalize_turn_input_ids(ids: list[int]) -> list[int]:
    """Ensure token IDs conform to canonical Reckoner turn-boundary newlines (`harness_bridge.py`).

    - `<|turn:user|>` (2) is immediately followed by `\\n` (326).
    - `<|turn:system|>` (5) is immediately followed by `\\n` (326) unless followed by `<|tools|>` (6).
    - `<|/turn|>` (4) is immediately followed by `\\n` (326) when followed by another turn (2, 3, 5).
    Leaves sequences containing any token outside Reckoner's vocabulary (`[0, 36864)`) untouched,
    and is idempotent on already-canonical token sequences.
    """
    if not ids:
        return ids
    if any(not isinstance(t, int) or t < 0 or t >= RECKONER_VOCAB_SIZE for t in ids):
        return ids
    out: list[int] = []
    n = len(ids)
    for i, tok in enumerate(ids):
        out.append(tok)
        nxt = ids[i + 1] if i + 1 < n else -1
        if tok == 2:
            if nxt not in (326, -1):
                out.append(326)
        elif tok == 5:
            if nxt not in (326, 6, -1):
                out.append(326)
        elif tok == 4:
            if nxt in (2, 3, 5):
                out.append(326)
    return out


def _maybe_canonicalize_seq(seq: list[int], rid: Any = None) -> list[int]:
    new_seq = canonicalize_turn_input_ids(seq)
    if new_seq != seq:
        logger.warning(
            "SGLANG_RECKONER_CANONICALIZE_TURNS rewrote non-canonical turn boundaries for rid=%s (len %d -> %d)",
            rid,
            len(seq),
            len(new_seq),
        )
    return new_seq


def _render_raw_chat_fallback(messages: list[dict[str, Any]]) -> str:
    """Fallback canonical prompt renderer used only when `--allow-raw-chat` is explicitly enabled."""
    system_text: Optional[str] = None
    turns: list[tuple[str, str]] = []
    for msg in messages:
        role = str(msg.get("role", "user"))
        content = str(msg.get("content") or "")
        if role == "system" and system_text is None and not turns:
            system_text = content
        elif role == "assistant":
            if not content.startswith(("<|ready|>", "<|act|>", "<|deliberate|>", "<|clarify|>")):
                content = f"<|ready|>{content}"
            turns.append(("agent", content))
        else:
            turns.append(("user", content))
    if len(turns) == 1 and turns[0][0] == "user":
        return format_prompt(turns[0][1], system=system_text)
    return format_dialogue(turns, system=system_text)


def install_front_door_chat_guard() -> None:
    """Hook `OpenAIServingChat._apply_jinja_template` so raw `/v1/chat/completions` on SGLang
    returns a clear error naming `reckoner-harness serve` unless `--allow-raw-chat` is enabled.
    """
    try:
        from sglang.srt.entrypoints.openai.serving_chat import (
            MessageProcessingResult,
            OpenAIServingChat,
        )
    except ImportError:
        return

    if getattr(OpenAIServingChat, "_reckoner_front_door_guard_installed", False):
        return

    _orig_apply_jinja = OpenAIServingChat._apply_jinja_template

    def _guarded_apply_jinja_template(
        self: Any,
        request: Any,
        messages: list[dict[str, Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if is_reckoner_model_active():
            if not is_raw_chat_allowed():
                raise ValueError(RECKONER_FRONT_DOOR_ERROR_MSG)
            prompt_str = _render_raw_chat_fallback(messages)
            tokenizer = self.tokenizer_manager.tokenizer
            prompt_ids = list(tokenizer.encode(prompt_str, add_special_tokens=False))
            return MessageProcessingResult(
                prompt=prompt_str,
                prompt_ids=prompt_ids,
                image_data=None,
                video_data=None,
                audio_data=None,
                modalities=[],
                stop=getattr(request, "stop", None),
            )
        try:
            return _orig_apply_jinja(
                self,
                request,
                messages,
                *args,
                **kwargs,
            )
        except Exception as exc:
            if "reckoner-harness serve" in str(exc):
                if is_raw_chat_allowed():
                    prompt_str = _render_raw_chat_fallback(messages)
                    tokenizer = self.tokenizer_manager.tokenizer
                    prompt_ids = list(tokenizer.encode(prompt_str, add_special_tokens=False))
                    return MessageProcessingResult(
                        prompt=prompt_str,
                        prompt_ids=prompt_ids,
                        image_data=None,
                        video_data=None,
                        audio_data=None,
                        modalities=[],
                        stop=getattr(request, "stop", None),
                    )
                raise ValueError(RECKONER_FRONT_DOOR_ERROR_MSG) from exc
            raise

    OpenAIServingChat._apply_jinja_template = _guarded_apply_jinja_template
    OpenAIServingChat._reckoner_front_door_guard_installed = True


def register_plugin() -> bool:
    """Idempotently register `ReckonerConfig`, `Qwen35GDN2Config`, `ReckonerForCausalLM`, `Qwen35GDN2ForCausalLM`, and `ReckonerGDN2AttnBackend`."""
    global _PLUGIN_REGISTERED
    from transformers import AutoConfig

    for model_type_str, cfg_cls in (
        ("reckoner", ReckonerConfig),
        ("qwen35_gdn2", Qwen35GDN2Config),
    ):
        try:
            AutoConfig.register(model_type_str, cfg_cls, exist_ok=True)
        except TypeError:
            try:
                AutoConfig.register(model_type_str, cfg_cls)
            except ValueError:
                pass

    try:
        from sglang.srt.configs.linear_attn_model_registry import (
            LinearAttnModelSpec,
            register_linear_attn_model,
        )
        from sglang.srt.managers.io_struct import GenerateReqInput
        from sglang.srt.models.registry import ModelRegistry

        ModelRegistry.models["ReckonerForCausalLM"] = ReckonerForCausalLM
        ModelRegistry.models["Qwen35GDN2ForCausalLM"] = Qwen35GDN2ForCausalLM
        register_linear_attn_model(
            LinearAttnModelSpec(
                config_class=ReckonerConfig,
                backend_class_name="serve.sglang_reckoner.gdn2_backend.ReckonerGDN2AttnBackend",
                arch_names=["ReckonerForCausalLM"],
                uses_mamba_radix_cache=True,
                support_mamba_cache=True,
            )
        )
        register_linear_attn_model(
            LinearAttnModelSpec(
                config_class=Qwen35GDN2Config,
                backend_class_name="serve.sglang_reckoner.gdn2_backend.ReckonerGDN2AttnBackend",
                arch_names=["Qwen35GDN2ForCausalLM"],
                uses_mamba_radix_cache=True,
                support_mamba_cache=True,
            )
        )
        if not getattr(GenerateReqInput, "_reckoner_turn_canonicalizer_installed", False):
            _orig_normalize = GenerateReqInput.normalize_batch_and_arguments

            def _normalize_with_canonical_turns(self: GenerateReqInput) -> None:
                if (
                    self.input_ids is not None
                    and os.environ.get("SGLANG_RECKONER_CANONICALIZE_TURNS", "0") == "1"
                    and is_reckoner_model_active()
                ):
                    rid = getattr(self, "rid", None)
                    if isinstance(self.input_ids, list) and self.input_ids:
                        if isinstance(self.input_ids[0], int):
                            self.input_ids = _maybe_canonicalize_seq(self.input_ids, rid=rid)
                        elif isinstance(self.input_ids[0], list):
                            self.input_ids = [
                                _maybe_canonicalize_seq(seq, rid=rid) if isinstance(seq, list) else seq
                                for seq in self.input_ids
                            ]
                _orig_normalize(self)

            GenerateReqInput.normalize_batch_and_arguments = _normalize_with_canonical_turns
            GenerateReqInput._reckoner_turn_canonicalizer_installed = True
        install_front_door_chat_guard()
        ReckonerToolLogitProcessor.log_active_path()
        _PLUGIN_REGISTERED = True
    except ImportError:
        _PLUGIN_REGISTERED = False

    return _PLUGIN_REGISTERED


# Alias for server/autoload callers
register_reckoner_with_sglang = register_plugin

# Automatically register on import
register_plugin()

__all__ = [
    "RECKONER_ERRORING_CHAT_TEMPLATE",
    "RECKONER_FRONT_DOOR_ERROR_MSG",
    "RECKONER_VOCAB_SIZE",
    "Qwen35FullAttentionLayer",
    "Qwen35GDN2Config",
    "Qwen35GDN2DecoderLayer",
    "Qwen35GDN2ForCausalLM",
    "Qwen35GDN2Layer",
    "Qwen35MLP",
    "Qwen35RMSNorm",
    "Qwen35RMSNormGated",
    "ReckonerConfig",
    "ReckonerDecoderLayer",
    "ReckonerForCausalLM",
    "ReckonerFullAttentionLayer",
    "ReckonerGDN2AttnBackend",
    "ReckonerGDN2Layer",
    "ReckonerMLP",
    "ReckonerRMSNorm",
    "ReckonerToolLogitProcessor",
    "canonicalize_turn_input_ids",
    "get_active_model_arch",
    "install_front_door_chat_guard",
    "is_raw_chat_allowed",
    "is_reckoner_model_active",
    "register_plugin",
    "register_reckoner_with_sglang",
    "set_active_model_arch",
    "supports_inflight_extend",
]
