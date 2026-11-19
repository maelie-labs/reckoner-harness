# Copyright 2026 Maelie Labs
# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
#
# Notice of Modification (Apache-2.0 Section 4(b)):
# Modified by Maelie Labs (2026): subclasses SGLang v0.5.22
# `sglang.srt.sampling.custom_logit_processor.CustomLogitProcessor` to intercept Reckoner
# tool calls and inject observations in-engine.
"""Out-of-tree SGLang CustomLogitProcessor for Reckoner Go tool harness execution.

Implements deterministic tool call interception and observation token forcing inside
SGLang's token-by-token decode loop without modifying SGLang source code:
- Detects `<|/call|>` (token 14) or `<|/quote|>` (token 34) emitted in `req.output_ids`.
- Parses the preceding `<|call:calc|>`, `<|call:solve|>`, `<|call:lua|>`, `<|call:str|>`,
  `<|call:date|>`, `<|call:find|>`, or `<|quote|>` span.
- Dispatches the payload to a per-session `GoToolHarness` subprocess (`sglang_reckoner/harness_bridge.py`),
  preserving `<|r0|>..<|r15|>` register state and anchored context per request.
- Tokenizes `<|obs|>result<|/obs|>` (or `<|obs:err|>error<|/obs|>`) and deterministically
  forces those tokens via `logits[i, :] = -inf; logits[i, next_obs_tok] = 0.0` on subsequent
  decode steps.
- Enforces `<|deliberate|>` thinking token budget (`thinking_budget`) by forcing `<|ready|>`.
- Enforces tool-mode masks (`tool_mode` in `{"all", "none", "lua_only"}` and `<|deliberate|>`
  restraint) and records gate distributions (`ready`, `act`, `deliberate`, `clarify`),
  `is_generated_mask`, `forced_positions`, and `tool_calls` for `RolloutSample` assembly.
"""

from __future__ import annotations

import ast
from collections import deque
from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

try:
    from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor
except ImportError:

    class CustomLogitProcessor:  # type: ignore[no-redef]
        def __call__(self, logits: torch.Tensor, custom_param_list: List[Dict[str, Any]]) -> torch.Tensor:
            raise NotImplementedError

        def to_str(self) -> str:
            return ""

        @classmethod
        def from_str(cls, processor_str: str) -> "CustomLogitProcessor":
            return cls()


def supports_inflight_extend() -> bool:
    """Return True if the active SGLang runtime carries `0001-inflight-extend.patch` (and is not overridden off)."""
    env_override = os.environ.get("SGLANG_RECKONER_INFLIGHT_EXTEND", "").strip().lower()
    if env_override in ("0", "false", "no", "off", "option_a", "logit_forcing"):
        return False
    if os.environ.get("SGLANG_RECKONER_FORCE_INFLIGHT_EXTEND", "").strip().lower() in ("1", "true", "yes", "on"):
        return True
    try:
        from sglang.srt.managers.schedule_batch import Req

        return bool(getattr(Req, "SGLANG_SUPPORTS_INFLIGHT_EXTEND", False))
    except ImportError:
        return False


from .harness_bridge import resolve_harness_binary

try:
    from serve.rollout import (
        EFFORT_MAP,
        MODE_TOKEN_MAP,
        TOK_ACT,
        TOK_CALL,
        TOK_CALL_CALC,
        TOK_CALL_DATE,
        TOK_CALL_FIND,
        TOK_CALL_LOOKUP,
        TOK_CALL_LUA,
        TOK_CALL_SOLVE,
        TOK_CALL_STR,
        TOK_CLARIFY,
        TOK_DELIBERATE,
        TOK_END_CALL,
        TOK_END_OBS,
        TOK_END_QUOTE,
        TOK_END_TURN,
        TOK_L_ANCHOR_BASE,
        TOK_M_ANCHOR_BASE,
        TOK_OBS,
        TOK_OBS_ERR,
        TOK_QUOTE,
        TOK_READY,
        TOK_TURN_AGENT,
        TOK_TURN_SYSTEM,
        TOK_TURN_USER,
        TOOL_TOKEN_MAP,
    )
    from tools.tool_harness import GoToolHarness
except Exception:
    from .harness_bridge import (
        EFFORT_MAP,
        MODE_TOKEN_MAP,
        TOK_ACT,
        TOK_CALL,
        TOK_CALL_CALC,
        TOK_CALL_DATE,
        TOK_CALL_FIND,
        TOK_CALL_LOOKUP,
        TOK_CALL_LUA,
        TOK_CALL_SOLVE,
        TOK_CALL_STR,
        TOK_CLARIFY,
        TOK_DELIBERATE,
        TOK_END_CALL,
        TOK_END_OBS,
        TOK_END_QUOTE,
        TOK_END_TURN,
        TOK_L_ANCHOR_BASE,
        TOK_M_ANCHOR_BASE,
        TOK_OBS,
        TOK_OBS_ERR,
        TOK_QUOTE,
        TOK_READY,
        TOK_TURN_AGENT,
        TOK_TURN_SYSTEM,
        TOK_TURN_USER,
        TOOL_TOKEN_MAP,
        GoToolHarness,
    )

CALL_START_TO_TOOL: Dict[int, str] = {
    TOK_CALL_CALC: "calc",
    TOK_CALL_SOLVE: "solve",
    TOK_CALL_LUA: "lua",
    TOK_CALL_STR: "str",
    TOK_CALL_DATE: "date",
    TOK_CALL_FIND: "find",
    TOK_CALL: "external",
}

MODE_TOKEN_IDS: List[int] = [TOK_READY, TOK_ACT, TOK_DELIBERATE, TOK_CLARIFY]


@dataclass(frozen=True)
class TokenIdTable:
    """Resolved special token ID table for Reckoner (36,864 vocab) or Qwen3.5 (248,320 vocab)."""

    tok_endoftext: int = 0
    tok_pad: int = 1
    tok_turn_user: int = TOK_TURN_USER
    tok_turn_agent: int = TOK_TURN_AGENT
    tok_end_turn: int = TOK_END_TURN
    tok_turn_system: int = TOK_TURN_SYSTEM
    tok_tools: int = 6
    tok_end_tools: int = 7
    tok_deliberate: int = TOK_DELIBERATE
    tok_act: int = TOK_ACT
    tok_ready: int = TOK_READY
    tok_clarify: int = TOK_CLARIFY
    tok_pause: int = 12
    tok_call: int = TOK_CALL
    tok_end_call: int = TOK_END_CALL
    tok_obs: int = TOK_OBS
    tok_obs_err: int = TOK_OBS_ERR
    tok_end_obs: int = TOK_END_OBS
    tok_call_calc: int = TOK_CALL_CALC
    tok_call_solve: int = TOK_CALL_SOLVE
    tok_call_lua: int = TOK_CALL_LUA
    tok_call_str: int = TOK_CALL_STR
    tok_call_date: int = TOK_CALL_DATE
    tok_call_lookup: int = TOK_CALL_LOOKUP
    tok_call_find: int = TOK_CALL_FIND
    tok_quote: int = TOK_QUOTE
    tok_end_quote: int = TOK_END_QUOTE
    tok_call_tools: int = 35
    tok_call_def: int = 36
    tok_m_anchor_base: int = TOK_M_ANCHOR_BASE
    tok_l_anchor_base: int = TOK_L_ANCHOR_BASE
    register_token_ids: Tuple[int, ...] = (
        21, 22, 23, 24, 25, 26, 27, 28,
        37, 38, 39, 40, 41, 42, 43, 44,
    )
    banned_token_ids: Tuple[int, ...] = ()
    no_anchors: bool = False
    is_qwen_vocab: bool = False
    vocab_size: int = 36864

    @property
    def mode_token_ids(self) -> List[int]:
        return [self.tok_ready, self.tok_act, self.tok_deliberate, self.tok_clarify]

    @property
    def mode_token_map(self) -> Dict[int, str]:
        return {
            self.tok_ready: "READY",
            self.tok_act: "ACT",
            self.tok_deliberate: "DELIBERATE",
            self.tok_clarify: "CLARIFY",
        }

    @property
    def call_start_to_tool(self) -> Dict[int, str]:
        base = {
            self.tok_call_calc: "calc",
            self.tok_call_solve: "solve",
            self.tok_call_lua: "lua",
            self.tok_call_str: "str",
            self.tok_call_date: "date",
            self.tok_call_find: "find",
            self.tok_call: "external",
        }
        if self.is_qwen_vocab:
            base[self.tok_call_lookup] = "lookup"
            base[self.tok_call_tools] = "tools"
            base[self.tok_call_def] = "def"
        return base

    @property
    def obs_delim_ids(self) -> List[int]:
        return [self.tok_obs, self.tok_obs_err, self.tok_end_obs]

    @property
    def all_call_start_ids(self) -> List[int]:
        return list(self.call_start_to_tool.keys())

    @property
    def non_lua_call_start_ids(self) -> List[int]:
        return [tid for tid, tname in self.call_start_to_tool.items() if tname != "lua"]

    @property
    def delib_forbid_ids(self) -> List[int]:
        ids = list(self.all_call_start_ids)
        if not self.no_anchors and self.tok_quote >= 0:
            ids.append(self.tok_quote)
        return ids

    @property
    def none_mode_forbid_ids(self) -> List[int]:
        return [self.tok_act] + self.all_call_start_ids


DEFAULT_RECKONER_TOKEN_TABLE = TokenIdTable()

QWEN35_TOKEN_TABLE = TokenIdTable(
    tok_endoftext=248044,
    tok_pad=248070,
    tok_turn_user=248071,
    tok_turn_agent=248072,
    tok_end_turn=248073,
    tok_turn_system=248074,
    tok_tools=248075,
    tok_end_tools=248076,
    tok_deliberate=248077,
    tok_act=248078,
    tok_ready=248079,
    tok_clarify=248080,
    tok_pause=248081,
    tok_call=248082,
    tok_end_call=248083,
    tok_obs=248084,
    tok_obs_err=248085,
    tok_end_obs=248086,
    tok_call_calc=248087,
    tok_call_solve=248088,
    tok_call_lua=248089,
    tok_call_str=248098,
    tok_call_date=248099,
    tok_call_lookup=248100,
    tok_call_find=248101,
    tok_quote=-1,
    tok_end_quote=-1,
    tok_call_tools=248102,
    tok_call_def=248103,
    tok_m_anchor_base=-1,
    tok_l_anchor_base=-1,
    register_token_ids=(
        248090, 248091, 248092, 248093, 248094, 248095, 248096, 248097,
        248104, 248105, 248106, 248107, 248108, 248109, 248110, 248111,
    ),
    banned_token_ids=(248068, 248069),
    no_anchors=True,
    is_qwen_vocab=True,
    vocab_size=248320,
)


def resolve_token_id_table(tokenizer: Any) -> TokenIdTable:
    """Resolve `TokenIdTable` from a loaded tokenizer (defaulting to `DEFAULT_RECKONER_TOKEN_TABLE`)."""
    if tokenizer is None or not hasattr(tokenizer, "encode"):
        return DEFAULT_RECKONER_TOKEN_TABLE

    def _one(tok_str: str, default_id: int) -> int:
        try:
            ids = tokenizer.encode(tok_str, add_special_tokens=False)
            if len(ids) == 1:
                return int(ids[0])
        except Exception:
            pass
        return default_id

    tok_obs = _one("<|obs|>", TOK_OBS)
    if tok_obs < 36864:
        return DEFAULT_RECKONER_TOKEN_TABLE

    reg_ids = tuple(
        _one(f"<|r{idx}|>", QWEN35_TOKEN_TABLE.register_token_ids[idx])
        for idx in range(16)
    )
    banned_list: List[int] = []
    for bt_str, default_bt_id in (("<think>", 248068), ("</think>", 248069)):
        bt_id = _one(bt_str, default_bt_id)
        if bt_id >= 0:
            banned_list.append(bt_id)

    return TokenIdTable(
        tok_endoftext=_one("<|endoftext|>", 248044),
        tok_pad=_one("<|pad|>", 248070),
        tok_turn_user=_one("<|turn:user|>", 248071),
        tok_turn_agent=_one("<|turn:agent|>", 248072),
        tok_end_turn=_one("<|/turn|>", 248073),
        tok_turn_system=_one("<|turn:system|>", 248074),
        tok_tools=_one("<|tools|>", 248075),
        tok_end_tools=_one("<|/tools|>", 248076),
        tok_deliberate=_one("<|deliberate|>", 248077),
        tok_act=_one("<|act|>", 248078),
        tok_ready=_one("<|ready|>", 248079),
        tok_clarify=_one("<|clarify|>", 248080),
        tok_pause=_one("<|pause|>", 248081),
        tok_call=_one("<|call|>", 248082),
        tok_end_call=_one("<|/call|>", 248083),
        tok_obs=tok_obs,
        tok_obs_err=_one("<|obs:err|>", 248085),
        tok_end_obs=_one("<|/obs|>", 248086),
        tok_call_calc=_one("<|call:calc|>", 248087),
        tok_call_solve=_one("<|call:solve|>", 248088),
        tok_call_lua=_one("<|call:lua|>", 248089),
        tok_call_str=_one("<|call:str|>", 248098),
        tok_call_date=_one("<|call:date|>", 248099),
        tok_call_lookup=_one("<|call:lookup|>", 248100),
        tok_call_find=_one("<|call:find|>", 248101),
        tok_quote=-1,
        tok_end_quote=-1,
        tok_call_tools=_one("<|call:tools|>", 248102),
        tok_call_def=_one("<|call:def|>", 248103),
        tok_m_anchor_base=-1,
        tok_l_anchor_base=-1,
        register_token_ids=reg_ids,
        banned_token_ids=tuple(banned_list),
        no_anchors=True,
        is_qwen_vocab=True,
        vocab_size=248320,
    )


def _default_tokenizer_path(prefer_qwen: bool = False) -> str:
    env_path = os.environ.get("RECKONER_TOKENIZER_PATH")
    if env_path and os.path.exists(env_path):
        return env_path
    repo_root = Path(__file__).resolve().parents[2]
    if not prefer_qwen:
        try:
            from . import get_active_model_arch

            if get_active_model_arch() == "Qwen35GDN2ForCausalLM":
                prefer_qwen = True
        except Exception:
            pass
    if prefer_qwen:
        qwen_env = os.environ.get("RECKONER_QWEN35_TEST_MODEL_DIR")
        if qwen_env and os.path.exists(qwen_env):
            return qwen_env
        qwen_dir = repo_root / "tokenizer" / "qwen35_reckoner_tokenizer"
        if qwen_dir.exists():
            return str(qwen_dir)
        return "maelie-labs/reckoner-q2b"
    test_env = os.environ.get("RECKONER_TEST_MODEL_DIR")
    if test_env and os.path.exists(test_env):
        return test_env
    v2_dir = repo_root / "tokenizer" / "tokenizer_weights_v2"
    if v2_dir.exists():
        return str(v2_dir)
    return "maelie-labs/reckoner-1.4b"


@dataclass
class RequestToolSession:
    """Tracks per-request state, GoToolHarness worker, and forced observation token queue."""

    session_id: str
    thinking_budget: int = 1024
    gate_mask_deliberate: bool = False
    force_system: Optional[str] = None
    tool_mode: str = "all"
    max_tool_calls: int = 16
    processed_output_len: int = 0
    pending_forced_tokens: deque[Tuple[int, bool]] = field(default_factory=deque)
    # Per-token masks/metadata aligned with `output_ids`
    is_generated_mask: List[bool] = field(default_factory=list)
    forced_positions: List[int] = field(default_factory=list)
    logprobs: List[float] = field(default_factory=list)
    last_step_logits: Optional[torch.Tensor] = None
    # Mode & segment tracking
    action_mode: str = "READY"
    current_mode: str = "ready"
    segments: List[Dict[str, str]] = field(default_factory=list)
    thinking_tokens: int = 0
    in_tool_call: bool = False
    current_tool_name: str = ""
    call_tokens: List[int] = field(default_factory=list)
    in_quote: bool = False
    quote_tokens: List[int] = field(default_factory=list)
    # Tool telemetry
    num_tool_calls: int = 0
    exceeded_tool_limit: bool = False
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    format_errors: List[str] = field(default_factory=list)
    total_tool_latency_us: float = 0.0
    allow_external_tools: bool = False
    active_obs_call_idx: Optional[int] = None
    active_call_enter_ts: float = 0.0
    active_first_force_ts: float = 0.0
    # Gate distribution & banned <think> telemetry at position 0
    gate_dist: Dict[str, float] = field(default_factory=dict)
    p_system1: float = 0.0
    p_system2: float = 0.0
    unmasked_think_prob: float = 0.0
    unmasked_think_top1: bool = False
    harness: Optional[Any] = None
    pending_context_messages: List[Any] = field(default_factory=list)
    req_ref: Optional[Any] = None


_OBS_DELIM_IDS = [TOK_OBS, TOK_OBS_ERR, TOK_END_OBS]
_ALL_CALL_START_IDS = list(CALL_START_TO_TOOL.keys())
_NON_LUA_CALL_START_IDS = [tid for tid, tname in CALL_START_TO_TOOL.items() if tname != "lua"]
_DELIB_FORBID_IDS = _ALL_CALL_START_IDS + [TOK_QUOTE]
_NONE_MODE_FORBID_IDS = [TOK_ACT] + _ALL_CALL_START_IDS


class ReckonerToolLogitProcessor(CustomLogitProcessor):
    """Out-of-tree SGLang `CustomLogitProcessor` executing Go tool calls in the decode loop.

    Usage in SGLang:
    - Pass `custom_logit_processor=ReckonerToolLogitProcessor.to_str()` and
      `custom_params={"session_id": ..., "effort": "medium", "tool_mode": "all"}`
      in `SamplingParams`.
    - SGLang calls `__call__(logits, custom_param_list)` at every decode step,
      passing `"__req__"` inside each element of `custom_param_list` when running inside
      the scheduler/tp_worker, or an explicit `output_ids` list in standalone tests.
    """

    _shared_lock = threading.RLock()
    _shared_tokenizer: Any = None
    _shared_token_table: Optional[TokenIdTable] = None
    _reckoner_v2_tokenizer: Any = None
    _shared_harness_bin: Optional[str] = None
    _shared_base_harness: Optional[Any] = None
    _harness_pool: List[Any] = []
    _sessions: Dict[str, RequestToolSession] = {}
    _max_sessions: int = 128
    # Reckoner-Q2B RL rollouts run up to 64 * 16 = 1,024 concurrent sequences per step, so the Qwen
    # vocab gets a larger default cap. Reckoner-1.4B keeps 128; `max_sessions` / the env var override both.
    _qwen_max_sessions: int = 4096
    _max_sessions_configured: bool = False
    _path_logged: bool = False
    _pool_prewarmed: bool = False
    _instance_cache: Optional["ReckonerToolLogitProcessor"] = None
    _warned_unknown_efforts: set = set()

    @classmethod
    def from_str(cls, json_str: str) -> "ReckonerToolLogitProcessor":
        if cls._instance_cache is None:
            cls._instance_cache = cls()
        return cls._instance_cache

    @classmethod
    def supports_inflight_extend(cls) -> bool:
        return supports_inflight_extend()

    @classmethod
    def get_token_table(cls, tokenizer: Any = None) -> TokenIdTable:
        with cls._shared_lock:
            tok = tokenizer if tokenizer is not None else cls.get_tokenizer()
            if tok is cls._shared_tokenizer and cls._shared_token_table is not None:
                return cls._shared_token_table
            tbl = resolve_token_id_table(tok)
            if tok is cls._shared_tokenizer:
                cls._shared_token_table = tbl
            return tbl

    @classmethod
    def _get_reckoner_v2_tokenizer(cls) -> Any:
        with cls._shared_lock:
            if cls._reckoner_v2_tokenizer is None:
                from transformers import AutoTokenizer

                repo_root = Path(__file__).resolve().parents[2]
                v2_dir = repo_root / "tokenizer" / "tokenizer_weights_v2"
                env_v2 = (
                    os.environ.get("RECKONER_V2_TOKENIZER_PATH")
                    or os.environ.get("RECKONER_TEST_MODEL_DIR")
                    or os.environ.get("RECKONER_TOKENIZER_PATH")
                )
                tok_path = (
                    env_v2
                    if (env_v2 and os.path.exists(env_v2))
                    else (str(v2_dir) if v2_dir.exists() else "maelie-labs/reckoner-1.4b")
                )
                cls._reckoner_v2_tokenizer = AutoTokenizer.from_pretrained(
                    tok_path, trust_remote_code=True
                )
            return cls._reckoner_v2_tokenizer

    @classmethod
    def prewarm_harness_pool(
        cls, count: int = 36, no_anchors: Optional[bool] = None
    ) -> None:
        with cls._shared_lock:
            if cls._shared_base_harness is not None:
                return
            if no_anchors is None:
                if cls._shared_token_table is not None:
                    no_anchors = cls._shared_token_table.no_anchors
                elif cls._shared_tokenizer is not None:
                    no_anchors = cls.get_token_table(cls._shared_tokenizer).no_anchors
                else:
                    try:
                        from . import get_active_model_arch

                        no_anchors = get_active_model_arch() == "Qwen35GDN2ForCausalLM"
                    except Exception:
                        no_anchors = False
            while len(cls._harness_pool) < count:
                try:
                    h = GoToolHarness(
                        binary_path=resolve_harness_binary(cls._shared_harness_bin),
                        no_anchors=bool(no_anchors),
                    )
                    h._is_clean = True
                    cls._harness_pool.append(h)
                except Exception:
                    break
            cls._pool_prewarmed = True

    @classmethod
    def log_active_path(cls) -> str:
        active = cls.supports_inflight_extend()
        mode_str = (
            "inflight_extend (Option C, 1-step EXTEND)"
            if active
            else "logit_forcing (Option A, token-by-token DECODE fallback)"
        )
        with cls._shared_lock:
            if not cls._path_logged:
                logger.info("ReckonerToolLogitProcessor observation injection path: %s", mode_str)
                cls._path_logged = True
            if not cls._pool_prewarmed and os.environ.get("SGLANG_RECKONER_AUTOLOAD") == "1":
                cls.prewarm_harness_pool(36)
        return mode_str

    @classmethod
    def configure_shared(
        cls,
        tokenizer: Any = None,
        harness_bin: Optional[str] = None,
        base_harness: Optional[Any] = None,
        inflight_extend: Optional[bool] = None,
        max_sessions: Optional[int] = None,
    ) -> None:
        with cls._shared_lock:
            cls._shared_tokenizer = tokenizer
            cls._shared_token_table = (
                resolve_token_id_table(tokenizer) if tokenizer is not None else None
            )
            if tokenizer is None and not os.environ.get("RECKONER_TOKENIZER_PATH"):
                try:
                    from . import set_active_model_arch

                    set_active_model_arch("ReckonerForCausalLM")
                except Exception:
                    pass
            if harness_bin is not None:
                cls._shared_harness_bin = resolve_harness_binary(harness_bin)
            if base_harness is not None or harness_bin is not None:
                cls._shared_base_harness = base_harness
            if inflight_extend is not None:
                os.environ["SGLANG_RECKONER_INFLIGHT_EXTEND"] = "1" if inflight_extend else "0"
                cls._path_logged = False
            if max_sessions is not None:
                cls._max_sessions = max(1, int(max_sessions))
                cls._max_sessions_configured = True

    @classmethod
    def get_tokenizer(
        cls,
        model_path: Optional[str] = None,
        vocab_size_hint: Optional[int] = None,
    ) -> Any:
        with cls._shared_lock:
            prefer_qwen = bool(vocab_size_hint is not None and vocab_size_hint >= 100000)
            if cls._shared_tokenizer is not None and vocab_size_hint is not None:
                try:
                    cur_len = len(cls._shared_tokenizer)
                    if (vocab_size_hint >= 100000 and cur_len < 100000) or (
                        vocab_size_hint < 100000 and cur_len >= 100000
                    ):
                        cls._shared_tokenizer = None
                        cls._shared_token_table = None
                except Exception:
                    pass
            if cls._shared_tokenizer is None:
                from transformers import AutoTokenizer

                path = model_path or _default_tokenizer_path(prefer_qwen=prefer_qwen)
                cls._shared_tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
                cls._shared_token_table = resolve_token_id_table(cls._shared_tokenizer)
            return cls._shared_tokenizer

    @classmethod
    def _acquire_harness(cls, no_anchors: Optional[bool] = None) -> Any:
        with cls._shared_lock:
            if cls._shared_base_harness is not None:
                if hasattr(cls._shared_base_harness, "reset"):
                    cls._shared_base_harness.reset()
                return cls._shared_base_harness
            if no_anchors is None:
                no_anchors = cls.get_token_table().no_anchors
            target_no_anchors = bool(no_anchors)
            while cls._harness_pool:
                h = cls._harness_pool.pop()
                if bool(getattr(h, "no_anchors", False)) != target_no_anchors:
                    try:
                        h.close()
                    except Exception:
                        pass
                    continue
                if not getattr(h, "_is_clean", False):
                    h.reset()
                h._is_clean = False
                return h
            h = GoToolHarness(
                binary_path=resolve_harness_binary(cls._shared_harness_bin),
                no_anchors=target_no_anchors,
            )
            h._is_clean = False
            return h

    @classmethod
    def _release_harness(cls, harness: Any) -> None:
        if harness is None:
            return
        with cls._shared_lock:
            if harness is cls._shared_base_harness:
                return
            try:
                harness.reset()
                harness._is_clean = True
                cls._harness_pool.append(harness)
            except Exception:
                pass

    @classmethod
    def _ensure_session_harness(
        cls,
        sess: RequestToolSession,
        flush_anchors: bool = False,
        tool_name: str = "anchor",
        tokenizer: Any = None,
    ) -> Any:
        tids = cls.get_token_table(tokenizer)
        if sess.harness is None:
            sess.harness = cls._acquire_harness(no_anchors=tids.no_anchors)
        if flush_anchors and sess.pending_context_messages and hasattr(sess.harness, "dispatch_full"):
            for msg_idx, msg_item in enumerate(sess.pending_context_messages):
                try:
                    if isinstance(msg_item, str):
                        rec_tok = cls._get_reckoner_v2_tokenizer()
                        harness_toks = [int(t) for t in rec_tok.encode(msg_item, add_special_tokens=False)]
                    elif isinstance(msg_item, list) and msg_item:
                        raw_toks = [int(t) for t in msg_item]
                        if tids.no_anchors or tids.is_qwen_vocab or any(t >= 36864 for t in raw_toks):
                            tok = tokenizer if tokenizer is not None else cls.get_tokenizer()
                            text_msg = tok.decode(raw_toks, skip_special_tokens=False)
                            rec_tok = cls._get_reckoner_v2_tokenizer()
                            harness_toks = [int(t) for t in rec_tok.encode(text_msg, add_special_tokens=False)]
                        else:
                            harness_toks = raw_toks
                    else:
                        continue
                    if not harness_toks:
                        continue
                    anchor_payload = json.dumps(
                        {"tokens": harness_toks, "message_id": msg_idx}
                    )
                    resp = sess.harness.dispatch_full("anchor", anchor_payload)
                    if isinstance(resp, dict) and not resp.get("success", True):
                        logger.warning(
                            "ReckonerToolLogitProcessor: anchor dispatch returned error before tool=%s (session=%s, msg_idx=%d): %s",
                            tool_name,
                            sess.session_id,
                            msg_idx,
                            resp.get("error", resp),
                        )
                except Exception as exc:
                    logger.warning(
                        "ReckonerToolLogitProcessor: anchor dispatch failed before tool=%s (session=%s, msg_idx=%d): %s",
                        tool_name,
                        sess.session_id,
                        msg_idx,
                        exc,
                    )
            sess.pending_context_messages.clear()
        return sess.harness

    @classmethod
    def _shm_path(cls, session_id: str) -> str:
        safe_id = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in session_id)
        return f"/dev/shm/reckoner_sgl_sess_{safe_id}.pt"

    @classmethod
    def _shm_json_path(cls, session_id: str) -> str:
        safe_id = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in session_id)
        return f"/dev/shm/reckoner_sgl_sess_{safe_id}.json"

    @classmethod
    def _save_session_shm(cls, sess: RequestToolSession) -> None:
        if os.environ.get("SGLANG_RECKONER_AUTOLOAD") != "1":
            return
        try:
            json_path = cls._shm_json_path(sess.session_id)
            json_tmp = f"{json_path}.tmp.{os.getpid()}"
            json_payload = {
                "session_id": sess.session_id,
                "thinking_budget": sess.thinking_budget,
                "gate_mask_deliberate": sess.gate_mask_deliberate,
                "force_system": sess.force_system,
                "tool_mode": sess.tool_mode,
                "max_tool_calls": sess.max_tool_calls,
                "processed_output_len": sess.processed_output_len,
                "is_generated_mask": list(sess.is_generated_mask),
                "forced_positions": list(sess.forced_positions),
                "logprobs": list(sess.logprobs),
                "action_mode": sess.action_mode,
                "current_mode": sess.current_mode,
                "segments": list(sess.segments),
                "thinking_tokens": sess.thinking_tokens,
                "in_tool_call": sess.in_tool_call,
                "current_tool_name": sess.current_tool_name,
                "call_tokens": list(sess.call_tokens),
                "in_quote": sess.in_quote,
                "quote_tokens": list(sess.quote_tokens),
                "num_tool_calls": sess.num_tool_calls,
                "exceeded_tool_limit": sess.exceeded_tool_limit,
                "tool_calls": list(sess.tool_calls),
                "format_errors": list(sess.format_errors),
                "total_tool_latency_us": sess.total_tool_latency_us,
                "gate_dist": dict(sess.gate_dist),
                "p_system1": sess.p_system1,
                "p_system2": sess.p_system2,
                "unmasked_think_prob": sess.unmasked_think_prob,
                "unmasked_think_top1": sess.unmasked_think_top1,
            }
            with open(json_tmp, "w", encoding="utf-8") as jf:
                json.dump(json_payload, jf)
            os.replace(json_tmp, json_path)

            if sess.last_step_logits is not None:
                path = cls._shm_path(sess.session_id)
                tmp_path = f"{path}.tmp.{os.getpid()}"
                pt_payload = dict(json_payload)
                pt_payload["last_step_logits"] = sess.last_step_logits
                torch.save(pt_payload, tmp_path)
                os.replace(tmp_path, path)
        except Exception:
            pass

    @classmethod
    def _load_session_shm(cls, session_id: str, unlink: bool = True) -> Optional[RequestToolSession]:
        pt_path = cls._shm_path(session_id)
        json_path = cls._shm_json_path(session_id)
        has_pt = os.path.isfile(pt_path)
        has_json = os.path.isfile(json_path)
        if not has_pt and not has_json:
            return None
        try:
            # `.json` is updated on every SHM save (`_save_session_shm`), whereas `.pt` is only
            # written on steps where `last_step_logits` is non-None (e.g., CPU or `record_logits`).
            # Always read the freshest state from `.json` when available, attaching `last_step_logits`
            # from `.pt` if present.
            if has_json:
                with open(json_path, "r", encoding="utf-8") as jf:
                    data = json.load(jf)
                if has_pt and "last_step_logits" not in data:
                    try:
                        pt_data = torch.load(pt_path, map_location="cpu", weights_only=False)
                        if "last_step_logits" in pt_data:
                            data["last_step_logits"] = pt_data["last_step_logits"]
                    except Exception:
                        pass
            else:
                data = torch.load(pt_path, map_location="cpu", weights_only=False)
            if unlink:
                for p in (pt_path, json_path):
                    try:
                        os.unlink(p)
                    except OSError:
                        pass
            return RequestToolSession(
                session_id=data["session_id"],
                thinking_budget=data.get("thinking_budget", 1024),
                gate_mask_deliberate=data.get("gate_mask_deliberate", False),
                force_system=data.get("force_system"),
                tool_mode=data.get("tool_mode", "all"),
                max_tool_calls=data.get("max_tool_calls", 16),
                processed_output_len=data.get("processed_output_len", 0),
                is_generated_mask=list(data.get("is_generated_mask", [])),
                forced_positions=list(data.get("forced_positions", [])),
                logprobs=list(data.get("logprobs", [])),
                last_step_logits=data.get("last_step_logits"),
                action_mode=data.get("action_mode", "READY"),
                current_mode=data.get("current_mode", "ready"),
                segments=list(data.get("segments", [])),
                thinking_tokens=data.get("thinking_tokens", 0),
                in_tool_call=bool(data.get("in_tool_call", False)),
                current_tool_name=str(data.get("current_tool_name", "")),
                call_tokens=list(data.get("call_tokens", [])),
                in_quote=bool(data.get("in_quote", False)),
                quote_tokens=list(data.get("quote_tokens", [])),
                num_tool_calls=int(data.get("num_tool_calls", 0)),
                exceeded_tool_limit=bool(data.get("exceeded_tool_limit", False)),
                tool_calls=list(data.get("tool_calls", [])),
                format_errors=list(data.get("format_errors", [])),
                total_tool_latency_us=float(data.get("total_tool_latency_us", 0.0)),
                gate_dist=dict(data.get("gate_dist", {})),
                p_system1=float(data.get("p_system1", 0.0)),
                p_system2=float(data.get("p_system2", 0.0)),
                unmasked_think_prob=float(data.get("unmasked_think_prob", 0.0)),
                unmasked_think_top1=bool(data.get("unmasked_think_top1", False)),
                harness=None,
            )
        except Exception:
            return None

    @classmethod
    def _materialize_segments(cls, sess: Optional[RequestToolSession], tokenizer: Any = None) -> None:
        if sess is None or not sess.segments:
            return
        if not any("_toks" in seg for seg in sess.segments):
            return
        tok = tokenizer if tokenizer is not None else cls.get_tokenizer()
        for seg in sess.segments:
            toks = seg.pop("_toks", None)
            if toks and tok is not None and hasattr(tok, "decode"):
                seg["text"] = seg.get("text", "") + tok.decode(toks, skip_special_tokens=False)

    @classmethod
    def get_session(cls, session_id: str) -> Optional[RequestToolSession]:
        with cls._shared_lock:
            sess = cls._sessions.get(session_id)
            if sess is not None:
                cls._materialize_segments(sess)
                return sess
            sess = cls._load_session_shm(session_id, unlink=False)
            cls._materialize_segments(sess)
            return sess

    @classmethod
    def pop_session(cls, session_id: str) -> Optional[RequestToolSession]:
        with cls._shared_lock:
            sess = cls._sessions.pop(session_id, None)
            if sess is not None:
                if sess.harness is not None:
                    cls._release_harness(sess.harness)
                    sess.harness = None
                sess.req_ref = None
                for shm_p in (cls._shm_path(session_id), cls._shm_json_path(session_id)):
                    if os.path.isfile(shm_p):
                        try:
                            os.unlink(shm_p)
                        except OSError:
                            pass
                cls._materialize_segments(sess)
                return sess
            sess = cls._load_session_shm(session_id, unlink=True)
            cls._materialize_segments(sess)
            return sess

    @classmethod
    def clear_sessions(cls) -> None:
        with cls._shared_lock:
            for sess in cls._sessions.values():
                if sess.harness is not None:
                    cls._release_harness(sess.harness)
                    sess.harness = None
                sess.req_ref = None
            cls._sessions.clear()
            cls._warned_unknown_efforts.clear()

    @classmethod
    def _sweep_finished_sessions_locked(cls) -> None:
        for sid, old_sess in list(cls._sessions.items()):
            r = old_sess.req_ref
            if r is not None:
                fin_fn = getattr(r, "finished", None)
                if callable(fin_fn):
                    try:
                        if fin_fn():
                            if old_sess.harness is not None:
                                cls._release_harness(old_sess.harness)
                                old_sess.harness = None
                            old_sess.req_ref = None
                            cls._sessions.pop(sid, None)
                    except Exception:
                        pass

    @staticmethod
    def _req_is_finished(r: Any) -> bool:
        fin_fn = getattr(r, "finished", None)
        if not callable(fin_fn):
            return False
        try:
            return bool(fin_fn())
        except Exception:
            return False

    @classmethod
    def _get_or_create_session(cls, params: Dict[str, Any], req: Any = None) -> RequestToolSession:
        # Inside SGLang each `Req` owns its own `custom_params` dict (`Req.__init__` copies it to add
        # `__req__`), so the session bound to that dict is exactly this request's state. Looking it up
        # by `session_id` alone let two requests share one session: a client that reuses a
        # `session_id` (concurrently, or after an earlier request finished but before it was swept)
        # got another request's `processed_output_len`, call state and forced-token queue, so the
        # external-call `<|/turn|>` pause was skipped or landed on the wrong row.
        if req is not None:
            bound = params.get("__reckoner_session__")
            if isinstance(bound, RequestToolSession):
                return bound
        session_id = str(
            params.get("session_id")
            or (getattr(req, "rid", None) if req is not None else None)
            or id(params)
        )
        with cls._shared_lock:
            sess = cls._sessions.get(session_id)
            if (
                sess is not None
                and req is not None
                and sess.req_ref is not None
                and sess.req_ref is not req
            ):
                if cls._req_is_finished(sess.req_ref):
                    # Stale session left by an earlier request with the same id: start fresh.
                    cls._sessions.pop(session_id, None)
                    if sess.harness is not None:
                        cls._release_harness(sess.harness)
                        sess.harness = None
                    sess.req_ref = None
                else:
                    # Two live requests with the same id: keep their state apart.
                    logger.warning(
                        "ReckonerToolLogitProcessor: session_id %r is already used by a running request; "
                        "giving request %r its own session state",
                        session_id,
                        getattr(req, "rid", None),
                    )
                    session_id = f"{session_id}#{getattr(req, 'rid', None) or id(req)}"
                sess = cls._sessions.get(session_id)
            if sess is None:
                cls._sweep_finished_sessions_locked()
                max_sess_env = os.environ.get("SGLANG_RECKONER_MAX_SESSIONS")
                max_sess = max(1, int(max_sess_env)) if max_sess_env else max(1, int(cls._max_sessions))
                if (
                    not max_sess_env
                    and not cls._max_sessions_configured
                    and len(cls._sessions) >= max_sess
                    and cls.get_token_table().is_qwen_vocab
                ):
                    max_sess = max(1, int(cls._qwen_max_sessions))
                while len(cls._sessions) >= max_sess:
                    # Prefer evicting sessions that are finished or unbound (`req_ref is None`)
                    # before evicting any session bound to an active request.
                    evict_id: Optional[str] = None
                    for cand_id, cand_sess in cls._sessions.items():
                        if cand_sess.req_ref is None or cls._req_is_finished(cand_sess.req_ref):
                            evict_id = cand_id
                            break
                    if evict_id is None:
                        evict_id = next(iter(cls._sessions))
                    old_sess = cls._sessions.pop(evict_id)
                    if old_sess.harness is not None:
                        cls._release_harness(old_sess.harness)
                        old_sess.harness = None
                    old_sess.req_ref = None
                effort_name = str(params.get("effort") or params.get("reasoning_effort") or "medium").strip().lower()
                if effort_name not in EFFORT_MAP:
                    if effort_name not in cls._warned_unknown_efforts:
                        cls._warned_unknown_efforts.add(effort_name)
                        logger.warning(
                            "ReckonerToolLogitProcessor: unknown reasoning_effort %r in custom_params; "
                            "falling back to 'medium' (valid: %s)",
                            effort_name,
                            ", ".join(sorted(EFFORT_MAP.keys())),
                        )
                    effort_cfg = EFFORT_MAP["medium"]
                else:
                    effort_cfg = EFFORT_MAP[effort_name]
                thinking_budget = int(
                    params.get("thinking_budget", params.get("max_thinking_tokens", effort_cfg["budget"]))
                )
                gate_mask_deliberate = bool(
                    params.get("gate_mask_deliberate", effort_cfg["gate_mask_deliberate"])
                )
                ctx_msgs = params.get("context_messages_tokens")
                pending_ctx = list(ctx_msgs) if isinstance(ctx_msgs, list) else []
                # Auto-anchor system/user spans from the prompt only for the Qwen vocab (Reckoner-Q2B, anchor-free
                # harness); the Reckoner-1.4B path keeps explicit anchor tags.
                if not pending_ctx and req is not None and cls.get_token_table().is_qwen_vocab:
                    origin_ids = getattr(req, "origin_input_ids", None)
                    if isinstance(origin_ids, list) and origin_ids:
                        tids = cls.get_token_table()
                        turn_starts = (tids.tok_turn_system, tids.tok_turn_user)
                        end_tok = tids.tok_end_turn
                        nl_toks = (198, 203)  # Qwen3.5 newline IDs
                        idx_p = 0
                        n_orig = len(origin_ids)
                        while idx_p < n_orig:
                            if origin_ids[idx_p] in turn_starts:
                                span_start = idx_p + 1
                                if span_start < n_orig and origin_ids[span_start] in nl_toks:
                                    span_start += 1
                                span_end = span_start
                                while span_end < n_orig and origin_ids[span_end] != end_tok:
                                    span_end += 1
                                if span_end > span_start:
                                    pending_ctx.append(list(origin_ids[span_start:span_end]))
                                idx_p = span_end + 1
                            else:
                                idx_p += 1
                sess = RequestToolSession(
                    session_id=session_id,
                    thinking_budget=thinking_budget,
                    gate_mask_deliberate=gate_mask_deliberate,
                    force_system=params.get("force_system"),
                    tool_mode=str(params.get("tool_mode", "all")),
                    max_tool_calls=int(params.get("max_tool_calls", 16)),
                    allow_external_tools=bool(params.get("allow_external_tools", False)),
                    harness=None,
                    pending_context_messages=pending_ctx,
                    req_ref=req,
                )
                cls._sessions[session_id] = sess
            elif req is not None and sess.req_ref is None:
                sess.req_ref = req
            if req is not None:
                params["__reckoner_session__"] = sess
            return sess

    @staticmethod
    def _escape_non_pipe_angle_added_tokens_for_tokenizer(text: str, tokenizer: Any, tids: TokenIdTable) -> str:
        if "<" not in text:
            return text
        tags: Tuple[str, ...] = (
            (
                "</tool_response>",
                "<tool_response>",
                "</tool_call>",
                "<tool_call>",
                "</think>",
                "<think>",
            )
            if tids.is_qwen_vocab
            else ()
        )
        added_dec = getattr(tokenizer, "added_tokens_decoder", None)
        if isinstance(added_dec, dict) and added_dec:
            dyn_tags = [
                str(getattr(v, "content", v))
                for v in added_dec.values()
                if str(getattr(v, "content", v)).startswith("<")
                and not str(getattr(v, "content", v)).startswith("<|")
            ]
            if dyn_tags:
                tags = tuple(sorted(set(dyn_tags), key=lambda s: (-len(s), s)))
        esc = text
        for tag in tags:
            if tag in esc:
                esc = esc.replace(tag, "<\\" + tag[1:])
        return esc

    @classmethod
    def _escape_tool_body_for_tokenizer(
        cls,
        text: str,
        tokenizer: Any,
        tids: TokenIdTable,
    ) -> str:
        if not text:
            return text
        escaped = text.replace("<|", "<\\|")
        return cls._escape_non_pipe_angle_added_tokens_for_tokenizer(
            escaped, tokenizer, tids
        )

    @staticmethod
    def _split_leading_register_prefix(text: str) -> Tuple[str, str, bool]:
        if not text.startswith("<|r"):
            return "", text, False
        end_idx = text.find("|> = ")
        if end_idx not in (4, 5):
            return "", text, False
        num_str = text[3:end_idx]
        if not num_str.isdigit():
            return "", text, False
        reg_num = int(num_str)
        if reg_num < 0 or reg_num >= 16:
            return "", text, False
        cut = end_idx + len("|> = ")
        return text[:cut], text[cut:], True

    @classmethod
    def _escape_solve_success_body_for_tokenizer(
        cls,
        text: str,
        tokenizer: Any,
        tids: TokenIdTable,
    ) -> str:
        parts: List[str] = []
        rem = text
        while True:
            idx = rem.find("(<|r")
            if idx < 0:
                parts.append(cls._escape_tool_body_for_tokenizer(rem, tokenizer, tids))
                break
            after = rem[idx + 4 :]
            close_idx = after.find("|>")
            if close_idx in (1, 2) and after[:close_idx].isdigit():
                reg_num = int(after[:close_idx])
                tail = after[close_idx + 2 :]
                if 0 <= reg_num < 16 and (
                    tail.startswith(")") or tail.startswith(", replaced ")
                ):
                    parts.append(
                        cls._escape_tool_body_for_tokenizer(rem[:idx], tokenizer, tids)
                    )
                    parts.append(rem[idx : idx + 4 + close_idx + 2])
                    rem = tail
                    continue
            parts.append(
                cls._escape_tool_body_for_tokenizer(rem[: idx + 4], tokenizer, tids)
            )
            rem = after
        return "".join(parts)

    @classmethod
    def _sanitize_14b_find_tokens(
        cls,
        tokenizer: Any,
        tids: TokenIdTable,
        raw_find_tokens: List[int],
    ) -> List[int]:
        if not raw_find_tokens:
            return list(raw_find_tokens)
        l_max = tids.tok_l_anchor_base + 1023 if tids.tok_l_anchor_base >= 0 else 34175
        if all(128 <= tid <= l_max for tid in raw_find_tokens):
            return list(raw_find_tokens)
        clean: List[int] = []
        for tid in raw_find_tokens:
            if 128 <= tid <= l_max:
                clean.append(tid)
            else:
                tok_str = tokenizer.decode([tid], skip_special_tokens=False)
                esc_str = cls._escape_tool_body_for_tokenizer(tok_str, tokenizer, tids)
                clean.extend(tokenizer.encode(esc_str, add_special_tokens=False))
        return clean

    @classmethod
    def _encode_tool_observation(
        cls,
        tokenizer: Any,
        tids: TokenIdTable,
        tool_name: str,
        result_str: str,
        success: bool,
        raw_find_tokens: Optional[List[int]] = None,
    ) -> List[int]:
        """Single choke point for encoding in-path tool observations into token IDs."""
        if not success:
            enc_err = cls._escape_tool_body_for_tokenizer(result_str, tokenizer, tids)
            return list(
                tokenizer.encode(
                    f"<|obs:err|>{enc_err}<|/obs|>",
                    add_special_tokens=False,
                )
            )

        if (
            not tids.is_qwen_vocab
            and tool_name in ("find", "lines")
            and not tids.no_anchors
        ):
            if not raw_find_tokens and result_str not in ("", "no matches"):
                enc_body = cls._escape_tool_body_for_tokenizer(
                    result_str, tokenizer, tids
                )
                return list(
                    tokenizer.encode(
                        f"<|obs|>{enc_body}<|/obs|>", add_special_tokens=False
                    )
                )
            clean_tokens = cls._sanitize_14b_find_tokens(
                tokenizer, tids, raw_find_tokens or []
            )
            return [tids.tok_obs] + clean_tokens + [tids.tok_end_obs]

        if tool_name in ("calc", "str", "date", "lua", "lookup", "reg"):
            reg_prefix, body, ok = cls._split_leading_register_prefix(result_str)
            if ok:
                enc_body = cls._escape_tool_body_for_tokenizer(body, tokenizer, tids)
                return list(
                    tokenizer.encode(
                        f"<|obs|>{reg_prefix}{enc_body}<|/obs|>",
                        add_special_tokens=False,
                    )
                )
            enc_body = cls._escape_tool_body_for_tokenizer(result_str, tokenizer, tids)
            return list(
                tokenizer.encode(f"<|obs|>{enc_body}<|/obs|>", add_special_tokens=False)
            )

        if tool_name == "solve":
            reg_prefix, body, ok = cls._split_leading_register_prefix(result_str)
            if ok:
                enc_body = cls._escape_tool_body_for_tokenizer(body, tokenizer, tids)
                return list(
                    tokenizer.encode(
                        f"<|obs|>{reg_prefix}{enc_body}<|/obs|>",
                        add_special_tokens=False,
                    )
                )
            enc_body = cls._escape_solve_success_body_for_tokenizer(
                result_str, tokenizer, tids
            )
            return list(
                tokenizer.encode(f"<|obs|>{enc_body}<|/obs|>", add_special_tokens=False)
            )

        if tool_name == "quote":
            body = result_str
            if body.startswith("<|obs|>") and body.endswith("<|/obs|>"):
                body = body[len("<|obs|>") : -len("<|/obs|>")]
            enc_body = cls._escape_tool_body_for_tokenizer(body, tokenizer, tids)
            return list(
                tokenizer.encode(f"<|obs|>{enc_body}<|/obs|>", add_special_tokens=False)
            )

        enc_body = cls._escape_tool_body_for_tokenizer(result_str, tokenizer, tids)
        return list(
            tokenizer.encode(f"<|obs|>{enc_body}<|/obs|>", add_special_tokens=False)
        )

    @classmethod
    def _execute_tool_call(
        cls,
        sess: RequestToolSession,
        tool_name: str,
        call_tokens: List[int],
        tokenizer: Any,
        t_call_enter: float = 0.0,
    ) -> Tuple[List[int], bool]:
        tids = cls.get_token_table(tokenizer)
        payload_text = tokenizer.decode(call_tokens, skip_special_tokens=False).strip()
        harness = cls._ensure_session_harness(
            sess,
            flush_anchors=(tool_name in ("quote", "find", "lines")),
            tool_name=tool_name,
            tokenizer=tokenizer,
        )

        t_ipc_start = time.perf_counter()
        raw_find_tokens: Optional[List[int]] = None
        if tool_name == "external":
            if sess.allow_external_tools:
                sess.tool_calls.append(
                    {
                        "tool": "external",
                        "payload": payload_text,
                        "result": "",
                        "is_error": False,
                        "latency_us": 0.0,
                        "harness_ipc_ms": 0.0,
                        "pause_resume_ms": 0.0,
                        "obs_prefill_ms": 0.0,
                        "call_to_next_token_ms": 0.0,
                        "obs_tokens": 0,
                    }
                )
                return [tids.tok_end_turn], True
            success = False
            t_res = f"ErrUnknownTool: external tools not supported ({payload_text})"
            t_lat_us = 0.0
        elif tool_name == "find":
            harness_tool = "lines" if payload_text.startswith("lines(") else "find"
            if hasattr(harness, "dispatch_full"):
                resp = harness.dispatch_full(harness_tool, payload_text)
            else:
                ok, res, _ = harness.dispatch_structured(harness_tool, payload_text)
                resp = {"success": ok, "result": res if ok else "", "error": "" if ok else res}
            success = bool(resp.get("success"))
            t_res = str(resp.get("result", "") if success else resp.get("error", ""))
            t_lat_us = float(resp.get("elapsed_us", 0.0))
            if success and "tokens" in resp:
                raw_find_tokens = list(resp["tokens"])
            tool_name = harness_tool
        else:
            success, t_res, t_lat_ms = harness.dispatch_structured(tool_name, payload_text)
            t_lat_us = float(t_lat_ms) * 1000.0

        obs_ids = cls._encode_tool_observation(
            tokenizer=tokenizer,
            tids=tids,
            tool_name=tool_name,
            result_str=t_res,
            success=success,
            raw_find_tokens=raw_find_tokens,
        )

        t_ipc_end = time.perf_counter()
        harness_ipc_ms = (t_ipc_end - t_ipc_start) * 1000.0
        enter_ts = t_call_enter if t_call_enter > 0.0 else t_ipc_start
        pause_resume_ms = max(0.0, (t_ipc_end - enter_ts) * 1000.0 - (t_lat_us / 1000.0))

        sess.total_tool_latency_us += t_lat_us
        sess.tool_calls.append(
            {
                "tool": tool_name,
                "payload": payload_text,
                "result": t_res,
                "is_error": not success,
                "latency_us": t_lat_us,
                "harness_ipc_ms": round(harness_ipc_ms, 4),
                "pause_resume_ms": round(pause_resume_ms, 4),
                "obs_prefill_ms": 0.0,
                "call_to_next_token_ms": round((t_ipc_end - enter_ts) * 1000.0, 4),
                "obs_tokens": len(obs_ids),
            }
        )
        sess.active_obs_call_idx = len(sess.tool_calls) - 1
        sess.active_call_enter_ts = enter_ts
        sess.active_first_force_ts = t_ipc_end
        return obs_ids, False

    @classmethod
    def _execute_quote(
        cls,
        sess: RequestToolSession,
        quote_tokens: List[int],
        tokenizer: Any,
        t_call_enter: float = 0.0,
    ) -> List[int]:
        tids = cls.get_token_table(tokenizer)
        harness = cls._ensure_session_harness(
            sess, flush_anchors=True, tool_name="quote", tokenizer=tokenizer
        )

        t_ipc_start = time.perf_counter()
        m_id = None
        line_ids: List[int] = []
        if not tids.no_anchors and tids.tok_m_anchor_base >= 0 and tids.tok_l_anchor_base >= 0:
            for qt in quote_tokens:
                if tids.tok_m_anchor_base <= qt < tids.tok_m_anchor_base + 256:
                    m_id = qt - tids.tok_m_anchor_base
                elif tids.tok_l_anchor_base <= qt < tids.tok_l_anchor_base + 1024:
                    line_ids.append(qt - tids.tok_l_anchor_base)

        if m_id is not None and line_ids:
            lo, hi = min(line_ids), max(line_ids)
            qpayload = f"m{m_id}, L{lo}, L{hi}"
            if hasattr(harness, "dispatch_full"):
                resp = harness.dispatch_full("quote", qpayload)
            else:
                ok, res, _ = harness.dispatch_structured("quote", qpayload)
                resp = {"success": ok, "result": res if ok else "", "error": "" if ok else res}
            success = bool(resp.get("success"))
            t_lat_us = float(resp.get("elapsed_us", 0.0))
            t_ipc_end = time.perf_counter()
            enter_ts = t_call_enter if t_call_enter > 0.0 else t_ipc_start
            pause_resume_ms = max(0.0, (t_ipc_end - enter_ts) * 1000.0 - (t_lat_us / 1000.0))
            receipt = resp.get("receipt", "")
            if success and receipt:
                obs_ids = cls._encode_tool_observation(
                    tokenizer, tids, "quote", receipt, True
                )
            else:
                err_msg = resp.get("error", "quote failed")
                sess.format_errors.append(f"quote dispatch failed: {err_msg}")
                obs_ids = cls._encode_tool_observation(
                    tokenizer, tids, "quote", err_msg, False
                )
            sess.total_tool_latency_us += t_lat_us
            sess.tool_calls.append(
                {
                    "tool": "quote",
                    "payload": qpayload,
                    "result": resp.get("text", resp.get("result", "")) if success else resp.get("error", ""),
                    "is_error": not success,
                    "latency_us": t_lat_us,
                    "harness_ipc_ms": round((t_ipc_end - t_ipc_start) * 1000.0, 4),
                    "pause_resume_ms": round(pause_resume_ms, 4),
                    "obs_prefill_ms": 0.0,
                    "call_to_next_token_ms": round((t_ipc_end - enter_ts) * 1000.0, 4),
                    "obs_tokens": len(obs_ids),
                }
            )
            sess.active_obs_call_idx = len(sess.tool_calls) - 1
            sess.active_call_enter_ts = enter_ts
            sess.active_first_force_ts = t_ipc_end
            return obs_ids

        sess.format_errors.append("malformed quote anchors (no message/line anchor tokens)")
        return cls._encode_tool_observation(
            tokenizer, tids, "quote", "malformed quote", False
        )

    @staticmethod
    def _external_call_complete(call_tokens: List[int], tokenizer: Any) -> bool:
        """True when the open generic call's payload is a complete `name(args)` call expression."""
        if not call_tokens:
            return False
        try:
            if ")" not in tokenizer.decode([int(call_tokens[-1])], skip_special_tokens=False):
                return False
            text = tokenizer.decode([int(t) for t in call_tokens], skip_special_tokens=False).strip()
            if not text.endswith(")"):
                return False
            node = ast.parse(text, mode="eval").body
        except Exception:
            return False
        return isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute))

    @classmethod
    def _advance_session_state(
        cls,
        sess: RequestToolSession,
        output_ids: List[int],
        tokenizer: Any,
        allow_tool_exec: bool = True,
        t_step_enter: float = 0.0,
    ) -> None:
        tids = cls.get_token_table(tokenizer)
        mode_map = tids.mode_token_map
        call_map = tids.call_start_to_tool
        start_idx = sess.processed_output_len
        n = len(output_ids)
        if start_idx >= n:
            return

        for pos in range(start_idx, n):
            tok = int(output_ids[pos])
            # Record logprob of chosen token from the previous step's unmasked logits if available
            if pos == n - 1 and sess.last_step_logits is not None:
                is_gen = sess.is_generated_mask[pos] if pos < len(sess.is_generated_mask) else True
                if not is_gen:
                    sess.logprobs.append(0.0)
                else:
                    lp = float(F.log_softmax(sess.last_step_logits.float(), dim=-1)[tok].item())
                    sess.logprobs.append(lp)
            elif len(sess.logprobs) <= pos:
                sess.logprobs.append(0.0)

            # Ensure is_generated_mask is populated if not already set by forced queue
            if len(sess.is_generated_mask) <= pos:
                sess.is_generated_mask.append(True)

            is_gen_tok = sess.is_generated_mask[pos]
            if not is_gen_tok:
                continue

            if pos == 0:
                if tok in mode_map:
                    sess.action_mode = mode_map[tok]
                    sess.current_mode = sess.action_mode.lower()
                    sess.segments.append({"mode": sess.current_mode, "text": ""})
                    continue
                if tok in call_map:
                    # The prompt already ends with the mode token (e.g. a continuation cut after `<|act|>`), so the
                    # first output token opens a call: track it as one instead of swallowing it, or the call is never
                    # executed and an external call never pauses.
                    sess.action_mode = "ACT"
                    sess.current_mode = "act"
                    sess.segments.append({"mode": "act", "text": ""})
                else:
                    sess.action_mode = "DIRECT"
                    sess.current_mode = "none"
                    sess.segments.append({"mode": "none", "text": ""})
                    sess.format_errors.append("missing mode token after <|turn:agent|>")
                    continue

            if sess.in_tool_call:
                if tok == tids.tok_end_turn:
                    sess.format_errors.append("Unclosed tool call at turn end")
                    sess.in_tool_call = False
                    sess.call_tokens.clear()
                elif tok == tids.tok_end_call:
                    if allow_tool_exec:
                        obs_ids, is_gen_stop = cls._execute_tool_call(
                            sess,
                            sess.current_tool_name,
                            list(sess.call_tokens),
                            tokenizer,
                            t_call_enter=t_step_enter,
                        )
                        for obs_tok in obs_ids:
                            sess.pending_forced_tokens.append((int(obs_tok), is_gen_stop))
                    sess.in_tool_call = False
                    sess.call_tokens.clear()
                else:
                    sess.call_tokens.append(tok)
                    # Qwen vocab (Reckoner-Q2B): a generic `<|call|>name(args)` is closed by `<|/call|>` as
                    # soon as the payload is a complete call expression. Nothing else masked the turn end
                    # inside an open call, so in long prompts the model could sample `<|/turn|>` (or a second
                    # `<|call|>`) straight after `)`, and the external-call pause never fired.
                    if (
                        allow_tool_exec
                        and pos == n - 1
                        and tids.is_qwen_vocab
                        and sess.current_tool_name == "external"
                        and not sess.pending_forced_tokens
                        and cls._external_call_complete(sess.call_tokens, tokenizer)
                    ):
                        sess.pending_forced_tokens.append((tids.tok_end_call, True))
                continue

            if sess.in_quote:
                if tok == tids.tok_end_turn:
                    sess.format_errors.append("Unclosed quote at turn end")
                    sess.in_quote = False
                    sess.quote_tokens.clear()
                elif not tids.no_anchors and tok == tids.tok_end_quote:
                    if allow_tool_exec:
                        obs_ids = cls._execute_quote(
                            sess,
                            list(sess.quote_tokens),
                            tokenizer,
                            t_call_enter=t_step_enter,
                        )
                        for obs_tok in obs_ids:
                            sess.pending_forced_tokens.append((int(obs_tok), False))
                    sess.in_quote = False
                    sess.quote_tokens.clear()
                else:
                    sess.quote_tokens.append(tok)
                continue

            if tok == tids.tok_end_turn:
                continue

            if tok in mode_map:
                sess.current_mode = mode_map[tok].lower()
                sess.segments.append({"mode": sess.current_mode, "text": ""})
                continue

            if not tids.no_anchors and tok == tids.tok_quote:
                if sess.num_tool_calls >= sess.max_tool_calls:
                    sess.exceeded_tool_limit = True
                    sess.format_errors.append("Exceeded 16 tool/quote calls limit")
                    sess.pending_forced_tokens.append((tids.tok_end_turn, True))
                    break
                sess.num_tool_calls += 1
                sess.in_quote = True
                sess.quote_tokens = []
                continue

            if tok in call_map:
                if sess.current_mode == "deliberate":
                    sess.format_errors.append("Tool call attempted inside <|deliberate|> segment")
                    sess.current_mode = "ready"
                    sess.segments.append({"mode": "ready", "text": ""})
                    sess.pending_forced_tokens.append((tids.tok_ready, True))
                elif sess.num_tool_calls >= sess.max_tool_calls:
                    sess.exceeded_tool_limit = True
                    sess.format_errors.append("Exceeded 16 tool calls limit")
                    sess.pending_forced_tokens.append((tids.tok_end_turn, True))
                    break
                else:
                    sess.num_tool_calls += 1
                    sess.in_tool_call = True
                    sess.current_tool_name = call_map[tok]
                    sess.call_tokens = []
                continue

            if sess.current_mode == "deliberate":
                sess.thinking_tokens += 1
            if sess.segments:
                sess.segments[-1].setdefault("_toks", []).append(tok)

        sess.processed_output_len = n
        if not allow_tool_exec:
            cls._materialize_segments(sess, tokenizer)

        # Check thinking budget forcing after processing newly emitted tokens
        if (
            not sess.pending_forced_tokens
            and sess.current_mode == "deliberate"
            and sess.thinking_tokens >= sess.thinking_budget
        ):
            sess.pending_forced_tokens.append((tids.tok_ready, True))

    @classmethod
    def _pop_inflight_obs_tokens(
        cls,
        sess: RequestToolSession,
        current_output_len: int,
        max_new_tokens: Optional[int] = None,
        req: Any = None,
    ) -> List[int]:
        obs_ids: List[int] = []
        while sess.pending_forced_tokens and not sess.pending_forced_tokens[0][1]:
            tok_id, _ = sess.pending_forced_tokens.popleft()
            obs_ids.append(int(tok_id))
        if not obs_ids:
            return []

        if max_new_tokens is not None:
            rem = max(0, int(max_new_tokens) - int(current_output_len))
            if len(obs_ids) > rem:
                obs_ids = obs_ids[:rem]
                sess.pending_forced_tokens.clear()

        while len(sess.is_generated_mask) < current_output_len:
            sess.is_generated_mask.append(True)
        while len(sess.logprobs) < current_output_len:
            sess.logprobs.append(0.0)

        req_lp = getattr(req, "logprob", None) if (req is not None and getattr(req, "return_logprob", False)) else None
        for obs_tok in obs_ids:
            sess.is_generated_mask.append(False)
            sess.logprobs.append(0.0)
            if req_lp is not None:
                if req_lp.output_token_logprobs_val is not None:
                    req_lp.output_token_logprobs_val.append(0.0)
                if req_lp.output_token_logprobs_idx is not None:
                    req_lp.output_token_logprobs_idx.append(int(obs_tok))
                if getattr(req_lp, "top_logprobs_num", 0) > 0:
                    if req_lp.output_top_logprobs_val is not None:
                        req_lp.output_top_logprobs_val.append([])
                    if req_lp.output_top_logprobs_idx is not None:
                        req_lp.output_top_logprobs_idx.append([])
                if getattr(req_lp, "token_ids_logprob", None) is not None:
                    if req_lp.output_token_ids_logprobs_val is not None:
                        req_lp.output_token_ids_logprobs_val.append([])
                    if req_lp.output_token_ids_logprobs_idx is not None:
                        req_lp.output_token_ids_logprobs_idx.append([])

        sess.processed_output_len = current_output_len + len(obs_ids)
        if sess.active_obs_call_idx is not None and 0 <= sess.active_obs_call_idx < len(sess.tool_calls):
            t_inject_done = time.perf_counter()
            tc_entry = sess.tool_calls[sess.active_obs_call_idx]
            lat_ms = float(tc_entry.get("latency_us", 0.0)) / 1000.0
            tc_entry["pause_resume_ms"] = round(
                max(0.0, (t_inject_done - sess.active_call_enter_ts) * 1000.0 - lat_ms),
                4,
            )
            tc_entry["injection_mode"] = "inflight_extend"
            cls._save_session_shm(sess)
            sess.active_first_force_ts = time.perf_counter()
        else:
            cls._save_session_shm(sess)
        return obs_ids

    @classmethod
    def _on_inflight_token(cls, req: Any) -> Optional[List[int]]:
        """Scheduler result-processor callback: executes tool call on `<|/call|>` and returns `obs_ids` for 1-step EXTEND."""
        out_ids = getattr(req, "output_ids", None)
        if not out_ids:
            return None
        tokenizer = cls.get_tokenizer()
        tids = cls.get_token_table(tokenizer)
        trigger_ids = (tids.tok_end_call,) if tids.no_anchors else (tids.tok_end_call, tids.tok_end_quote)
        if int(out_ids[-1]) not in trigger_ids:
            return None
        t_enter = time.perf_counter()
        sp = getattr(req, "sampling_params", None)
        params = getattr(sp, "custom_params", None)
        if not isinstance(params, dict):
            return None
        sess = cls._get_or_create_session(params, req=req)
        cls._advance_session_state(sess, out_ids, tokenizer, t_step_enter=t_enter)
        max_new = getattr(sp, "max_new_tokens", None)
        obs_ids = cls._pop_inflight_obs_tokens(
            sess,
            current_output_len=len(out_ids),
            max_new_tokens=max_new,
            req=req,
        )
        return obs_ids if obs_ids else None

    @classmethod
    def maybe_inflight_extend_params(
        cls,
        params: Dict[str, Any],
        max_new_tokens: Optional[int] = None,
    ) -> List[int]:
        """In-process / CPU helper mirroring `_on_inflight_token` when `inflight_extend` is active."""
        use_inflight = bool(params.get("inflight_extend", cls.supports_inflight_extend()))
        if not use_inflight:
            return []
        out_ids = params.get("output_ids", [])
        if not out_ids:
            return []
        tokenizer = cls.get_tokenizer()
        tids = cls.get_token_table(tokenizer)
        trigger_ids = (tids.tok_end_call,) if tids.no_anchors else (tids.tok_end_call, tids.tok_end_quote)
        if int(out_ids[-1]) not in trigger_ids:
            return []
        t_enter = time.perf_counter()
        sess = cls._get_or_create_session(params, req=params.get("__req__"))
        output_ids_list = list(out_ids)
        cls._advance_session_state(sess, output_ids_list, tokenizer, t_step_enter=t_enter)
        return cls._pop_inflight_obs_tokens(
            sess,
            current_output_len=len(output_ids_list),
            max_new_tokens=max_new_tokens,
            req=params.get("__req__"),
        )

    _call_step_counter: int = 0
    _device_mask_cache: Dict[Tuple[Any, ...], Dict[str, torch.Tensor]] = {}

    @classmethod
    def _get_device_masks(
        cls,
        device: torch.device,
        tids: TokenIdTable,
        vocab_limit: int,
    ) -> Dict[str, torch.Tensor]:
        key = (
            device,
            int(vocab_limit),
            id(tids),
            tuple(tids.obs_delim_ids),
            tuple(tids.banned_token_ids),
            tuple(tids.none_mode_forbid_ids),
            tuple(tids.non_lua_call_start_ids),
            tuple(tids.delib_forbid_ids),
        )
        cached = cls._device_mask_cache.get(key)
        if cached is not None:
            return cached
        always_ids = [x for x in (list(tids.obs_delim_ids) + list(tids.banned_token_ids)) if 0 <= x < vocab_limit]
        none_ids = [x for x in tids.none_mode_forbid_ids if 0 <= x < vocab_limit]
        lua_only_ids = [x for x in tids.non_lua_call_start_ids if 0 <= x < vocab_limit]
        delib_ids = [x for x in tids.delib_forbid_ids if 0 <= x < vocab_limit]
        masks = {
            "always": torch.tensor(always_ids, device=device, dtype=torch.long),
            "none": torch.tensor(none_ids, device=device, dtype=torch.long),
            "lua_only": torch.tensor(lua_only_ids, device=device, dtype=torch.long),
            "delib": torch.tensor(delib_ids, device=device, dtype=torch.long),
        }
        cls._device_mask_cache[key] = masks
        return masks

    @staticmethod
    def _mask_rows_cols(logits: torch.Tensor, rows: List[int], col_t: torch.Tensor) -> None:
        if not rows or col_t.numel() == 0:
            return
        if len(rows) == logits.shape[0]:
            logits.index_fill_(1, col_t, -float("inf"))
        else:
            row_t = torch.tensor(rows, device=logits.device, dtype=torch.long).unsqueeze(1)
            logits[row_t, col_t.unsqueeze(0)] = -float("inf")

    def __call__(
        self,
        logits: torch.Tensor,
        custom_param_list: List[Dict[str, Any]],
    ) -> torch.Tensor:
        t_step_enter = time.perf_counter()
        assert logits.shape[0] == len(custom_param_list)
        self.log_active_path()
        vocab_limit = int(logits.shape[-1])
        tokenizer = self.get_tokenizer(vocab_size_hint=vocab_limit)
        tids = self.get_token_table(tokenizer)
        mode_token_ids = tids.mode_token_ids
        banned_ids = list(tids.banned_token_ids)

        _is_cpu_ovr = getattr(self, "_is_cpu_override", None)
        is_cpu = (logits.device.type == "cpu") if _is_cpu_ovr is None else bool(_is_cpu_ovr)
        default_inflight = self.supports_inflight_extend()

        # If every request in this batch is an intermediate chunk of a chunked prefill
        # (`req.inflight_middle_chunks > 0`), SGLang discards the sampled token anyway;
        # return immediately so chunked prefill pays zero custom logit processor overhead.
        if custom_param_list and all(
            getattr(p.get("__req__"), "inflight_middle_chunks", 0) > 0
            for p in custom_param_list
        ):
            return logits

        # Pre-check which rows in this batch are at cur_pos == 0 AND need gate_dist recorded
        # so we only compute softmax/argmax on those specific rows (zero work on pure decode steps).
        pos0_rows: List[int] = []
        need_think_stats = False
        for r_i, params in enumerate(custom_param_list):
            req = params.get("__req__")
            if getattr(req, "inflight_middle_chunks", 0) > 0:
                continue
            out_ids = getattr(req, "output_ids", None) if req is not None else params.get("output_ids", None)
            if not out_ids and (
                is_cpu
                or req is None
                or bool(params.get("record_gate_dist", False))
                or bool(params.get("record_logits", False))
            ):
                pos0_rows.append(r_i)
                if is_cpu or req is None or bool(params.get("record_logits", False)) or not bool(params.get("skip_think_stats", False)):
                    need_think_stats = True

        pos0_mode_probs: Dict[int, List[float]] = {}
        pos0_think_probs: Dict[int, float] = {}
        pos0_think_top1: Dict[int, bool] = {}
        if pos0_rows:
            logits_p0 = logits[pos0_rows] if len(pos0_rows) < logits.shape[0] else logits
            mp_list = F.softmax(logits_p0[:, mode_token_ids].detach().float(), dim=-1).cpu().tolist()
            for idx_in_p0, r_i in enumerate(pos0_rows):
                pos0_mode_probs[r_i] = mp_list[idx_in_p0]
            if need_think_stats and banned_ids and 0 <= banned_ids[0] < vocab_limit:
                think_id = int(banned_ids[0])
                top1_list = (torch.argmax(logits_p0, dim=-1) == think_id).cpu().tolist()
                lse = torch.logsumexp(logits_p0.detach().float(), dim=-1)
                tp_list = torch.exp(logits_p0[:, think_id].detach().float() - lse).cpu().tolist()
                for idx_in_p0, r_i in enumerate(pos0_rows):
                    pos0_think_top1[r_i] = bool(top1_list[idx_in_p0])
                    pos0_think_probs[r_i] = float(tp_list[idx_in_p0])

        if not is_cpu and os.environ.get("RECKONER_RC_WATCHDOG_BREATHER") == "1":
            type(self)._call_step_counter += 1
            if type(self)._call_step_counter % 32 == 0:
                torch.cuda.synchronize(logits.device)
                time.sleep(0.015)

        dmasks = self._get_device_masks(logits.device, tids, vocab_limit)
        active_rows: List[int] = []
        none_rows: List[int] = []
        lua_only_rows: List[int] = []
        delib_rows: List[int] = []
        forced_rows: List[Tuple[int, int]] = []

        for i, params in enumerate(custom_param_list):
            req = params.get("__req__")
            # Skip intermediate chunked-prefill rows (`inflight_middle_chunks > 0`): SGLang
            # discards their sampled token, so leaving their logits untouched matches `main`.
            if getattr(req, "inflight_middle_chunks", 0) > 0:
                continue
            active_rows.append(i)
            use_inflight = bool(params.get("inflight_extend", default_inflight))
            if req is not None:
                if use_inflight:
                    params["__inflight_extend_cb__"] = self._on_inflight_token
                else:
                    params.pop("__inflight_extend_cb__", None)
            sess = self._get_or_create_session(params, req=req)

            if req is not None and hasattr(req, "output_ids"):
                output_ids = req.output_ids
            else:
                output_ids = params.get("output_ids", [])

            prev_calls = len(sess.tool_calls)
            prev_mode = sess.current_mode
            prev_err = len(sess.format_errors)

            # Process any newly generated token(s) since last step
            self._advance_session_state(
                sess, output_ids, tokenizer, t_step_enter=t_step_enter
            )

            # If we just finished forcing all observation tokens for an active tool call,
            # record the exact observation prefill / decode-step duration and total call-to-next-token time!
            obs_timing_closed = False
            if (
                not sess.pending_forced_tokens
                and sess.active_obs_call_idx is not None
                and 0 <= sess.active_obs_call_idx < len(sess.tool_calls)
            ):
                tc_entry = sess.tool_calls[sess.active_obs_call_idx]
                obs_prefill_ms = max(0.0, (t_step_enter - sess.active_first_force_ts) * 1000.0)
                total_ms = (
                    (sess.active_first_force_ts - sess.active_call_enter_ts) * 1000.0
                    + obs_prefill_ms
                )
                tc_entry["obs_prefill_ms"] = round(obs_prefill_ms, 4)
                tc_entry["call_to_next_token_ms"] = round(total_ms, 4)
                sess.active_obs_call_idx = None
                obs_timing_closed = True

            cur_pos = len(output_ids)
            record_logits = bool(params.get("record_logits", False))
            # `simulate_gpu_shm` (tests only) keeps CPU logits but follows the GPU `/dev/shm` save schedule.
            shm_cpu = is_cpu and not bool(params.get("simulate_gpu_shm", False))
            if is_cpu or record_logits:
                sess.last_step_logits = logits[i].detach().float().cpu().clone()
            else:
                sess.last_step_logits = None

            max_new_req = getattr(getattr(req, "sampling_params", None), "max_new_tokens", 2) if req is not None else 1
            shm_dirty = (
                (cur_pos == 0 and (shm_cpu or record_logits or req is None or max_new_req <= 1))
                or cur_pos == 1
                or (cur_pos + 1 >= max_new_req)
                or record_logits
                or obs_timing_closed
                or sess.current_mode != prev_mode
                or len(sess.format_errors) != prev_err
                or (len(sess.tool_calls) != prev_calls and not sess.pending_forced_tokens)
            )

            # Position 0: record gate distribution (when requested) and apply force_system / effort gate masks
            if cur_pos == 0:
                if not sess.gate_dist and i in pos0_mode_probs:
                    mprobs = pos0_mode_probs[i]
                    sess.gate_dist = {
                        "ready": float(mprobs[0]),
                        "act": float(mprobs[1]),
                        "deliberate": float(mprobs[2]),
                        "clarify": float(mprobs[3]),
                    }
                    sess.p_system1 = sess.gate_dist["ready"] + sess.gate_dist["act"]
                    sess.p_system2 = sess.gate_dist["deliberate"]
                    if i in pos0_think_probs and i in pos0_think_top1:
                        sess.unmasked_think_prob = float(pos0_think_probs[i])
                        sess.unmasked_think_top1 = bool(pos0_think_top1[i])

                force_sys = sess.force_system
                if force_sys in ("sys1", "system1"):
                    keep_act = logits[i, tids.tok_act].clone()
                    keep_ready = logits[i, tids.tok_ready].clone()
                    logits[i, :] = -float("inf")
                    logits[i, tids.tok_act] = keep_act
                    logits[i, tids.tok_ready] = keep_ready
                elif force_sys in ("sys2", "system2", "deliberate"):
                    keep = logits[i, tids.tok_deliberate].clone()
                    logits[i, :] = -float("inf")
                    logits[i, tids.tok_deliberate] = keep
                elif force_sys == "ready":
                    keep = logits[i, tids.tok_ready].clone()
                    logits[i, :] = -float("inf")
                    logits[i, tids.tok_ready] = keep
                elif force_sys == "act":
                    keep = logits[i, tids.tok_act].clone()
                    logits[i, :] = -float("inf")
                    logits[i, tids.tok_act] = keep
                elif sess.gate_mask_deliberate:
                    logits[i, tids.tok_deliberate] = -float("inf")

                if sess.tool_mode == "none" and force_sys != "act":
                    logits[i, tids.tok_act] = -float("inf")

            # If a forced token (`<|obs|>...<|/obs|>` or budget `<|ready|>`) is queued, force it now
            if sess.pending_forced_tokens:
                forced_tok, is_gen_forced = sess.pending_forced_tokens.popleft()
                while len(sess.is_generated_mask) < cur_pos:
                    sess.is_generated_mask.append(True)
                sess.is_generated_mask.append(bool(is_gen_forced))
                if is_gen_forced:
                    sess.forced_positions.append(cur_pos)
                forced_rows.append((i, int(forced_tok)))
                first_obs_pop = False
                if (
                    sess.active_obs_call_idx is not None
                    and 0 <= sess.active_obs_call_idx < len(sess.tool_calls)
                    and int(forced_tok) in (tids.tok_obs, tids.tok_obs_err)
                ):
                    t_force_done = time.perf_counter()
                    sess.active_first_force_ts = t_force_done
                    tc_entry = sess.tool_calls[sess.active_obs_call_idx]
                    lat_ms = float(tc_entry.get("latency_us", 0.0)) / 1000.0
                    tc_entry["pause_resume_ms"] = round(
                        max(0.0, (t_force_done - sess.active_call_enter_ts) * 1000.0 - lat_ms),
                        4,
                    )
                    tc_entry["injection_mode"] = "logit_forcing"
                    first_obs_pop = True
                # Save SHM on the first forced observation token (`first_obs_pop`), the last
                # forced token (`not sess.pending_forced_tokens`), or when this step reaches
                # `max_new_tokens` (`cur_pos + 1 >= max_new_req`), so a request truncated
                # mid-injection under `logit_forcing` still persists its `tool_calls` and partial
                # `is_generated_mask` to `/dev/shm`.
                if (
                    not sess.pending_forced_tokens
                    or first_obs_pop
                    or (cur_pos + 1 >= max_new_req)
                    or shm_cpu
                    or record_logits
                ):
                    self._save_session_shm(sess)
                continue

            if shm_dirty:
                self._save_session_shm(sess)

            # Collect rows for vectorized tool_mode and deliberate masks
            if sess.tool_mode == "none":
                none_rows.append(i)
            elif sess.tool_mode == "lua_only":
                lua_only_rows.append(i)

            if sess.current_mode == "deliberate" and bool(params.get("forbid_tools_in_deliberate", True)):
                delib_rows.append(i)

        # Apply observation delimiter + banned thinking token masks across active rows
        # (skipping rows with `inflight_middle_chunks > 0`).
        self._mask_rows_cols(logits, active_rows, dmasks["always"])

        # Apply tool_mode and deliberate restraint masks in 1 batched op per group
        self._mask_rows_cols(logits, none_rows, dmasks["none"])
        self._mask_rows_cols(logits, lua_only_rows, dmasks["lua_only"])
        self._mask_rows_cols(logits, delib_rows, dmasks["delib"])

        # Apply forced tokens last so forced <|obs|> / <|/obs|> overwrite the batch-wide always mask
        for r_i, f_tok in forced_rows:
            logits[r_i, :] = -float("inf")
            logits[r_i, f_tok] = 0.0

        return logits

