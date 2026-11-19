"""Self-contained token layout, prompt formatting, and Go harness subprocess bridge.

Eliminates dependencies on the research repository layout (`serve.rollout`,
`serve.turn_format`, `tools.tool_harness`, `scripts.harness_bin`) when running
`integrations/sglang/` standalone from the `reckoner-harness` repository.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import re
import shutil
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

# Canonical Reckoner Special Token IDs (tokenizer v2, vocab_size=36,864)
TOK_TURN_USER = 2
TOK_TURN_AGENT = 3
TOK_END_TURN = 4
TOK_TURN_SYSTEM = 5
TOK_TOOLS = 6
TOK_DELIBERATE = 8
TOK_ACT = 9
TOK_READY = 10
TOK_CLARIFY = 11
TOK_CALL = 13
TOK_END_CALL = 14
TOK_OBS = 15
TOK_OBS_ERR = 16
TOK_END_OBS = 17
TOK_CALL_CALC = 18
TOK_CALL_SOLVE = 19
TOK_CALL_LUA = 20
TOK_CALL_STR = 29
TOK_CALL_DATE = 30
TOK_CALL_LOOKUP = 31
TOK_CALL_FIND = 32
TOK_QUOTE = 33
TOK_END_QUOTE = 34
TOK_NEWLINE = 326
TOK_M_ANCHOR_BASE = 32896
TOK_L_ANCHOR_BASE = 33152

MODE_TOKEN_MAP: Dict[int, str] = {
    TOK_READY: "READY",
    TOK_ACT: "ACT",
    TOK_DELIBERATE: "DELIBERATE",
    TOK_CLARIFY: "CLARIFY",
}

TOOL_TOKEN_MAP: Dict[int, str] = {
    TOK_CALL_CALC: "calc",
    TOK_CALL_SOLVE: "solve",
    TOK_CALL_LUA: "lua",
    TOK_CALL_STR: "str",
    TOK_CALL_DATE: "date",
    TOK_CALL_LOOKUP: "lookup",
    TOK_CALL: "external",
}

EFFORT_MAP: Dict[str, Dict[str, Any]] = {
    "none": {"gate_mask_deliberate": True, "budget": 0, "loop_cap": 1},
    "minimal": {"gate_mask_deliberate": True, "budget": 0, "loop_cap": 1},
    "low": {"gate_mask_deliberate": False, "budget": 256, "loop_cap": 2},
    "medium": {"gate_mask_deliberate": False, "budget": 1024, "loop_cap": 3},
    "high": {"gate_mask_deliberate": False, "budget": 4096, "loop_cap": 4},
    "fast": {"gate_mask_deliberate": False, "budget": 256, "loop_cap": 2},
    "careful": {"gate_mask_deliberate": False, "budget": 4096, "loop_cap": 4},
    "xhigh": {"gate_mask_deliberate": False, "budget": 4096, "loop_cap": 4},
}


_ADDED_TOKENS_CACHE_LOCK = threading.Lock()
_ADDED_TOKENS_CACHE: Dict[str, Tuple[int, int, List[str]]] = {}


def _resolve_non_pipe_angle_added_tokens(
    tokenizer_path: Optional[str] = None,
    added_tokens: Optional[List[str]] = None,
) -> List[str]:
    if added_tokens is not None:
        filtered = [
            t
            for t in added_tokens
            if isinstance(t, str) and t.startswith("<") and not t.startswith("<|")
        ]
        return sorted(filtered, key=lambda s: (-len(s), s))

    if tokenizer_path is not None:
        cand = tokenizer_path.strip()
        if not cand:
            raise ValueError(
                "ErrArgs: explicit tokenizer_path cannot be empty or whitespace"
            )
    else:
        raw_env = os.environ.get("RECKONER_TOKENIZER_PATH")
        if raw_env is None or raw_env == "":
            return []
        cand = raw_env.strip()
        if not cand:
            raise ValueError(
                "ErrArgs: RECKONER_TOKENIZER_PATH cannot be whitespace-only"
            )

    if os.path.isdir(cand):
        cand = os.path.join(cand, "tokenizer.json")
    if not os.path.isfile(cand):
        raise FileNotFoundError(
            f"ErrArgs: configured tokenizer file not found at {cand!r}"
        )
    abs_cand = os.path.abspath(cand)
    try:
        st = os.stat(abs_cand)
    except OSError as exc:
        raise FileNotFoundError(
            f"ErrArgs: failed to stat configured tokenizer file {abs_cand!r}: {exc}"
        ) from exc

    with _ADDED_TOKENS_CACHE_LOCK:
        cached = _ADDED_TOKENS_CACHE.get(abs_cand)
        if cached is not None and cached[0] == st.st_mtime_ns and cached[1] == st.st_size:
            return list(cached[2])

    try:
        with open(abs_cand, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        raise ValueError(
            f"ErrArgs: failed to parse configured tokenizer JSON {abs_cand!r}: {exc}"
        ) from exc

    if (
        not isinstance(data, dict)
        or "added_tokens" not in data
        or not isinstance(data.get("added_tokens"), list)
    ):
        raise ValueError(
            f"ErrArgs: invalid tokenizer JSON structure in {abs_cand!r} (missing added_tokens list)"
        )

    raw_added = data["added_tokens"]
    tags = [
        at["content"]
        for at in raw_added
        if isinstance(at, dict)
        and isinstance(at.get("content"), str)
        and at["content"].startswith("<")
        and not at["content"].startswith("<|")
    ]
    sorted_tags = sorted(tags, key=lambda s: (-len(s), s))
    with _ADDED_TOKENS_CACHE_LOCK:
        _ADDED_TOKENS_CACHE[abs_cand] = (st.st_mtime_ns, st.st_size, sorted_tags)
    return list(sorted_tags)


def escape_control_tokens(
    text: str,
    tokenizer_path: Optional[str] = None,
    added_tokens: Optional[List[str]] = None,
) -> str:
    """Escapes control-token opening markers (<| -> <\\|) and external non-<|...|> angle-bracket added_tokens."""
    tags = _resolve_non_pipe_angle_added_tokens(
        tokenizer_path=tokenizer_path, added_tokens=added_tokens
    )
    escaped = text.replace("<|", "<\\|")
    if "<" not in escaped:
        return escaped
    for tag in tags:
        if tag in escaped:
            escaped = escaped.replace(tag, "<\\" + tag[1:])
    return escaped


def format_prompt(
    query: str,
    system: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    added_tokens: Optional[List[str]] = None,
) -> str:
    """Renders canonical single-turn evaluation / inference prompt ending in <|turn:agent|>."""
    esc_query = escape_control_tokens(
        query, tokenizer_path=tokenizer_path, added_tokens=added_tokens
    )
    if system:
        esc_sys = escape_control_tokens(
            system, tokenizer_path=tokenizer_path, added_tokens=added_tokens
        )
        return (
            f"<|turn:system|>\n{esc_sys}<|/turn|>\n"
            f"<|turn:user|>\n{esc_query}<|/turn|>\n"
            f"<|turn:agent|>"
        )
    return (
        f"<|turn:user|>\n{esc_query}<|/turn|>\n"
        f"<|turn:agent|>"
    )


def format_dialogue(
    turns: List[Tuple[str, str]],
    system: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    added_tokens: Optional[List[str]] = None,
) -> str:
    """Renders multi-turn dialogue prompt ending in <|turn:agent|>, ready for generation."""
    out = ""
    if system:
        esc_sys = escape_control_tokens(
            system, tokenizer_path=tokenizer_path, added_tokens=added_tokens
        )
        out += f"<|turn:system|>\n{esc_sys}<|/turn|>\n"
    for role, content in turns:
        if role == "user":
            esc_user = escape_control_tokens(
                content, tokenizer_path=tokenizer_path, added_tokens=added_tokens
            )
            out += f"<|turn:user|>\n{esc_user}<|/turn|>\n"
        elif role in ("agent", "assistant"):
            out += f"<|turn:agent|>{content}<|/turn|>\n"
        else:
            raise ValueError(f"Unknown turn role: {role!r}")
    out += "<|turn:agent|>"
    return out


def extract_boxed_answer(text: str) -> str:
    """Extracts final boxed answer from text or returns empty string."""
    m = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    return m[-1].strip() if m else ""


@dataclass
class RolloutSample:
    """Structured output per generation sample."""

    token_ids: List[int]
    is_generated_mask: List[bool]
    logprobs: List[float]
    segments: List[Dict[str, str]]
    tool_calls: List[Dict[str, Any]]
    gate_dist: Dict[str, float]
    p_system1: float
    p_system2: float
    thinking_tokens: int
    loop_passes: int
    truncated: bool
    format_errors: List[str]
    action_mode: str
    prompt_str: str
    final_response: str
    answer: str
    prompt_tokens: int
    tokens_generated: int
    tool_latency_us: float
    forced_positions: List[int] = field(default_factory=list)
    p_think_first_pos: float = 0.0
    p_think_gate_pos: float = 0.0
    p_think_max: float = 0.0
    think_was_top1_anywhere: bool = False

    @property
    def text(self) -> str:
        return self.final_response


def resolve_harness_binary(binary_path: Optional[str] = None) -> str:
    """Resolves a valid `reckoner-harness` executable path across standalone and repo layouts."""
    if binary_path and os.path.isfile(binary_path):
        return os.path.abspath(binary_path)

    env_bin = os.environ.get("RECKONER_HARNESS_BIN")
    if env_bin and os.path.isfile(env_bin):
        return os.path.abspath(env_bin)

    # Search relative to harness repo root (`integrations/sglang/sglang_reckoner/../../..`)
    harness_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    candidates = [
        os.path.join(harness_repo_root, "reckoner-harness"),
        os.path.join(harness_repo_root, "build", "bin", "reckoner-harness-linux-amd64"),
        os.path.join(harness_repo_root, "build", "harness", "reckoner-harness-linux-amd64"),
    ]
    if binary_path:
        candidates.insert(0, os.path.join(harness_repo_root, binary_path))
    for cand in candidates:
        if os.path.isfile(cand):
            return os.path.abspath(cand)

    which_bin = shutil.which("reckoner-harness") or shutil.which("reckoner-harness-linux-amd64")
    if which_bin:
        return which_bin
    return binary_path or "reckoner-harness"


class GoToolHarness:
    """Microsecond-scale Go Executive Harness stdio client with worker recycling."""

    def __init__(
        self,
        binary_path: Optional[str] = None,
        max_calls: int = 200,
        hard_max_calls: Optional[int] = None,
        no_anchors: bool = False,
    ):
        self.binary_path = resolve_harness_binary(binary_path)
        self.no_anchors = bool(no_anchors)
        self.max_calls = max_calls
        self.hard_max_calls = hard_max_calls if hard_max_calls is not None else max(max_calls * 50, max_calls)
        self.call_count = 0
        self._lock = threading.RLock()
        self._ensure_proc()

    def _ensure_proc(self) -> None:
        with self._lock:
            if not hasattr(self, "proc") or self.proc is None or self.proc.poll() is not None:
                cmd = [self.binary_path, "--no-anchors"] if self.no_anchors else [self.binary_path]
                self.proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                )
                self.call_count = 0

    def recycle(self) -> None:
        with self._lock:
            self.close()
            self._ensure_proc()

    def dispatch_structured(self, tool_name: str, payload: str) -> Tuple[bool, str, float]:
        t0 = time.perf_counter()
        req = json.dumps({"tool_name": tool_name, "payload": payload})
        with self._lock:
            try:
                self._ensure_proc()
                assert self.proc.stdin is not None and self.proc.stdout is not None
                self.proc.stdin.write(req + "\n")
                self.proc.stdin.flush()
                line = self.proc.stdout.readline().strip()
                elapsed_ms = (time.perf_counter() - t0) * 1000
                resp = json.loads(line)
                success = bool(resp.get("success"))
                val = resp.get("result", "") if success else resp.get("error", "")
                self.call_count += 1
                if self.call_count >= self.hard_max_calls:
                    self.recycle()
                return success, str(val), elapsed_ms
            except Exception as e:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                return False, str(e), elapsed_ms

    def dispatch_full(self, tool_name: str, payload: str) -> Dict[str, Any]:
        req = json.dumps({"tool_name": tool_name, "payload": payload})
        with self._lock:
            try:
                self._ensure_proc()
                assert self.proc.stdin is not None and self.proc.stdout is not None
                self.proc.stdin.write(req + "\n")
                self.proc.stdin.flush()
                line = self.proc.stdout.readline().strip()
                resp = json.loads(line)
                self.call_count += 1
                if self.call_count >= self.hard_max_calls:
                    self.recycle()
                return resp
            except Exception as e:
                return {"success": False, "error": str(e), "result": ""}

    def dispatch_sync(self, tool_name: str, payload: str) -> Tuple[str, float]:
        success, val, elapsed_ms = self.dispatch_structured(tool_name, payload)
        if success:
            return val, elapsed_ms
        return f"Error: {val}", elapsed_ms

    def reset(self) -> None:
        with self._lock:
            if self.call_count >= self.max_calls:
                self.recycle()
            else:
                self.dispatch_sync("reset", "")

    def close(self) -> None:
        with self._lock:
            if hasattr(self, "proc") and self.proc is not None:
                try:
                    if self.proc.stdin:
                        self.proc.stdin.close()
                    self.proc.terminate()
                    self.proc.wait(timeout=1.0)
                except Exception:
                    try:
                        self.proc.kill()
                    except Exception:
                        pass
                self.proc = None
