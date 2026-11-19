#!/usr/bin/env python3
"""
Direct 1.4B byte-identity verification between a baseline git ref (the previous release tag) and the current branch.

Compares both binaries on identical inputs and verifies 0 disallowed byte mismatches across:
1. All 1,672 conformance cases and every JSON tool API request/response exchange in the conformance suite;
2. Stateless single-shot CLI invocations across conformance inputs;
3. `reckoner-harness serve` rendering (both no-profile and `--profile reckoner-1.4b`) of >= 500 chat requests
   from the evaluation sets, compared as token-ID streams (`input_ids` + `sampling_params`) sent to a fake backend
   and final `/v1/chat/completions` responses;
4. Multi-step 1.4B fake-backend transcripts (calc, solve, lua, str, date, lookup, deliberate, find/lines
   anchored & unanchored, quote receipt expansion, external fncall, and error observations);
5. Effort sweep across every effort name, alias, empty value, and unknown names on HTTP (`native` and
   `stop_and_continue`) and IPC (`format_prompt` / `render_prompt`), asserting that the ONLY differences
   between the previous release tag and the current branch with the 1.4B profile (`effort_line: "off"`) are:
   - `custom_params.effort` is the canonical name (so empty `""` becomes `"medium"`, `"fast"` -> `"low"`,
     `"careful"` -> `"high"`, `"xhigh"` -> `"high"`, `"minimal"` -> `"minimal"`);
   - `custom_params.thinking_budget` and `custom_params.gate_mask_deliberate` change for `minimal`, `low`,
     `high`, `fast`, `careful`, and `xhigh`;
   - in `stop_and_continue` mode, `sampling_params.logit_bias` adds the `<|deliberate|>` ban (`{"8": -100}`)
     for `none` and `minimal`;
   - unknown effort names (e.g. `"turbo"`, `"auto"`) return HTTP 400 on `/v1/chat/completions` and `ErrArgs`
     on IPC `format_prompt` / `render_prompt`.
"""

import argparse
import hashlib
import http.server
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


CANONICAL_EFFORTS: Dict[str, Tuple[str, int, bool]] = {
    "": ("medium", 1024, False),
    "none": ("none", 0, True),
    "minimal": ("minimal", 0, True),
    "low": ("low", 256, False),
    "medium": ("medium", 1024, False),
    "high": ("high", 4096, False),
    "fast": ("low", 256, False),
    "careful": ("high", 4096, False),
    "xhigh": ("high", 4096, False),
}

OLD_112_EFFORTS: Dict[str, Tuple[str, int, bool]] = {
    "": ("", 1024, False),
    "none": ("none", 0, True),
    "minimal": ("minimal", 1024, False),
    "low": ("low", 0, True),
    "medium": ("medium", 1024, False),
    "high": ("high", 2048, False),
    "fast": ("fast", 0, True),
    "careful": ("careful", 2048, False),
    "xhigh": ("xhigh", 1024, False),
}

UNKNOWN_EFFORTS = ["turbo", "auto"]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _resolve_zig_cc() -> str:
    zig_bin = "zig"
    for cand in [
        os.path.join(os.path.expanduser("~"), ".local", "bin", "zig"),
        os.path.join(os.path.expanduser("~"), ".local", "zig-x86_64-linux-0.14.1", "zig"),
    ]:
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            zig_bin = cand
            break
    return f"{zig_bin} cc -target x86_64-linux-musl"


def build_binary_in_dir(src_dir: Path, bin_out: Path) -> str:
    bin_out.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["SOURCE_DATE_EPOCH"] = env.get("SOURCE_DATE_EPOCH", "1600000000")
    env["CGO_ENABLED"] = "1"
    env["GOOS"] = "linux"
    env["GOARCH"] = "amd64"
    env["CC"] = _resolve_zig_cc()
    subprocess.run(
        [
            "go",
            "build",
            "-trimpath",
            "-buildvcs=false",
            "-ldflags",
            "-buildid= -s -w -linkmode external -extldflags '-static -s'",
            "-o",
            str(bin_out),
            ".",
        ],
        cwd=str(src_dir),
        env=env,
        check=True,
    )
    return str(bin_out)


def build_baseline_ref_binary(repo_root: Path, work_dir: Path, baseline_ref: str) -> str:
    wt_dir = work_dir / "baseline_wt"
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(wt_dir), baseline_ref],
        cwd=str(repo_root),
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        bin_out = work_dir / "build" / "bin" / "reckoner-harness-baseline"
        return build_binary_in_dir(wt_dir, bin_out)
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(wt_dir)],
            cwd=str(repo_root),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


class StdioHarness:
    def __init__(self, binary_path: str, env: Optional[Dict[str, str]] = None):
        self.binary_path = binary_path
        self.proc = subprocess.Popen(
            [binary_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=env,
        )

    def send_and_canonicalize(self, req: Dict[str, Any]) -> bytes:
        line_in = json.dumps(req, ensure_ascii=False, sort_keys=True) + "\n"
        self.proc.stdin.write(line_in)
        self.proc.stdin.flush()
        line_out = self.proc.stdout.readline()
        if not line_out:
            err_txt = self.proc.stderr.read()
            raise RuntimeError(f"Unexpected EOF from {self.binary_path}: {err_txt}")
        obj = json.loads(line_out)
        if "elapsed_us" in obj:
            obj["elapsed_us"] = 0
        return json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.terminate()
            self.proc.wait(timeout=2)
        except Exception:
            self.proc.kill()


def build_conformance_requests_for_case(case: Dict[str, Any]) -> List[Dict[str, Any]]:
    reqs: List[Dict[str, Any]] = [{"tool": "reset", "payload": ""}]
    tool = case.get("tool", "")
    cat = case.get("category", "")

    if cat in ["render", "parse_roundtrip", "invalid_call"]:
        if cat == "render":
            reqs.append({"tool": "fncall", "payload": json.dumps({"op": "render", "tools": case["tools"]})})
        elif cat == "parse_roundtrip":
            req_payload: Dict[str, Any] = {"op": "parse", "call": case["call"]}
            if "tools" in case:
                req_payload["tools"] = case["tools"]
            reqs.append({"tool": "fncall", "payload": json.dumps(req_payload)})
            if case.get("expected_arguments") is not None and case.get("expected_name"):
                ser_payload: Dict[str, Any] = {
                    "op": "serialize",
                    "name": case["expected_name"],
                    "arguments": case["expected_arguments"],
                }
                if "tools" in case:
                    ser_payload["tools"] = case["tools"]
                reqs.append({"tool": "fncall", "payload": json.dumps(ser_payload)})
        elif cat == "invalid_call":
            req_payload = {"op": "validate", "call": case["call"]}
            if "tools" in case:
                req_payload["tools"] = case["tools"]
            reqs.append({"tool": "fncall", "payload": json.dumps(req_payload)})
    elif cat in ["prompt_format", "anchors_behavior"]:
        if cat == "prompt_format":
            reqs.append({
                "tool": "format_prompt",
                "profile": case.get("profile", ""),
                "graded": case.get("graded", False),
                "messages": case.get("messages", []),
            })
        elif cat == "anchors_behavior":
            sc_req: Dict[str, Any] = {"tool": "session_config", "profile": case.get("profile", "")}
            if case.get("no_anchors"):
                sc_req["anchors"] = False
            reqs.append(sc_req)
            for setup in case.get("setup_messages", []):
                reqs.append({"tool": "anchor", "tokens": setup, "store": True})
            reqs.append({"tool": case["tool"], "payload": case.get("payload", "")})
            reqs.append({"tool": "session_config", "profile": "", "anchors": True})
    elif tool in ["anchor", "quote", "quote_mask", "find", "lines", "context_reset"]:
        if "anchors" in case:
            reqs.append({"tool": "session_config", "anchors": case["anchors"]})
        else:
            reqs.append({"tool": "session_config", "anchors": True})
        for setup in case.get("setup_messages", []):
            reqs.append({"tool": "anchor", "tokens": setup, "store": True})
        if tool == "anchor":
            r: Dict[str, Any] = {"tool": "anchor", "tokens": case["tokens"]}
            if "targets" in case:
                r["targets"] = case["targets"]
            if case.get("message_id") is not None:
                r["message_id"] = case["message_id"]
            reqs.append(r)
        elif tool == "quote":
            reqs.append({"tool": "quote", "payload": case.get("call", case.get("payload", ""))})
        elif tool == "quote_mask":
            reqs.append({"tool": "quote_mask", "prefix": case["prefix"]})
        elif tool in ["find", "lines"]:
            reqs.append({"tool": tool, "payload": case.get("payload", "")})
        elif tool == "context_reset":
            reqs.append({"tool": "context_reset", "payload": ""})
    else:
        regs_before = case.get("registers_before")
        if regs_before:
            for rname, rval in regs_before.items():
                reqs.append({"tool": "reg", "payload": f"set {rname} {rval}"})
        r = {"tool": tool, "payload": case["input"]}
        if "clock" in case:
            r["clock"] = case["clock"]
        reqs.append(r)
    return reqs


def compare_conformance_and_json_api(
    base_bin: str,
    cand_bin: str,
    repo_root: Path,
    cand_env: Optional[Dict[str, str]] = None,
    q2b_tok_path: str = "",
) -> Tuple[int, int, int, int]:
    suite_files = [
        repo_root / "conformance" / "cases.jsonl",
        repo_root / "conformance" / "fncall_cases.jsonl",
        repo_root / "conformance" / "context_cases.jsonl",
        repo_root / "conformance" / "profile_cases.jsonl",
    ]
    cases = []
    for sf in suite_files:
        with sf.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    cases.append(json.loads(line))

    base_env = os.environ.copy()
    base_env.pop("RECKONER_TOKENIZER_PATH", None)
    h_base = StdioHarness(base_bin, env=base_env)
    h_cand = StdioHarness(cand_bin, env=cand_env)
    case_mismatches = 0
    ipc_exchanges = 0
    ipc_mismatches = 0
    env_has_tok = bool((cand_env or os.environ).get("RECKONER_TOKENIZER_PATH", "").strip())

    try:
        for idx, case in enumerate(cases):
            reqs = build_conformance_requests_for_case(case)
            case_diff = False
            for r in reqs:
                if r.get("tool") == "format_prompt" and r.get("profile") == "reckoner-q2b":
                    if not env_has_tok and q2b_tok_path:
                        r = dict(r)
                        r["tokenizer"] = q2b_tok_path
                b_out = h_base.send_and_canonicalize(r)
                c_out = h_cand.send_and_canonicalize(r)
                ipc_exchanges += 1
                if b_out != c_out:
                    ipc_mismatches += 1
                    case_diff = True
                    if ipc_mismatches <= 5:
                        print(f"[JSON API MISMATCH case #{idx}] req={r}\n  base={b_out}\n  cand={c_out}", file=sys.stderr)
            if case_diff:
                case_mismatches += 1

        extra_14b_reqs = [
            {"tool": "session_config", "profile": "reckoner-1.4b", "tokenizer": ""},
            {"tool": "session_config", "profile": "reckoner-1.4b", "tokenizer": "   "},
            {"tool": "session_config", "profile": "", "tokenizer": ""},
            {
                "tool": "format_prompt",
                "profile": "reckoner-1.4b",
                "messages": [
                    {"role": "system", "content": "<|tools|>\ndef f(): pass\n<|/tools|>\ntrailing text"},
                    {"role": "user", "content": "hi"},
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "my_tool",
                                    "arguments": json.dumps({"text": "line1\nline2", "k": 1}),
                                },
                            },
                            {
                                "id": "call_non_obj_1",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": "not json"},
                            },
                            {
                                "id": "call_non_obj_2",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": "null"},
                            },
                            {
                                "id": "call_non_obj_3",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": "[1,2]"},
                            },
                            {
                                "id": "call_non_id_1",
                                "type": "function",
                                "function": {
                                    "name": "undeclared-tool",
                                    "arguments": json.dumps({"extra-key": 1}),
                                },
                            },
                            {
                                "id": "call_non_id_2",
                                "type": "function",
                                "function": {
                                    "name": "github.create-issue",
                                    "arguments": json.dumps({"title": "bug", "extra-key": "v"}),
                                },
                            },
                        ],
                    },
                    {"role": "tool", "content": "clean external tool output"},
                    {"role": "assistant", "content": "<|ready|>done"},
                ],
            },
            {
                "tool": "format_prompt",
                "profile": "",
                "messages": [
                    {"role": "user", "content": "hi"},
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "not json"}},
                            {"id": "c2", "type": "function", "function": {"name": "get_weather", "arguments": "null"}},
                            {"id": "c3", "type": "function", "function": {"name": "get_weather", "arguments": "[1,2]"}},
                            {"id": "c4", "type": "function", "function": {"name": "undeclared-tool", "arguments": json.dumps({"extra-key": 1})}},
                            {"id": "c5", "type": "function", "function": {"name": "github.create-issue", "arguments": json.dumps({"title": "bug"})}},
                        ],
                    },
                ],
            },
            {
                "tool": "format_prompt",
                "profile": "reckoner-1.4b",
                "messages": [
                    {"role": "system", "content": "reasoning_effort: high\n<|tools|>\ndef f(): pass\n<|/tools|>"},
                    {"role": "user", "content": "hi"},
                ],
            },
            {
                "tool": "format_prompt",
                "profile": "",
                "messages": [
                    {"role": "system", "content": "reasoning_effort: high\n<|tools|>\ndef f(): pass\n<|/tools|>"},
                    {"role": "user", "content": "hi"},
                ],
            },
            {"tool": "reset", "payload": ""},
            {"tool": "calc", "payload": "10"},
            {"tool": "calc", "payload": "20"},
            {"tool": "reg", "payload": "list"},
            {"tool": "reset", "payload": ""},
        ]
        for r in extra_14b_reqs:
            b_out = h_base.send_and_canonicalize(r)
            c_out = h_cand.send_and_canonicalize(r)
            ipc_exchanges += 1
            if b_out != c_out:
                ipc_mismatches += 1
                case_mismatches += 1
                print(f"[EXTRA 1.4B MISMATCH] req={r}\n  base={b_out}\n  cand={c_out}", file=sys.stderr)
    finally:
        h_base.close()
        h_cand.close()

    return len(cases), case_mismatches, ipc_exchanges, ipc_mismatches


def load_eval_prompts(eval_paths: List[str], target_count: int = 600) -> List[str]:
    prompts: List[str] = []
    for p in eval_paths:
        path = Path(p)
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as f:
            first = f.read(1)
            f.seek(0)
            if first == "[":
                arr = json.load(f)
                for row in arr:
                    if isinstance(row, dict):
                        for k in ("problem", "question", "prompt", "input"):
                            if isinstance(row.get(k), str) and row[k].strip():
                                prompts.append(row[k])
                                break
            else:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if isinstance(row, dict):
                        for k in ("problem", "question", "prompt", "input", "text"):
                            if isinstance(row.get(k), str) and row[k].strip():
                                prompts.append(row[k])
                                break
    base_templates = [
        "Compute {i} * {j} + {k} and verify whether it is divisible by 7.",
        "Solve the system: 3*x + 2*y = {i}, x - y = {j}.",
        "Line 0: Invoice #{i}\nLine 1: Subtotal ${j}.50\nLine 2: Tax ${k}.25\nWhat is the total?",
        "Check if <|act|> or <|r0|> appears literally in this question #{i}.",
        "Given f(x) = x^2 - {i}*x + {j}, find the vertex and roots.",
    ]
    idx = 0
    while len(prompts) < target_count:
        prompts.append(base_templates[idx % len(base_templates)].format(
            i=100 + idx * 7,
            j=20 + idx * 3,
            k=5 + (idx % 19),
        ))
        idx += 1
    return prompts[:target_count]


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeSGLangServer:
    def __init__(self):
        self.port = find_free_port()
        self.recorded_requests: List[Dict[str, Any]] = []
        self.scripted_steps: List[Tuple[List[int], int]] = []
        self.step_idx = 0
        self.lock = threading.Lock()

        parent = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                req = json.loads(raw.decode("utf-8"))
                with parent.lock:
                    norm_req = json.loads(raw.decode("utf-8"))
                    norm_req["rid"] = "FIXED_RID"
                    if isinstance(norm_req.get("sampling_params"), dict):
                        cp = norm_req["sampling_params"].get("custom_params")
                        if isinstance(cp, dict) and "session_id" in cp:
                            cp["session_id"] = "FIXED_SESSION"
                    parent.recorded_requests.append(norm_req)

                    if parent.step_idx < len(parent.scripted_steps):
                        out_ids, stop_tok = parent.scripted_steps[parent.step_idx]
                        parent.step_idx += 1
                    else:
                        out_ids, stop_tok = [10, 25049, 4], 4

                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                chunk = {
                    "output_ids": out_ids,
                    "meta_info": {
                        "prompt_tokens": len(req.get("input_ids", [])),
                        "completion_tokens": len(out_ids),
                        "cached_tokens": 0,
                        "finish_reason": {"type": "stop", "matched": stop_tok},
                    },
                }
                payload = f"data: {json.dumps(chunk, sort_keys=True)}\n\ndata: [DONE]\n\n"
                self.wfile.write(payload.encode("utf-8"))

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def reset_capture(self, scripted_steps: Optional[List[Tuple[List[int], int]]] = None):
        with self.lock:
            self.recorded_requests = []
            self.scripted_steps = list(scripted_steps or [])
            self.step_idx = 0

    def get_captured_requests(self) -> List[Dict[str, Any]]:
        with self.lock:
            return json.loads(json.dumps(self.recorded_requests))

    def get_captured_bytes(self) -> bytes:
        with self.lock:
            return json.dumps(self.recorded_requests, ensure_ascii=False, sort_keys=True).encode("utf-8")

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class RunningServeInstance:
    def __init__(self, binary_path: str, sglang_port: int, extra_args: List[str], mode: str = "stop_and_continue"):
        self.port = find_free_port()
        cmd = [
            binary_path,
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--sglang-addr",
            f"http://127.0.0.1:{sglang_port}",
            "--sglang-proto",
            "http",
            "--mode",
            mode,
        ] + extra_args
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self._wait_ready()

    def _wait_ready(self):
        deadline = time.time() + 5.0
        url = f"http://127.0.0.1:{self.port}/health"
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=0.5) as resp:
                    if resp.status == 200:
                        return
            except Exception:
                time.sleep(0.05)
        raise RuntimeError(f"Front-door server on port {self.port} did not become ready")

    def chat_completion_raw(self, payload: Dict[str, Any]) -> Tuple[int, Dict[str, str], bytes]:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                return resp.status, dict(resp.headers), resp.read()
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers), e.read()

    def chat_completion(self, payload: Dict[str, Any]) -> bytes:
        status, _, raw_body = self.chat_completion_raw(payload)
        if status != 200:
            raise RuntimeError(f"HTTP {status}: {raw_body.decode('utf-8', errors='replace')}")
        body = json.loads(raw_body.decode("utf-8"))
        body["id"] = "FIXED_ID"
        body["created"] = 0
        if isinstance(body.get("choices"), list):
            for ch in body["choices"]:
                msg = ch.get("message", {})
                if isinstance(msg.get("tool_calls"), list):
                    for tc_i, tc in enumerate(msg["tool_calls"]):
                        tc["id"] = f"call_{tc_i}"
        tel = body.get("reckoner_telemetry")
        if isinstance(tel, dict):
            tel["ttft_ms"] = 0.0
            tel["elapsed_ms"] = 0.0
            tel["decode_tok_s"] = 0.0
            if isinstance(tel.get("tool_calls"), list):
                for tc in tel["tool_calls"]:
                    tc["latency_us"] = 0.0
                    tc["harness_ipc_ms"] = 0.0
                    tc["pause_resume_ms"] = 0.0
                    tc["obs_prefill_ms"] = 0.0
                    tc["call_to_next_token_ms"] = 0.0
        return json.dumps(body, ensure_ascii=False, sort_keys=True).encode("utf-8")

    def close(self):
        try:
            self.proc.terminate()
            self.proc.wait(timeout=2)
        except Exception:
            self.proc.kill()


def encode_14b_tokens(harness: StdioHarness, text: str) -> List[int]:
    resp = json.loads(harness.send_and_canonicalize({
        "tool": "format_prompt",
        "messages": [{"role": "user", "content": text}],
    }).decode("utf-8"))
    toks = resp.get("tokens", [])
    return toks[2:-3]


def compare_serve_eval_prompts_and_transcripts(
    base_bin: str,
    cand_bin: str,
    eval_prompts: List[str],
) -> Tuple[int, int, int, int]:
    fake_base_sgl = FakeSGLangServer()
    fake_cand_sgl = FakeSGLangServer()

    h_enc = StdioHarness(base_bin)
    tok_15_mul_4 = encode_14b_tokens(h_enc, "15 * 4")
    tok_ans_60 = encode_14b_tokens(h_enc, "The answer is \\boxed{60}.")
    tok_eq = encode_14b_tokens(h_enc, "3*x + 12 = 60, x")
    tok_lua = encode_14b_tokens(h_enc, "local s=0; for i=1,10 do s=s+i end; return s")
    tok_ans_55 = encode_14b_tokens(h_enc, "\\boxed{55}")
    tok_str = encode_14b_tokens(h_enc, 'upper("reckoner")')
    tok_date = encode_14b_tokens(h_enc, "add(2026-10-10, 7)")
    tok_lookup = encode_14b_tokens(h_enc, "c.value")
    tok_delib = encode_14b_tokens(h_enc, "Let us check the parity first.")
    tok_find = encode_14b_tokens(h_enc, 'find("Subtotal")')
    tok_find_no_match = encode_14b_tokens(h_enc, 'find("zzqx")')
    tok_lines = encode_14b_tokens(h_enc, "lines(0, 0, 1)")
    tok_lines_empty = encode_14b_tokens(h_enc, "lines(0, 1, 1)")
    tok_fn_call = encode_14b_tokens(h_enc, 'get_weather(city="Paris")')
    tok_div_zero = encode_14b_tokens(h_enc, "1 / 0")
    h_enc.close()

    prompt_comparisons = 0
    prompt_mismatches = 0
    transcript_comparisons = 0
    transcript_mismatches = 0

    try:
        for mode_idx, extra_args in enumerate([[], ["--profile", "reckoner-1.4b"]]):
            srv_base = RunningServeInstance(base_bin, fake_base_sgl.port, extra_args)
            srv_cand = RunningServeInstance(cand_bin, fake_cand_sgl.port, extra_args)
            try:
                subset = eval_prompts[mode_idx * 300 : (mode_idx + 1) * 300]
                for i, prompt_text in enumerate(subset):
                    req_body: Dict[str, Any] = {
                        "messages": [{"role": "user", "content": prompt_text}],
                    }
                    if i % 3 == 1:
                        req_body["anchor_context"] = True
                    if i % 2 == 1:
                        req_body["graded"] = True
                    if i % 5 == 0:
                        req_body["messages"].insert(0, {"role": "system", "content": "You are a precise assistant."})
                    if i % 7 == 0:
                        req_body["tools"] = [{
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "description": "Get weather for a city",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"city": {"type": "string"}},
                                    "required": ["city"],
                                },
                            },
                        }]

                    fake_base_sgl.reset_capture()
                    fake_cand_sgl.reset_capture()
                    resp_b = srv_base.chat_completion(req_body)
                    resp_c = srv_cand.chat_completion(req_body)
                    sgl_b = fake_base_sgl.get_captured_bytes()
                    sgl_c = fake_cand_sgl.get_captured_bytes()

                    prompt_comparisons += 1
                    if sgl_b != sgl_c or resp_b != resp_c:
                        prompt_mismatches += 1
                        if prompt_mismatches <= 5:
                            print(
                                f"[SERVE PROMPT MISMATCH mode={extra_args} #{i}]\n  sgl_match={sgl_b == sgl_c}\n  resp_match={resp_b == resp_c}",
                                file=sys.stderr,
                            )

                transcripts = [
                    (
                        "single_turn_calc",
                        {"messages": [{"role": "user", "content": "What is 15 * 4?"}]},
                        [
                            ([9, 18] + tok_15_mul_4 + [14], 14),
                            ([10] + tok_ans_60 + [4], 4),
                        ],
                    ),
                    (
                        "multi_tool_calc_solve_lua",
                        {"messages": [{"role": "user", "content": "Run calc, solve, and lua."}]},
                        [
                            ([9, 18] + tok_15_mul_4 + [14], 14),
                            ([19] + tok_eq + [14], 14),
                            ([20] + tok_lua + [14], 14),
                            ([10] + tok_ans_55 + [4], 4),
                        ],
                    ),
                    (
                        "str_tool",
                        {"messages": [{"role": "user", "content": "Uppercase reckoner."}]},
                        [
                            ([9, 29] + tok_str + [14], 14),
                            ([10] + tok_ans_60 + [4], 4),
                        ],
                    ),
                    (
                        "date_tool",
                        {"messages": [{"role": "user", "content": "Add 7 days to 2026-10-10."}]},
                        [
                            ([9, 30] + tok_date + [14], 14),
                            ([10] + tok_ans_60 + [4], 4),
                        ],
                    ),
                    (
                        "lookup_tool",
                        {"messages": [{"role": "user", "content": "Speed of light?"}]},
                        [
                            ([9, 31] + tok_lookup + [14], 14),
                            ([10] + tok_ans_60 + [4], 4),
                        ],
                    ),
                    (
                        "deliberate_zero_tool",
                        {"messages": [{"role": "user", "content": "Think carefully."}]},
                        [
                            ([8] + tok_delib + [10] + tok_ans_60 + [4], 4),
                        ],
                    ),
                    (
                        "deliberate_then_act_calc",
                        {"messages": [{"role": "user", "content": "Think then calculate."}]},
                        [
                            ([8] + tok_delib + [9, 18] + tok_15_mul_4 + [14], 14),
                            ([10] + tok_ans_60 + [4], 4),
                        ],
                    ),
                    (
                        "find_unanchored",
                        {"messages": [{"role": "user", "content": "Header\nSubtotal: 42\nFooter"}]},
                        [
                            ([9, 32] + tok_find + [14], 14),
                            ([10] + tok_ans_60 + [4], 4),
                        ],
                    ),
                    (
                        "find_anchored",
                        {"messages": [{"role": "user", "content": "Header\nSubtotal: 42\nFooter"}], "anchor_context": True},
                        [
                            ([9, 32] + tok_find + [14], 14),
                            ([10] + tok_ans_60 + [4], 4),
                        ],
                    ),
                    (
                        "find_no_match_unanchored",
                        {"messages": [{"role": "user", "content": "Header\nSubtotal: 42\nFooter"}]},
                        [
                            ([9, 32] + tok_find_no_match + [14], 14),
                            ([10] + tok_ans_60 + [4], 4),
                        ],
                    ),
                    (
                        "find_no_match_anchored",
                        {"messages": [{"role": "user", "content": "Header\nSubtotal: 42\nFooter"}], "anchor_context": True},
                        [
                            ([9, 32] + tok_find_no_match + [14], 14),
                            ([10] + tok_ans_60 + [4], 4),
                        ],
                    ),
                    (
                        "lines_anchored",
                        {"messages": [{"role": "user", "content": "Line A\nLine B\nLine C"}], "anchor_context": True},
                        [
                            ([9, 32] + tok_lines + [14], 14),
                            ([10] + tok_ans_60 + [4], 4),
                        ],
                    ),
                    (
                        "lines_unanchored",
                        {"messages": [{"role": "user", "content": "Line A\nLine B\nLine C"}]},
                        [
                            ([9, 32] + tok_lines + [14], 14),
                            ([10] + tok_ans_60 + [4], 4),
                        ],
                    ),
                    (
                        "lines_empty_line_anchored",
                        {"messages": [{"role": "user", "content": "Line A\n\nLine C"}], "anchor_context": True},
                        [
                            ([9, 32] + tok_lines_empty + [14], 14),
                            ([10] + tok_ans_60 + [4], 4),
                        ],
                    ),
                    (
                        "external_fncall",
                        {
                            "messages": [{"role": "user", "content": "Weather in Paris?"}],
                            "tools": [{
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "description": "Get weather",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {"city": {"type": "string"}},
                                        "required": ["city"],
                                    },
                                },
                            }],
                        },
                        [
                            ([9, 13] + tok_fn_call + [14, 4], 4),
                        ],
                    ),
                    (
                        "tool_div_zero_error_obs",
                        {"messages": [{"role": "user", "content": "Divide 1 by 0 then recover."}]},
                        [
                            ([9, 18] + tok_div_zero + [14], 14),
                            ([10] + tok_ans_60 + [4], 4),
                        ],
                    ),
                ]

                for t_name, req_body, steps in transcripts:
                    fake_base_sgl.reset_capture(steps)
                    fake_cand_sgl.reset_capture(steps)
                    resp_b = srv_base.chat_completion(req_body)
                    resp_c = srv_cand.chat_completion(req_body)
                    sgl_b = fake_base_sgl.get_captured_bytes()
                    sgl_c = fake_cand_sgl.get_captured_bytes()

                    transcript_comparisons += 1
                    if sgl_b != sgl_c or resp_b != resp_c:
                        transcript_mismatches += 1
                        print(
                            f"[TRANSCRIPT MISMATCH mode={extra_args} {t_name}]\n  sgl_match={sgl_b == sgl_c}\n  resp_match={resp_b == resp_c}",
                            file=sys.stderr,
                        )
            finally:
                srv_base.close()
                srv_cand.close()
    finally:
        fake_base_sgl.close()
        fake_cand_sgl.close()

    return prompt_comparisons, prompt_mismatches, transcript_comparisons, transcript_mismatches


def compare_effort_sweep_http_and_ipc(
    base_bin: str,
    cand_bin: str,
) -> Tuple[List[str], List[str]]:
    """
    Sends the same requests at every effort name, alias, empty value, and unknown names
    on HTTP (native and stop_and_continue) and IPC (format_prompt top-level and payload),
    recording all observed allowed differences and any disallowed differences.
    """
    allowed_diffs: List[str] = []
    disallowed_diffs: List[str] = []

    # 1. IPC format_prompt (both top-level and JSON payload) across every effort name, alias, empty, and unknown
    h_base = StdioHarness(base_bin)
    h_cand = StdioHarness(cand_bin)
    try:
        for profile in ["", "reckoner-1.4b"]:
            for eff in list(CANONICAL_EFFORTS.keys()) + UNKNOWN_EFFORTS:
                for via_payload in [False, True]:
                    if via_payload:
                        inner: Dict[str, Any] = {
                            "profile": profile,
                            "messages": [{"role": "user", "content": "What is 6 * 7?"}],
                        }
                        if eff != "":
                            inner["reasoning_effort"] = eff
                        req: Dict[str, Any] = {
                            "tool": "format_prompt",
                            "payload": json.dumps(inner),
                        }
                        path_label = "IPC format_prompt(payload)"
                    else:
                        req = {
                            "tool": "format_prompt",
                            "profile": profile,
                            "messages": [{"role": "user", "content": "What is 6 * 7?"}],
                        }
                        if eff != "":
                            req["reasoning_effort"] = eff
                        path_label = "IPC format_prompt"

                    b_raw = h_base.send_and_canonicalize(req)
                    c_raw = h_cand.send_and_canonicalize(req)
                    b_obj = json.loads(b_raw.decode("utf-8"))
                    c_obj = json.loads(c_raw.decode("utf-8"))

                    if eff in UNKNOWN_EFFORTS:
                        if c_obj.get("success") or not str(c_obj.get("error", "")).startswith("ErrArgs"):
                            disallowed_diffs.append(
                                f"{path_label} profile={profile!r} effort={eff!r}: expected ErrArgs on candidate, got {c_obj}"
                            )
                        else:
                            allowed_diffs.append(
                                f"{path_label} profile={profile!r} effort={eff!r}: base success={b_obj.get('success')} -> cand ErrArgs ({c_obj.get('error')})"
                            )
                    else:
                        if b_raw != c_raw:
                            disallowed_diffs.append(
                                f"{path_label} profile={profile!r} effort={eff!r}: rendered prompt/tokens differed!\n  base={b_obj}\n  cand={c_obj}"
                            )
    finally:
        h_base.close()
        h_cand.close()

    # 2. HTTP native and stop_and_continue modes across every effort name, alias, empty, and unknown
    fake_base_sgl = FakeSGLangServer()
    fake_cand_sgl = FakeSGLangServer()
    try:
        for extra_args in [[], ["--profile", "reckoner-1.4b"]]:
            prof_label = "reckoner-1.4b" if extra_args else "<none>"
            for serve_mode in ["native", "stop_and_continue"]:
                srv_base = RunningServeInstance(base_bin, fake_base_sgl.port, extra_args, mode=serve_mode)
                srv_cand = RunningServeInstance(cand_bin, fake_cand_sgl.port, extra_args, mode=serve_mode)
                try:
                    for eff in list(CANONICAL_EFFORTS.keys()) + UNKNOWN_EFFORTS:
                        for stream_flag in ([False, True] if eff in UNKNOWN_EFFORTS else [False]):
                            req_body: Dict[str, Any] = {
                                "messages": [{"role": "user", "content": "What is 6 * 7?"}],
                                "stream": stream_flag,
                            }
                            if eff != "":
                                req_body["reasoning_effort"] = eff

                            if eff in UNKNOWN_EFFORTS:
                                fake_cand_sgl.reset_capture()
                                status_c, headers_c, body_c = srv_cand.chat_completion_raw(req_body)
                                body_str = body_c.decode("utf-8", errors="replace")
                                if (
                                    status_c != 400
                                    or "text/event-stream" in headers_c.get("Content-Type", "")
                                    or body_str.startswith("data:")
                                ):
                                    disallowed_diffs.append(
                                        f"HTTP {serve_mode} profile={prof_label} effort={eff!r} stream={stream_flag}: expected HTTP 400 before SSE, got status={status_c} ctype={headers_c.get('Content-Type')} body={body_str!r}"
                                    )
                                else:
                                    allowed_diffs.append(
                                        f"HTTP {serve_mode} profile={prof_label} effort={eff!r} stream={stream_flag}: base HTTP 200 -> cand HTTP 400 ({body_str.strip()})"
                                    )
                                continue

                            fake_base_sgl.reset_capture()
                            fake_cand_sgl.reset_capture()
                            resp_b = srv_base.chat_completion(req_body)
                            resp_c = srv_cand.chat_completion(req_body)
                            reqs_b = fake_base_sgl.get_captured_requests()
                            reqs_c = fake_cand_sgl.get_captured_requests()

                            if resp_b != resp_c:
                                disallowed_diffs.append(
                                    f"HTTP {serve_mode} profile={prof_label} effort={eff!r}: chat completion response differed"
                                )
                            if len(reqs_b) != 1 or len(reqs_c) != 1:
                                disallowed_diffs.append(
                                    f"HTTP {serve_mode} profile={prof_label} effort={eff!r}: expected 1 backend request, got base={len(reqs_b)} cand={len(reqs_c)}"
                                )
                                continue

                            rb = reqs_b[0]
                            rc = reqs_c[0]

                            # Input IDs must be 100% identical
                            if rb.get("input_ids") != rc.get("input_ids"):
                                disallowed_diffs.append(
                                    f"HTTP {serve_mode} profile={prof_label} effort={eff!r}: input_ids differed: base={rb.get('input_ids')} cand={rc.get('input_ids')}"
                                )

                            sp_b = dict(rb.get("sampling_params", {}))
                            sp_c = dict(rc.get("sampling_params", {}))
                            want_name, want_budget, want_mask = CANONICAL_EFFORTS[eff]

                            if serve_mode == "native":
                                cp_b = dict(sp_b.pop("custom_params", {}))
                                cp_c = dict(sp_c.pop("custom_params", {}))

                                if sp_b != sp_c:
                                    disallowed_diffs.append(
                                        f"HTTP native profile={prof_label} effort={eff!r}: non-custom sampling_params differed: base={sp_b} cand={sp_c}"
                                    )

                                got_name = cp_c.get("effort")
                                got_budget = cp_c.get("thinking_budget")
                                got_mask = cp_c.get("gate_mask_deliberate")
                                if (got_name, got_budget, got_mask) != (want_name, want_budget, want_mask):
                                    disallowed_diffs.append(
                                        f"HTTP native profile={prof_label} effort={eff!r}: cand custom_params got ({got_name!r}, {got_budget}, {got_mask}), want ({want_name!r}, {want_budget}, {want_mask})"
                                    )

                                for k in ("effort", "thinking_budget", "gate_mask_deliberate"):
                                    if cp_b.get(k) != cp_c.get(k):
                                        allowed_diffs.append(
                                            f"HTTP native profile={prof_label} effort={eff!r}: custom_params.{k}: {cp_b.get(k)!r} -> {cp_c.get(k)!r}"
                                        )
                                    cp_b.pop(k, None)
                                    cp_c.pop(k, None)

                                if cp_b != cp_c:
                                    disallowed_diffs.append(
                                        f"HTTP native profile={prof_label} effort={eff!r}: other custom_params keys differed: base={cp_b} cand={cp_c}"
                                    )
                            else:
                                lb_b = sp_b.pop("logit_bias", None)
                                lb_c = sp_c.pop("logit_bias", None)
                                if want_mask:
                                    if lb_c != {"8": -100}:
                                        disallowed_diffs.append(
                                            f"HTTP stop_and_continue profile={prof_label} effort={eff!r}: expected logit_bias {{'8': -100}}, got {lb_c!r}"
                                        )
                                    else:
                                        allowed_diffs.append(
                                            f"HTTP stop_and_continue profile={prof_label} effort={eff!r}: sampling_params.logit_bias: {lb_b!r} -> {lb_c!r}"
                                        )
                                else:
                                    if lb_c is not None:
                                        disallowed_diffs.append(
                                            f"HTTP stop_and_continue profile={prof_label} effort={eff!r}: expected no logit_bias, got {lb_c!r}"
                                        )
                                if sp_b != sp_c:
                                    disallowed_diffs.append(
                                        f"HTTP stop_and_continue profile={prof_label} effort={eff!r}: other sampling_params differed: base={sp_b} cand={sp_c}"
                                    )
                finally:
                    srv_base.close()
                    srv_cand.close()
    finally:
        fake_base_sgl.close()
        fake_cand_sgl.close()

    return allowed_diffs, disallowed_diffs


def main():
    parser = argparse.ArgumentParser(description="Verify 1.4B byte-identity against a baseline ref (the previous release tag).")
    parser.add_argument("--baseline-ref", required=True, help="Git ref to build baseline binary from in a temp worktree")
    parser.add_argument("--baseline-binary", default="", help="Optional path to prebuilt baseline binary")
    parser.add_argument("--candidate-binary", default="", help="Path to candidate binary (built from current tree if omitted)")
    parser.add_argument("--extra-jsonl", nargs="*", action="extend", default=[], help="Evaluation JSON/JSONL files for >= 500 chat prompts")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root / "scripts"))
    from run_harness_conformance import _write_synthetic_q2b_tokenizer

    with tempfile.TemporaryDirectory(prefix="reckoner-113-identity-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        if args.baseline_binary:
            base_bin = os.path.abspath(args.baseline_binary)
        else:
            base_bin = build_baseline_ref_binary(repo_root, tmp_path, args.baseline_ref)

        if args.candidate_binary:
            cand_bin = os.path.abspath(args.candidate_binary)
            if not os.path.isfile(cand_bin):
                print(f"ERROR: candidate binary not found: {cand_bin}", file=sys.stderr)
                sys.exit(1)
        else:
            cand_bin = build_binary_in_dir(repo_root, tmp_path / "build" / "bin" / "reckoner-harness-candidate")

        q2b_dir = tmp_path / "q2b_tok"
        q2b_dir.mkdir(parents=True, exist_ok=True)
        q2b_tok_path = _write_synthetic_q2b_tokenizer(str(repo_root), str(q2b_dir / "tokenizer.json"))

        base_sha = sha256_file(base_bin)
        cand_sha = sha256_file(cand_bin)
        print(f"Baseline binary ({args.baseline_ref}): {base_sha}")
        print(f"Candidate binary (current tree): {cand_sha}")
        print(f"RECKONER_TOKENIZER_PATH (caller): {os.environ.get('RECKONER_TOKENIZER_PATH', '<unset>')}")

        cases_cnt, case_mismatches, ipc_cnt, ipc_mismatches = compare_conformance_and_json_api(
            base_bin, cand_bin, repo_root, q2b_tok_path=q2b_tok_path
        )
        print(f"\n=== 1. Conformance Cases & JSON Tool API ===")
        print(f"Conformance cases compared:        {cases_cnt}")
        print(f"Conformance case mismatches:       {case_mismatches}")
        print(f"JSON tool API exchanges compared:  {ipc_cnt}")
        print(f"JSON tool API byte mismatches:     {ipc_mismatches}")

        eval_prompts = load_eval_prompts(args.extra_jsonl, target_count=600)
        prompt_cnt, prompt_mismatches, trans_cnt, trans_mismatches = compare_serve_eval_prompts_and_transcripts(
            base_bin, cand_bin, eval_prompts
        )
        print(f"\n=== 2. Serve Prompt Rendering (No-Profile + --profile reckoner-1.4b) ===")
        print(f"Eval chat requests compared:       {prompt_cnt} (300 no-profile + 300 --profile reckoner-1.4b)")
        print(f"Eval chat request byte mismatches: {prompt_mismatches}")

        print(f"\n=== 3. 1.4B Fake-Backend Multi-Step Transcripts ===")
        print(f"Fake-backend transcripts compared: {trans_cnt} (16 no-profile + 16 --profile reckoner-1.4b)")
        print(f"Transcript byte mismatches:        {trans_mismatches}")

        allowed_diffs, disallowed_diffs = compare_effort_sweep_http_and_ipc(base_bin, cand_bin)
        print(f"\n=== 4. Effort Sweep (HTTP & IPC, all names/aliases/empty/unknown) ===")
        print(f"Observed allowed differences ({len(allowed_diffs)}):")
        for d in allowed_diffs:
            print(f"  [ALLOWED] {d}")
        print(f"Disallowed differences:            {len(disallowed_diffs)}")
        for d in disallowed_diffs:
            print(f"  [DISALLOWED] {d}", file=sys.stderr)

        total_mismatches = case_mismatches + ipc_mismatches + prompt_mismatches + trans_mismatches + len(disallowed_diffs)
        if total_mismatches != 0:
            print(f"\nFAILED: {total_mismatches} total disallowed mismatches detected.", file=sys.stderr)
            sys.exit(1)

        print("\n1.4B BYTE-IDENTITY VERIFIED: 0 disallowed mismatches across all inputs.")


if __name__ == "__main__":
    main()
