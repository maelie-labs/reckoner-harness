#!/usr/bin/env python3
"""
Conformance test runner for the Reckoner Tool Harness.
Runs cases against any harness binary via the stdio streaming protocol.
Reports pass rate, latency percentiles (p50, p90, p99), and diffs.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple


def percentile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    k = (len(data) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return data[int(k)]
    d0 = data[int(f)] * (c - k)
    d1 = data[int(c)] * (k - f)
    return d0 + d1


class HarnessClient:
    def __init__(self, binary_path: str):
        if not os.path.isfile(binary_path):
            raise FileNotFoundError(f"Binary not found: {binary_path}")
        self.proc = subprocess.Popen(
            [binary_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )

    def dispatch_raw(self, req: Dict[str, Any]) -> Dict[str, Any]:
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            stderr_out = self.proc.stderr.read()
            raise RuntimeError(f"Harness process terminated unexpectedly. stderr: {stderr_out}")
        return json.loads(line.strip())

    def dispatch(self, tool: str, payload: str) -> Dict[str, Any]:
        return self.dispatch_raw({"tool": tool, "payload": payload})

    def reset(self):
        return self.dispatch("reset", "")

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.terminate()
            self.proc.wait(timeout=1.0)
        except Exception:
            self.proc.kill()


QWEN35_SPLIT_REGEX = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}|"
    r" ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def _gpt2_bytes_to_unicode() -> Dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def _write_synthetic_q2b_tokenizer(repo_root: str, dest_path: str) -> str:
    b2u = _gpt2_bytes_to_unicode()
    vocab: Dict[str, int] = {b2u[10]: 198}
    next_id = 0
    for b in range(256):
        if b == 10:
            continue
        if next_id == 198:
            next_id += 1
        vocab[b2u[b]] = next_id
        next_id += 1

    prof_path = os.path.join(repo_root, "profiles", "reckoner-q2b.json")
    with open(prof_path, "r", encoding="utf-8") as pf:
        prof = json.load(pf)

    added_list = []
    for k, tid in prof.get("control_tokens", {}).items():
        vocab[k] = tid
        added_list.append(
            {
                "id": tid,
                "content": k,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
        )

    raw = {
        "version": "1.0",
        "added_tokens": added_list,
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": {"Regex": QWEN35_SPLIT_REGEX},
                    "behavior": "Isolated",
                    "invert": False,
                },
                {
                    "type": "ByteLevel",
                    "add_prefix_space": False,
                    "trim_offsets": True,
                    "use_regex": False,
                },
            ],
        },
        "post_processor": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": False,
            "use_regex": False,
        },
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": False,
            "use_regex": False,
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": False,
            "vocab": vocab,
            "merges": [],
        },
    }
    with open(dest_path, "w", encoding="utf-8") as out_f:
        json.dump(raw, out_f)
    return dest_path


def run_conformance(binary_path: str, cases_paths: List[str], verbose: bool = False) -> bool:
    print(f"Running conformance test suite...")
    print(f"  Binary: {binary_path}")
    print(f"  Suites: {cases_paths}")

    cases = []
    for c_path in cases_paths:
        with open(c_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    cases.append(json.loads(line))

    print(f"Loaded {len(cases)} total test cases.\n")

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    import tempfile
    tmp_tok_dir = tempfile.TemporaryDirectory()
    q2b_tok_path = os.environ.get("RECKONER_TOKENIZER_PATH") or _write_synthetic_q2b_tokenizer(
        repo_root, os.path.join(tmp_tok_dir.name, "tokenizer.json")
    )

    client = HarnessClient(binary_path)

    passed = 0
    failed = 0
    failures: List[Dict[str, Any]] = []
    latencies_us: List[float] = []
    tool_stats: Dict[str, Dict[str, Any]] = {}

    for idx, case in enumerate(cases):
        client.reset()
        t0 = time.perf_counter()
        case_passed = True
        reason = ""
        tool = case.get("tool", "")

        # 1. Fncall category cases
        cat = case.get("category", "")
        if cat in ["render", "parse_roundtrip", "invalid_call"]:
            tool = "fncall"
            if tool not in tool_stats:
                tool_stats[tool] = {"passed": 0, "failed": 0, "latencies": []}

            if cat == "render":
                resp = client.dispatch("fncall", json.dumps({"op": "render", "tools": case["tools"]}))
                exp_render = case.get("expected_render", "")
                if not resp.get("success"):
                    case_passed = False
                    reason = f"Render failed: {resp.get('error')}"
                elif resp.get("result") != exp_render:
                    case_passed = False
                    reason = f"Render mismatch:\n  Expected: {exp_render!r}\n  Actual:   {resp.get('result')!r}"
            elif cat == "parse_roundtrip":
                req_payload = {"op": "parse", "call": case["call"]}
                if "tools" in case:
                    req_payload["tools"] = case["tools"]
                resp = client.dispatch("fncall", json.dumps(req_payload))
                if not resp.get("success"):
                    case_passed = False
                    reason = f"Parse failed: {resp.get('error')}"
                else:
                    parse_obj = json.loads(resp.get("result", "{}"))
                    if case.get("expected_name") and parse_obj.get("name") != case["expected_name"]:
                        case_passed = False
                        reason = f"Name mismatch: got {parse_obj.get('name')}, expected {case['expected_name']}"
                    # Check serialization
                    ser_payload = {
                        "op": "serialize",
                        "name": parse_obj.get("name", ""),
                        "arguments": parse_obj.get("arguments", {}),
                    }
                    if "tools" in case:
                        ser_payload["tools"] = case["tools"]
                    ser_resp = client.dispatch("fncall", json.dumps(ser_payload))
                    if not ser_resp.get("success"):
                        case_passed = False
                        reason = f"Serialize failed: {ser_resp.get('error')}"
                    elif case.get("canonical_call") and ser_resp.get("result") != case["canonical_call"]:
                        case_passed = False
                        reason = f"Canonical call mismatch: got {ser_resp.get('result')!r}, expected {case['canonical_call']!r}"

            elif cat == "invalid_call":
                req_payload = {"op": "validate", "call": case["call"]}
                if "tools" in case:
                    req_payload["tools"] = case["tools"]
                resp = client.dispatch("fncall", json.dumps(req_payload))
                exp_err = case.get("expected_error_prefix", "")
                if not resp.get("success"):
                    # Command-level failure with expected prefix
                    if exp_err and not resp.get("error", "").startswith(exp_err):
                        case_passed = False
                        reason = f"Expected error prefix {exp_err!r}, got error {resp.get('error')!r}"
                else:
                    val_res = json.loads(resp.get("result", "{}"))
                    if val_res.get("valid") is True:
                        case_passed = False
                        reason = f"Expected call to be invalid, but validation succeeded: {case['call']}"
                    elif exp_err and not val_res.get("error", "").startswith(exp_err):
                        case_passed = False
                        reason = f"Expected validation error prefix {exp_err!r}, got {val_res.get('error')!r}"

        # 1b. Profile cases (profile_cases.jsonl)
        elif cat in ["prompt_format", "anchors_behavior"]:
            tool = "profile"
            if tool not in tool_stats:
                tool_stats[tool] = {"passed": 0, "failed": 0, "latencies": []}

            if cat == "prompt_format":
                req = {
                    "tool": "format_prompt",
                    "profile": case.get("profile", ""),
                    "graded": case.get("graded", False),
                    "messages": case.get("messages", []),
                }
                if case.get("profile") == "reckoner-q2b" and not os.environ.get("RECKONER_TOKENIZER_PATH"):
                    req["tokenizer"] = q2b_tok_path
                resp = client.dispatch_raw(req)
                if not resp.get("success"):
                    case_passed = False
                    reason = f"format_prompt failed: {resp.get('error')}"
                else:
                    if case.get("expected_prompt") and resp.get("result") != case["expected_prompt"]:
                        case_passed = False
                        reason = f"Prompt mismatch:\n  Expected: {case['expected_prompt']!r}\n  Actual:   {resp.get('result')!r}"
                    if case.get("expected_tokens") and resp.get("tokens") != case["expected_tokens"]:
                        case_passed = False
                        reason = f"Tokens mismatch: got {resp.get('tokens')}, expected {case['expected_tokens']}"
            elif cat == "anchors_behavior":
                sc_req: Dict[str, Any] = {"tool": "session_config", "profile": case.get("profile", "")}
                if case.get("no_anchors"):
                    sc_req["anchors"] = False
                sc_resp = client.dispatch_raw(sc_req)
                if not sc_resp.get("success"):
                    case_passed = False
                    reason = f"session_config failed: {sc_resp.get('error')}"
                else:
                    for setup in case.get("setup_messages", []):
                        s_resp = client.dispatch_raw({"tool": "anchor", "tokens": setup, "store": True})
                        if not s_resp.get("success"):
                            case_passed = False
                            reason = f"Setup message store failed: {s_resp.get('error')}"
                            break
                    if case_passed:
                        resp = client.dispatch_raw({"tool": case["tool"], "payload": case.get("payload", "")})
                        exp_err = case.get("expected_error_prefix")
                        if exp_err:
                            if resp.get("success"):
                                case_passed = False
                                reason = f"Expected error {exp_err!r}, but succeeded"
                            elif not resp.get("error", "").startswith(exp_err):
                                case_passed = False
                                reason = f"Expected error prefix {exp_err!r}, got {resp.get('error')!r}"
                        else:
                            if not resp.get("success"):
                                case_passed = False
                                reason = f"Tool {case['tool']} failed: {resp.get('error')}"
                            if case.get("expected_prefix") and not resp.get("result", "").startswith(case["expected_prefix"]):
                                case_passed = False
                                reason = f"Prefix mismatch: got {resp.get('result')!r}, expected prefix {case['expected_prefix']!r}"
                            if case.get("expected_text") and resp.get("text") != case["expected_text"]:
                                case_passed = False
                                reason = f"Text mismatch: got {resp.get('text')!r}, expected {case['expected_text']!r}"
                client.dispatch_raw({"tool": "session_config", "profile": "", "anchors": True})

        # 2. Context cases
        elif tool in ["anchor", "quote", "quote_mask", "find", "lines", "context_reset"]:
            if tool not in tool_stats:
                tool_stats[tool] = {"passed": 0, "failed": 0, "latencies": []}

            # Configure anchor mode if specified (default: True)
            if "anchors" in case:
                client.dispatch_raw({"tool": "session_config", "anchors": case["anchors"]})
            else:
                client.dispatch_raw({"tool": "session_config", "anchors": True})

            # Run setup messages
            for setup in case.get("setup_messages", []):
                s_resp = client.dispatch_raw({"tool": "anchor", "tokens": setup, "store": True})
                if not s_resp.get("success"):
                    case_passed = False
                    reason = f"Setup message store failed: {s_resp.get('error')}"
                    break

            if case_passed:
                if tool == "anchor":
                    req = {"tool": "anchor", "tokens": case["tokens"]}
                    if "targets" in case:
                        req["targets"] = case["targets"]
                    if case.get("message_id") is not None:
                        req["message_id"] = case["message_id"]
                    resp = client.dispatch_raw(req)

                    exp_err = case.get("expected_error_prefix")
                    if exp_err:
                        if resp.get("success"):
                            case_passed = False
                            reason = f"Expected anchor error {exp_err!r}, but succeeded"
                        elif not resp.get("error", "").startswith(exp_err):
                            case_passed = False
                            reason = f"Expected anchor error prefix {exp_err!r}, got {resp.get('error')!r}"
                    else:
                        if not resp.get("success"):
                            case_passed = False
                            reason = f"Anchor failed: {resp.get('error')}"
                        if case.get("expected_tokens") and resp.get("tokens") != case["expected_tokens"]:
                            case_passed = False
                            reason = f"Anchor tokens mismatch: got {resp.get('tokens')}, expected {case['expected_tokens']}"
                        if case.get("expected_targets") and resp.get("targets") != case["expected_targets"]:
                            case_passed = False
                            reason = f"Anchor targets mismatch"
                        if case.get("expected_next_message_id") is not None and resp.get("next_message_id") != case["expected_next_message_id"]:
                            case_passed = False
                            reason = f"Next message id mismatch: got {resp.get('next_message_id')}, expected {case['expected_next_message_id']}"

                elif tool == "quote":
                    req = {"tool": "quote", "payload": case.get("call", case.get("payload", ""))}
                    resp = client.dispatch_raw(req)
                    exp_err = case.get("expected_error_prefix")
                    if exp_err:
                        if resp.get("success"):
                            case_passed = False
                            reason = f"Expected quote error {exp_err!r}, but succeeded"
                        elif not resp.get("error", "").startswith(exp_err):
                            case_passed = False
                            reason = f"Expected quote error prefix {exp_err!r}, got {resp.get('error')!r}"
                    else:
                        if not resp.get("success"):
                            case_passed = False
                            reason = f"Quote failed: {resp.get('error')}"
                        if case.get("expected_text") and resp.get("text") != case["expected_text"]:
                            case_passed = False
                            reason = f"Quote text mismatch: got {resp.get('text')!r}, expected {case['expected_text']!r}"
                        if case.get("expected_receipt") and resp.get("receipt") != case["expected_receipt"]:
                            case_passed = False
                            reason = f"Quote receipt exact mismatch: got {resp.get('receipt')!r}, expected {case['expected_receipt']!r}"
                        if case.get("expected_receipt_prefix") and not resp.get("receipt", "").startswith(case["expected_receipt_prefix"]):
                            case_passed = False
                            reason = f"Quote receipt prefix mismatch: got {resp.get('receipt')!r}, expected prefix {case['expected_receipt_prefix']!r}"

                elif tool == "quote_mask":
                    req = {"tool": "quote_mask", "prefix": case["prefix"]}
                    resp = client.dispatch_raw(req)
                    exp_err = case.get("expected_error_prefix")
                    if exp_err:
                        if resp.get("success"):
                            case_passed = False
                            reason = f"Expected quote_mask error {exp_err!r}, but succeeded"
                        elif not resp.get("error", "").startswith(exp_err):
                            case_passed = False
                            reason = f"Expected quote_mask error prefix {exp_err!r}, got {resp.get('error')!r}"
                    else:
                        if not resp.get("success"):
                            case_passed = False
                            reason = f"Quote_mask failed: {resp.get('error')}"
                        if case.get("expected_allowed_tokens") and resp.get("allowed_tokens") != case["expected_allowed_tokens"]:
                            case_passed = False
                            reason = f"Quote_mask allowed tokens mismatch: got {resp.get('allowed_tokens')}, expected {case['expected_allowed_tokens']}"
                        if case.get("expected_step") and resp.get("step") != case["expected_step"]:
                            case_passed = False
                            reason = f"Quote_mask step mismatch: got {resp.get('step')}, expected {case['expected_step']}"

                elif tool in ["find", "lines"]:
                    req = {"tool": tool, "payload": case.get("payload", "")}
                    resp = client.dispatch_raw(req)
                    exp_err = case.get("expected_error_prefix")
                    if exp_err:
                        if resp.get("success"):
                            case_passed = False
                            reason = f"Expected {tool} error {exp_err!r}, but succeeded"
                        elif not resp.get("error", "").startswith(exp_err):
                            case_passed = False
                            reason = f"Expected {tool} error prefix {exp_err!r}, got {resp.get('error')!r}"
                    else:
                        if not resp.get("success"):
                            case_passed = False
                            reason = f"{tool} failed: {resp.get('error')}"
                        if case.get("expected_prefix") and not resp.get("result", "").startswith(case["expected_prefix"]):
                            case_passed = False
                            reason = f"{tool} prefix mismatch: got {resp.get('result')!r}, expected prefix {case['expected_prefix']!r}"
                        if case.get("expected_result") and resp.get("result") != case["expected_result"]:
                            case_passed = False
                            reason = f"{tool} result mismatch: got {resp.get('result')!r}, expected {case['expected_result']!r}"
                        if case.get("expected_matches") is not None and resp.get("num_matches", 0) != case["expected_matches"]:
                            case_passed = False
                            reason = f"{tool} matches count mismatch: got {resp.get('num_matches', 0)}, expected {case['expected_matches']}"

        # 3. Standard cases (cases.jsonl)
        else:
            inp = case["input"]
            exp_out = case.get("expected_output")
            exp_err = case.get("expected_error_prefix")
            regs_before = case.get("registers_before")

            if tool not in tool_stats:
                tool_stats[tool] = {"passed": 0, "failed": 0, "latencies": []}

            if regs_before:
                for rname, rval in regs_before.items():
                    client.dispatch("reg", f"set {rname} {rval}")

            req = {"tool": tool, "payload": inp}
            if "clock" in case:
                req["clock"] = case["clock"]
            resp = client.dispatch_raw(req)
            success = resp.get("success", False)
            actual_res = resp.get("result", "")
            actual_err = resp.get("error", "")

            if exp_err:
                if success:
                    case_passed = False
                    reason = f"Expected error starting with {exp_err!r}, but command succeeded with {actual_res!r}"
                elif not actual_err.startswith(exp_err):
                    case_passed = False
                    reason = f"Expected error prefix {exp_err!r}, got error {actual_err!r}"
            else:
                if not success:
                    case_passed = False
                    reason = f"Command failed unexpectedly with error: {actual_err!r}"
                elif actual_res != exp_out:
                    case_passed = False
                    reason = f"Output mismatch:\n  Expected: {exp_out!r}\n  Actual:   {actual_res!r}"

        elapsed_us = (time.perf_counter() - t0) * 1e6
        latencies_us.append(elapsed_us)
        tool_stats[tool]["latencies"].append(elapsed_us)

        if case_passed:
            passed += 1
            tool_stats[tool]["passed"] += 1
        else:
            failed += 1
            tool_stats[tool]["failed"] += 1
            failures.append({
                "index": idx,
                "tool": tool,
                "case_id": case.get("id", f"case_{idx}"),
                "reason": reason,
                "tags": case.get("tags", []),
            })

    client.close()

    # Reporting
    latencies_us.sort()
    total = len(cases)
    pass_rate = (passed / total) * 100 if total > 0 else 0.0

    print("=" * 70)
    print("CONFORMANCE TEST RESULTS")
    print("=" * 70)
    print(f"Total:      {total}")
    print(f"Passed:     {passed} ({pass_rate:.2f}%)")
    print(f"Failed:     {failed}")
    print("-" * 70)
    print("Per-Tool Breakdown:")
    for tname, stats in sorted(tool_stats.items()):
        t_total = stats["passed"] + stats["failed"]
        t_rate = (stats["passed"] / t_total) * 100 if t_total > 0 else 0.0
        t_lats = sorted(stats["latencies"])
        t_p50 = percentile(t_lats, 50)
        t_p90 = percentile(t_lats, 90)
        print(f"  {tname:<12} {stats['passed']:>4}/{t_total:<4} ({t_rate:>6.2f}%) | "
              f"p50: {t_p50:>6.1f}µs, p90: {t_p90:>6.1f}µs")

    print("-" * 70)
    print("Latency Percentiles (end-to-end Python stdio round-trip):")
    print(f"  Min:  {percentile(latencies_us, 0):.1f} µs")
    print(f"  p50:  {percentile(latencies_us, 50):.1f} µs")
    print(f"  p90:  {percentile(latencies_us, 90):.1f} µs")
    print(f"  p99:  {percentile(latencies_us, 99):.1f} µs")
    print(f"  Max:  {percentile(latencies_us, 100):.1f} µs")

    if failed > 0:
        print("\n" + "=" * 70)
        print(f"FAILURES ({failed} total):")
        print("=" * 70)
        for f_item in failures[:20]:
            print(f"[{f_item['index']}] ID: {f_item['case_id']} | Tool: {f_item['tool']}")
            print(f"  {f_item['reason']}")
            print(f"  Tags: {f_item['tags']}")
            print()
        if len(failures) > 20:
            print(f"... and {len(failures) - 20} more failures.")

    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        try:
            with open(summary_file, "a", encoding="utf-8") as sf:
                sf.write(f"#### Conformance Results: {passed}/{total} ({pass_rate:.2f}%)\n\n")
                sf.write(f"- **Passed**: {passed}\n")
                sf.write(f"- **Failed**: {failed}\n\n")
        except Exception:
            pass

    return failed == 0


def main():
    parser = argparse.ArgumentParser(description="Run harness conformance tests.")
    parser.add_argument("--binary", "--bin", default="reckoner-harness", help="Path to harness binary")
    parser.add_argument("--cases", default="all", help="Path to cases.jsonl, or 'all' (default: all)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    binary = os.path.join(repo_root, args.binary) if not os.path.isabs(args.binary) else args.binary
    if not os.path.isfile(binary):
        alt_bin = os.path.join(repo_root, "build", "bin", "reckoner-harness-linux-amd64")
        if os.path.isfile(alt_bin):
            binary = alt_bin

    if args.cases == "all":
        suites = [
            os.path.join(repo_root, "conformance", "cases.jsonl"),
            os.path.join(repo_root, "conformance", "fncall_cases.jsonl"),
            os.path.join(repo_root, "conformance", "context_cases.jsonl"),
            os.path.join(repo_root, "conformance", "profile_cases.jsonl"),
        ]
    else:
        suites = [os.path.join(repo_root, args.cases) if not os.path.isabs(args.cases) else args.cases]

    success = run_conformance(binary, suites, args.verbose)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
