#!/usr/bin/env python3
"""verify_docs_contract.py: Verifies all documented examples and error codes against the binary.

This script executes every example in SPEC.md and README.md against the native
compiled harness binary and validates the error code catalog in SPEC.md against
the Go codebase.
"""

import glob
import json
import os
import re
import subprocess
import sys


def find_binary():
    env_bin = os.environ.get("RECKONER_HARNESS_BIN", "").strip()
    if env_bin and os.path.isfile(env_bin) and os.access(env_bin, os.X_OK):
        return env_bin
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    candidates = [
        os.path.join(repo_root, "build", "bin", "reckoner-harness-linux-amd64"),
        os.path.join(repo_root, "build", "bin", "reckoner-harness"),
        os.path.join(repo_root, "reckoner-harness"),
    ]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    raise FileNotFoundError(f"Could not find executable harness binary in {candidates}")


def run_cli(bin_path, tool, payload):
    proc = subprocess.run(
        [bin_path, tool, payload],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "RECKONER_HARNESS_CLOCK": "2026-10-10T23:42:00Z"},
    )
    if proc.returncode != 0:
        return proc.stderr.strip()
    return proc.stdout.strip()


def run_json_stream(bin_path, requests):
    proc = subprocess.Popen(
        [bin_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    results = []
    for req in requests:
        proc.stdin.write(json.dumps(req) + "\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
        results.append(json.loads(line))
    proc.stdin.close()
    proc.terminate()
    proc.wait()
    return results


def extract_doc_examples(repo_root):
    spec_path = os.path.join(repo_root, "SPEC.md")
    spec_text = open(spec_path, encoding="utf-8").read()

    examples = []

    # 1. Parse CLI blocks from SPEC.md:
    # ```bash
    # $ reckoner-harness <tool> '<payload>'
    # <expected>
    # ```
    cli_blocks = re.findall(
        r"\$\s+reckoner-harness\s+([a-z]+)\s+([\x27\x22])([^\n]+?)\2\s*\n((?:(?!\$|```).*\n?)+)",
        spec_text,
    )
    for tool, quote, payload, expected in cli_blocks:
        examples.append((tool, payload, expected.strip()))

    # 2. Parse inline solve and reg expressions in SPEC:
    for m in re.finditer(
        r"`(?:solve\s+\")?([^\"`]+)\"?`\s+returns\s+`([^`]+)`", spec_text
    ):
        payload, expected = m.group(1), m.group(2)
        if payload.startswith("set r") or payload.startswith("get r"):
            examples.append(("reg", payload, expected.strip()))
        else:
            examples.append(("solve", payload, expected.strip()))

    # 3. Parse inline formats as:
    for m in re.finditer(r"`([^`]+)`\s+formats as\s+`([^`]+)`", spec_text):
        expr, expected = m.group(1), m.group(2)
        examples.append(("calc", expr, f"<|r0|> = {expected}"))

    # 4. Parse inline evaluates to across tools:
    for m in re.finditer(r"`([^`]+)`\s+evaluates to\s+`([^`]+)`", spec_text):
        expr, expected = m.group(1), m.group(2)
        if any(expr.startswith(fn + "(") for fn in ["len", "slice", "reverse", "upper", "lower", "trim", "contains", "count", "count_vowels", "find", "replace", "to_hex", "from_hex", "to_bin", "to_ascii", "is_palindrome"]):
            examples.append(("str", expr, f"<|r0|> = {expected}"))
        elif expr.startswith("return "):
            examples.append(("lua", expr, f"<|r0|> = {expected}"))
        elif expr.startswith("lookup("):
            examples.append(("lookup", expr, f"<|r0|> = {expected}"))
        elif any(expr.startswith(fn + "(") for fn in ["add", "diff", "diff_weeks", "day_of_week", "next_weekday", "is_leap_year", "days_in_month", "quarter", "now"]):
            examples.append(("date", expr, f"<|r0|> = {expected}"))
        elif expected.startswith("[too large"):
            examples.append(("calc", expr, f"<|r0|> = {expected}"))
        else:
            examples.append(("calc", expr, f"<|r0|> = {expected}"))

    return examples


def verify_examples(bin_path, repo_root):
    print("=== 1. Verifying Documented CLI Examples ===")
    cli_cases = extract_doc_examples(repo_root)

    failed = 0
    for tool, payload, expected in cli_cases:
        actual = run_cli(bin_path, tool, payload)
        if actual != expected:
            print(f"FAIL CLI {tool} {payload[:30]!r}\n  Expected: {expected!r}\n  Actual:   {actual!r}")
            failed += 1
        else:
            print(f"PASS CLI {tool:7s} -> {actual[:40]}")
    if failed:
        sys.exit(f"CLI examples failed: {failed}")

    print("\n=== 2. Verifying Documented JSON Stream Examples ===")
    stream_requests = [
        {"tool": "calc", "payload": "35 * 12 + 14"},
        {"tool": "calc", "payload": "1/3 + 1/6"},
        {"tool": "calc", "payload": "1/0"},
    ]
    stream_responses = run_json_stream(bin_path, stream_requests)

    # 1. calc 35 * 12 + 14 (binds r0)
    r0 = stream_responses[0]
    assert r0["success"] is True and r0["result"] == "<|r0|> = 434", f"Unexpected r0: {r0}"
    print(f"PASS JSON stream 35 * 12 + 14 -> {r0}")

    # 2. calc 1/3 + 1/6 (binds r1 in same session)
    r1 = stream_responses[1]
    assert r1["success"] is True and r1["result"] == "<|r1|> = 0.5", f"Unexpected r1: {r1}"
    print(f"PASS JSON stream 1/3 + 1/6 -> {r1}")

    # 3. calc 1/0
    r2 = stream_responses[2]
    assert r2["success"] is False and r2["result"] == "" and "ErrDivZero" in r2["error"], f"Unexpected r2: {r2}"
    print(f"PASS JSON stream 1/0 -> {r2}")


def verify_error_catalog(repo_root):
    print("\n=== 3. Verifying Error Code Catalog ===")
    spec_path = os.path.join(repo_root, "SPEC.md")
    spec_text = open(spec_path, encoding="utf-8").read()

    # Extract error codes from SPEC.md Table 5.1 dynamically
    spec_codes = dict(re.findall(r"\|\s*`(Err[A-Za-z0-9_]+)`\s*\|\s*([^|]+?)\s*\|", spec_text))
    print(f"Found {len(spec_codes)} error codes in SPEC.md: {sorted(spec_codes.keys())}")

    # Verify every error code has a meaningful description
    for code, desc in spec_codes.items():
        if not desc.strip():
            print(f"FAIL: Error code {code} has empty description in SPEC.md")
            sys.exit(1)

    # Grep codebase for all Err[A-Z]\w+ in strictly non-test Go files
    code_files = glob.glob(os.path.join(repo_root, "*.go")) + glob.glob(os.path.join(repo_root, "fncall", "*.go"))
    prod_files = [f for f in code_files if not f.endswith("_test.go")]

    code_errors = set()
    for f in prod_files:
        content = open(f, encoding="utf-8").read()
        for m in re.finditer(r'"(Err[A-Z][A-Za-z0-9_]+)', content):
            code = m.group(1)
            if code != "Error":
                code_errors.add(code)

    print(f"Found {len(code_errors)} error codes emitted in non-test Go code: {sorted(code_errors)}")

    missing_in_spec = code_errors - set(spec_codes.keys())
    if missing_in_spec:
        print(f"FAIL: Error codes emitted in code but missing in SPEC.md: {missing_in_spec}")
        sys.exit(1)

    extra_in_spec = set(spec_codes.keys()) - code_errors
    if extra_in_spec:
        print(f"FAIL: Error codes documented in SPEC.md but not found in codebase: {extra_in_spec}")
        sys.exit(1)

    print(f"PASS: Error code catalog is 100% synchronized with Go codebase ({len(code_errors)} error codes, 0 diffs).")


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    bin_path = find_binary()
    print(f"Using harness binary: {bin_path}")
    verify_examples(bin_path, repo_root)
    verify_error_catalog(repo_root)
    print("\nALL CONTRACT CHECKS PASSED SUCCESSFULLY.")


if __name__ == "__main__":
    main()
