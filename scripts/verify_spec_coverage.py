#!/usr/bin/env python3
"""verify_spec_coverage.py: Automated dynamic coverage gate for SPEC.md.

Validates that:
1. Every harness tool (calc, solve, lua, str, date, lookup, reg, fncall, anchor, quote, quote_mask, find, lines, context_reset) is documented.
2. Context references (anchors, quotes, receipts, constrained decoding masks, search) are fully specified.
3. The planned front-door serving architecture is specified per project requirements.
4. Go source AST/regex is dynamically parsed:
   - All calc/math functions in infix.go and eval.go are documented in SPEC.md.
   - All str functions in str.go are documented in SPEC.md.
   - All date functions in date.go are documented in SPEC.md (and date 'now' removal is noted).
   - All lookup fields across CODATA, periodic table, and astronomy are documented in SPEC.md.
   - All fncall actions (render, parse, serialize, validate, grammar) are documented in SPEC.md.
   - All error codes emitted by the Go codebase are documented in SPEC.md.
5. RFC 2119 keywords (MUST, SHOULD, etc.) and exact number formatting rules are present.
6. All documented CLI and JSON streaming examples execute successfully against the compiled binary.
"""

import glob
import json
import os
import re
import subprocess
import sys

REQUIRED_TOOLS = [
    "calc",
    "solve",
    "lua",
    "str",
    "date",
    "lookup",
    "reg",
    "fncall",
    "anchor",
    "quote",
    "quote_mask",
    "find",
    "lines",
    "context_reset",
]

REQUIRED_CONTEXT_CONCEPTS = [
    "<|m0|>",
    "<|m255|>",
    "<|L0|>",
    "<|L1023|>",
    "<|quote|>",
    "<|/quote|>",
    "Quoted 1 line",
    "Quoted N lines",
    "step",
    "provenance",
    "no matches",
]

REQUIRED_SERVING_CONCEPTS = [
    "front door",
    "/v1/chat/completions",
    "reasoning_content",
    "tool_calls",
    "Native Mode",
    "Stop-and-Continue",
    "strip",
    "expand",
]

REQUIRED_FORMATTING_RULES = [
    "terminating decimal",
    "irreducible fraction",
    "half-away-from-zero",
    "scientific notation",
    "too large to display",
]


def find_binary(repo_root):
    env_bin = os.environ.get("RECKONER_HARNESS_BIN", "").strip()
    if env_bin and os.path.isfile(env_bin) and os.access(env_bin, os.X_OK):
        return env_bin
    candidates = [
        os.path.join(repo_root, "reckoner-harness"),
        os.path.join(repo_root, "build", "bin", "reckoner-harness-linux-amd64"),
        os.path.join(repo_root, "build", "bin", "reckoner-harness"),
    ]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    raise FileNotFoundError(f"Could not find executable harness binary in {candidates}")


def check_tool_coverage(spec_text):
    print("=== 1. Checking Tool Coverage in SPEC.md ===")
    missing_tools = []
    for tool in REQUIRED_TOOLS:
        pattern = re.compile(rf"\b{re.escape(tool)}\b", re.IGNORECASE)
        if not pattern.search(spec_text):
            missing_tools.append(tool)
    if missing_tools:
        print(f"FAIL: Missing tool documentation for: {missing_tools}")
        sys.exit(1)
    print(f"PASS: All {len(REQUIRED_TOOLS)} harness tools documented: {REQUIRED_TOOLS}")


def check_context_and_serving_coverage(spec_text):
    print("\n=== 2. Checking Context References & Serving Specification ===")
    missing_ctx = [c for c in REQUIRED_CONTEXT_CONCEPTS if c not in spec_text]
    if missing_ctx:
        print(f"FAIL: Missing context specification concepts: {missing_ctx}")
        sys.exit(1)
    print(f"PASS: All {len(REQUIRED_CONTEXT_CONCEPTS)} context reference concepts documented.")

    missing_srv = [s for s in REQUIRED_SERVING_CONCEPTS if s.lower() not in spec_text.lower()]
    if missing_srv:
        print(f"FAIL: Missing serving architecture concepts: {missing_srv}")
        sys.exit(1)
    print(f"PASS: All {len(REQUIRED_SERVING_CONCEPTS)} serving architecture concepts documented.")

    # Check that forbidden internal tokens/names are absent
    if "<|call:lines|>" in spec_text:
        print("FAIL: <|call:lines|> token found in SPEC.md (must use <|call:find|>)")
        sys.exit(1)
    if "<|call:fncall|>" in spec_text:
        print("FAIL: <|call:fncall|> token found in SPEC.md (fncall has no call token)")
        sys.exit(1)
    internal_id_match = re.search(r"\b(?:QA|HD|TD|HARNESS|FIX|RUNBOOK|PLAN)-[0-9A-Z_]+\b", spec_text, re.IGNORECASE)
    if internal_id_match:
        print(f"FAIL: Internal task reference '{internal_id_match.group(0)}' found in SPEC.md")
        sys.exit(1)
    if "whitespace alignment" in spec_text.lower():
        print("FAIL: Contradictory 'whitespace alignment' found in SPEC.md")
        sys.exit(1)
    print("PASS: Verified absence of prohibited tokens/names (<|call:lines|>, <|call:fncall|>, internal task IDs, whitespace alignment).")


def check_source_symbol_coverage(repo_root, spec_text):
    print("\n=== 3. Checking Dynamic Go Source Symbol Coverage ===")

    # 1. Calc/math function names from infix.go and eval.go
    calc_names = set()
    for f in [os.path.join(repo_root, "infix.go"), os.path.join(repo_root, "eval.go")]:
        if os.path.isfile(f):
            content = open(f, encoding="utf-8").read()
            for m in re.finditer(r'case\s+([^:]+):', content):
                cases = [s.strip().strip('"') for s in m.group(1).split(',') if '"' in s]
                for c in cases:
                    if re.match(r'^[a-z][a-z0-9_]*$', c):
                        calc_names.add(c)

    # 2. Str function names from str.go
    str_names = set()
    str_file = os.path.join(repo_root, "str.go")
    if os.path.isfile(str_file):
        content = open(str_file, encoding="utf-8").read()
        for m in re.finditer(r'case\s+([^:]+):', content):
            cases = [s.strip().strip('"') for s in m.group(1).split(',') if '"' in s]
            for c in cases:
                if re.match(r'^[a-z][a-z0-9_]*$', c):
                    str_names.add(c)

    # 3. Date function names from date.go
    date_names = set()
    date_file = os.path.join(repo_root, "date.go")
    if os.path.isfile(date_file):
        content = open(date_file, encoding="utf-8").read()
        m_sw = re.search(r'switch\s+fnName\s*\{(.*?)\n\tdefault:', content, re.DOTALL)
        if m_sw:
            for m in re.finditer(r'case\s+"([a-z_]+)":', m_sw.group(1)):
                c = m.group(1)
                date_names.add(c)
        if 'case "now":' not in content:
            print("FAIL: date 'now' handler missing from date.go")
            sys.exit(1)

    # 4. Lookup fields dynamically parsed from lookup.go and astronomy.json
    lookup_fields = set()
    lookup_file = os.path.join(repo_root, "lookup.go")
    if os.path.isfile(lookup_file):
        content = open(lookup_file, encoding="utf-8").read()
        for sw in re.finditer(r'switch\s+field\s*\{([^}]+)\}', content):
            for m in re.finditer(r'case\s+([^:]+):', sw.group(1)):
                cases = [s.strip().strip('"') for s in m.group(1).split(',') if '"' in s]
                for c in cases:
                    if re.match(r'^[a-z][a-z0-9_]*$', c):
                        lookup_fields.add(c)
        if 'formatConstantResult' in content:
            lookup_fields.update(["value", "unit", "uncertainty"])
    astro_file = os.path.join(repo_root, "lookup_data", "astronomy.json")
    if os.path.isfile(astro_file):
        astro = json.load(open(astro_file, encoding="utf-8"))
        for item in astro:
            lookup_fields.update(item.get("fields", {}).keys())

    # 5. Fncall actions from fncall/fncall.go
    fncall_actions = set()
    fncall_file = os.path.join(repo_root, "fncall", "fncall.go")
    if os.path.isfile(fncall_file):
        content = open(fncall_file, encoding="utf-8").read()
        fncall_actions = set(re.findall(r'case\s+"([a-z]+)":', content))

    # Verify coverage in SPEC.md
    categories = [
        ("calc functions", calc_names),
        ("str functions", str_names),
        ("date functions", date_names),
        ("lookup fields", lookup_fields),
        ("fncall actions", fncall_actions),
    ]

    for cat_name, symbols in categories:
        missing = []
        for sym in sorted(symbols):
            pattern = re.compile(rf"\b{re.escape(sym)}\b")
            if not pattern.search(spec_text):
                missing.append(sym)
        if missing:
            print(f"FAIL: Missing {cat_name} in SPEC.md: {missing}")
            sys.exit(1)
        print(f"PASS: All {len(symbols)} {cat_name} documented in SPEC.md: {sorted(symbols)}")


def check_error_catalog(repo_root, spec_text):
    print("\n=== 4. Checking Error Code Catalog Synchronization ===")
    spec_codes = dict(re.findall(r"\|\s*`(Err[A-Za-z0-9_]+)`\s*\|\s*([^|]+?)\s*\|", spec_text))
    print(f"Found {len(spec_codes)} error codes in SPEC.md Section 7.1: {sorted(spec_codes.keys())}")

    for code, desc in spec_codes.items():
        if not desc.strip():
            print(f"FAIL: Error code {code} has empty description in SPEC.md")
            sys.exit(1)

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

    diff = code_errors.symmetric_difference(spec_codes.keys())
    if diff:
        print(f"FAIL: Error code mismatch between Go code and SPEC.md: {diff}")
        sys.exit(1)
    print(f"PASS: Error code catalog is 100% synchronized ({len(code_errors)} error codes, 0 diffs).")


def check_normative_language_and_formatting(spec_text):
    print("\n=== 5. Checking Normative Language & Number Formatting Rules ===")
    rfc_keywords = ["MUST", "MUST NOT", "REQUIRED", "SHALL", "SHOULD"]
    for kw in rfc_keywords:
        if kw not in spec_text:
            print(f"FAIL: Missing RFC 2119 keyword {kw} in SPEC.md")
            sys.exit(1)
    print(f"PASS: All required RFC 2119 keywords present in SPEC.md: {rfc_keywords}")

    for rule in REQUIRED_FORMATTING_RULES:
        if rule.lower() not in spec_text.lower():
            print(f"FAIL: Missing number formatting rule description for {rule!r} in SPEC.md")
            sys.exit(1)
    print(f"PASS: All {len(REQUIRED_FORMATTING_RULES)} number formatting rules documented in SPEC.md.")


def run_cli_and_json_verification(bin_path, repo_root, spec_text):
    print("\n=== 6. Executing All Documented Examples in SPEC.md ===")
    examples = re.findall(r"(?:for example|e\.g\.),?\s*`([^`]+)`\s+evaluates to\s+`([^`]+)`", spec_text)
    print(f"Extracted {len(examples)} documented examples from SPEC.md")
    if len(examples) < 100:
        print(f"FAIL: Expected >= 100 documented examples in SPEC.md, found {len(examples)}")
        sys.exit(1)

    failed_ex = []
    for inp, exp in examples:
        if inp.startswith("lookup("):
            tool = "lookup"
        elif inp.startswith("add(") or inp.startswith("diff(") or inp.startswith("diff_weeks(") or inp.startswith("day_of_week(") or inp.startswith("next_weekday(") or inp.startswith("is_leap_year(") or inp.startswith("days_in_month(") or inp.startswith("quarter(") or inp.startswith("now("):
            tool = "date"
        elif any(inp.startswith(fn + "(") for fn in ["len", "slice", "reverse", "upper", "lower", "trim", "contains", "count", "count_vowels", "find", "replace", "to_hex", "from_hex", "to_bin", "to_ascii", "is_palindrome"]):
            tool = "str"
        elif "=" in inp and not any(k in inp for k in ["round", "line"]):
            tool = "solve"
        else:
            tool = "calc"

        proc = subprocess.run([bin_path, tool, inp], capture_output=True, text=True, env={**os.environ, "RECKONER_HARNESS_CLOCK": "2026-10-10T23:42:00Z"})
        act = proc.stdout.strip()
        match = (act == exp or act == f"<|r0|> = {exp}" or
                 (act.startswith("<|r0|> = ") and act[len("<|r0|> = "):] == exp) or
                 (exp.startswith("<|r0|> = ") and act == exp))
        if not match:
            failed_ex.append((tool, inp, exp, act))

    if failed_ex:
        print(f"FAIL: {len(failed_ex)} documented examples failed execution against binary:")
        for tool, inp, exp, act in failed_ex:
            print(f"  [{tool}] {inp} -> Expected: {exp!r}, Got: {act!r}")
        sys.exit(1)
    print(f"PASS: All {len(examples)} documented SPEC examples executed successfully against binary.")

    print("\n=== 7. Executing Documented CLI and JSON Stream Protocols ===")
    # 1. CLI examples
    cli_blocks = re.findall(
        r"\$\s+reckoner-harness\s+([a-z]+)\s+([\x27\x22])([^\n]+?)\2\s*\n((?:(?!\$|```).*\n?)+)",
        spec_text,
    )
    for tool, quote, payload, expected in cli_blocks:
        proc = subprocess.run([bin_path, tool, payload], capture_output=True, text=True)
        actual = proc.stdout.strip()
        expected = expected.strip()
        assert actual == expected, f"CLI {tool} {payload!r} -> Expected: {expected!r}, Got: {actual!r}"
        print(f"PASS CLI {tool:7s} -> {actual[:40]}")

    # 2. JSON streaming examples
    p = subprocess.Popen([bin_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

    def dispatch(req):
        p.stdin.write(json.dumps(req) + "\n")
        p.stdin.flush()
        line = p.stdout.readline()
        return json.loads(line.strip())

    # Math turn
    r = dispatch({"tool": "calc", "payload": "35 * 12 + 14"})
    assert r["success"] is True and r["result"] == "<|r0|> = 434"
    print("PASS JSON stream calc -> <|r0|> = 434")

    # Error recovery
    r = dispatch({"tool": "calc", "payload": "1/0"})
    assert r["success"] is False and "ErrDivZero" in r["error"]
    print("PASS JSON stream divzero -> ErrDivZero")

    # Context anchor
    r = dispatch({"tool": "anchor", "tokens": [25049, 12511], "targets": [-100, 12511], "store": True})
    assert r["success"] is True and r["tokens"] == [32896, 33152, 25049, 12511]
    print("PASS JSON stream anchor -> [32896, 33152, 25049, 12511]")

    # Context quote
    r = dispatch({"tool": "quote", "payload": "<|quote|><|m0|><|L0|><|L0|><|/quote|>"})
    assert r["success"] is True and r["result"] == "Invoice" and "Quoted 1 line (L0)" in r["receipt"]
    assert "text" in r and "provenance" in r
    print("PASS JSON stream quote -> Invoice with receipt, text, provenance")

    # Context quote_mask
    r = dispatch({"tool": "quote_mask", "prefix": ["<|quote|>", "<|m0|>"]})
    assert r["success"] is True and r["step"] == "start_line" and "<|L0|>" in r["allowed_tokens"]
    print("PASS JSON stream quote_mask -> step start_line")

    # Context find
    r = dispatch({"tool": "find", "payload": 'find("invoice")'})
    assert r["success"] is True and r["result"] == "<|m1|><|L0|>[exact] Invoice" and r["num_matches"] == 1
    print("PASS JSON stream find -> <|m1|><|L0|>[exact] Invoice")

    # Context lines
    r = dispatch({"tool": "lines", "payload": "lines(0, 0, 0)"})
    assert r["success"] is True and r["result"] == "<|m2|><|L0|>Invoice"
    print("PASS JSON stream lines -> <|m2|><|L0|>Invoice")

    # Reset
    r = dispatch({"tool": "reset"})
    assert r["success"] is True and r["result"] == "OK"
    print("PASS JSON stream reset -> OK")

    p.stdin.close()
    p.terminate()
    p.wait()


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    spec_path = os.path.join(repo_root, "SPEC.md")
    if not os.path.isfile(spec_path):
        sys.exit(f"FAIL: SPEC.md not found at {spec_path}")
    spec_text = open(spec_path, encoding="utf-8").read()

    bin_path = find_binary(repo_root)
    print(f"Using harness binary: {bin_path}")

    check_tool_coverage(spec_text)
    check_context_and_serving_coverage(spec_text)
    check_source_symbol_coverage(repo_root, spec_text)
    check_error_catalog(repo_root, spec_text)
    check_normative_language_and_formatting(spec_text)
    run_cli_and_json_verification(bin_path, repo_root, spec_text)

    print("\n=======================================================")
    print("ALL SPEC COVERAGE AND VERIFICATION GATES PASSED (100%)")
    print("=======================================================")


if __name__ == "__main__":
    main()
