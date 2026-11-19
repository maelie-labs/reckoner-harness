# Reckoner Tool Harness Conformance Suite

This directory contains the canonical conformance test suite for the Reckoner Tool Harness (`reckoner-harness` 1.1.3, `1,672` cases across four JSONL files plus two JSON effort fixtures, six files in total):

- `cases.jsonl` (`1,257` cases): single-turn and multi-turn `eval` protocol (`calc`, `solve`, `lua`, `str`, `date`, `lookup`, register ring `<|r0|>..<|r15|>`, error semantics, and output formatting).
- `fncall_cases.jsonl` (`352` cases): external function-calling schema validation, `<|tools|>` signature rendering, `<|call|>` call parsing, and bijective identifier sanitization (`NameMapper`, `fncall/render.go`) across categories `render`, `parse_roundtrip`, and `invalid_call`.
- `context_cases.jsonl` (`49` cases): context anchoring (`<|mN|>`, `<|Lk|>`), `find`, `lines`, `quote`, and `quote_mask` verification.
- `profile_cases.jsonl` (`14` cases): built-in model profiles (`reckoner-1.4b`, `reckoner-q2b`, and no profile `""`) verifying prompt formatting (`default_system`, `graded_answer_instruction`) and anchor behavior (`find`, `quote`, `no_anchors`).
- `effort_table.json` (`1` JSON fixture): canonical `reasoning_effort` mapping (`none`, `minimal`, `low`, `medium`, `high`), default (`medium`), aliases (`fast` -> `low`, `careful` -> `high`, `xhigh` -> `high`), and `loop_cap` (latent depth-loop cap `K_max`: reserved, not enforced).
- `effort_none_prefix.json` (`1` JSON fixture): canonical system-turn prefix `P` (`<|turn:system|>\nreasoning_effort: none<|/turn|>\n`) and token IDs for `effort_line: "none_only"` with `reasoning_effort: "none"` and no default or caller system prompt.

## Specification & Contract

The conformance suite is the behavioral contract for the Go harness (the project ships only the Go harness plus the portable [SPEC.md](../SPEC.md)).

Every test case validates:
1. Exact string output for successful evaluations (`expected_output` / `result`), including register tags (`<|r0|> = ...` or `<|x|> = ...`).
2. Exact error code prefixes for failed evaluations (`expected_error_prefix`), adhering to the deterministic error contract `Err<Code>: message`.
3. Correct register, context anchor, and profile state transformations across invocations.

### Case Schema (`cases.jsonl`)

Each line in `cases.jsonl` is a JSON object with these fields:

```json
{
  "tool": "calc | solve | str | date | lookup | lua | reg",
  "input": "<raw tool input string>",
  "expected_output": "<exact expected stdout string on success>",
  "expected_error_prefix": "<ErrCode prefix if error expected, or null>",
  "registers_before": { "r0": "val", ... } | null,
  "registers_after": { "r0": "val", ... } | null,
  "tags": ["calc", "arithmetic", "decimals", ...]
}
```

## Running the Conformance Suite

Compile the harness binary:

```bash
go build -o build/bin/reckoner-harness .
```

Run the full conformance suite (`1,672` cases across all four files):

```bash
python3 scripts/run_harness_conformance.py --binary build/bin/reckoner-harness --cases all
```

To run a specific test suite file, provide the `--cases` argument:

```bash
python3 scripts/run_harness_conformance.py --binary build/bin/reckoner-harness --cases conformance/cases.jsonl
```

The runner exercises the stdio streaming protocol (`{"tool": "...", "payload": "..."}`) with state isolation between runs and reports:
- Pass rate per tool and overall (must be 100%).
- End-to-end latency percentiles (min, p50, p90, p99, max).
- Detailed diffs for any failing cases.

## Key Design Invariants Enforced

- **Registers**: Typed register storage and observation formatting.
- **Power Bounds**: Exact power with bit-depth caps and exact rounding away from zero.
- **Floor Division & Modulo**: Exact rational floor division (`//`) and modulo (`%`).
- **Chained Comparisons**: Type-strict chained comparisons, with list comparisons rejected.
- **Polynomial Solver**: Exact symbolic equation solver with distinct real roots and double root simplification.
- **Decimal Formatting**: Exact terminating decimal conversion for rationals with denominators `2^a 5^b`, and float formatting.
- **Error Contract**: Deterministic single-line `Err<Code>: <message>` error contract.
- **Resilience**: Zero panics, 1 MiB payload ceiling, and execution deadlines.
- **String Formatting**: Preserves whitespace in quoted string literals and formats hexadecimal in uppercase.
- **Civil Calendar**: Civil calendar date arithmetic with month-end clamping and weekday formatting `YYYY-MM-DD (Weekday)`.
- **Rune Operations**: Rune-aware string length and slice indexing.
- **Register Ring Buffer**: 16 registers (`<|r0|>`..`<|r15|>`) with ring-buffer overwrite announcements (`<|r0|> = val (replaced oldVal)`).
