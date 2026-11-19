# Reckoner Harness v1.1.3 Release Notes

`v1.1.3` is the first public release of the Reckoner Harness (`reckoner-harness`). It runs the tools that Reckoner models call during generation and serves the models through an OpenAI-compatible API.

---

## What It Does

When a model emits a tool call (`<|call:calc|>`, `<|call:solve|>`, `<|call:lua|>`, `<|call:str|>`, `<|call:date|>`, `<|call:lookup|>`, `<|call:find|>` or `<|quote|>`), generation pauses at `<|/call|>` or `<|/quote|>`. The harness runs the tool, binds the result to one of 16 circular registers (`<|r0|>`..`<|r15|>`) and injects `<|obs|>...<|/obs|>` into the live sequence.

`reckoner-harness serve` is the front door (`POST /v1/chat/completions`). It renders prompts, applies reasoning-effort budgets and context anchors, and runs tools in front of SGLang `0.5.22` with the bundled `sglang_reckoner` plugin (`integrations/sglang/`). It listens on `127.0.0.1` by default; pass `--host 0.0.0.0` to expose it to other machines.

---

## What Is Included

### Tools and registers (`<|r0|>`..`<|r15|>`)

- `calc`: exact rational and integer arithmetic, quadratic surds, polynomials, vectors, matrices, sums and products, statistics, number theory and unit conversion.
- `solve`: linear, quadratic and polynomial equations, `n × n` linear systems, and univariate inequalities.
- `lua`: sandboxed Lua 5.4.9 (vendored C source, statically linked via `cgo`) with a 20,000,000-instruction budget and a 10 MB memory cap.
- `str`: Unicode rune-indexed inspection, slicing, search and transforms.
- `date`: proleptic Gregorian calendar arithmetic with a settable clock (`--clock`, `RECKONER_HARNESS_CLOCK`, or per-request `"clock"`).
- `lookup`: offline reference tables (NIST CODATA 2022 physical constants, PubChem / IUPAC elements, NASA GSFC / IAU 2015 planetary and astronomical data).
- `anchor`, `quote`, `quote_mask`, `find`, `lines`: context anchors (`<|m0|>`..`<|m255|>`, `<|L0|>`..`<|L1023|>`), citation receipts, constrained-decoding masks, and exact search with an Okapi BM25 fallback.
- `fncall`: schema rendering, parsing, Python-style argument serialization, JSON Schema validation and EBNF/GBNF grammar generation for external tools.

### Model profiles

- `reckoner-1.4b` for Reckoner-1.4B (`maelie-labs/reckoner-1.4b`, `reckoner-1.4b-w8a16`, `reckoner-1.4b-w4a16`): embedded 36,864-token tokenizer, `default_system: null`, `graded_answer_instruction: "Put your final answer within \\boxed{}."`, `anchors: true` and `effort_line: "off"`.
- `reckoner-q2b`: for an upcoming Qwen3.5-2B retrofit (weights not released yet). Runs anchor-free with an external `tokenizer.json`.

### SGLang plugin (`integrations/sglang/`)

`sglang_reckoner` (`v1.1.3`) and `integrations/sglang/patches/0001-inflight-extend.patch` for SGLang `0.5.22`. The plugin runs the GDN-2 + GQA hybrid models in-engine (`ReckonerForCausalLM`, plus `Qwen35GDN2ForCausalLM` for the upcoming retrofit). Fused Triton gate kernels are on by default through `sglang_reckoner.server`. Set `SGLANG_RECKONER_FUSED_GATES=0` to turn them off. Tool results are injected as `<|obs|>` tokens without leaving the scheduler.

### Conformance suite

1,672 conformance cases in `conformance/*.jsonl` (1,257 tool, 352 function calling, 49 context reference and 14 model profile cases), plus the effort fixtures `conformance/effort_table.json` and `conformance/effort_none_prefix.json`.

---

## Special-Token Escaping

Text from users, clients and tools can never turn into a control token. When that text contains `<|`, the harness writes it as `<\|`, and it escapes the tokenizer's other special tokens the same way. This applies to every prompt field and every tool result, including errors, in every profile and with no profile. The only control tokens in a prompt are the ones the harness places (turn framing, tool-call and observation markers, register prefixes, context anchors) and validated agent tokens such as `<|act|>` in a client's pre-formatted assistant turn. A pre-formatted assistant turn that contains a turn, system, tools or end-of-text token, or any other special token outside the agent set, is rejected.

---

## Reasoning Effort (`reasoning_effort`)

The front door (`reckoner-harness serve`), JSON IPC (`format_prompt` / `render_prompt`) and the SGLang plugin (`sglang_reckoner`) share one table (`conformance/effort_table.json`). Values are trimmed and case-insensitive. Empty or omitted means `medium`.

| `reasoning_effort` | Aliases | `<\|deliberate\|>` Gate | `thinking_budget` (tokens) |
| :--- | :--- | :--- | :--- |
| `none` | | Masked (`true`) | `0` |
| `minimal` | | Masked (`true`) | `0` |
| `low` | `fast` | Allowed (`false`) | `256` |
| `medium` (default) | `""` (omitted) | Allowed (`false`) | `1024` |
| `high` | `careful`, `xhigh` | Allowed (`false`) | `4096` |

An unrecognized value returns HTTP `400 Bad Request` (`ErrArgs`, `text/plain`) before any streaming bytes on `/v1/chat/completions`, and `ErrArgs` on JSON IPC `format_prompt` / `render_prompt`. In `stop_and_continue` mode (`--mode stop-and-continue` or `stop_and_continue`), `none` and `minimal` ban `<|deliberate|>` with `logit_bias: -100` in `sampling_params`. Positive `thinking_budget` caps need native SGLang mode.

---

## Binary Checksums (`v1.1.3`)

All six binaries are cross-compiled reproducibly on one Linux host with Go `1.27.3` and `zig cc 0.14.1`. The Linux binaries are fully static `musl` executables.

| Platform | Binary | SHA-256 Checksum |
| :--- | :--- | :--- |
| Linux x86_64 | `reckoner-harness-linux-amd64` | `ce187909a2b1e6a286e4a8a065b50829b3cede37fdb62161a028087d3d6bc3f6` |
| Linux ARM64 | `reckoner-harness-linux-arm64` | `a2faf0a443a607db60813c59d5d7b61a99101045dbc523325662a6ff27aa7029` |
| Windows x86_64 | `reckoner-harness-windows-amd64.exe` | `9e140e197eb93ea78399c187f67ca3b582b65ff728e2ee10258d020006ed27d3` |
| Windows ARM64 | `reckoner-harness-windows-arm64.exe` | `bb6f6b5e85d0bf3ccc6d68f3cdc13c6b451eeaf10b156e37649ba32dfc20a013` |
| macOS Intel | `reckoner-harness-darwin-amd64` | `c8a1cdb42b2e6d79ef74faa94579b29a41fdb27848b8cc11bf6ee58a898372f7` |
| macOS Apple Silicon | `reckoner-harness-darwin-arm64` | `4b9328fe62d1053a115a7f3b3e88ba73f70844be11dd522201ac2fc2576f81ec` |
