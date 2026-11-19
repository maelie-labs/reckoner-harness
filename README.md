# Reckoner Harness 1.1.3

*Implements Reckoner Protocol Specification v1.1.3 ([SPEC.md](SPEC.md))*

The Reckoner Harness runs the tools that Reckoner models call while they generate. When a model emits a tool call such as `<|call:calc|>`, the harness runs the tool, stores the result in a register (`<|r0|>` to `<|r15|>`) and returns the observation into the token stream.

## How it differs from standard tool calling

In most setups a tool call ends the model's turn: the model writes JSON, a client runs the tool, and a new request sends the conversation back. Reckoner keeps tools inside generation.

- **No round trip.** The model emits a control token such as `<|call:calc|>`, and the harness runs the tool and writes the result into the same response. Generation resumes a median 31 ms after a tool call (BF16, one request at a time, 16 GB consumer GPU).
- **Exact results.** Arithmetic, algebra, dates, strings, small programs and lookups run in deterministic tools. Each result is stored in a register, a numbered slot the model can refer to later.
- **Short answers.** On our own 880-task tool-use test, the BF16 build scores 586/880 with a median answer of 66.5 tokens (with a box request, one request at a time, 2,048-token budget). Under the same conditions, with thinking off and the same tools offered as JSON function calls (the best of three tool-prompt versions for each), Qwen3.5-2B scores 593/880 at a median of 583.5 tokens and Qwen3.5-0.8B scores 391/880 at 308 tokens.

How to read that comparison: the test covers the task types Reckoner was trained on, and its grader scores only the final answer inside `\boxed{}` (an answer without one counts as wrong). For Qwen, the box request is a system prompt that also says "Please solve the following problem step by step", and 140 of Qwen3.5-2B's 880 answers hit the 2,048-token budget, which raises its median length.

Full comparison: [EVALUATION.md](https://huggingface.co/maelie-labs/reckoner-1.4b/blob/main/EVALUATION.md) in the model repository.

## About Reckoner

Reckoner is an open-weights model project from [Maelie Labs](https://huggingface.co/maelie-labs). Its models hand exact work (arithmetic, algebra, small programs, lookups) to tools instead of approximating it in their weights. Each turn opens with a mode gate: answer right away (`<|ready|>`), call tools (`<|act|>`), think step by step (`<|deliberate|>`), or ask a clarifying question (`<|clarify|>`). Every model ships with open weights, its training recipe and its evaluations. Reckoner-1.4B is licensed under Apache-2.0.

- **Reckoner-1.4B**: trained from scratch; [BF16](https://huggingface.co/maelie-labs/reckoner-1.4b), [W8A16](https://huggingface.co/maelie-labs/reckoner-1.4b-w8a16), [W4A16](https://huggingface.co/maelie-labs/reckoner-1.4b-w4a16), [Base](https://huggingface.co/maelie-labs/reckoner-1.4b-base).
- **More models**: retrofits of existing open models, starting with [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B), are in the works.
- **Reckoner Harness**: [`maelie-labs/reckoner-harness`](https://github.com/maelie-labs/reckoner-harness) (v1.1.3).
- **Maelie Labs on Hugging Face**: [`maelie-labs`](https://huggingface.co/maelie-labs).

## Quick start: serve Reckoner-1.4B

You need a Linux x86_64 host with an NVIDIA GPU and a CUDA 13 driver, Python 3.13 (the pinned install below is tested on 3.13 only), `git` and `patch`. SGLang's first start can need up to about 10 GB of host RAM. Run all three terminals on the same host.

```bash
# 1. Get the repository and the harness binary
git clone https://github.com/maelie-labs/reckoner-harness.git && cd reckoner-harness
curl -LO https://github.com/maelie-labs/reckoner-harness/releases/download/v1.1.3/reckoner-harness-linux-amd64
curl -LO https://github.com/maelie-labs/reckoner-harness/releases/download/v1.1.3/SHA256SUMS
grep reckoner-harness-linux-amd64 SHA256SUMS | sha256sum -c -
install -m 755 reckoner-harness-linux-amd64 reckoner-harness

# 2. Install SGLang 0.5.22 and the plugin at the tested versions, apply the SGLang patch, download the weights
pip install "sglang==0.5.22" -e integrations/sglang -c integrations/sglang/sglang_reckoner/sglang_rl_env_lock.txt
SGL_SITE=$(python -c "import pathlib, sglang; print(pathlib.Path(sglang.__file__).resolve().parents[1])")
patch -p1 -d "$SGL_SITE" < integrations/sglang/patches/0001-inflight-extend.patch
hf download maelie-labs/reckoner-1.4b --local-dir ./reckoner-1.4b

# 3. Terminal 1: SGLang token backend
python -m sglang_reckoner.server --model-path ./reckoner-1.4b --harness-bin "$PWD/reckoner-harness" \
  --host 127.0.0.1 --port 30000 --context-length 66560

# 4. Terminal 2: OpenAI-compatible front door (run from the reckoner-harness folder)
./reckoner-harness serve --profile reckoner-1.4b --sglang-addr http://127.0.0.1:30000 --host 127.0.0.1 --port 8080

# 5. Terminal 3: once Terminal 1 reports the server is ready, ask a question
curl -s http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is the exact value of 987654321 * 123456789?"}]}'
```

For the quantized builds, `serve` flags and `reasoning_effort`, see [Serving Options](#serving-options).

---

## Serving Options

### Quantized variants

The W8A16 and W4A16 builds ship in `compressed-tensors` format:

| Model | `model.safetensors` | GPU weight memory |
| :--- | :--- | :--- |
| [`maelie-labs/reckoner-1.4b`](https://huggingface.co/maelie-labs/reckoner-1.4b) (BF16) | 2.89 GB | 2.87 GB |
| [`maelie-labs/reckoner-1.4b-w8a16`](https://huggingface.co/maelie-labs/reckoner-1.4b-w8a16) | 1.54 GB | 1.66 GB |
| [`maelie-labs/reckoner-1.4b-w4a16`](https://huggingface.co/maelie-labs/reckoner-1.4b-w4a16) | 0.85 GB | 0.93 GB |

To use one, download it (for example `hf download maelie-labs/reckoner-1.4b-w8a16 --local-dir ./reckoner-1.4b-w8a16`), point `--model-path` at it, and add `--quantization compressed-tensors` to the Terminal 1 command.

### Serving notes

- `model` in a request is optional; the front door answers as `reckoner-1.4b`. `GET /health` reports the active profile and SGLang address.
- Calling SGLang's own `/v1/chat/completions`, or `tokenizer.apply_chat_template(...)`, returns an error that points you to `reckoner-harness serve`.
- `sglang_reckoner.server` and `serve` exchange the logit-processor id and per-request tool telemetry through `/dev/shm`, which is why they must share a host.
- `sglang_reckoner.server` sets the `SGLANG_RECKONER_*` environment defaults it needs. Set `SGLANG_RECKONER_FUSED_GATES=0` to turn off the fused GDN-2 kernels.
- Once running, SGLang uses about 5.5 GB of host RAM.
- The patch `0001-inflight-extend.patch` lets SGLang prefill a tool result mid-decode instead of token by token. To undo it, run `patch -R -p1 -d "$SGL_SITE" < integrations/sglang/patches/0001-inflight-extend.patch`.
- The plugin pins `flash-linear-attention==0.6.0`, which runs on the pinned Triton 3.8 stack as is.
- The `hf` CLI comes with `huggingface_hub`, which the plugin installs as a dependency. To build the harness instead of downloading it, see [Building from Source](#building-from-source).
- SGLang is the only supported engine. `llama.cpp` is not supported in this release (no GGUF export).

### `serve` flags

| Flag | Default | Meaning |
| :--- | :--- | :--- |
| `--host` | `127.0.0.1` | Listen address. Use `0.0.0.0` only if you mean to expose the port to other machines. |
| `--port`, `-p` | `8753` | Listen port. |
| `--sglang-addr` | `127.0.0.1:30000` | SGLang backend address. An `http://`, `grpc://` or `unix://` prefix is accepted. |
| `--sglang-proto` | `auto` | `http` or `grpc`. `auto` picks `grpc` only for a `grpc://` address. |
| `--profile` | none | Built-in profile name (`reckoner-1.4b`, `reckoner-q2b`) or path to a JSON profile. |
| `--tokenizer` | none | File or directory path for an external Hugging Face `tokenizer.json`, used only by profiles with an external tokenizer such as `reckoner-q2b`. `RECKONER_TOKENIZER_PATH` works too. Rejected with `ErrArgs` when the profile uses the embedded tokenizer. |
| `--model-dir`, `--model-path`, `--model` | none | Model directory searched for the profile's tokenizer when neither `--tokenizer` nor `RECKONER_TOKENIZER_PATH` is set (external-tokenizer profiles only). |
| `--graded` | off | Treat every request as graded (append the profile's `graded_answer_instruction`). |
| `--anchor-context` | off | Insert context anchors into rendered prompts (never when the profile sets `anchors: false`). |
| `--no-anchors` | off | Anchor-free mode (see [Anchor-Free Mode](#anchor-free-mode)). |
| `--mode` | `native` | `native` (in-path tools) or `stop-and-continue` (`stop_and_continue` is also accepted). |
| `--processor-str <id>` | read from `/dev/shm` | SGLang logit-processor id. By default it is read from the file `sglang_reckoner.server` writes to `/dev/shm`, with a built-in id as fallback. |

`serve` has no `--help` (`serve --help` starts the server) and ignores unknown flags without an error, so check spelling against this table.

### Reasoning modes and `reasoning_effort`

Every agent turn starts right after `<|turn:agent|>` (no newline) with one gate token:
- `<|ready|>`: answer directly without tools.
- `<|act|>`: use harness tools, with short narration, `<|call:...|>`...`<|/call|>` and `<|obs|>`...`<|/obs|>`.
- `<|deliberate|>`: think step by step without tools, then emit `<|ready|>` and the final answer.
- `<|clarify|>`: ask the user a clarifying question.

`reasoning_effort` controls thinking per request. The front door, JSON IPC (`format_prompt` / `render_prompt`) and the SGLang plugin share one table (`conformance/effort_table.json`):

| `reasoning_effort` | Aliases | `<\|deliberate\|>` | `thinking_budget` (tokens) |
| :--- | :--- | :--- | :--- |
| `none`, `minimal` | | Masked at the gate | 0 |
| `low` | `fast` | Allowed | 256 |
| `medium` (default; also omitted or `""`) | | Allowed | 1,024 |
| `high` | `careful`, `xhigh` | Allowed | 4,096 |

- In native mode the budget is a hard limit: the harness forces `<|ready|>` once `thinking_tokens` reaches it.
- An unrecognized value returns HTTP `400 Bad Request` (`ErrArgs`, `text/plain`) before any SSE bytes on `/v1/chat/completions`, and `ErrArgs` on IPC `format_prompt` / `render_prompt`.
- In `stop_and_continue` mode, `none` and `minimal` apply `logit_bias: -100` to `<|deliberate|>` on every round. `low`, `medium` and `high` run without `custom_logit_processor`, so `thinking_budget` is not enforced, and the harness logs a one-time warning per process.
- The `loop_cap` field in `conformance/effort_table.json` (the latent depth-loop cap `K_max`) is reserved and not enforced. Tool calls are limited by the session `max_tool_calls` (default `16`).
- A profile can set `effort_line` (`"off"`, the default in the built-in profiles, `"none_only"` or `"all"`) to write `reasoning_effort: <name>` at the start of the `system` turn. `minimal` renders as `reasoning_effort: none`.

---

## Tools

| Tool | Control Token | Capabilities |
| :--- | :--- | :--- |
| `calc` | `<\|call:calc\|>` | Exact rational arithmetic, closed forms in `ℚ(i, π, √r)`, trigonometry on multiples of `π/12`, polynomials (`expand`, `factor`, `divide`, `roots`), 2D/3D geometry, linear algebra up to 10x10, finite sums and products, and unit conversion. |
| `solve` | `<\|call:solve\|>` | Linear, quadratic, cubic and quartic polynomial equations, systems of linear equations, and polynomial or rational inequalities (answers in interval notation). |
| `lua` | `<\|call:lua\|>` | Sandboxed Lua 5.4 with a 20,000,000-instruction cap and a 10 MB heap limit. No system calls, file I/O or external modules. |
| `str` | `<\|call:str\|>` | UTF-8 rune operations: `len`, `slice`, `reverse`, case folding, substring search and count, and hex encoding. |
| `date` | `<\|call:date\|>` | Gregorian calendar arithmetic: adding to dates, date differences, day of week, leap years, weekday stepping, and `now([zone])` with a settable clock (`--clock` CLI flag or request `clock` field). Reads and writes `YYYY-MM-DD (Weekday)` strings. |
| `lookup` | `<\|call:lookup\|>` | Offline reference data: NIST CODATA 2022 physical constants with units, PubChem elements 1 to 118, and NASA/IAU planetary data. |
| `reg` | `<\|r0\|>`..`<\|r15\|>` | 16 circular registers. Results bind in order. After 16 bindings the buffer wraps, overwrites earlier slots and says so in the observation. |
| `fncall` | CLI / IPC | Parses, validates (JSON Schema) and renders (as Python signatures) external tool calls. |
| `anchor` | `<\|m0\|>`..`<\|m255\|>`, `<\|L0\|>`..`<\|L1023\|>` | Inserts message and line anchors into context, with a mid-line anchor every 64 tokens and message continuation. |
| `quote` | `<\|quote\|>`..`<\|/quote\|>` | Byte-exact citations from anchored context, with a receipt and constrained-decoding masks. |
| `find` | `<\|call:find\|>` | Normalized exact substring search with a BM25 fallback. Returns the top 3 matches with ±2 lines as newly anchored context. |
| `lines` | `<\|call:find\|>` | Returns line ranges from anchored context as newly anchored context. |

---

## Standard Error Codes

Errors follow Section 7.1 of [SPEC.md](SPEC.md): `<|obs:err|>Err<Code>: <message><|/obs|>`.

| Error Code | Meaning |
| :--- | :--- |
| `ErrAmbiguous` | Ambiguous reference in lookup or resolution. |
| `ErrArgs` | Invalid arguments or argument count passed to tool. |
| `ErrDimension` | Matrix or vector dimension mismatch. |
| `ErrDivZero` | Division or modulo by zero. |
| `ErrDomain` | Mathematical domain error (such as `ln(0)` or `log(-1)`). |
| `ErrInfiniteSolutions` | System of equations has infinitely many solutions. |
| `ErrInternal` | Internal harness panic recovery or unexpected failure. |
| `ErrInvalidDate` | Invalid Gregorian calendar date format or nonexistent date. |
| `ErrInvalidUnit` | Unrecognized or malformed physical unit string. |
| `ErrLua` | Lua runtime error, memory limit exceeded, or prohibited operation. |
| `ErrNoSolution` | System of equations or inequality has no solution. |
| `ErrNotFound` | Requested element, constant, astronomical entity, or message slot not found. |
| `ErrOutOfBounds` | Line anchor, register index, or string slice index out of bounds. |
| `ErrOverflow` | Numeric overflow, infinite float, or value exceeding precision bounds. |
| `ErrRegisterBounds` | Register index outside permitted 0..15 range. |
| `ErrSyntax` | Parsing failure or malformed expression syntax. |
| `ErrTimeout` | Tool execution exceeded the 2-second dispatcher deadline. |
| `ErrTooLong` | Input sequence exceeds maximum permitted length. |
| `ErrTypeMismatch` | Operation attempted on incompatible value types. |
| `ErrUnitMismatch` | Incompatible physical units in arithmetic operation. |
| `ErrUnsetRegister` | Attempt to read from an uninitialized register. |
| `ErrUnsupported` | Tool or operation not supported by runtime configuration. |
| `ErrWorkLimitExceeded` | Evaluation exceeded bit-budget or iteration cap. |

---

## Latency

Measured on the static Linux `amd64` (`musl`) release binary: 27 test cases on an 8-core x86_64 desktop CPU, up to 1,000 iterations per case, 25,300 calls in total. "Internal" is the harness's own `elapsed_us` (1 µs resolution). "Pipe" is the round trip from Python over standard I/O.

| Tool | Samples | Internal p50 | Internal p90 | Pipe p50 | Pipe p90 | Pipe p99 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `reg` | 2,000 | 3.0 µs | 4.0 µs | 17.6 µs | 21.6 µs | 39.0 µs |
| `lookup` | 3,000 | 4.0 µs | 6.0 µs | 18.9 µs | 22.6 µs | 41.1 µs |
| `str` | 3,000 | 4.0 µs | 5.0 µs | 18.7 µs | 22.0 µs | 35.0 µs |
| `date` | 3,000 | 6.0 µs | 9.0 µs | 22.1 µs | 31.6 µs | 43.7 µs |
| `quote` | 1,000 | 7.0 µs | 11.0 µs | 45.0 µs | 56.3 µs | 87.6 µs |
| `calc` | 8,000 | 23.0 µs | 111.0 µs | 39.5 µs | 130.0 µs | 243.6 µs |
| `solve` | 3,000 | 27.0 µs | 47.0 µs | 43.7 µs | 72.1 µs | 137.0 µs |
| `lua` | 2,000 | 97.0 µs | 107.0 µs | 118.1 µs | 132.6 µs | 177.9 µs |
| `find` | 100 | 210.0 µs | 242.4 µs | 246.2 µs | 284.6 µs | 1257.7 µs |
| `anchor` | 200 | 407.0 µs | 555.3 µs | 630.3 µs | 867.8 µs | 2879.4 µs |

Source data is in `benchmarks/latency_results.json`. To reproduce:

```bash
python3 scripts/benchmark_latency.py --bin ./build/bin/reckoner-harness-linux-amd64 --iterations 1000 --output benchmarks/latency_results.json
```

---

## Serving Integration

### Standard I/O Streaming Protocol

The binary reads newline-delimited JSON requests on standard input and writes one JSON response per line on standard output.

Request:

```json
{"tool": "calc", "payload": "35 * 12 + 14"}
```

Response:

```json
{"success": true, "result": "<|r0|> = 434", "elapsed_us": 6}
```

Error response:

```json
{"success": false, "result": "", "error": "ErrDivZero: division by zero", "elapsed_us": 4}
```

Python example:

```python
import json
import subprocess

class HarnessClient:
    def __init__(self, binary_path: str = "./reckoner-harness"):
        self.proc = subprocess.Popen(
            [binary_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def execute(self, tool_name: str, payload: str) -> dict:
        req = json.dumps({"tool": tool_name, "payload": payload})
        self.proc.stdin.write(req + "\n")
        self.proc.stdin.flush()
        res_line = self.proc.stdout.readline()
        return json.loads(res_line)

    def close(self):
        self.proc.terminate()
        self.proc.wait()

client = HarnessClient()
response = client.execute("calc", "1/3 + 1/6")
print(response)  # {'success': True, 'result': '<|r0|> = 0.5', 'elapsed_us': 7}
client.close()
```

### In-Path Serving

`reckoner-harness serve` provides the OpenAI-compatible `/v1/chat/completions` API in front of SGLang. Inside SGLang, the plugin's logit processor (`ReckonerToolLogitProcessor`) runs each tool call during decoding and prefills the result into the same turn, so there is no turn restart. Measured end to end in SGLang, the time from a tool call to the next generated token is about 15–20 ms (p50 14.79 to 19.53 ms, one request at a time on a 16 GB consumer GPU, across the BF16, W8A16 and W4A16 builds; see `EVALUATION.md` in the model repositories).

### Model Profiles

A profile sets a model's default system prompt, graded-answer instruction, tokenizer and anchor setting. Select one with `--profile <name|path>` on the CLI or `reckoner-harness serve`, using a built-in name or a path to a JSON profile. Built-in profiles live in `profiles/` and are embedded in the binary.

`reckoner-1.4b` sets `default_system: null`, `graded_answer_instruction: "Put your final answer within \\boxed{}."` and `anchors: true`, and uses the embedded 36,864-token BPE tokenizer (`newline_token_id: 326`). Context anchors (`<|mN|>`, `<|Lk|>`) stay off by default in `serve`, which inserts them only when given `--anchor-context` or `"anchor_context": true` on the request.

A `reckoner-q2b` profile and plugin support for an upcoming [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) retrofit are included, though those weights are not released yet.

Without `--profile`, no profile is applied. When a request is graded (`"graded": true` in the request JSON or `--graded` on the CLI), the harness appends `"\n\n" + graded_answer_instruction` to the last user turn. Chat requests are ungraded by default. See Section 12 ("Model Profiles") in [SPEC.md](SPEC.md).

### Anchor-Free Mode

For models trained without the context anchor tokens (`<|mN|>`, `<|Lk|>`), use anchor-free mode: `--no-anchors`, JSON API `{"anchors": false}`, or a profile with `"anchors": false` (such as `reckoner-q2b`). The harness then inserts no anchors into messages or observations, `find` and `lines` return plain context, and `quote` returns `ErrUnsupported`. `--no-anchors` overrides the profile's `anchors` setting. See "Anchor-Free Mode" in [SPEC.md](SPEC.md).

---

## Building from Source

### Prerequisites

- Go 1.26 or later (the `go.mod` minimum). The release binaries were built with Go `1.27.3`, and only that version reproduces their checksums.
- Zig `0.14.1`, the C cross-compiler for the cgo Lua 5.4 build.

### Local Native Build

```bash
go build -o reckoner-harness .
```

### Cross-Compilation Matrix

`scripts/build_harness.sh` builds the release targets for Linux, macOS and Windows, using `zig cc` as the cgo cross-compiler:

```bash
bash scripts/build_harness.sh
```

Targets produced in `build/bin/`:
- `reckoner-harness-linux-amd64` and `reckoner-harness-linux-arm64` (statically linked with musl libc)
- `reckoner-harness-windows-amd64.exe` and `reckoner-harness-windows-arm64.exe`
- `reckoner-harness-darwin-amd64` and `reckoner-harness-darwin-arm64`

With Go `1.27.3` and Zig `0.14.1`, the build reproduces the checksums under [Downloads](#downloads) byte for byte. The manual CI workflow (`.github/workflows/test-binaries.yml`) builds with Go `1.26.x` for conformance testing, so its binaries do not match.

---

## Verification and Testing

### Unit Tests

```bash
go vet ./...
go test -v ./...
```

### Conformance Suite

The test suite in `conformance/` is the behavioral contract. The harness must pass all 1,672 cases byte for byte (1,257 tool, 352 function calling, 49 context reference and 14 model profile cases):

```bash
python3 scripts/run_harness_conformance.py --binary ./reckoner-harness --cases all
```

The unit tests also check the 352 function calling cases in `conformance/fncall_cases.jsonl`:

```bash
go test -v ./fncall
```

### SGLang Plugin CPU Tests

```bash
CUDA_VISIBLE_DEVICES="" pytest integrations/sglang/tests/ -v
```

---

## Downloads

Release binaries for `v1.1.3`, built with Go `1.27.3` and `zig cc 0.14.1`:

| Platform | Binary | SHA-256 Checksum |
| :--- | :--- | :--- |
| Linux x86_64 | `reckoner-harness-linux-amd64` | `ce187909a2b1e6a286e4a8a065b50829b3cede37fdb62161a028087d3d6bc3f6` |
| Linux ARM64 | `reckoner-harness-linux-arm64` | `a2faf0a443a607db60813c59d5d7b61a99101045dbc523325662a6ff27aa7029` |
| Windows x86_64 | `reckoner-harness-windows-amd64.exe` | `9e140e197eb93ea78399c187f67ca3b582b65ff728e2ee10258d020006ed27d3` |
| Windows ARM64 | `reckoner-harness-windows-arm64.exe` | `bb6f6b5e85d0bf3ccc6d68f3cdc13c6b451eeaf10b156e37649ba32dfc20a013` |
| macOS Intel | `reckoner-harness-darwin-amd64` | `c8a1cdb42b2e6d79ef74faa94579b29a41fdb27848b8cc11bf6ee58a898372f7` |
| macOS Apple Silicon | `reckoner-harness-darwin-arm64` | `4b9328fe62d1053a115a7f3b3e88ba73f70844be11dd522201ac2fc2576f81ec` |

---

## License

The Reckoner Harness and SGLang plugin are licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

Copyright notices and licenses for third-party components and bundled reference data are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). These cover Lua 5.4, the Go runtime, musl libc, the MinGW-w64 runtime, Zig `compiler_rt`, `jsonschema`, `golang.org/x/text`, NIST/PubChem/NASA/IAU reference data (`lookup_data/SOURCES.md`), Reckoner Tokenizer v2 (`tokenizer_data/SOURCES.md`) and the SGLang and Qwen3.5 adaptations.
