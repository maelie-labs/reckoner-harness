# Reckoner SGLang Engine Integration (`sglang-reckoner`)

This directory provides the out-of-tree **SGLang 0.5.22** integration plugin (`sglang_reckoner`) for serving both **Reckoner-1.4B** (`--profile reckoner-1.4b`, `ReckonerForCausalLM`) and **Reckoner-Q2B** (`--profile reckoner-q2b`, `Qwen35GDN2Config` / `Qwen35GDN2ForCausalLM`, built on [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) by Alibaba's Qwen team under `Apache-2.0`; model weights coming soon) behind the Go tool harness front door (`reckoner-harness serve`).

## Architecture Overview

For Reckoner-1.4B and Reckoner-Q2B, serving uses a two-process stack where **the Go harness is the user-facing front door** and **SGLang is the token-level generation backend**:

```text
Client (OpenAI HTTP /v1/chat/completions)
  │
  ▼
reckoner-harness serve --profile <reckoner-1.4b|reckoner-q2b>
  │  • Applies model profile defaults (default_system, graded_answer_instruction, anchors, tokenizer)
  │  • Renders canonical turn tokens (<|turn:system|>, <|turn:user|>, <|turn:agent|>)
  │  • Injects context anchor tokens (<|mN|>, <|Lk|>) when started with --anchor-context (on anchor-enabled profiles)
  ▼
sglang_reckoner.server (SGLang 0.5.22 token-in / token-out backend)
  │  • Runs ReckonerForCausalLM (1.4B) or Qwen35GDN2ForCausalLM (Q2B) with RadixLinearAttention + GDN-2
  │  • Runs ReckonerToolLogitProcessor in-engine to intercept <|/call|> & <|/quote|>,
  │    execute tools via GoToolHarness, and inject <|obs|>...<|/obs|> via inflight-extend
  └─► Returns token IDs + tool call sidecar metadata (/dev/shm) to reckoner-harness serve
```

> Calling SGLang's raw `/v1/chat/completions` endpoint without the front door is blocked by default with an error message pointing to `reckoner-harness serve` (unless `--allow-raw-chat` is explicitly passed for low-level debugging).

## Installation and Quick Start

You need a Linux x86_64 host with an NVIDIA GPU and a CUDA 13 driver, Python 3.13 (the pinned install uses the lockfile `sglang_reckoner/sglang_rl_env_lock.txt`, generated and tested on Python 3.13 with `torch==2.14.1+cu130` and `triton==3.8.0`), `git` and `patch`. Run `sglang_reckoner.server` and `reckoner-harness serve` on the same host: they exchange the logit-processor id and per-request tool telemetry through `/dev/shm`.

```bash
# 1. Get the repository and the harness binary (or build it: go build -o reckoner-harness .)
git clone https://github.com/maelie-labs/reckoner-harness.git && cd reckoner-harness
curl -LO https://github.com/maelie-labs/reckoner-harness/releases/download/v1.1.3/reckoner-harness-linux-amd64
curl -LO https://github.com/maelie-labs/reckoner-harness/releases/download/v1.1.3/SHA256SUMS
grep reckoner-harness-linux-amd64 SHA256SUMS | sha256sum -c -
install -m 755 reckoner-harness-linux-amd64 reckoner-harness

# 2. Install SGLang 0.5.22 and the plugin at the tested versions (includes flash-linear-attention 0.6.0), apply the SGLang patch, download the weights
pip install "sglang==0.5.22" -e integrations/sglang -c integrations/sglang/sglang_reckoner/sglang_rl_env_lock.txt
SGL_SITE=$(python -c "import pathlib, sglang; print(pathlib.Path(sglang.__file__).resolve().parents[1])")
patch -p1 -d "$SGL_SITE" < integrations/sglang/patches/0001-inflight-extend.patch
hf download maelie-labs/reckoner-1.4b --local-dir ./reckoner-1.4b   # the hf CLI comes with huggingface_hub, installed above as a dependency

# 3. Terminal 1: SGLang token backend
python -m sglang_reckoner.server --model-path ./reckoner-1.4b --harness-bin "$PWD/reckoner-harness" \
  --host 127.0.0.1 --port 30000 --context-length 66560

# 4. Terminal 2: OpenAI-compatible front door (run from the reckoner-harness folder)
./reckoner-harness serve --profile reckoner-1.4b --sglang-addr http://127.0.0.1:30000 --host 127.0.0.1 --port 8080

# 5. Terminal 3: once Terminal 1 reports the server is ready, ask a question
curl -s http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is the exact value of 987654321 * 123456789?"}]}'
```

The scheduler patch (`patches/0001-inflight-extend.patch`) lets SGLang's decode loop prefill a mid-generation tool observation (`<|obs|>...<|/obs|>`) in a single `EXTEND` pass instead of forcing it token by token. To undo it, run `patch -R -p1 -d "$SGL_SITE" < integrations/sglang/patches/0001-inflight-extend.patch`. The plugin pins `flash-linear-attention==0.6.0`, which runs on the pinned `torch 2.14.1` / `triton 3.8.0` stack without changes.

Available Reckoner-1.4B checkpoints on Hugging Face:
- BF16: [`maelie-labs/reckoner-1.4b`](https://huggingface.co/maelie-labs/reckoner-1.4b) (`2.89 GB` `model.safetensors`, `2.98 GB` GPU weight memory)
- W8A16: [`maelie-labs/reckoner-1.4b-w8a16`](https://huggingface.co/maelie-labs/reckoner-1.4b-w8a16) (`1.54 GB` `model.safetensors`, `1.72 GB` GPU weight memory)
- W4A16: [`maelie-labs/reckoner-1.4b-w4a16`](https://huggingface.co/maelie-labs/reckoner-1.4b-w4a16) (`0.85 GB` `model.safetensors`, `1.07 GB` GPU weight memory)

For a quantized build, download it instead (for example `hf download maelie-labs/reckoner-1.4b-w8a16 --local-dir ./reckoner-1.4b-w8a16`), point `--model-path` at it, and add `--quantization compressed-tensors` to the Terminal 1 command.

> `sglang_reckoner.server` accepts `--harness-bin <path>` (or reads `RECKONER_HARNESS_BIN` from the environment), sets `SGLANG_RECKONER_AUTOLOAD=1`, `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`, and `SGLANG_RECKONER_FUSED_GATES=1` when unset, always sets `RECKONER_TOKENIZER_PATH` to `--tokenizer-path` or `--model-path`, and `supports_inflight_extend()` (`tool_processor.py`) auto-detects `0001-inflight-extend.patch` when `SGLANG_RECKONER_INFLIGHT_EXTEND` is unset, so no environment variables are needed. Set `SGLANG_RECKONER_FUSED_GATES=0` to disable the fused GDN-2 kernels. On the Go front door (`reckoner-harness serve`) and in the Python plugin (`conformance/effort_table.json`), OpenAI-compatible `"reasoning_effort"` maps `"none"` and `"minimal"` to `0` thinking tokens (`<|deliberate|>` masked at the gate, or `logit_bias: -100` in `stop_and_continue` mode, where `--mode stop-and-continue` and `stop_and_continue` are both accepted), `"low"` (alias `"fast"`) to `256` tokens, `"medium"` (or omitted / `""`) to `1,024` tokens, and `"high"` (aliases `"careful"`, `"xhigh"`) to `4,096` tokens (`serve.go`). Unrecognized values return HTTP `400 Bad Request` (`ErrArgs`). The `reckoner-1.4b` profile allows context anchors, which stay off by default unless `serve` is started with `--anchor-context` (or a request sets `"anchor_context": true`). Both `reckoner-1.4b` and `reckoner-q2b` set `"effort_line": "off"` by default (`"none_only"` and `"all"` are opt-in per profile). In 1.1.3, `serve` has no `--help` and ignores unknown flags; see the `serve` flag table in the top-level [README](../../README.md).

### Reckoner-Q2B (`Qwen35GDN2ForCausalLM`, anchor-free context)

Reckoner-Q2B is built on [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) by Alibaba's Qwen team (`Apache-2.0`), retrofitted with GDN-2 and Reckoner control tokens (`248,320` vocabulary). Reckoner-Q2B weights are coming soon.

Profile values in `profiles/reckoner-q2b.json`:
- `default_system`: `"You are Reckoner-Q2B, a model by Maelie Labs."`
- `graded_answer_instruction`: `"Put your final answer within \\boxed{}."`
- `anchors`: `false`
- `tokenizer`: `"tokenizer.json"` (`newline_token_id: 198`, 43 control token IDs in `control_tokens`: `<|endoftext|>` at `248044` and `<|pad|>`..`<|r15|>` at `248070..248111`)

To serve Reckoner-Q2B through `sglang_reckoner.server` and the `reckoner-harness serve` front door:

```bash
# Terminal 1: SGLang token backend (automatically sets RECKONER_TOKENIZER_PATH to --model-path)
python -m sglang_reckoner.server --model-path ./reckoner-q2b --harness-bin "$PWD/reckoner-harness" \
  --host 127.0.0.1 --port 30000 --context-length 66560

# Terminal 2: OpenAI-compatible front door (loads the 248K Qwen3.5 tokenizer.json and control-token table)
./reckoner-harness serve --profile reckoner-q2b --tokenizer ./reckoner-q2b/tokenizer.json \
  --sglang-addr http://127.0.0.1:30000 --host 127.0.0.1 --port 8080
```

On the Go front door (and on CLI `format_prompt` / `tokenize`), pass `--tokenizer ./reckoner-q2b/tokenizer.json` (or `--model-dir ./reckoner-q2b`, or set `RECKONER_TOKENIZER_PATH=./reckoner-q2b`). In the Python plugin, `sglang_reckoner.server` sets `RECKONER_TOKENIZER_PATH` automatically from `--tokenizer-path` or `--model-path`. Set `RECKONER_TOKENIZER_PATH=/path/to/reckoner-q2b` explicitly if you import `sglang_reckoner` outside `sglang_reckoner.server`.

## Running the CPU Unit Test Suite

The plugin includes a CPU-only unit test suite (`tests/test_sglang_reckoner_cpu.py` and `tests/test_sglang_reckoner_quant_cpu.py`) that verifies model registration, numerical parity against golden reference fixtures (`tests/fixtures/`), quantized linear plumbing, front-door enforcement, and tool logit processor behavior without requiring a GPU:

```bash
CUDA_VISIBLE_DEVICES="" pytest integrations/sglang/tests/ -v
```

To also run the optional local-checkpoint and harness-binary integration tests, set:

```bash
export RECKONER_HARNESS_BIN="$PWD/reckoner-harness"
export RECKONER_TEST_MODEL_DIR="/path/to/reckoner-1.4b"
export RECKONER_QWEN35_TEST_MODEL_DIR="/path/to/reckoner-q2b"
CUDA_VISIBLE_DEVICES="" pytest integrations/sglang/tests/ -v
```
