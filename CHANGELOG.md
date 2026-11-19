# Changelog

All notable changes to the Reckoner Harness are documented in this file.

## [1.1.3] - 2026-11-06

First public release.

### Added
- **Tools**: `calc`, `solve`, `lua`, `str`, `date`, `lookup`, `anchor`, `quote`, `quote_mask`, `find`, `lines` and `fncall`, with 16 circular registers (`<|r0|>`..`<|r15|>`). `find` runs normalized exact matching first, then a BM25 ranked fallback, and labels results `[exact]` or `[ranked]`. `date` supports a settable clock (`--clock`) for `now([zone])`.
- **Front door (`reckoner-harness serve`)**: OpenAI-compatible `/v1/chat/completions` server with built-in BPE encoding and in-path tool dispatch.
- **SGLang plugin (`integrations/sglang/`)**: model adapter, server entry point, the SGLang in-flight extend patch and a CPU test suite. The plugin pins `flash-linear-attention==0.6.0` (`fla-core==0.6.0`), so serving needs only the one SGLang patch.
- **Model profiles (`profiles/<name>.json`)**: JSON profiles with `name`, `default_system`, `graded_answer_instruction`, `anchors` and `notes`, selected with `--profile <name|path>` on the CLI and `serve`, or `"profile"` in JSON IPC. With a profile active, `effective_anchors = profile.anchors && !no_anchors && (explicit_anchors != false)`, so a profile with `anchors: false` wins over `anchors: true` or `--anchors`. Sending `"profile": ""` clears the profile and resets the context store's anchors to the CLI default.
- **Built-in profile `reckoner-1.4b`**: `default_system: null`, `graded_answer_instruction: "Put your final answer within \\boxed{}."`, `anchors: true`.
- **Built-in profile `reckoner-q2b`** for an upcoming Qwen3.5-2B retrofit (weights not released yet): anchor-free, loading an external `tokenizer.json` through `--tokenizer` or `--model-dir` / `--model-path` / `--model`.
- **Graded requests (`--graded` / `"graded": true`)**: append `"\n\n" + graded_answer_instruction` to the final user turn. Chat requests are ungraded by default.
- **Anchor-free mode (`--no-anchors`, `{"anchors": false}`)** for models trained without context anchor tokens (`<|mN|>`, `<|Lk|>`).
- **IPC actions**: `format_prompt` / `render_prompt` render chat messages with profile rules and return the prompt and token IDs. `tokenize` (alias `encode`) returns `tokens` and decoded `text` for raw tokenizer checks, without prompt-level escaping.
- **`reasoning_effort` table (`conformance/effort_table.json`)**, shared by the front door (`serve.go`), JSON IPC and the SGLang plugin (`harness_bridge.py`, `tool_processor.py`): `none` and `minimal` mask `<|deliberate|>` (`thinking_budget: 0`), while `low`, `medium` and `high` allow it with `thinking_budget` 256, 1024 and 4096. Aliases: `fast` -> `low`, `careful` and `xhigh` -> `high`. Empty or omitted means `medium`. The table's `loop_cap` field (the latent depth-loop cap `K_max`) is reserved and not enforced. Tool calls are limited by the session `max_tool_calls` (default `16`).
- **Profile field `effort_line`** (`"off"`, `"none_only"`, `"all"`, defaulting to `"off"` in both built-in profiles). `"none_only"` writes `reasoning_effort: none` into the first `system` turn for `none` and `minimal` (see `conformance/effort_none_prefix.json`). `"all"` writes `reasoning_effort: <name>` for every level, with `minimal` written as `none`. The line goes at the start of the first `system` turn, after `ApplyProfileToMessages` and before any `<|tools|>...<|/tools|>` block, and is skipped if the turn already starts with `reasoning_effort:`.
- **Conformance suite**: 1,672 byte-exact cases (1,257 tool, 352 function calling, 49 context reference and 14 model profile cases).

### Behavior
- **Local by default**: `reckoner-harness serve` listens on `127.0.0.1` unless `--host` says otherwise. Pass `--host 0.0.0.0` to accept connections from other machines.
- **Special-token escaping**: prompt rendering and tool observations escape `<|` and the tokenizer's registered angle-bracket tokens in every client field and tool body, in every profile and with no profile. Inside error observations (`ErrUnsetRegister`, `ErrNotFound`, `ErrOutOfBounds`, `ErrSyntax`), `reg list` entries after the first prefix, and `external`/`fncall` bodies, `<|...|>` tags are written as escaped text. A `find` or `lines` hit on the `<|tools|>...<|/tools|>` line in the `system` turn returns escaped `<\|tools|>` and `<\|/tools|>` text.
- **Pre-formatted assistant turns** may hold only agent-turn control tokens. Turn, system/tools and end-of-text tokens, and other special tokens outside the agent set, are rejected with `ErrArgs`.
- **Fail-closed `reasoning_effort` validation**: an unrecognized value returns HTTP `400 Bad Request` (`ErrArgs`, `text/plain`) before any SSE bytes on `/v1/chat/completions`, and `ErrArgs` on IPC `format_prompt` / `render_prompt`. When called without the front door, `ReckonerToolLogitProcessor` logs a one-time warning per unknown value and falls back to `medium`.
- **`stop_and_continue` mode** (`--mode stop-and-continue` or `stop_and_continue`): `none` and `minimal` ban `<|deliberate|>` with `logit_bias: -100` in `sampling_params` on every round, without `custom_logit_processor`. `low`, `medium` and `high` run without `custom_logit_processor` and log a one-time warning per process that `thinking_budget` is not enforced in this mode.
- **Fail-closed tokenizer loading in the Python bridge**: `sglang_reckoner.harness_bridge.escape_control_tokens` raises an error when a configured tokenizer path or `RECKONER_TOKENIZER_PATH` is whitespace, missing, unreadable, or lacks `"added_tokens"`.
