#!/usr/bin/env python3
"""Verify exact byte-level BPE and control-token parity between Hugging Face
`tokenizers` (Rust) and the Go `reckoner-harness` tokenizer for Reckoner-Q2B.

Checks across >= 20,000 strings covering 7 categories:
  1. eval_and_sft_prompts_completions
  2. math_and_latex
  3. code_snippets
  4. numbers_and_dates
  5. unicode_multilingual_emoji_rtl
  6. whitespace_edge_cases
  7. control_tokens_and_lookalikes

For every string s in the corpus, asserts:
  - Go encode(s) == HF encode(s, add_special_tokens=True).ids
  - Go decode(encode(s)) == HF decode(encode(s), skip_special_tokens=False)
"""

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer


CONTROL_TOKENS = [
    "<|endoftext|>",
    "<|pad|>",
    "<|turn:user|>",
    "<|turn:agent|>",
    "<|/turn|>",
    "<|turn:system|>",
    "<|tools|>",
    "<|/tools|>",
    "<|deliberate|>",
    "<|act|>",
    "<|ready|>",
    "<|clarify|>",
    "<|pause|>",
    "<|call|>",
    "<|/call|>",
    "<|obs|>",
    "<|obs:err|>",
    "<|/obs|>",
    "<|call:calc|>",
    "<|call:solve|>",
    "<|call:lua|>",
    "<|r0|>",
    "<|r1|>",
    "<|r2|>",
    "<|r3|>",
    "<|r4|>",
    "<|r5|>",
    "<|r6|>",
    "<|r7|>",
    "<|call:str|>",
    "<|call:date|>",
    "<|call:lookup|>",
    "<|call:find|>",
    "<|call:tools|>",
    "<|call:def|>",
    "<|r8|>",
    "<|r9|>",
    "<|r10|>",
    "<|r11|>",
    "<|r12|>",
    "<|r13|>",
    "<|r14|>",
    "<|r15|>",
]

QWEN_EXTRA_SPECIALS = [
    "<|im_start|>",
    "<|im_end|>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
    "<think>",
    "</think>",
]


def load_extra_jsonl(paths, max_items=3500):
    items = []
    if not paths:
        return items
    existing = [Path(p) for p in paths if Path(p).exists()]
    if not existing:
        return items
    per_file_cap = max(500, (max_items // len(existing)) + 200)
    for path in existing:
        file_items = 0
        with path.open("r", encoding="utf-8", errors="replace") as f:
            first_char = ""
            while True:
                ch = f.read(1)
                if not ch or not ch.isspace():
                    first_char = ch
                    break
            f.seek(0)
            if first_char == "[":
                try:
                    arr = json.load(f)
                    for row in arr:
                        if isinstance(row, dict):
                            for k in ("problem", "question", "prompt", "input", "solution", "answer", "rendered", "trajectory"):
                                if k in row and isinstance(row[k], str) and row[k]:
                                    items.append(row[k])
                                    file_items += 1
                                    if file_items >= per_file_cap:
                                        break
                        if file_items >= per_file_cap:
                            break
                except Exception:
                    pass
            else:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                        if isinstance(row, dict):
                            for k in ("problem", "question", "prompt", "input", "text", "completion", "output", "response", "rendered", "trajectory", "trace"):
                                if k in row and isinstance(row[k], str) and row[k]:
                                    items.append(row[k])
                                    file_items += 1
                            if "messages" in row and isinstance(row["messages"], list):
                                for m in row["messages"]:
                                    if isinstance(m, dict) and isinstance(m.get("content"), str):
                                        items.append(m["content"])
                                        file_items += 1
                            if "turns" in row and isinstance(row["turns"], list):
                                for m in row["turns"]:
                                    if isinstance(m, dict) and isinstance(m.get("content"), str):
                                        items.append(m["content"])
                                        file_items += 1
                    except Exception:
                        pass
                    if file_items >= per_file_cap:
                        break
        if len(items) >= max_items:
            break
    return items[:max_items]


def build_corpus(seed: int = 42, extra_paths=None):
    rng = random.Random(seed)
    corpus = []  # list of (category, text)

    # 1. Prompts & completions (eval suites + SFT trajectories + multi-turn tool traces)
    if extra_paths:
        for s in load_extra_jsonl(extra_paths, max_items=3500):
            corpus.append(("eval_and_sft_prompts_completions", s))

    verbs = ["Compute", "Find", "Evaluate", "Determine", "Solve for x in", "Simplify", "Calculate"]
    nouns = ["the sum", "the product", "the remainder", "the exact value", "the derivative", "the integral", "the greatest common divisor"]
    contexts = [
        "If a store sells {a} items at ${b}.{c:02d} each and offers a {d}% discount, what is the total cost?",
        "A train travels {a} km in {b} hours, then {c} km at {d} km/h. Find the average speed.",
        "<|turn:system|>\nYou are Reckoner-Q2B, a model by Maelie Labs.<|/turn|>\n<|turn:user|>\n{v} {n} of {a} and {b}.\n\nPut your final answer within \\boxed{{}}.<|/turn|>\n<|turn:agent|><|act|><|call:calc|>{a} * {b} + {c}<|/call|><|obs|><|r0|>={res}<|/obs|><|ready|>The answer is \\boxed{{{res}}}.<|/turn|>",
        "<|turn:user|>\nSolve {a}*x + {b} = {c}.<|/turn|>\n<|turn:agent|><|act|><|call:solve|>{a}*x + {b} = {c}, x<|/call|><|obs|>x=<|r0|>={res}<|/obs|><|ready|>\\boxed{{{res}}}<|/turn|>",
        "<|turn:user|>\nWrite a short note explaining why {a} + {b} equals {res}, isn't it?<|/turn|>\n<|turn:agent|><|deliberate|>We'll check {a}+{b}={res}. They're equal.<|ready|>Yes, it's \\boxed{{{res}}}.<|/turn|>",
    ]
    while sum(1 for c, _ in corpus if c == "eval_and_sft_prompts_completions") < 3500:
        a = rng.randint(-5000, 50000)
        b = rng.randint(1, 9999)
        c = rng.randint(0, 99)
        d = rng.randint(1, 75)
        v = rng.choice(verbs)
        n = rng.choice(nouns)
        tmpl = rng.choice(contexts)
        corpus.append((
            "eval_and_sft_prompts_completions",
            tmpl.format(a=a, b=b, c=c, d=d, v=v, n=n, res=a * b + c),
        ))

    # 2. Math & LaTeX (\boxed{...}, \frac{a}{b}, Greek letters, subscripts, matrices)
    greek = ["\\alpha", "\\beta", "\\gamma", "\\delta", "\\epsilon", "\\zeta", "\\eta", "\\theta", "\\lambda", "\\mu", "\\pi", "\\sigma", "\\phi", "\\omega", "α", "β", "γ", "δ", "π", "∑", "∏", "∫", "√", "∞", "≤", "≥", "≠", "±", "∈", "⊂", "∪", "∩"]
    latex_tmpls = [
        "$$\\frac{{{a}}}{{{b}}} + \\sqrt{{{c}}} = \\boxed{{\\frac{{{a}\\sqrt{{{d}}}}}{{{b}}}}}$$",
        "Let $x_{{{i}}} = {g}_{{{j}}}^{{{k}}} \\cdot \\binom{{{a}}}{{{b}}}$ for $i \\in \\{{1, \\dots, {c}\\}}$.",
        "\\begin{{pmatrix}} {a} & {b} \\\\ {c} & {d} \\end{{pmatrix}} \\begin{{pmatrix}} x_1 \\\\ x_2 \\end{{pmatrix}} = \\boxed{{{g}}}",
        "Given $f(x) = \\int_{{0}}^{{\\infty}} e^{{-{a}t}} t^{{{b}-1}}\\,dt = \\frac{{\\Gamma({b})}}{{{a}^{{{b}}}}}$, evaluate $\\boxed{{{c}/{d}}}$.",
        "Subscripts and superscripts: a_0 + a_1 x + a_2 x^2 + x₁ + x₂ + x₃ = y⁰ + y¹ + y².",
    ]
    for _ in range(3200):
        t = rng.choice(latex_tmpls).format(
            a=rng.randint(-999, 999),
            b=rng.randint(1, 999),
            c=rng.randint(2, 500),
            d=rng.randint(2, 99),
            i=rng.randint(0, 20),
            j=rng.randint(0, 20),
            k=rng.randint(1, 10),
            g=rng.choice(greek),
        )
        corpus.append(("math_and_latex", t))

    # 3. Code snippets (Python, Lua, Go, JSON, shell)
    code_tmpls = [
        "def solve_{n}(x: int, y: float = {f:.4f}) -> dict:\n    # Calculate result\n    items = [i ** 2 for i in range({a}) if i % {b} == 0]\n    return {{'sum': sum(items), 'msg': f\"don't stop at {{len(items)}}\"}}\n",
        "local acc = 0\nfor i = 1, {a}, {b} do\n  if i % {c} == 0 then\n    acc = acc + (i * {d}) // {b}\n  end\nend\nreturn acc\n",
        "func compute{n}(vals []int) (int, error) {{\n\tif len(vals) == 0 {{\n\t\treturn 0, fmt.Errorf(\"empty slice at %d\", {a})\n\t}}\n\treturn vals[0] + {b}, nil\n}}\n",
        '{{"id": "case_{n}", "values": [{a}, {b}, -{c}, {f}], "flags": {{"active": true, "mode": "strict"}}, "expr": "<|r0|> + {a}"}}',
        "CUDA_VISIBLE_DEVICES=\"\" bash -c 'for f in *.jsonl; do wc -l \"$f\" | awk \"{{print \\$1 + {a}}}\"; done' && echo \"done {n}\"",
    ]
    for idx in range(3200):
        corpus.append((
            "code_snippets",
            rng.choice(code_tmpls).format(
                n=idx,
                a=rng.randint(1, 10000),
                b=rng.randint(1, 99),
                c=rng.randint(2, 37),
                d=rng.randint(-500, 500),
                f=rng.uniform(-1e4, 1e4),
            ),
        ))

    # 4. Long numbers, decimals, negative numbers, scientific notation, dates
    num_tmpls = [
        "{big1} * {big2} = {prod}",
        "-{big1}.{frac1} + {big2}.{frac2}e-{exp} / -0.000{frac1}E+{exp2}",
        "2026-{m:02d}-{day:02d}T{h:02d}:{mi:02d}:{s:02d}Z -> add_days(2026-{m:02d}-{day:02d}, -{big2})",
        "0x{big1:x}, 0b{day:b}, {big1:,}, ${big2:,.2f}, -{exp}%",
    ]
    for _ in range(3200):
        big1 = rng.randint(10**6, 10**18)
        big2 = rng.randint(10**3, 10**12)
        corpus.append((
            "numbers_and_dates",
            rng.choice(num_tmpls).format(
                big1=big1,
                big2=big2,
                prod=big1 * big2,
                frac1=rng.randint(1000, 99999999),
                frac2=rng.randint(10, 999999),
                exp=rng.randint(1, 308),
                exp2=rng.randint(1, 100),
                m=rng.randint(1, 12),
                day=rng.randint(1, 28),
                h=rng.randint(0, 23),
                mi=rng.randint(0, 59),
                s=rng.randint(0, 59),
            ),
        ))

    # 5. Unicode: CJK, emoji, combining marks (including >30 non-starters / Zalgo without Stream-Safe CGJ), right-to-left text, accented NFC/NFD
    unicode_samples = [
        "计算：{a} + {b} = {c}。请将最终答案写在 \\boxed{{{c}}} 中。",
        "日本語のテスト：{a}個のりんごと{b}個のみかんがあります。合計はいくつですか？",
        "한국어 테스트: {a} 더하기 {b}는 얼마입니까? 정답: {c}",
        "العربية: احسب ناتج {a} + {b} = {c} في المعادلة التالية.",
        "עברית: חשב את הסכום של {a} ו-{b} והצג את התוצאה {c}.",
        "Emoji math 🚀🔢: {a} 🍎 + {b} 🍊 = {c} 🎉! Family: 👨‍👩‍👧‍👦, flags: 🏳️‍🌈 🇺🇸 🇯🇵, skin tones: 👍🏽 👩🏾‍🔬.",
        # Combining marks (NFD inputs that NFC normalizes or partially composes)
        "Cafe\u0301 vs Café, na\u00efve vs nai\u0308ve, A\u030angstro\u0308m, e\u0301\u0327, o\u0302\u0323, Z\u0335a\u0300l\u0316g\u0321o\u0308 {a}",
        "Кириллица: Вычислите значение выражения {a} * {b} - {c}, й vs и\u0306, ё vs е\u0308.",
        "Mixed script contraction: Café's menu, टीचर's book, ת'א, l'État, d'abord, 'tis, 'twas, 'S, 'T, 'RE, 'VE, 'M, 'LL, 'D ({a}).",
    ]
    # Explicit >30 combining marks and Zalgo text (verifying no Stream-Safe U+034F CGJ insertion up to 500 marks):
    combining_pool = [
        "\u0300", "\u0301", "\u0302", "\u0303", "\u0304", "\u0306", "\u0307", "\u0308",
        "\u030a", "\u030b", "\u030c", "\u0316", "\u0321", "\u0323", "\u0327", "\u0328",
        "\u0335", "\u0336", "\u0591", "\u05b0", "\u064b", "\u0651", "\u093c",
    ]
    for n_marks in [29, 30, 31, 32, 35, 50, 64, 100, 150, 250, 500]:
        corpus.append(("unicode_multilingual_emoji_rtl", "\u0301" * n_marks))
        corpus.append(("unicode_multilingual_emoji_rtl", "a" + ("\u0301" * n_marks)))
        corpus.append(("unicode_multilingual_emoji_rtl", "e" + ("\u0301\u0323\u0302\u0327" * ((n_marks // 4) + 1))[:n_marks]))
        mixed_marks = "".join(rng.choice(combining_pool) for _ in range(n_marks))
        corpus.append(("unicode_multilingual_emoji_rtl", f"Z{mixed_marks}a{mixed_marks}l{mixed_marks}g{mixed_marks}o"))
    # Hangul Jamo + >30 combining marks and explicit U+034F in input:
    corpus.append(("unicode_multilingual_emoji_rtl", "\u1100\u1161\u11a8" + ("\u0301\u0323" * 20)))
    corpus.append(("unicode_multilingual_emoji_rtl", "a\u034f" + ("\u0301" * 40)))
    while sum(1 for c, _ in corpus if c == "unicode_multilingual_emoji_rtl") < 3200:
        a = rng.randint(1, 9999)
        b = rng.randint(1, 9999)
        corpus.append((
            "unicode_multilingual_emoji_rtl",
            rng.choice(unicode_samples).format(a=a, b=b, c=a + b),
        ))

    # 6. Whitespace edge cases: leading/trailing spaces, multiple spaces, \r\n, tabs, empty strings, NEL/NBSP
    corpus.append(("whitespace_edge_cases", ""))
    ws_atoms = [" ", "  ", "   ", "    ", "\t", "\t\t", "\n", "\n\n", "\r\n", "\r\n\r\n", "\r", "\r\r", " \n ", " \r\n\t ", "\u0085", "\u00a0", "\u2003", "\u3000"]
    words = ["hello", "world", "123", "x=1", "a+b", "'s", "!", "?", "---", "<|act|>", "foo_bar"]
    for w in ws_atoms:
        corpus.append(("whitespace_edge_cases", w))
    while sum(1 for c, _ in corpus if c == "whitespace_edge_cases") < 2500:
        parts = [rng.choice(ws_atoms if rng.random() < 0.6 else words) for _ in range(rng.randint(1, 10))]
        corpus.append(("whitespace_edge_cases", "".join(parts)))

    # 7. Every control-token string, and text that looks like a control token
    for ct in CONTROL_TOKENS + QWEN_EXTRA_SPECIALS:
        corpus.append(("control_tokens_and_lookalikes", ct))
        corpus.append(("control_tokens_and_lookalikes", f" {ct} "))
        corpus.append(("control_tokens_and_lookalikes", f"{ct}\n{ct}"))

    lookalikes = [
        "<|act|>",
        "</|ready|>",
        "<|r0",
        "<|r16|>",
        "<|r99|>",
        "<|call:unknown|>",
        "<|call:external|>",
        "<|turn:assistant|>",
        "<|obs:ok|>",
        "<|quote|>",
        "<|/quote|>",
        "<|m0|>",
        "<|L0|>",
        "<|m255|><|L1023|>",
        "<|",
        "|>",
        "<| |>",
        "<|act",
        "act|>",
        "<<|act|>>",
        "<|call:calc|>2+2<|/call|><|obs|><|r0|>=4<|/obs|><|ready|>\\boxed{4}<|/turn|>",
        "<tool_call>{\"name\": \"calc\"}</tool_call>",
        "<tool_call",
        "</think>",
        "<think>let's think</think>",
        "<tool_response>ok</tool_response>",
    ]
    for lk in lookalikes:
        corpus.append(("control_tokens_and_lookalikes", lk))

    while sum(1 for c, _ in corpus if c == "control_tokens_and_lookalikes") < 2500:
        tok1 = rng.choice(CONTROL_TOKENS + lookalikes)
        tok2 = rng.choice(CONTROL_TOKENS + lookalikes)
        mid = rng.choice(["", " ", "\n", " 42 * 19 ", "<|", "|>", "<|r0", "</|ready|>", " 's "])
        corpus.append(("control_tokens_and_lookalikes", f"{tok1}{mid}{tok2}"))

    return corpus


def extract_non_pipe_angle_added_tokens(hf_tok: Tokenizer):
    data = json.loads(hf_tok.to_str())
    raw_added = data.get("added_tokens", [])
    tags = [
        at["content"]
        for at in raw_added
        if isinstance(at, dict)
        and isinstance(at.get("content"), str)
        and at["content"].startswith("<")
        and not at["content"].startswith("<|")
    ]
    return sorted(tags, key=lambda s: (-len(s), s))


def escape_control_tokens_py(s: str, non_pipe_angle_added_tokens=None, hf_tok: Tokenizer = None) -> str:
    if non_pipe_angle_added_tokens is None and hf_tok is not None:
        non_pipe_angle_added_tokens = extract_non_pipe_angle_added_tokens(hf_tok)
    esc = s.replace("<|", "<\\|")
    if "<" in esc and non_pipe_angle_added_tokens:
        for tag in non_pipe_angle_added_tokens:
            if tag in esc:
                esc = esc.replace(tag, "<\\" + tag[1:])
    return esc


def render_q2b_prompt_hf(
    hf_tok: Tokenizer,
    messages,
    graded: bool = False,
    non_pipe_angle_added_tokens=None,
):
    if non_pipe_angle_added_tokens is None:
        non_pipe_angle_added_tokens = extract_non_pipe_angle_added_tokens(hf_tok)
    default_sys = "You are Reckoner-Q2B, a model by Maelie Labs."
    graded_instr = "Put your final answer within \\boxed{}."
    msgs = [dict(m) for m in messages]
    if not any(m.get("role") == "system" for m in msgs):
        msgs.insert(0, {"role": "system", "content": default_sys})
    if graded:
        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i].get("role") == "user":
                msgs[i] = dict(msgs[i])
                msgs[i]["content"] = msgs[i].get("content", "") + "\n\n" + graded_instr
                break
    ids = []
    for idx, m in enumerate(msgs):
        if idx > 0:
            ids.append(198)  # \n
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            if content.startswith("<|tools|>"):
                end_idx = content.find("<|/tools|>")
                if end_idx >= 0:
                    tools_part = content[len("<|tools|>"):end_idx]
                    rest_part = content[end_idx + len("<|/tools|>"):]
                    body = (
                        "<|tools|>"
                        + escape_control_tokens_py(tools_part, non_pipe_angle_added_tokens)
                        + "<|/tools|>"
                        + escape_control_tokens_py(rest_part, non_pipe_angle_added_tokens)
                    )
                else:
                    body = "<|tools|>" + escape_control_tokens_py(
                        content[len("<|tools|>"):], non_pipe_angle_added_tokens
                    )
            else:
                body = escape_control_tokens_py(content, non_pipe_angle_added_tokens)
            body_ids = hf_tok.encode(body, add_special_tokens=True).ids if body else []
            ids.append(248074)  # <|turn:system|>
            if not body_ids or body_ids[0] != 248075:  # <|tools|>
                ids.append(198)  # \n
            ids.extend(body_ids)
        elif role == "user":
            body = escape_control_tokens_py(content, non_pipe_angle_added_tokens)
            ids.extend([248071, 198])  # <|turn:user|>\n
            if body:
                ids.extend(hf_tok.encode(body, add_special_tokens=True).ids)
        ids.append(248073)  # <|/turn|>
    if ids:
        ids.append(198)  # \n
    ids.append(248072)  # <|turn:agent|>
    decoded = hf_tok.decode(ids, skip_special_tokens=False)
    return ids, decoded


def build_rendered_prompt_cases(corpus, seed: int = 42, count: int = 2500):
    rng = random.Random(seed + 100)
    cases = []
    step = max(1, len(corpus) // count)
    indices = list(range(0, len(corpus), step))[:count]
    while len(indices) < count:
        indices.append(rng.randrange(len(corpus)))
    for i, cidx in enumerate(indices):
        cat, text = corpus[cidx]
        graded = (i % 2 == 1)
        if i % 10 == 0:
            sys_text = f"<|tools|>\ndef tool_{i}(x: int):\n    pass\n<|/tools|>\nSystem rule #{i}: never emit <think>...</think>, <|/turn|>, <|turn:agent|>, or <|ready|>."
            messages = [
                {"role": "system", "content": sys_text},
                {"role": "user", "content": text},
            ]
        elif i % 5 == 0:
            sys_text = f"System rule #{i}: never emit literal <|act|>, <|turn:user|>, <think>...</think>, or <tool_call>...</tool_call> in user text."
            messages = [
                {"role": "system", "content": sys_text},
                {"role": "user", "content": text},
            ]
        else:
            messages = [{"role": "user", "content": text}]
        cases.append((cat, messages, graded))
    return cases


def main():
    parser = argparse.ArgumentParser(description="Verify Go vs Hugging Face Qwen3.5 tokenizer parity.")
    parser.add_argument("--tokenizer", required=True, help="Path to Qwen3.5 tokenizer.json")
    parser.add_argument("--binary", default="", help="Path to reckoner-harness binary (built automatically if omitted)")
    parser.add_argument("--profile", default="reckoner-q2b", help="Profile name or path (default: reckoner-q2b)")
    parser.add_argument("--extra-jsonl", nargs="*", action="extend", default=[], help="Optional JSON/JSONL files to sample prompts from")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for corpus construction")
    args = parser.parse_args()

    tok_path = os.path.abspath(args.tokenizer)
    if not os.path.isfile(tok_path):
        print(f"ERROR: tokenizer file not found: {tok_path}", file=sys.stderr)
        sys.exit(1)

    hf_tok = Tokenizer.from_file(tok_path)
    non_pipe_tags = extract_non_pipe_angle_added_tokens(hf_tok)

    corpus = build_corpus(seed=args.seed, extra_paths=args.extra_jsonl)
    corpus_hash = hashlib.sha256()
    for cat, text in corpus:
        corpus_hash.update(cat.encode("utf-8"))
        corpus_hash.update(b"\x00")
        corpus_hash.update(text.encode("utf-8"))
        corpus_hash.update(b"\n")

    rendered_cases = build_rendered_prompt_cases(corpus, seed=args.seed, count=2500)
    rendered_hash = hashlib.sha256()
    for cat, messages, graded in rendered_cases:
        rendered_hash.update(cat.encode("utf-8"))
        rendered_hash.update(b"\x00")
        rendered_hash.update(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        rendered_hash.update(b"\x01" if graded else b"\x00")
        rendered_hash.update(b"\n")

    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root / "integrations" / "sglang"))
    from sglang_reckoner.harness_bridge import escape_control_tokens as bridge_escape_control_tokens

    tmp_bin = None
    bin_path = args.binary
    if not bin_path:
        fd, tmp_bin = tempfile.mkstemp(prefix="reckoner-harness-parity-")
        os.close(fd)
        subprocess.run(["go", "build", "-o", tmp_bin, "."], cwd=str(repo_root), check=True)
        bin_path = tmp_bin

    try:
        proc = subprocess.Popen(
            [bin_path, "--profile", args.profile, "--tokenizer", tok_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        cat_counts = Counter()
        id_mismatches = []
        decode_mismatches = []
        bridge_escape_mismatches = []

        for idx, (cat, text) in enumerate(corpus):
            cat_counts[cat] += 1
            py_esc = escape_control_tokens_py(text, non_pipe_tags)
            br_esc = bridge_escape_control_tokens(text, tokenizer_path=tok_path)
            if py_esc != br_esc:
                bridge_escape_mismatches.append((idx, cat, text, py_esc, br_esc))

            hf_ids = hf_tok.encode(text, add_special_tokens=True).ids
            hf_dec = hf_tok.decode(hf_ids, skip_special_tokens=False)

            req_line = json.dumps({"tool": "tokenize", "payload": text}, ensure_ascii=False) + "\n"
            proc.stdin.write(req_line)
            proc.stdin.flush()
            resp_line = proc.stdout.readline()
            if not resp_line:
                err_out = proc.stderr.read()
                raise RuntimeError(f"Harness EOF at index {idx}: {err_out}")
            resp = json.loads(resp_line)
            if not resp.get("success"):
                raise RuntimeError(f"Harness error at index {idx}: {resp.get('error')}")

            go_ids = resp.get("tokens") or []
            go_dec = resp.get("text", "")

            if go_ids != hf_ids:
                id_mismatches.append((idx, cat, text, hf_ids, go_ids))
            if go_dec != hf_dec:
                decode_mismatches.append((idx, cat, text, hf_dec, go_dec))

        rendered_cat_counts = Counter()
        rendered_id_mismatches = []
        rendered_str_mismatches = []

        for ridx, (cat, messages, graded) in enumerate(rendered_cases):
            rendered_cat_counts[cat] += 1
            expected_ids, expected_str = render_q2b_prompt_hf(
                hf_tok, messages, graded=graded, non_pipe_angle_added_tokens=non_pipe_tags
            )

            req_line = json.dumps({
                "tool": "format_prompt",
                "profile": args.profile,
                "tokenizer": tok_path,
                "graded": graded,
                "messages": messages,
            }, ensure_ascii=False) + "\n"
            proc.stdin.write(req_line)
            proc.stdin.flush()
            resp_line = proc.stdout.readline()
            if not resp_line:
                err_out = proc.stderr.read()
                raise RuntimeError(f"Harness EOF at rendered index {ridx}: {err_out}")
            resp = json.loads(resp_line)
            if not resp.get("success"):
                raise RuntimeError(f"Harness error at rendered index {ridx}: {resp.get('error')}")

            go_r_ids = resp.get("tokens") or []
            go_r_str = resp.get("result", "")

            if go_r_ids != expected_ids:
                rendered_id_mismatches.append((ridx, cat, messages, expected_ids, go_r_ids))
            if go_r_str != expected_str:
                rendered_str_mismatches.append((ridx, cat, messages, expected_str, go_r_str))

        # Layer 3: effort_none_prefix.json conformance check (when effort_line="none_only" and default_system=null)
        none_prefix_path = repo_root / "conformance" / "effort_none_prefix.json"
        none_prefix_mismatches = []
        if none_prefix_path.is_file():
            none_prefix_fix = json.loads(none_prefix_path.read_text(encoding="utf-8"))
            want_prefix_str = none_prefix_fix["prefix_text"]
            want_prefix_ids = none_prefix_fix["qwen35_prefix_token_ids"]
            q2b_base = json.loads((repo_root / "profiles" / "reckoner-q2b.json").read_text(encoding="utf-8"))
            q2b_base["name"] = "q2b-none-only"
            q2b_base["default_system"] = None
            q2b_base["effort_line"] = "none_only"
            fd_p, tmp_prof = tempfile.mkstemp(prefix="q2b-none-only-", suffix=".json")
            os.close(fd_p)
            try:
                Path(tmp_prof).write_text(
                    json.dumps(q2b_base),
                    encoding="utf-8",
                )
                for req_eff in ("none", "minimal"):
                    req_line = json.dumps({
                        "tool": "format_prompt",
                        "profile": tmp_prof,
                        "tokenizer": tok_path,
                        "reasoning_effort": req_eff,
                        "messages": [{"role": "user", "content": "What is 2+2?"}],
                    }, ensure_ascii=False) + "\n"
                    proc.stdin.write(req_line)
                    proc.stdin.flush()
                    resp = json.loads(proc.stdout.readline())
                    if not resp.get("success"):
                        none_prefix_mismatches.append((req_eff, "error", resp.get("error")))
                    else:
                        got_ids = (resp.get("tokens") or [])[:len(want_prefix_ids)]
                        got_str = resp.get("result", "")
                        if got_ids != want_prefix_ids or not got_str.startswith(want_prefix_str):
                            none_prefix_mismatches.append((req_eff, want_prefix_ids, got_ids, got_str))
            finally:
                if os.path.exists(tmp_prof):
                    os.remove(tmp_prof)

        proc.stdin.close()
        proc.wait(timeout=5)
    finally:
        if tmp_bin and os.path.exists(tmp_bin):
            os.remove(tmp_bin)

    print("=== Layer 1: Raw Encode / DecodeTokens Parity (tokenize IPC vs Hugging Face tokenizers) ===")
    print(f"Total corpus strings: {len(corpus)}")
    print(f"Corpus SHA-256: {corpus_hash.hexdigest()}")
    print("Category breakdown:")
    for cat, cnt in sorted(cat_counts.items()):
        print(f"  {cat}: {cnt}")
    print(f"Token ID sequence mismatches: {len(id_mismatches)}")
    print(f"Decode(encode(s)) mismatches: {len(decode_mismatches)}")
    print(f"harness_bridge vs qwen_tokenizer_parity escape mismatches: {len(bridge_escape_mismatches)}")

    print("\n=== Layer 2: Rendered Prompt Parity (format_prompt IPC with escapeControlTokens vs Hugging Face tokenizers) ===")
    print(f"Total rendered chat prompts: {len(rendered_cases)}")
    print(f"Rendered corpus SHA-256: {rendered_hash.hexdigest()}")
    print("Rendered category breakdown:")
    for cat, cnt in sorted(rendered_cat_counts.items()):
        print(f"  {cat}: {cnt}")
    print(f"Rendered token ID sequence mismatches: {len(rendered_id_mismatches)}")
    print(f"Rendered prompt string mismatches: {len(rendered_str_mismatches)}")
    print(f"effort_none_prefix.json mismatches: {len(none_prefix_mismatches)}")

    if (
        id_mismatches
        or decode_mismatches
        or bridge_escape_mismatches
        or rendered_id_mismatches
        or rendered_str_mismatches
        or none_prefix_mismatches
    ):
        for idx, cat, text, want, got in id_mismatches[:10]:
            print(f"\n[RAW ID MISMATCH #{idx} ({cat})] text={text!r}\n  HF: {want}\n  Go: {got}", file=sys.stderr)
        for idx, cat, text, want, got in decode_mismatches[:10]:
            print(f"\n[RAW DECODE MISMATCH #{idx} ({cat})] text={text!r}\n  HF: {want!r}\n  Go: {got!r}", file=sys.stderr)
        for idx, cat, text, want, got in bridge_escape_mismatches[:10]:
            print(f"\n[BRIDGE ESCAPE MISMATCH #{idx} ({cat})] text={text!r}\n  parity: {want!r}\n  bridge: {got!r}", file=sys.stderr)
        for idx, cat, msgs, want, got in rendered_id_mismatches[:10]:
            print(f"\n[RENDERED ID MISMATCH #{idx} ({cat})] msgs={msgs!r}\n  HF: {want}\n  Go: {got}", file=sys.stderr)
        for idx, cat, msgs, want, got in rendered_str_mismatches[:10]:
            print(f"\n[RENDERED STR MISMATCH #{idx} ({cat})] msgs={msgs!r}\n  HF: {want!r}\n  Go: {got!r}", file=sys.stderr)
        for item in none_prefix_mismatches[:10]:
            print(f"\n[EFFORT NONE PREFIX MISMATCH] {item!r}", file=sys.stderr)
        sys.exit(1)

    print("PARITY VERIFIED: 0 mismatches across both raw Encode and rendered prompts.")


if __name__ == "__main__":
    main()


