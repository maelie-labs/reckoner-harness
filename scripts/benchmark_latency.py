#!/usr/bin/env python3
"""
Latency microbenchmark for Reckoner Harness.
Measures per-call latency in microseconds (µs) across all native tools via stdio streaming IPC.
Generates exact percentile distributions (p50, p90, p99) used in README and SPEC.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from typing import Any, Dict, List


def get_cpu_model() -> str:
    try:
        if os.path.exists("/proc/cpuinfo"):
            cores = set()
            with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                phys_id = "0"
                for line in f:
                    if line.startswith("physical id"):
                        phys_id = line.split(":", 1)[1].strip()
                    elif line.startswith("core id"):
                        cores.add((phys_id, line.split(":", 1)[1].strip()))
            if cores:
                return f"{len(cores)}-core {platform.machine()} desktop CPU"
    except Exception:
        pass
    return f"{platform.machine() or 'Unknown'} CPU"


def get_go_version() -> str:
    try:
        res = subprocess.run(["go", "version"], stdout=subprocess.PIPE, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "Unknown Go"


def get_binary_sha256(binary_path: str) -> str:
    try:
        h = hashlib.sha256()
        with open(binary_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


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
            bufsize=1,
        )

    def dispatch(self, tool: str, payload: str) -> Dict[str, Any]:
        req = json.dumps({"tool": tool, "payload": payload})
        self.proc.stdin.write(req + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            err = self.proc.stderr.read()
            raise RuntimeError(f"Harness process terminated. stderr: {err}")
        return json.loads(line.strip())

    def reset(self):
        return self.dispatch("reset", "")

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.terminate()
            self.proc.wait(timeout=1.0)
        except Exception:
            self.proc.kill()


BENCHMARK_SUITE = [
    # tool, payload, category
    ("calc", "482 * 19 - 14", "arithmetic"),
    ("calc", "sqrt(144) * (2^3)", "arithmetic_powers"),
    ("calc", "expand((x+1)*(x+2))", "polynomial_expand"),
    ("calc", "factor(x^2 - 5*x + 6)", "polynomial_factor"),
    ("calc", "det([[1, 2], [3, 4]])", "linalg_det"),
    ("calc", "dist([1, 2], [4, 6])", "geometry_dist"),
    ("calc", 'convert(1000, "m", "km")', "units_convert"),
    ("calc", "sum(k, k, 1, 100)", "sums"),
    ("solve", "3*x + 10 = 25, x", "solve_linear"),
    ("solve", "x^2 - 4 = 0, x", "solve_quadratic"),
    ("solve", "x^2 - 16 > 0", "solve_inequality"),
    ("lua", "local x = 25 * 4; return x", "lua_eval"),
    ("lua", "local s = 0; for i=1,10 do s = s + i end; return s", "lua_loop"),
    ("str", 'len("The quick brown fox jumps over the lazy dog")', "str_len"),
    ("str", 'slice("reproducibility", 0, 7)', "str_slice"),
    ("str", 'replace("hello world", "world", "reckoner")', "str_replace"),
    ("date", 'add("2024-01-31", 1, "month")', "date_add"),
    ("date", 'diff("2024-01-01", "2024-12-31")', "date_diff"),
    ("date", 'day_of_week("2026-10-10")', "date_weekday"),
    ("lookup", 'lookup("speed of light")', "lookup_constant"),
    ("lookup", 'lookup("Fe", "atomic_weight")', "lookup_element"),
    ("lookup", 'lookup("Mars", "mass")', "lookup_astronomy"),
    ("reg", "set r0 42", "reg_set"),
    ("reg", "r0", "reg_get"),
]


def run_benchmark(binary_path: str, iterations_per_case: int = 1000) -> Dict[str, Any]:
    cpu_model = get_cpu_model()
    go_ver = get_go_version()
    os_info = f"{platform.system()} ({platform.machine()})"
    binary_sha = get_binary_sha256(binary_path)

    print("=" * 75)
    print("Reckoner Harness: Latency Microbenchmark")
    print(f"CPU:     {cpu_model}")
    print(f"OS:      {os_info}")
    print(f"Go:      {go_ver}")
    print(f"Binary:  {binary_path} (sha256: {binary_sha[:16]}...)")
    print(f"Cases:   {len(BENCHMARK_SUITE)} queries across 7 tools")
    print(f"Iter:    {iterations_per_case} per case")
    print("=" * 75)

    client = HarnessClient(binary_path)

    # Warmup
    for _ in range(50):
        client.dispatch("calc", "1 + 1")

    tool_latencies: Dict[str, List[float]] = {}
    case_latencies: Dict[str, List[float]] = {}
    internal_latencies: Dict[str, List[float]] = {}

    tokens_1k = []
    for i in range(1000):
        if i % 20 == 19:
            tokens_1k.append(326)
        else:
            tokens_1k.append(1000 + i)
    anchor_1k_payload = json.dumps({"tokens": tokens_1k, "store": False})

    # Prepare context benchmark cases
    BENCHMARK_CASES = list(BENCHMARK_SUITE) + [
        ("anchor", anchor_1k_payload, "anchor_1k_tokens"),
        ("quote", "<|quote|><|m0|><|L0|><|L10|><|/quote|>", "quote_10_lines"),
        ("find", 'find("Invoice ch_9812")', "find_100k_context"),
    ]

    for tool, payload, label in BENCHMARK_CASES:
        if tool not in tool_latencies:
            tool_latencies[tool] = []
            internal_latencies[tool] = []
        case_latencies[label] = []

        client.reset()
        if label == "reg_get":
            client.dispatch("reg", "set r0 42")
        elif label == "quote_10_lines":
            client.dispatch("context_reset", "")
            client.dispatch("anchor", json.dumps({"tokens": tokens_1k, "store": True}))
        elif label == "find_100k_context":
            client.dispatch("context_reset", "")
            for m in range(50):
                tokens_2k = []
                for i in range(2000):
                    if i % 25 == 24:
                        tokens_2k.append(326)
                    elif m == 25 and i == 1000:
                        tokens_2k.extend([25049, 12511, 659, 190, 152, 151, 144, 145])
                    else:
                        tokens_2k.append(1000 + (i % 500))
                client.dispatch("anchor", json.dumps({"tokens": tokens_2k, "store": True}))

        # Scale iterations for heavy context cases to keep benchmark responsive
        iters = iterations_per_case
        if label == "find_100k_context":
            iters = min(iterations_per_case, 100)
        elif label == "anchor_1k_tokens":
            iters = min(iterations_per_case, 200)

        for _ in range(iters):
            t0 = time.perf_counter()
            resp = client.dispatch(tool, payload)
            elapsed_us = (time.perf_counter() - t0) * 1e6

            if not resp.get("success", False):
                raise RuntimeError(f"Benchmark case failed: {tool} {payload} -> {resp}")

            tool_latencies[tool].append(elapsed_us)
            case_latencies[label].append(elapsed_us)
            internal_latencies[tool].append(resp.get("elapsed_us", 0))

    client.close()

    results = {
        "system": {
            "cpu": cpu_model,
            "os": os_info,
            "go_version": go_ver,
            "binary_sha256": binary_sha,
        },
        "binary": binary_path,
        "iterations_per_case": iterations_per_case,
        "tools": {},
        "cases": {},
    }

    print("\nPer-Tool Latency (Internal harness execution vs Python stdio pipe IPC, µs):")
    print(f"{'Tool':<8} | {'Samples':<7} | {'Internal p50':<12} | {'Internal p90':<12} | {'Pipe p50':<10} | {'Pipe p90':<10} | {'Pipe p99':<10}")
    print("-" * 80)

    for tool, lats in sorted(tool_latencies.items()):
        lats.sort()
        p50 = percentile(lats, 50)
        p90 = percentile(lats, 90)
        p99 = percentile(lats, 99)
        inlats = sorted(internal_latencies[tool])
        inp50 = percentile(inlats, 50)
        inp90 = percentile(inlats, 90)

        results["tools"][tool] = {
            "samples": len(lats),
            "internal_p50_us": round(inp50, 1),
            "internal_p90_us": round(inp90, 1),
            "pipe_p50_us": round(p50, 1),
            "pipe_p90_us": round(p90, 1),
            "pipe_p99_us": round(p99, 1),
            "p50_us": round(p50, 1),
            "p90_us": round(p90, 1),
            "p99_us": round(p99, 1),
        }
        print(f"{tool:<8} | {len(lats):<7} | {inp50:<12.1f} | {inp90:<12.1f} | {p50:<10.1f} | {p90:<10.1f} | {p99:<10.1f}")

    print("\nDetailed Per-Query Benchmarks:")
    print(f"{'Label':<24} | {'Tool':<8} | {'Pipe p50 (µs)':<14} | {'Pipe p90 (µs)':<14}")
    print("-" * 65)
    for tool, payload, label in BENCHMARK_CASES:
        clats = sorted(case_latencies[label])
        cp50 = percentile(clats, 50)
        cp90 = percentile(clats, 90)
        results["cases"][label] = {
            "tool": tool,
            "p50_us": round(cp50, 1),
            "p90_us": round(cp90, 1),
        }
        print(f"{label:<24} | {tool:<8} | {cp50:<14.1f} | {cp90:<14.1f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run harness latency microbenchmarks.")
    parser.add_argument("--binary", "--bin", dest="binary", default="build/bin/reckoner-harness-linux-amd64", help="Path to harness binary")
    parser.add_argument("--iterations", type=int, default=1000, help="Iterations per benchmark query")
    parser.add_argument("--out", "--output", dest="out", default="benchmarks/latency_results.json", help="Output JSON path")
    args = parser.parse_args()

    results = run_benchmark(args.binary, args.iterations)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved benchmark results to {args.out}")


if __name__ == "__main__":
    main()
