#!/usr/bin/env python3
import hashlib
import os
import sys

def main():
    if len(sys.argv) < 3:
        print("Usage: ci_check_binary.py <binary_path> <platform>")
        sys.exit(1)

    bpath = sys.argv[1]
    platform = sys.argv[2]

    if not os.path.isfile(bpath):
        print(f"ERROR: Binary not found: {bpath}")
        sys.exit(1)

    with open(bpath, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()

    print(f"=== Platform: {platform} ===")
    print(f"Binary: {bpath}")
    print(f"SHA-256: {sha256}")

    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a", encoding="utf-8") as f:
            f.write(f"### Platform: {platform}\n\n")
            f.write(f"- **Binary Path**: `{bpath}`\n")
            f.write(f"- **SHA-256**: `{sha256}`\n\n")

if __name__ == "__main__":
    main()
