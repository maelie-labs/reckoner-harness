#!/usr/bin/env bash
# Cross-compiles the Reckoner pure Go + Lua 5.4 harness for all 6 target platforms using zig cc as cgo CC.
# Produces fully static Linux binaries (musl), Windows binaries (system DLLs only), and macOS binaries.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-1600000000}"
GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

ZIG_VERSION="0.14.1"
OUT_DIR="${1:-$REPO_ROOT/build/bin}"
mkdir -p "$OUT_DIR"

# Locate zig binary (user home or PATH)
ZIG_BIN=""
if command -v zig >/dev/null 2>&1; then
    ZIG_BIN="$(command -v zig)"
elif [ -x "$HOME/.local/bin/zig" ]; then
    ZIG_BIN="$HOME/.local/bin/zig"
elif [ -x "$HOME/.local/zig-x86_64-linux-0.14.1/zig" ]; then
    ZIG_BIN="$HOME/.local/zig-x86_64-linux-0.14.1/zig"
fi

if [ -z "$ZIG_BIN" ]; then
    echo "ERROR: zig not found in PATH or ~/.local/bin/zig."
    echo "Please install zig $ZIG_VERSION in user's home (e.g. ~/.local/zig-x86_64-linux-$ZIG_VERSION/zig)."
    exit 1
fi

ZIG_ACTUAL_VERSION="$("$ZIG_BIN" version)"
if [ "$ZIG_ACTUAL_VERSION" != "$ZIG_VERSION" ]; then
    echo "ERROR: zig version $ZIG_ACTUAL_VERSION does not match required $ZIG_VERSION"
    exit 1
fi

echo "Using zig binary: $ZIG_BIN ($ZIG_ACTUAL_VERSION)"

# Prepare macOS stub directory for libresolv, CoreFoundation, and Security framework linking
DARWIN_STUBS_DIR="/tmp/reckoner_darwin_stubs"
mkdir -p "$DARWIN_STUBS_DIR/CoreFoundation.framework" "$DARWIN_STUBS_DIR/Security.framework"

cat << 'EOF' > "$DARWIN_STUBS_DIR/libresolv.tbd"
--- !tapi-tbd
tbd-version: 4
targets: [ x86_64-macos, arm64-macos ]
install-name: '/usr/lib/libresolv.9.dylib'
current-version: 1
exports:
  - targets: [ x86_64-macos, arm64-macos ]
    symbols: [ _res_9_init ]
EOF

cat << 'EOF' > "$DARWIN_STUBS_DIR/CoreFoundation.framework/CoreFoundation.tbd"
--- !tapi-tbd
tbd-version: 4
targets: [ x86_64-macos, arm64-macos ]
install-name: '/System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation'
current-version: 1
exports:
  - targets: [ x86_64-macos, arm64-macos ]
    symbols: [ _kCFTypeDictionaryKeyCallBacks, _CFArrayAppendValue, _CFArrayCreateMutable, _CFArrayGetCount, _CFArrayGetValueAtIndex, _CFDataCreate, _CFDataGetBytePtr, _CFDataGetLength, _CFDateCreate, _CFErrorCopyDescription, _CFErrorGetCode, _CFRelease, _CFStringCreateExternalRepresentation, _CFStringCreateWithBytes ]
EOF

cat << 'EOF' > "$DARWIN_STUBS_DIR/Security.framework/Security.tbd"
--- !tapi-tbd
tbd-version: 4
targets: [ x86_64-macos, arm64-macos ]
install-name: '/System/Library/Frameworks/Security.framework/Versions/A/Security'
current-version: 1
exports:
  - targets: [ x86_64-macos, arm64-macos ]
    symbols: [ _SecCertificateCopyData, _SecCertificateCreateWithData, _SecPolicyCreateSSL, _SecTrustCopyCertificateChain, _SecTrustCreateWithCertificates, _SecTrustSetVerifyDate, _SecTrustEvaluateWithError ]
EOF

TARGETS=(
    "linux amd64 x86_64-linux-musl reckoner-harness-linux-amd64 static"
    "linux arm64 aarch64-linux-musl reckoner-harness-linux-arm64 static"
    "windows amd64 x86_64-windows-gnu reckoner-harness-windows-amd64.exe dynamic"
    "windows arm64 aarch64-windows-gnu reckoner-harness-windows-arm64.exe dynamic"
    "darwin amd64 x86_64-macos reckoner-harness-darwin-amd64 darwin"
    "darwin arm64 aarch64-macos reckoner-harness-darwin-arm64 darwin"
)

echo "=========================================================="
echo "Building 6-target cross-platform harness matrix"
echo "=========================================================="

for entry in "${TARGETS[@]}"; do
    read -r goos goarch zig_target bin_name link_mode <<< "$entry"
    echo -n "Building $goos/$goarch ($bin_name)... "
    out_file="$OUT_DIR/$bin_name"

    if [ "$link_mode" = "static" ]; then
        (cd "$REPO_ROOT" && CGO_ENABLED=1 GOOS="$goos" GOARCH="$goarch" \
        CC="$ZIG_BIN cc -target $zig_target" \
        go build -trimpath -buildvcs=false \
        -ldflags "-buildid= -s -w -linkmode external -extldflags '-static -s'" \
        -o "$out_file" .)
    elif [ "$link_mode" = "darwin" ]; then
        (cd "$REPO_ROOT" && CGO_ENABLED=1 GOOS="$goos" GOARCH="$goarch" \
        CC="$ZIG_BIN cc -target $zig_target -L$DARWIN_STUBS_DIR -F$DARWIN_STUBS_DIR" \
        go build -trimpath -buildvcs=false \
        -ldflags "-buildid= -w -s -extldflags '-L$DARWIN_STUBS_DIR -F$DARWIN_STUBS_DIR'" \
        -o "$out_file" .)
    else
        (cd "$REPO_ROOT" && CGO_ENABLED=1 GOOS="$goos" GOARCH="$goarch" \
        CC="$ZIG_BIN cc -target $zig_target -s" \
        go build -trimpath -buildvcs=false \
        -ldflags "-buildid= -s -w -extldflags -s" \
        -o "$out_file" .)
    fi

    echo "DONE: $(ls -lh "$out_file" | awk '{print $5}')"
done

# Copy third-party notices next to binaries
cp "$REPO_ROOT/THIRD_PARTY_NOTICES.md" "$OUT_DIR/THIRD_PARTY_NOTICES.md"
echo "Copied THIRD_PARTY_NOTICES.md to $OUT_DIR"

echo "=========================================================="
echo "Static analysis & verification"
echo "=========================================================="
for entry in "${TARGETS[@]}"; do
    read -r goos goarch zig_target bin_name link_mode <<< "$entry"
    out_file="$OUT_DIR/$bin_name"
    echo "--- $bin_name ---"
    file "$out_file"
    if [ "$goos" = "linux" ]; then
        if command -v ldd >/dev/null 2>&1; then
            ldd "$out_file" 2>&1 || true
        fi
    fi
    echo -n "Checking $bin_name for /home/ paths... "
    num_home=$(strings "$out_file" | grep -c "/home/" || true)
    if [ "$num_home" -ne 0 ]; then
        echo "ERROR: found $num_home /home/ paths in $bin_name"
        exit 1
    fi
    echo "CLEAN (0)"
done

# Execution test for native linux-amd64
echo "=========================================================="
echo "Running smoke test on native linux-amd64 binary:"
echo "=========================================================="
NATIVE_BIN="$OUT_DIR/reckoner-harness-linux-amd64"
res="$("$NATIVE_BIN" lua 'return 1+1')"
echo "Output: $res"
if [ "$res" != "<|r0|> = 2" ]; then
    echo "ERROR: Unexpected output from smoke test: $res"
    exit 1
fi
echo "Smoke test passed successfully!"
