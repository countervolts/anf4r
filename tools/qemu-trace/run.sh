#!/usr/bin/env bash
# Build the shims + harness with the NDK and run libanf.so under qemu-aarch64 against them.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$(pwd)
NDK=${NDK:-/opt/android-ndk}
CC=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang
BIONIC=$ROOT/vendor/bionic
OUT=$ROOT/build/trace
mkdir -p "$OUT" "$OUT/dump"
cp "$ROOT/vendor/anf-sdk/lib/arm64-v8a/libanf.so" "$OUT/"
$CC -shared -fPIC -Wl,-z,nodelete -o "$OUT/libOpenCL.so" tools/qemu-trace/cl_rec.c
$CC -shared -fPIC -Wl,-z,nodelete -o "$OUT/libvulkan.so" tools/qemu-trace/vk_rec.c
$CC -shared -fPIC -Wl,-z,nodelete -o "$OUT/libsysprop.so" tools/qemu-trace/sysprop.c
CXX=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang++
$CXX -fPIC -static-libstdc++ -o "$OUT/harness" tools/qemu-trace/harness.cpp -ldl
# libanf dlopens libvulkan.so/libOpenCL.so by soname; our shims provide them on LD_LIBRARY_PATH.
export QEMU_LD_PREFIX="$BIONIC"
LIBS="$OUT:$BIONIC/lib64:$OUT"
qemu-aarch64 -L "$BIONIC" \
  -E LD_LIBRARY_PATH="$LIBS" \
  -E LD_PRELOAD="$OUT/libsysprop.so" \
  -E ANF_TRACE_LOG="$OUT/trace.log" \
  -E ANF_TRACE_DUMP="$OUT/dump" \
  "$BIONIC/linker64" "$OUT/harness" "$OUT/libanf.so" 2>"$OUT/harness.err" || true
echo "=== harness.err (tail) ==="; tail -30 "$OUT/harness.err"
echo "=== trace.log (first 40) ==="; head -40 "$OUT/trace.log" 2>/dev/null
echo "=== dumped modules ==="; ls "$OUT/dump" 2>/dev/null | head
