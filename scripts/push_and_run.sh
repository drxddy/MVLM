#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build/android"
DEVICE_DIR="/data/local/tmp/mgpu"

BINARY="${1:-mgpu_device_info}"
WEIGHTS="${2:-}"

echo ">>> Creating device directory $DEVICE_DIR..."
adb shell "mkdir -p $DEVICE_DIR/kernels"

# Push binaries
echo ">>> Pushing binaries..."
for bin in mgpu_device_info mgpu_cli mgpu_bench; do
    if [ -f "$BUILD_DIR/$bin" ]; then
        adb push "$BUILD_DIR/$bin" "$DEVICE_DIR/"
        adb shell "chmod +x $DEVICE_DIR/$bin"
    fi
done

# Push OpenCL kernel files
echo ">>> Pushing kernel files..."
for cl in "$PROJECT_DIR"/src/kernels/*.cl; do
    [ -f "$cl" ] && adb push "$cl" "$DEVICE_DIR/kernels/"
done

# Push model weights if specified
if [ -n "$WEIGHTS" ]; then
    if [ -f "$WEIGHTS" ]; then
        echo ">>> Pushing weights: $WEIGHTS..."
        adb push "$WEIGHTS" "$DEVICE_DIR/"
    else
        echo "WARNING: Weights file not found: $WEIGHTS"
    fi
fi

# Run the specified binary
echo ">>> Running $BINARY on device..."
adb shell "cd $DEVICE_DIR && ./$BINARY"
