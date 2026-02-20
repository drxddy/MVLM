#!/usr/bin/env bash
set -euo pipefail

# TODO: Set this to your Android NDK path
ANDROID_NDK="${ANDROID_NDK:-$HOME/Android/Sdk/ndk/27.0.12077973}"

if [ ! -d "$ANDROID_NDK" ]; then
    echo "ERROR: Android NDK not found at $ANDROID_NDK"
    echo "Set ANDROID_NDK environment variable or edit this script."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build/android"
THIRD_PARTY="$PROJECT_DIR/third_party"
TOOLCHAIN="$ANDROID_NDK/build/cmake/android.toolchain.cmake"

ABI="arm64-v8a"
API_LEVEL=28

# --- Fetch OpenCL headers if missing ---
if [ ! -d "$THIRD_PARTY/OpenCL-Headers" ]; then
    echo ">>> Cloning Khronos OpenCL-Headers..."
    git clone --depth 1 https://github.com/KhronosGroup/OpenCL-Headers.git \
        "$THIRD_PARTY/OpenCL-Headers"
fi

# --- Build OpenCL ICD Loader if missing ---
ICD_BUILD="$THIRD_PARTY/OpenCL-ICD-Loader/build"
if [ ! -f "$ICD_BUILD/libOpenCL.so" ]; then
    if [ ! -d "$THIRD_PARTY/OpenCL-ICD-Loader" ]; then
        echo ">>> Cloning Khronos OpenCL-ICD-Loader..."
        git clone --depth 1 https://github.com/KhronosGroup/OpenCL-ICD-Loader.git \
            "$THIRD_PARTY/OpenCL-ICD-Loader"
    fi
    echo ">>> Building OpenCL ICD Loader for Android ($ABI)..."
    cmake -S "$THIRD_PARTY/OpenCL-ICD-Loader" -B "$ICD_BUILD" -G Ninja \
        -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
        -DANDROID_ABI="$ABI" \
        -DANDROID_PLATFORM="android-$API_LEVEL" \
        -DOPENCL_ICD_LOADER_HEADERS_DIR="$THIRD_PARTY/OpenCL-Headers" \
        -DCMAKE_BUILD_TYPE=Release
    cmake --build "$ICD_BUILD" -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu)"
fi

# --- Build MGPU ---
echo ">>> Configuring MGPU for Android ($ABI, API $API_LEVEL)..."
cmake -S "$PROJECT_DIR" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
    -DANDROID_ABI="$ABI" \
    -DANDROID_PLATFORM="android-$API_LEVEL" \
    -DMGPU_ANDROID=ON \
    -DOPENCL_INCLUDE_DIR="$THIRD_PARTY/OpenCL-Headers" \
    -DOPENCL_LIB="$ICD_BUILD/libOpenCL.so" \
    -DCMAKE_BUILD_TYPE=Release

echo ">>> Building MGPU..."
cmake --build "$BUILD_DIR" -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu)"

echo ">>> Build complete. Binaries in $BUILD_DIR"
