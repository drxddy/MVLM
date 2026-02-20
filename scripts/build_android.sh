#!/usr/bin/env bash
set -euo pipefail

# MGPU Android Build Script
# This script builds MGPU for Android with Adreno GPU support
#
# Usage:
#   ./build_android.sh              # Build with default settings
#   ./build_android.sh --help       # Show help
#   ./build_android.sh --clean      # Clean and rebuild
#   ./build_android.sh --abi arm64-v8a  # Specify ABI

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build/android"
THIRD_PARTY="$PROJECT_DIR/third_party"

# Default settings
ABI="arm64-v8a"
API_LEVEL=28
CMAKE_BUILD_TYPE="Release"
CLEAN=0
JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            echo "MGPU Android Build Script"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --abi <abi>          Target ABI (default: arm64-v8a)"
            echo "  --api <level>        Android API level (default: 28)"
            echo "  --ndk <path>        Android NDK path (default: ANDROID_NDK env or auto-detect)"
            echo "  --clean              Clean before building"
            echo "  --debug              Build with debug symbols"
            echo "  --jobs <n>          Number of parallel jobs (default: auto)"
            echo "  --help               Show this help"
            exit 0
            ;;
        --abi)
            ABI="$2"
            shift 2
            ;;
        --api)
            API_LEVEL="$2"
            shift 2
            ;;
        --ndk)
            ANDROID_NDK="$2"
            shift 2
            ;;
        --clean)
            CLEAN=1
            shift
            ;;
        --debug)
            CMAKE_BUILD_TYPE="Debug"
            shift
            ;;
        --jobs)
            JOBS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage"
            exit 1
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1" >&2; }

# --- Detect Android NDK ---
find_ndk() {
    local ndk_path=""

    # Check environment variable
    if [[ -n "${ANDROID_NDK:-}" ]]; then
        if [[ -d "$ANDROID_NDK" ]]; then
            echo "$ANDROID_NDK"
            return 0
        fi
        log_warn "ANDROID_NDK=$ANDROID_NDK not found"
    fi

    # Check common locations
    local android_home="${ANDROID_HOME:-${ANDROID_SDK_ROOT:-}}"
    if [[ -n "$android_home" && -d "$android_home/ndk" ]]; then
        # Find latest NDK version
        ndk_path=$(ls -td "$android_home/ndk"/[0-9]* 2>/dev/null | head -1)
        if [[ -n "$ndk_path" ]]; then
            echo "$ndk_path"
            return 0
        fi
    fi

    # Check standard installation paths
    for path in \
        "$HOME/Android/Sdk/ndk" \
        "$HOME/Library/Android/sdk/ndk" \
        "/opt/android-ndk" \
        "/usr/local/android-ndk"; do
        if [[ -d "$path" ]]; then
            ndk_path=$(ls -td "$path"/[0-9]* 2>/dev/null | head -1)
            if [[ -n "$ndk_path" ]]; then
                echo "$ndk_path"
                return 0
            fi
        fi
    done

    return 1
}

# --- Install NDK if not found ---
install_ndk() {
    log_warn "Android NDK not found!"

    # Check if sdkmanager is available
    if command -v sdkmanager &> /dev/null; then
        log_info "Installing NDK via sdkmanager..."
        local android_home="${ANDROID_HOME:-${ANDROID_SDK_ROOT:-$HOME/Android/Sdk}}"
        export ANDROID_HOME="${android_home:-$HOME/Android/Sdk}"
        export ANDROID_SDK_ROOT="$ANDROID_HOME"
        sdkmanager "ndk;27.0.12077973" || true

        local ndk_path="$ANDROID_HOME/ndk/27.0.12077973"
        if [[ -d "$ndk_path" ]]; then
            echo "$ndk_path"
            return 0
        fi
    fi

    # Try commandlinetools
    if command -v cmdline-tools &> /dev/null || [[ -d "$HOME/android-sdk/cmdline-tools" ]]; then
        log_info "You can install NDK manually with:"
        echo "  sdkmanager 'ndk;27.0.12077973'"
    fi

    return 1
}

# --- Main ---

log_info "MGPU Android Build"
log_info "  ABI: $ABI"
log_info "  API: android-$API_LEVEL"
log_info "  Build: $CMAKE_BUILD_TYPE"

# Find or install NDK
ANDROID_NDK=$(find_ndk) || ANDROID_NDK=$(install_ndk)

if [[ -z "$ANDROID_NDK" || ! -d "$ANDROID_NDK" ]]; then
    log_error "Android NDK not found. Please install Android NDK:"
    echo ""
    echo "  1. Download from: https://developer.android.com/ndk/downloads"
    echo "  2. Extract to: ~/Android/Sdk/ndk/"
    echo "  3. Or set ANDROID_NDK environment variable"
    exit 1
fi

log_info "Using NDK: $ANDROID_NDK"

# Check for cmake toolchain
TOOLCHAIN="$ANDROID_NDK/build/cmake/android.toolchain.cmake"
if [[ ! -f "$TOOLCHAIN" ]]; then
    log_error "Android NDK CMake toolchain not found at $TOOLCHAIN"
    exit 1
fi

# Create build directory
mkdir -p "$BUILD_DIR"

# Clean if requested
if [[ $CLEAN == 1 ]]; then
    log_info "Cleaning build directory..."
    rm -rf "$BUILD_DIR"/*
fi

# Create third_party directory
mkdir -p "$THIRD_PARTY"

# --- Fetch OpenCL headers ---
if [[ ! -d "$THIRD_PARTY/OpenCL-Headers" ]]; then
    log_info "Cloning OpenCL-Headers..."
    git clone --depth 1 https://github.com/KhronosGroup/OpenCL-Headers.git \
        "$THIRD_PARTY/OpenCL-Headers"
else
    log_info "OpenCL-Headers already present"
fi

# --- Build OpenCL ICD Loader ---
ICD_BUILD="$THIRD_PARTY/OpenCL-ICD-Loader/build"
if [[ ! -f "$ICD_BUILD/libOpenCL.so" ]]; then
    mkdir -p "$ICD_BUILD"

    if [[ ! -d "$THIRD_PARTY/OpenCL-ICD-Loader" ]]; then
        log_info "Cloning OpenCL-ICD-Loader..."
        git clone --depth 1 https://github.com/KhronosGroup/OpenCL-ICD-Loader.git \
            "$THIRD_PARTY/OpenCL-ICD-Loader"
    fi

    log_info "Building OpenCL ICD Loader for Android..."
    cmake -S "$THIRD_PARTY/OpenCL-ICD-Loader" -B "$ICD_BUILD" -G Ninja \
        -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
        -DANDROID_ABI="$ABI" \
        -DANDROID_PLATFORM="android-$API_LEVEL" \
        -DOPENCL_ICD_LOADER_HEADERS_DIR="$THIRD_PARTY/OpenCL-Headers" \
        -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
        -DANDROID_STL=c++_shared \
        -DANDROID_TOOLCHAIN_VERSION=clang

    cmake --build "$ICD_BUILD" -j"$JOBS"
fi

log_info "OpenCL ICD Loader ready"

# --- Configure MGPU ---
log_info "Configuring MGPU for Android..."

cmake -S "$PROJECT_DIR" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
    -DANDROID_ABI="$ABI" \
    -DANDROID_PLATFORM="android-$API_LEVEL" \
    -DANDROID_STL=c++_shared \
    -DANDROID_TOOLCHAIN_VERSION=clang \
    -DMGPU_ANDROID=ON \
    -DOPENCL_INCLUDE_DIR="$THIRD_PARTY/OpenCL-Headers" \
    -DOPENCL_LIB="$ICD_BUILD/libOpenCL.so" \
    -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE"

# --- Build MGPU ---
log_info "Building MGPU (using $JOBS jobs)..."
cmake --build "$BUILD_DIR" -j"$JOBS"

# --- Summary ---
echo ""
log_info "=========================================="
log_info "Build complete!"
log_info "=========================================="
log_info "Output directory: $BUILD_DIR"
log_info ""
log_info "Binaries:"
ls -la "$BUILD_DIR"/mgpu_* 2>/dev/null || true

echo ""
log_info "To deploy and run on device:"
echo "  ./scripts/push_and_run.sh mgpu_device_info"
echo "  ./scripts/push_and_run.sh mgpu_cli"
echo ""
