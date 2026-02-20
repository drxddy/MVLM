# MGPU: On-Device Vision-Language Model Inference Engine for Adreno GPUs

## Project: Real-Time VLM on Mobile via Custom OpenCL Inference Engine

**Status:** Phase 5 Complete (End-to-End Pipeline)
**Date:** February 2026
**Target:** Snapdragon 8 Gen 3 / 8 Elite (Adreno A7x GPUs)

---

## 1. The Problem

Mobile VLMs are broken. Every framework (llama.cpp, MLC, mllm) either runs CPU-only or crashes trying to use the GPU.

EPFL tested on OnePlus 13R: CPU-bound inference hits 80–95°C, 10–12W power, and >100s latency. Qualcomm's llama.cpp backend works for text-only. No framework runs a full VLM (vision + language) on Adreno GPU efficiently.

**Goal:** Build an OpenCL engine that runs Moondream2 end-to-end on the Adreno GPU in <300ms, using <4W power, sub-65°C temperature.

---

## 2. Why Moondream2

Small, proven, Apache 2.0. Only 1.86B params, ~900MB quantized. Runs on 2GB devices. Splits cleanly: SigLIP encoder (12 layers) + Phi-1.5 decoder (24 layers). Aggressively compresses vision to 729 tokens (27×27 patches). GGUF weights available.

**After this works**, port to SmolVLM (even fewer tokens) or LFM2-VL-3B (better quality).

Reference: EPFL study shows GPU offload reduces power 10x (1.3W vs 12W). Qualcomm's llama.cpp backend proves OpenCL works on Adreno—just not for VLMs yet.

---

## 3. Architecture

**Three phases of inference:**

1. **Vision Prefill** (~50-80ms): Camera frame → patches → ViT encoder (12 layers) → projection → 729 visual tokens
2. **Text Prefill** (~20-40ms): Prompt tokens + visual tokens → LLM encoder (24 layers) → KV-cache
3. **Autoregressive Decode** (~3-5ms/token): Loop 1 token at a time, use cached KV for speed

---

## 4. Implemented Features

### OpenCL Engine (`src/engine/`)

- **Device Initialization** (`device.cpp`)
  - Platform/device enumeration
  - Extension probing for Adreno-specific features
  - OpenCL 1.2+ with Qualcomm extensions

- **Memory Management** (`memory.cpp`)
  - Buffer allocation with pool support
  - 2D Image objects for weight matrices (L1 texture cache optimization)
  - On-chip global memory buffers (QCOM extension)

- **Kernel Dispatch** (`compute.cpp`)
  - Unified kernel dispatch interface
  - Event-based synchronization
  - Profiling integration

- **Pipeline** (`pipeline.cpp`)
  - Event-driven inference pipeline
  - Recordable queues for decode loop (QCOM extension)

### OpenCL Kernels (`src/kernels/`)

- **GEMM** (`gemm.cl`)
  - Naive GEMM (correctness baseline)
  - Tiled GEMM (local memory, workgroup tiling)
  - Image-based GEMM (TP/L1 texture cache for weights)
  - GEMV (M=1 decode, workgroup reduction)

- **Attention** (`attention.cl`)
  - Prefill attention (full sequence)
  - Decode attention (single token vs KV-cache)
  - Subgroup-optimized softmax

- **Normalization** (`layernorm.cl`)
  - RMSNorm with subgroup reduction
  - Local memory fallback

- **Activations** (`activations.cl`)
  - SiLU (SiLU-gated MLP)
  - GELU
  - Vectorized operations
  - Residual add (`vector_add`)

- **RoPE** (`rope.cl`)
  - Rotary position embeddings
  - Precomputed sin/cos tables

- **Embedding** (`embedding.cl`)
  - Token embedding lookup
  - Vectorized

- **Vision** (`vision.cl`)
  - Image preprocessing (resize + normalize)
  - Patch embedding (conv2d-like)

### Model Support (`src/models/`)

- **GGUF Loader** (`gguf_loader.cpp`)
  - GGUF v2/v3 format parsing
  - Memory-mapped file loading
  - Tensor metadata extraction
  - Support for F16, F32, Q4_0-Q8_1, Q2_K-Q6_K quantization

- **Tokenizer** (`tokenizer.cpp`)
  - BPE encoding/decoding
  - Load from GGUF metadata or separate vocab file
  - UTF-8 handling

- **Moondream2** (`moondream2.cpp`)
  - SigLIP vision encoder
  - Phi-1.5 LLM decoder
  - KV-cache management
  - Forward pass implementation

---

## 5. Qualcomm/OpenCL Technical Details

### Key OpenCL Features Used

#### Image-Based Weights (Texture Cache)
```
clCreateImage() with CL_MEM_OBJECT_IMAGE2D
```
Weight matrices stored as 2D images leverage Adreno's L1/L2 texture cache, providing 2-3x speedup over buffer-based GEMM.

#### Subgroup Operations
```
get_sub_group_size()
sub_group_reduce_add()
sub_group_broadcast()
```
Eliminate local memory synchronization barriers. Adreno supports up to 64 threads per subgroup.

#### Qualcomm-Specific Extensions

| Extension | Purpose |
|-----------|---------|
| `cl_qcom_subgroup_shuffle` | Efficient cross-thread data exchange |
| `cl_qcom_onchip_global_memory` | Fast on-chip memory for KV-cache |
| `cl_qcom_recordable_queues` | Record decode loop for replay |
| `cl_qcom_perf_hint` | GPU performance hints |
| `cl_qcom_dot_product8` | Int8 dot product for quantized inference |
| `cl_qcom_ahb` | Direct AHB memory access |

#### FP16 Optimization
```
__opencl_c_fp16=1
```
All kernels use half-precision floats for 2x throughput on Adreno.

### Kernel Optimization Patterns

1. **Vectorized Memory Access**: 128-bit loads/stores (`float4`)
2. **Workgroup Tiling**: 16×16 or 32×32 tile sizes
3. **Local Memory Reduction**: Minimize global memory traffic
4. **Branch-Free Code**: Avoid warp divergence
5. **Memory Coalescing**: Aligned accesses, sequential patterns

---

## 6. Build Instructions

### Native (macOS/Linux with OpenCL)
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
```

### Android (NDK Cross-Compilation)
```bash
./scripts/build_android.sh --abi arm64-v8a --api 28
```

Or use the Android app:
```bash
cd android
./gradlew assembleDebug
# APK: android/app/build/outputs/apk/debug/app-debug.apk
```

### Testing
```bash
cd build
./test_gguf        # GGUF loader tests
./test_tokenizer   # Tokenizer tests
./test_device      # Device/OpenCL tests (requires GPU)
```

---

## 7. Usage

### CLI Tool
```bash
./mgpu_cli --model weights/moondream2.gguf \
            --kernels src/kernels \
            --prompt "Describe this image" \
            --max-tokens 128
```

### Output
```
[forward] seq_len=128, pos_offset=0
[forward] embedding lookup done
[forward] layer 0/24 done
...
Prefill: 45ms (2844 tok/s)
Decode: 3ms/token (333 tok/s)
Total: 50 tokens in 194ms
```

---

## 8. Performance Targets

| Metric | Target | Today (CPU) |
|--------|--------|------------|
| Vision prefill | <100ms | ~6s |
| Time to first token | <200ms | >30s |
| Decode speed | >15 tok/s | ~3 tok/s |
| End-to-end (50 tokens) | <500ms | >40s |
| Power | <4W | 10-12W |
| Temp | <65°C | 80-95°C |

---

## 9. Repository Structure

```
MGPU/
├── PLAN.md                     # Implementation phases
├── README.md                   # This file
├── qualcom.md                  # Qualcomm optimization guide
├── CMakeLists.txt              # Build configuration
├── src/
│   ├── engine/
│   │   ├── device.cpp/h        # OpenCL device init, extension query
│   │   ├── pipeline.cpp/h     # Event-driven inference pipeline
│   │   ├── memory.cpp/h       # Buffer/image allocation, KV-cache
│   │   ├── compute.cpp/h      # Kernel dispatch layer
│   │   └── profiler.cpp/h     # GPU timer integration
│   ├── kernels/
│   │   ├── gemm.cl            # GEMM/GEMV (naive, tiled, image)
│   │   ├── attention.cl        # Prefill and decode attention
│   │   ├── layernorm.cl       # RMSNorm with subgroup
│   │   ├── activations.cl     # SiLU, GELU, vector_add
│   │   ├── rope.cl            # Rotary position embeddings
│   │   ├── embedding.cl       # Token embedding lookup
│   │   └── vision.cl          # Patch embedding, preprocess
│   ├── models/
│   │   ├── moondream2.cpp/h  # Moondream2 model graph
│   │   ├── gguf_loader.cpp/h # GGUF weight parser
│   │   └── tokenizer.cpp/h   # BPE tokenizer
│   └── app/
│       ├── main.cpp           # CLI tool
│       └── device_info.cpp    # Device info dump
├── android/
│   ├── app/
│   │   ├── src/main/
│   │   │   ├── cpp/           # JNI wrapper + CMake
│   │   │   ├── java/          # Kotlin activities
│   │   │   └── res/           # Layouts, strings
│   │   └── build.gradle
│   └── build.gradle
├── tests/
│   ├── test_gguf.cpp          # GGUF loader tests
│   ├── test_tokenizer.cpp     # Tokenizer tests
│   ├── test_device.cpp        # Device tests
│   └── test_utils.h           # Test helpers
├── benchmarks/
│   └── gemm_bench.cpp         # GEMM microbenchmark
├── scripts/
│   ├── build_android.sh       # NDK cross-compilation
│   ├── push_and_run.sh        # adb deploy + execute
│   └── perf_mode.sh           # Adreno perf mode
└── third_party/
    ├── OpenCL-Headers/        # Khronos OpenCL headers
    └── OpenCL-ICD-Loader/    # OpenCL ICD loader
```

---

## 10. References

1. Guerrero et al., "Efficient Deployment of VLMs on Mobile Devices: OnePlus 13R Case Study," arXiv:2507.08505, Jul 2025
2. Nota AI, "Deploying an Efficient VLM on Mobile Devices" (PhiVA), 2024
3. Xue et al., "PowerInfer-2: Fast LLM Inference on a Smartphone," arXiv:2406.06282, 2024
4. Li et al., "mllm-NPU: 1000 tokens/second on-device LLM prefilling," arXiv:2407.05858, 2024
5. Qualcomm, "New OpenCL GPU Backend in llama.cpp for Adreno GPUs," Feb 2025
6. Qualcomm, "Harnessing Adreno GPU for Generative AI: Open-Source Approach," Feb 2025
7. Qualcomm, "Snapdragon OpenCL General Programming and Optimization," 80-NB295-11 Rev C, Feb 2023
8. Sharshar et al., "Vision-Language Models for Edge Networks: A Survey," IEEE JIOT, arXiv:2502.07855, 2025
9. Chu et al., "MobileVLM V2: Faster and Stronger Baseline for VLM," arXiv:2402.03766, 2024
10. Marafioti et al., "SmolVLM: Redefining small and efficient multimodal models," arXiv:2504.05299, 2025
11. LiquidAI, "LFM2-VL-3B," Oct 2025
12. LearnOpenCV, "VLM on Edge: Worth the Hype or Just a Novelty?" Sep 2025
13. Trelis Research, "Top Vision Models 2025," Feb 2025
