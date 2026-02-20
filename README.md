# MVLM: On-Device Vision-Language Model Inference Engine for Adreno GPUs

## Project: Real-Time VLM on Mobile via Custom OpenCL Inference Engine

**Status:** Planning  
**Date:** February 2026  
**Target:** Snapdragon 8 Gen 3 / 8 Elite (Adreno A7x GPUs)

---

## 1. Problem Statement

Current mobile VLM deployment is broken. The July 2025 EPFL study on the OnePlus 13R
(Snapdragon 8 Gen 3) found that **every existing framework** (llama.cpp, MLC-Imp, mllm) either:

- Runs entirely CPU-bound (GPU sits at 0% utilization), hitting 80–95°C and 10–12W
- Crashes or freezes when attempting GPU offload
- Achieves >100 seconds end-to-end latency for a single VLM query

The root cause: no framework properly targets the Adreno GPU's OpenCL compute pipeline
for full VLM inference. Qualcomm's own OpenCL backend for llama.cpp (Feb 2025) supports
text-only LLMs but not vision encoders. MLC-LLM's OpenCL backend handles LLMs but
VLM support is experimental.

**Our goal:** Build a from-scratch OpenCL inference engine that runs an entire VLM — vision
encoder + projection + language decoder — on the Adreno GPU, achieving <300ms end-to-end
latency for camera-to-text inference.

---

## 2. Research Landscape (as of Feb 2026)

### 2.1 Small VLMs Available

| Model | Params | Vision Encoder | LLM Backbone | Int4 Size | Min RAM | License |
|-------|--------|---------------|--------------|-----------|---------|---------|
| **Moondream2** | 1.86B | SigLIP | Phi-1.5 | ~900MB | 2GB | Apache 2.0 |
| **SmolVLM** | 2B | SigLIP-384 | SmolLM2-1.7B | ~1GB | 5GB | Apache 2.0 |
| **InternVL2-1B** | 0.9B | InternViT-300M | InternLM2-1.8B | ~500MB | <2GB | MIT |
| **MobileVLM V2** | 1.7B | CLIP ViT | MobileLLaMA-1.4B | ~850MB | 3GB | Apache 2.0 |
| **LFM2-VL-3B** | 3B | SigLIP2 NaFlex | LFM2-2.6B (hybrid conv+attn) | ~1.9GB | 4GB | Apache 2.0 |
| **Qwen2.5-VL-3B** | 3B | Dynamic ViT | Qwen2.5-3B | ~1.6GB | 5GB | Apache 2.0 |

**Primary target: Moondream2** (smallest, proven on edge, 2GB RAM, Apache 2.0)  
**Stretch target: SmolVLM or LFM2-VL-3B** (better quality, still fits in mobile RAM)

### 2.2 Key Research Papers

| Paper | Key Finding | Relevance |
|-------|-------------|-----------|
| **EPFL VLM Mobile Study** (Jul 2025) | GPU offload reduces power by 10x (1.3W vs 12W), matches latency of smaller models | Validates our approach — GPU is mandatory |
| **Nota AI PhiVA** (2024) | Swapping ViT-L→ViT-B/16 cut vision encoding 87% (37s→6s) on Galaxy S24 | Smaller vision encoder = critical |
| **PowerInfer-2** (SJTU, 2024) | 47B model at 11.68 tok/s on smartphone via neuron-cluster pipelining | Neuron-cluster I/O pipelining technique |
| **mllm-NPU** (2024) | 1000+ tok/s prefill on Qwen1.5-1.8B using NPU + shadow outlier execution | NPU for prefill, CPU/GPU for decode |
| **Qualcomm OpenCL llama.cpp** (Feb 2025) | Official Adreno OpenCL backend for llama.cpp, Q4_0 quantized, upstreamed | Proves OpenCL LLM inference works on Adreno |
| **Qualcomm MLC-LLM** (Feb 2025) | MLC+TVM on Adreno via OpenCL, supports LLaMA/Qwen/etc | Reference for OpenCL kernel patterns |
| **VLMs for Edge Networks Survey** (IEEE JIOT, 2025) | Comprehensive taxonomy of VLM compression for edge | Quantization + pruning strategies |
| **LearnOpenCV VLM on Edge** (Sep 2025) | Moondream2 runs on 2GB devices, Qwen2.5-VL needs 5GB, Jetson Orin matches RTX 3060 | Real perf numbers for our target models |
| **LFM2-VL-3B** (LiquidAI, Oct 2025) | Hybrid conv+attention, 96 image tokens, ~1.9GB Q8, runs on-device | Most efficient new architecture |
| **DeepFusionKernel** (2025) | Fusing SwiGLU MLP into single kernel: 9.7% speedup on A100 | Kernel fusion strategy for MLP blocks |

### 2.3 Existing Frameworks (and why they're insufficient)

| Framework | GPU Support | VLM Support | Problem |
|-----------|------------|-------------|---------|
| llama.cpp + Adreno OpenCL | ✅ Adreno | ❌ Text LLM only | No vision encoder |
| MLC-LLM + Adreno OpenCL | ✅ Adreno | ⚠️ Experimental | VLM not optimized for Adreno |
| mllm | ⚠️ Partial | ✅ | GPU mostly unused, CPU-bound |
| MLC-Imp | ✅ Mobile GPU | ✅ | Not Adreno-optimized, limited models |
| Qualcomm QNN/SNPE | ✅ Adreno + NPU | ⚠️ | Proprietary, closed-source, complex |

**Gap:** No open-source engine does full VLM inference on Adreno GPU via OpenCL with
hardware-specific optimizations (image objects, subgroup ops, on-chip memory, etc.)

---

## 3. Architecture

### 3.1 System Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Android NDK App                       │
│  ┌──────────────────────────────────────────────────┐   │
│  │              MVLM Inference Engine                │   │
│  │                                                    │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  │   │
│  │  │  Weight     │  │  Kernel    │  │  Pipeline   │  │   │
│  │  │  Loader     │  │  Library   │  │  Manager    │  │   │
│  │  │  (GGUF/     │  │  (OpenCL)  │  │  (event-   │  │   │
│  │  │   custom)   │  │            │  │   driven)   │  │   │
│  │  └────────────┘  └────────────┘  └────────────┘  │   │
│  │                                                    │   │
│  │  ┌────────────────────────────────────────────┐   │   │
│  │  │         Adreno OpenCL Runtime               │   │   │
│  │  │  • Image objects for weights (L1 cache)     │   │   │
│  │  │  • Subgroup shuffle for reductions          │   │   │
│  │  │  • On-chip global memory for intermediates  │   │   │
│  │  │  • Recordable queues for layer replay       │   │   │
│  │  │  • Zero-copy AHB for camera input           │   │   │
│  │  └────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────┘   │
│                                                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐               │
│  │  Camera  │  │  Adreno  │  │  System  │               │
│  │  (AHB)   │  │  GPU     │  │  RAM     │               │
│  └──────────┘  └──────────┘  └──────────┘               │
└─────────────────────────────────────────────────────────┘
```

### 3.2 Inference Pipeline

```
Phase 1: Vision Prefill (~50-80ms target)
  Camera Frame (zero-copy AHB)
    → Patch Embedding (image→patches, GEMM)
    → ViT Encoder (6-12 transformer layers)
    → Projection MLP (small, <2M params)
    → Visual Tokens (81-196 tokens)

Phase 2: Text Prefill (~20-40ms target)
  User Prompt Tokens + Visual Tokens
    → Embedding Lookup
    → N Transformer Layers (batch prefill)
    → KV-Cache populated

Phase 3: Autoregressive Decode (~3-5ms/token target)
  Loop:
    → Embedding Lookup (1 token)
    → N Transformer Layers (KV-cache attention)
    → LM Head → Logits → Sample
    → Yield token
  Until EOS or max_tokens
```

### 3.3 Key Adreno-Specific Optimizations

Based on the Qualcomm OpenCL programming guide:

| Optimization | Adreno Feature Used | Expected Impact |
|-------------|-------------------|-----------------|
| Weight matrices as 2D image objects | TP/L1 texture cache (read-only, hardware prefetch) | 2-3x faster weight loads vs buffer |
| FP16 compute throughout | Adreno half-precision ALUs at 2x throughput | 2x ALU throughput |
| Subgroup reductions for softmax/layernorm | `cl_khr_subgroups`, hardware-accelerated `sub_group_reduce_add` | Eliminates local memory barriers |
| Subgroup shuffle for attention | `cl_qcom_subgroup_shuffle` | Fast inter-workitem data exchange |
| On-chip global memory for KV-cache (visual tokens) | `cl_qcom_onchip_global_memory` | Eliminates DRAM round-trip for fixed visual KV |
| Recordable queues for decoder layers | `cl_qcom_recordable_queues` | Record once, replay 24x per token, near-zero dispatch overhead |
| Constant memory for norm params | `__constant` + `max_constant_size` attribute | LayerNorm/RMSNorm weights in fast on-chip constant RAM |
| Int8 dot products for quantized GEMM | `cl_qcom_dot_product8` / `cl_khr_integer_dot_product` | Hardware-accelerated int8×int8→int32 |
| Vectorized load/store (128-bit) | `vload4`/`vstore4`, `read_imageh` with CL_RGBA | Maximize memory bandwidth utilization |
| Zero-copy camera input | `cl_qcom_android_ahardwarebuffer_host_ptr` | Zero memcpy from camera ISP to GPU |
| Branch-free activations | ALU ops instead of ternary (per Epsilon filter case study) | SiLU/GELU without divergence |
| mul24/mad24 for index math | Native 24-bit integer multiply hardware | Faster than 32-bit multiply |
| Avoid size_t in kernels | Use int/uint instead (doc §8.7) | Saves 2 registers per variable |
| Performance hint | `cl_qcom_perf_hint` HIGH | Lock GPU to max frequency during inference |

---

## 4. Implementation Plan

### Phase 0: Environment Setup (Week 1-2)

- [ ] Set up Android NDK build environment with OpenCL headers
- [ ] Build and run Qualcomm's llama.cpp OpenCL backend on target device
- [ ] Profile baseline: run Moondream2 via llama.cpp (CPU-only) to get baseline numbers
- [ ] Query device capabilities: extensions, wave size, L2 cache size, local memory, max workgroup
- [ ] Write OpenCL device info dump tool
- [ ] Verify all required extensions are present on target device

**Deliverable:** Working dev environment, baseline perf numbers, extension availability report

### Phase 1: GEMM Kernel (Week 3-6) ⭐ CRITICAL PATH

The GEMM kernel is 80-90% of inference time. This is the make-or-break component.

- [ ] **v1: Naive GEMM** — Validate correctness, establish baseline
- [ ] **v2: Tiled GEMM** — Workgroup-level tiling, local memory for shared tiles
- [ ] **v3: Image-based weights** — Store weight matrices as 2D image objects, read via TP/L1
- [ ] **v4: Quantized GEMM** — Int4 weights (dequantize to fp16 in-kernel), fp16 activations
  - Use `cl_qcom_dot_product8` for int8 path
  - Pack int4 weights as int8 pairs for dot product
- [ ] **v5: Vectorized + coalesced** — 128-bit loads, coalesced access patterns
- [ ] **v6: Workgroup size tuning** — Brute-force search across WG sizes per matrix shape
- [ ] Benchmark each version, profile with Snapdragon Profiler (SDP)
  - Track: ALU utilization %, L1/L2 hit %, GPU busy %, register footprint

**Target:** >50% of theoretical peak fp16 GFLOPS for the decode GEMV case

**Reference implementations to study:**
- Qualcomm's llama.cpp OpenCL kernels (upstreamed, open source)
- MLC-LLM TVM-generated Adreno OpenCL kernels
- The matrix multiply blog series from Qualcomm Developer Network

### Phase 2: Transformer Primitives (Week 5-8, overlaps with Phase 1)

Build the remaining ops needed for a transformer layer:

- [ ] **RMSNorm** — fp16, use subgroup reduction for mean-square, constant memory for weights
- [ ] **RoPE** — Rotary positional embeddings, precompute sin/cos tables in constant memory
- [ ] **SiLU/GELU activation** — Branch-free using ALU ops (native_exp, native_recip)
- [ ] **Softmax** — Subgroup reduction for max and sum, numerically stable (subtract max first)
- [ ] **Attention**
  - Prefill: batched Q×K^T, softmax, ×V (standard GEMM-based)
  - Decode: single-query attention against KV-cache (memory-bound, optimize loads)
- [ ] **Elementwise ops** — Add, multiply, concat (fuse with preceding ops where possible)
- [ ] **Embedding lookup** — Simple buffer index, trivial kernel
- [ ] **LM Head** — Final GEMM (vocab_size output), argmax/sampling

**Key insight from DeepFusionKernel paper:** Fuse SwiGLU/SiLU-gated MLP into a single
kernel (gate_proj, up_proj, SiLU, elementwise multiply, down_proj) to eliminate intermediate
buffer round-trips. This gave 9.7-13.2% speedup on datacenter GPUs — should be even
more impactful on bandwidth-constrained mobile.

### Phase 3: Model Loading & Memory Management (Week 7-9)

- [ ] GGUF weight loader (parse Moondream2 GGUF format)
  - Map quantized weights to OpenCL image objects
  - Pre-allocate all buffers at startup
- [ ] KV-cache allocator
  - Visual token KV-cache: allocate in on-chip global memory (`cl_qcom_onchip_global_memory`)
  - Text token KV-cache: allocate in system RAM, ring buffer with max context length
- [ ] Activation buffer pool (ping-pong between two buffers per layer)
- [ ] Zero-copy camera integration (`cl_qcom_android_ahardwarebuffer_host_ptr`)

### Phase 4: Vision Encoder (Week 9-11)

- [ ] Patch embedding layer (conv2d or GEMM-based, depends on model)
- [ ] ViT transformer layers (reuse Phase 2 primitives)
  - Moondream2 uses SigLIP: 12 layers, 768-dim, 12 heads
  - SmolVLM uses SigLIP-384: 27 layers, 1152-dim, 16 heads
- [ ] Projection MLP (1-2 linear layers, trivial)
- [ ] Image preprocessing kernel (resize, normalize, patch extraction — all on GPU)

### Phase 5: End-to-End Pipeline (Week 11-13)

- [ ] Wire vision encoder → projection → LLM decoder
- [ ] Implement recordable queue for decoder layer loop
  - Record the kernel sequence for one decoder layer
  - Replay with updated KV-cache pointers per layer
- [ ] Event-driven pipeline (non-blocking enqueues, minimize CPU-GPU sync)
- [ ] Token sampling (greedy, top-k, temperature) — can be CPU-side
- [ ] Tokenizer integration (SentencePiece/BPE, runs on CPU)

### Phase 6: Optimization & Profiling (Week 13-16)

- [ ] Profile with Snapdragon Profiler (SDP)
  - Per-kernel ALU utilization, cache hit ratios, register footprint
  - Identify memory-bound vs compute-bound kernels
- [ ] Kernel fusion pass
  - LayerNorm + first GEMM
  - SiLU-gated MLP fusion
  - Attention score + softmax
- [ ] Workgroup size auto-tuning (runtime brute-force search, cache results per device)
- [ ] Power/thermal profiling (target <3W sustained, <60°C)
- [ ] Memory optimization (minimize peak allocation, reuse buffers)

### Phase 7: Demo App (Week 15-17)

- [ ] Android camera preview → VLM inference → text overlay
- [ ] Benchmarking mode: display tokens/sec, latency breakdown, power consumption
- [ ] Batch evaluation mode for accuracy testing (VQA benchmarks)

---

## 5. Performance Targets

| Metric | Target | Baseline (CPU llama.cpp) | Stretch |
|--------|--------|------------------------|---------|
| Vision prefill | <100ms | ~6s (Nota AI PhiVA) | <50ms |
| Time to first token | <200ms | >30s | <100ms |
| Decode speed | >15 tok/s | ~3-5 tok/s | >30 tok/s |
| End-to-end (50 tokens) | <500ms | >40s | <250ms |
| Peak power | <4W | 10-12W | <2W |
| Peak temperature | <65°C | 80-95°C | <55°C |
| RAM usage | <2GB | ~3-5GB | <1.5GB |

These targets are informed by:
- EPFL study: GPU offload reduced power by 10x, temp by 30°C
- Nota AI: ViT-B/16 encoder in 6s on Galaxy S24 (we target 10-20x faster with OpenCL)
- Qualcomm llama.cpp blog: Q4_0 models running at competitive speeds on Adreno
- LearnOpenCV: Qwen2.5-VL 3B at 2.48s on Jetson Orin Nano (our target device is faster)

---

## 6. Model Selection: Why Moondream2 First

1. **Smallest real VLM** — 1.86B params, proven to work on 2GB devices
2. **Clean architecture** — SigLIP encoder (12 layers) + Phi-1.5 decoder (24 layers)
3. **Aggressive token compression** — Only 729 visual tokens (27×27 patches)
4. **Apache 2.0** — Full commercial freedom
5. **Active community** — GGUF weights available, llama.cpp integration exists
6. **Proven on edge** — LearnOpenCV tested it successfully on Raspberry Pi 4GB

**After Moondream2 works**, port to:
- **SmolVLM** (81 visual tokens via 9x pixel shuffle — even fewer tokens than Moondream2)
- **LFM2-VL-3B** (hybrid conv+attention architecture, only 96 image tokens, best quality/size)

---

## 7. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| GEMM kernel can't reach target perf | Medium | Critical | Study Qualcomm's upstreamed llama.cpp kernels; fall back to TVM code generation |
| On-chip global memory too small for visual KV-cache | Medium | High | Fall back to L2-cached system memory; profile cache hit rates |
| Recordable queues don't work with dynamic KV-cache pointers | Low | Medium | Fall back to regular enqueue with event-driven pipeline |
| Int4 quantization degrades model quality unacceptably | Low | High | Use int8 as fallback; benchmark accuracy on VQA datasets |
| Thermal throttling during sustained inference | Medium | Medium | Use `cl_qcom_perf_hint` NORMAL (not HIGH); profile thermal budget |
| GGUF format doesn't map cleanly to image objects | Low | Low | Write custom weight format; one-time conversion at first launch |

---

## 8. Success Criteria

### Minimum Viable Demo
- Moondream2 running entirely on Adreno GPU via OpenCL
- Camera frame → text description in <1 second
- Power consumption <5W during inference
- No crashes, no screen freezes, no thermal shutdown

### Publication-Ready
- Benchmark against llama.cpp (CPU), MLC-LLM (Vulkan), QNN (NPU)
- Per-kernel profiling breakdown with Snapdragon Profiler
- Accuracy evaluation on VQAv2, TextVQA, MMMU
- Power/thermal characterization over sustained workload
- Open-source release of engine + kernels

---

## 9. Repository Structure

```
MVLM/
├── PLAN.md                     # This document
├── docs/
│   └── qualcom.md              # Qualcomm OpenCL optimization guide (reference)
├── src/
│   ├── engine/
│   │   ├── device.cpp/h        # OpenCL device init, extension query
│   │   ├── pipeline.cpp/h      # Event-driven inference pipeline
│   │   ├── memory.cpp/h        # Buffer/image allocation, KV-cache management
│   │   └── profiler.cpp/h      # GPU timer, SDP integration
│   ├── kernels/
│   │   ├── gemm.cl             # GEMM/GEMV kernels (multiple versions)
│   │   ├── attention.cl        # Prefill and decode attention
│   │   ├── layernorm.cl        # RMSNorm with subgroup reduction
│   │   ├── activations.cl      # SiLU, GELU, softmax (branch-free)
│   │   ├── rope.cl             # Rotary position embeddings
│   │   ├── embedding.cl        # Token embedding lookup
│   │   └── vision.cl           # Patch embedding, image preprocessing
│   ├── models/
│   │   ├── moondream2.cpp/h    # Moondream2 model graph
│   │   ├── smolvlm.cpp/h       # SmolVLM model graph (future)
│   │   └── gguf_loader.cpp/h   # GGUF weight parser
│   └── app/
│       ├── main.cpp            # CLI benchmark tool
│       └── android/            # Android camera demo app
├── weights/                    # Downloaded GGUF files (gitignored)
├── benchmarks/
│   ├── gemm_bench.cpp          # GEMM kernel microbenchmark
│   └── e2e_bench.cpp           # End-to-end inference benchmark
└── scripts/
    ├── build_android.sh        # NDK cross-compilation
    ├── push_and_run.sh         # adb push + execute
    └── perf_mode.sh            # Enable Adreno performance mode
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
