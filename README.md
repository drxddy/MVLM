# MVLM: On-Device Vision-Language Model Inference Engine for Adreno GPUs

## Project: Real-Time VLM on Mobile via Custom OpenCL Inference Engine

**Status:** Planning  
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

**Key Adreno optimizations:**

- Weight matrices as 2D image objects (L1 texture cache, 2-3x faster)
- FP16 compute everywhere
- Subgroup operations for reductions (eliminates local memory sync)
- On-chip memory for visual token KV-cache
- Recordable queues for decoder layer replay (record once, replay 24x per decode)
- Zero-copy camera integration
- Int8 dot products for quantized weights
- Vectorized 128-bit loads/stores

---

---

## 4. Performance Targets

| Metric | Target | Today (CPU) |
|--------|--------|------------|
| Vision prefill | <100ms | ~6s |
| Time to first token | <200ms | >30s |
| Decode speed | >15 tok/s | ~3 tok/s |
| End-to-end (50 tokens) | <500ms | >40s |
| Power | <4W | 10-12W |
| Temp | <65°C | 80-95°C |

---

## 5. What Success Looks Like

**Minimum:** Moondream2 on Adreno GPU, camera → text in <1s, <5W power, no crashes.

**Good:** Benchmark against llama.cpp (CPU), MLC-LLM, QNN/NPU. Per-kernel profiling. Accuracy on VQAv2, TextVQA, MMMU.

---

## 6. Risks & Mitigations

| Risk | Mitigation |
|------|----------|
| GEMM perf not enough | Study Qualcomm's llama.cpp kernels, fall back to TVM |
| On-chip memory too small for KV-cache | Use L2-cached system memory |
| Quantization hurts model quality | Use int8 instead of int4 |
| Thermal throttling | Profile carefully, adjust perf hints |

---

## 7. Repository Structure

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

## 8. References

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
