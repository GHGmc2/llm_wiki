---
title: "Megatron-Core MoE: Scalable Training of Mixture-of-Experts Models"
type: source-note
tags: [llm-training, mixture-of-experts, distributed-systems, nvidia, parallelism, megatron]
created: 2026-05-02
updated: 2026-05-02
sources: [raw/scalable-training-moe-megatron-core.pdf]
status: stable
---

# Megatron-Core MoE: Scalable Training of Mixture-of-Experts Models

**Source**: NVIDIA Technical Report, March 2026. 88 pages, 42 figures. Open source under Megatron-Core.

**Authors**: Zijie Yan (project lead), Hongxiao Bai, Xin Yao, Dennis Liu, et al. (44 authors total). Corresponding: {zijiey, juney, jiajiey}@nvidia.com.

## Key Points

- MoE sparsity (only _K_ of _E_ experts activate per token) creates a **parameter-compute mismatch** that breaks traditional parallelism assumptions
- This mismatch manifests as three coupled constraints: the **Memory Wall**, **Communication Wall**, and **Compute Efficiency Wall**
- **Parallel Folding** decouples attention (dense) and MoE (sparse) layer parallelism, allowing each to use its optimal topology
- Achieves 1,233 TFLOPS/GPU for DeepSeek-V3-685B on GB300, 368 TFLOPS/GPU on H100 at 1,024 GPU scale
- Full open-source stack: Megatron-Core + Transformer Engine, used from billions to trillions of parameters

## Summary

This report presents Megatron-Core MoE, NVIDIA's open-source stack for training large-scale Mixture-of-Experts models. The core insight is that MoE sparsity creates fundamentally different systems challenges than dense models: because only a subset of experts activate per token, total parameters grow much faster than per-token computation, creating coupled constraints across memory, communication, and computation.

### The Three Walls Framework

The paper organizes its optimization strategies around three "walls" that emerge from MoE's sparsity:

1. **[Memory Wall](moe-memory-wall.md)**: Parameters, optimizer states, gradients, and activations must fit within GPU memory. DeepSeek-V3's full footprint is 199.5 GB/GPU — reduced to under 80 GB through fine-grained recomputation, activation offloading, FP8/FP4 precision, and FSDP.

2. **[Communication Wall](moe-communication-wall.md)**: Expert Parallelism (EP) introduces all-to-all communication that can consume 30-50% of step time. Solved via optimized dispatchers (DeepEP for NVL8, HybridEP for NVL72) and communication-computation overlap.

3. **[Compute Efficiency Wall](moe-compute-efficiency-wall.md)**: Small per-expert GEMMs underutilize Tensor Cores; dynamic token counts create host-device synchronization overhead. Solved via Grouped GEMM, kernel fusions, CUDA Graphs, and Sync-Free MoE with ECHO (Elastic Cloning for Hot Experts).

### Parallel Folding (Section 3)

The **dense-sparse mismatch**: attention layers benefit from high Tensor Parallelism (large QKV matrices), while MoE experts suffer from it (small per-expert dimensions). Traditional frameworks forced both to share one TP config. Parallel Folding decouples them, allowing attention to use TP=4 while MoE uses ETP=1 with EP=64. This breaks the EP ≤ DP constraint and reduces minimum GPU requirements.

### Reduced-Precision Training (Section 5)

Full FP8 and NVFP4 support across multiple recipes: per-tensor FP8, blockwise FP8 (Hopper), MXFP8 (Blackwell), and NVFP4 (Blackwell). Reduced precision impacts all three walls simultaneously — lower memory, less communication volume, faster computation. MoE-specific challenges include dynamic token shapes requiring grouped quantization.

### Performance (Section 8)

| Model | Hardware | GPUs | SeqLen | TFLOPS/GPU | Tokens/s/GPU |
|-------|----------|------|--------|------------|-------------|
| DeepSeek-V3-685B | GB300 | 256 | 4,096 | 1,233 | 4,730 |
| DeepSeek-V3-685B | GB200 | 256 | 4,096 | 1,048 | 4,020 |
| DeepSeek-V3-685B | H100 | 1,024 | 4,096 | 368 | 1,412 |
| Qwen3-235B | GB300 | 256 | 4,096 | 974 | 6,583 |
| Qwen3-235B (long ctx) | GB300 | 128 | 131,072 | 1,150 | 1,556 |

GB200 delivers ~3× H100 throughput at comparable or smaller GPU counts.

### Production Features (Section 7)

Load balancing strategies (aux-loss, Sinkhorn, bias-based), shared experts, Latent MoE (compressing dispatch dimension), parallelism-agnostic distributed checkpointing, flexible asymmetric VPP, dense-to-MoE upcycling, multi-token prediction, and Muon optimizer with MuonClip for attention stability.

### Best Practices (Section 9)

A systematic three-phase optimization workflow:
1. **Phase 1**: Establish memory-feasible parallelism
2. **Phase 2**: Select optimal parallelism strategy (minimize model parallelism, keep EP/TP within NVLink, prefer EP over TP for experts)
3. **Phase 3**: Profile and identify dominant bottleneck, apply targeted optimizations

Same model can require completely different strategies on different hardware (see DeepSeek-V3 case study on GB200 vs H100).

## Connections

- [Parallel Folding](moe-parallel-folding.md) — decoupling attention and MoE parallelism
- [Memory Wall](moe-memory-wall.md) — detailed breakdown of memory optimization techniques
- [Communication Wall](moe-communication-wall.md) — EP dispatchers and communication overlap
- [Compute Efficiency Wall](moe-compute-efficiency-wall.md) — Grouped GEMM, CUDA Graphs, kernel fusions
- [FP8/FP4 Training](moe-fp8-fp4-training.md) — reduced-precision recipes for MoE
- [Performance Best Practices](moe-performance-best-practices.md) — three-phase optimization workflow and case study
