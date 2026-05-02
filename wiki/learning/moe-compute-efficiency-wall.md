---
title: "MoE Compute Efficiency Wall"
type: concept
tags: [llm-training, mixture-of-experts, compute-optimization, cuda, megatron, gpu]
created: 2026-05-02
updated: 2026-05-02
sources: [raw/scalable-training-moe-megatron-core.pdf]
status: stable
---

# MoE Compute Efficiency Wall

**Source**: Megatron-Core MoE Technical Report, Section 4.3 [src](raw/scalable-training-moe-megatron-core.pdf)

## Key Points

- MoE compute inefficiency has two causes: **kernel inefficiency** (small per-expert GEMMs underutilize Tensor Cores) and **host overhead** (CPU cannot launch kernels fast enough for fine-grained operations)
- **Grouped GEMM** batches expert computations to increase matrix sizes and improve GPU utilization
- **CUDA Graphs** eliminate per-iteration CPU kernel launch overhead by capturing the execution graph
- **ECHO** (Elastic Cloning for Hot Experts) balances load across EP ranks, enabling static memory allocation for CUDA Graphs even with dropless MoE
- Faster compute (FP8, better hardware) exposes CPU overhead — the bottleneck shifts

## Compute Anatomy: Two Sources of Inefficiency

### 1. Kernel Inefficiency

In dense models, attention and FFN matrices are large enough to saturate Tensor Cores (M, N, K ≥ 1024). In fine-grained MoE, each expert processes only a fraction of tokens, producing much smaller GEMM dimensions:

```
Dense FFN:   [batch × seq_len, hidden_dim] × [hidden_dim, 4×hidden_dim]
             e.g., [32768, 7168] × [7168, 28672]  ← saturates Tensor Cores

MoE expert:  [tokens_to_expert, hidden_dim] × [hidden_dim, intermediate_dim]
             e.g., [512, 7168] × [7168, 2048]  ← underutilization
```

With 256 experts and top-8 routing, each expert receives ~3% of tokens on average. The resulting GEMMs are too small for peak throughput.

### 2. Host Overhead

MoE's fine-grained operations (router, permutation, per-expert compute) require many small GPU kernel launches. The CPU cannot dispatch them fast enough, creating gaps in GPU execution. This becomes the dominant bottleneck once communication and memory are resolved (e.g., on GB200/NVL72).

## 1. Grouped GEMM

**Problem**: Launching separate GEMM kernels for each expert wastes GPU resources (each kernel has launch overhead, small matrices underutilize Tensor Cores).

**Solution**: Batch all expert computations into a single kernel invocation. Instead of:

```
for expert in 1..256:
    output[expert] = matmul(tokens[expert], weight[expert])
```

Grouped GEMM executes all 256 matmuls in one kernel, with each expert's computation running on a subset of SMs. This increases aggregate matrix sizes and reduces kernel launch overhead.

**Implementation**: Uses CUTLASS Grouped GEMM with expert-aware tiling — tile sizes adapt to per-expert token counts for maximum SM utilization.

**Impact**: On fine-grained MoE (256 experts, top-8), Grouped GEMM provides 2-3× throughput improvement over sequential launches.

## 2. Kernel Fusions

**Problem**: Each MoE layer requires multiple small operations (router, permutation, quantization, dispatch), each a separate kernel launch.

**Solution**: Fuse adjacent operations into single kernels:

| Fusion | Operations Combined | Benefit |
|--------|-------------------|---------|
| **Permutation fusion** | Token permute → quantize → dispatch | Reduces intermediate buffers, fewer launches |
| **Router fusion** | Router logits → softmax → top-k → aux-loss | Eliminates router activation storage |
| **Aux-loss fusion** | Load balancing loss computed inline during routing | No separate loss computation pass |

These fusions reduce kernel launch count per MoE layer from ~15-20 to ~5-8.

## 3. CUDA Graphs

**Problem**: Even with fusions, the CPU must still launch each kernel and manage dependencies. For MoE with hundreds of experts, this CPU overhead can dominate step time.

**Solution**: CUDA Graphs capture the entire execution graph (kernels + dependencies) and replay it with a single CPU launch [src](raw/scalable-training-moe-megatron-core.pdf):

```
Without CUDA Graphs:
CPU: launch(K1) → launch(K2) → launch(K3) → ... → launch(K100)
GPU:         [K1]     [K2]     [K3]   ...   [K100]
             ↑ gaps where GPU waits for CPU ↑

With CUDA Graphs:
CPU: launch(graph)
GPU: [K1][K2][K3]...[K100]  ← no gaps
```

**Challenge**: CUDA Graphs require static tensor shapes, but dropless MoE has dynamic per-expert token counts.

**Megatron-Core's solution**: Partial CUDA Graphs capture the static portions (attention, router, MoE preprocessing) while leaving dynamic expert computation ungraphed. On GB200, this captures ~70% of operations.

## 4. Full CUDA Graphs for Dropless MoE

For dropless MoE (where all tokens are always processed, no capacity limits), three techniques enable full CUDA Graphs coverage:

### 4a. Sync-Free Kernels (Device-Initiated)

**Problem**: Dynamic token counts require host-device synchronization to query tensor shapes before launching kernels — this breaks CUDA Graphs.

**Solution**: Device-initiated kernels read dynamic shapes directly from GPU memory (written by a previous kernel), eliminating the host-device sync. The shape metadata lives in GPU global memory and is atomically updated by producer kernels.

### 4b. ECHO: Elastic Cloning for Hot Experts

**Problem**: In dropless MoE, some experts receive more tokens than others ("hot experts"). Worst-case buffer sizing for CUDA Graphs (required for static allocation) would reserve memory for the maximum possible tokens per expert, wasting large amounts of GPU memory.

**Solution**: ECHO dynamically clones hot experts across EP ranks [src](raw/scalable-training-moe-megatron-core.pdf):

1. After routing, identify experts with above-average token counts
2. Clone (replicate) those experts' weights to idle capacity on other EP ranks
3. Split the hot expert's tokens across the original + cloned instances
4. In backward, reduce gradients from clones back to the original expert

This narrows the gap between worst-case and average load, enabling smaller static allocations for CUDA Graphs.

**Overhead**: Extra communication for cloning (forward) and gradient reduction (backward). Only a small fraction of experts are typically cloned, so overhead is modest.

### 4c. Paged Stashing

**Problem**: Each layer needs temporary buffers and activation storage, multiplying the worst-case allocation by the number of layers.

**Solution**: Paged stashing shares a single worst-case temporary buffer across all layers and uses a paged allocation scheme for activations. Memory complexity drops from O(layers × worst_case) to O(worst_case + actual_total).

### Putting It All Together

Full CUDA Graphs coverage for dropless MoE:
- Sync-free kernels → no host-device sync
- ECHO → practical worst-case buffer sizes
- Paged stashing → memory-efficient across layers

## 5. Reduced-Precision Compute

FP8 and FP4 training accelerate GEMMs through Tensor Cores with native low-precision support:
- Blockwise FP8 on Hopper (H100)
- MXFP8 on Blackwell (GB200/GB300) with native Tensor Core acceleration
- NVFP4 on Blackwell for maximum throughput

See [Megatron-Core MoE](megatron-core-moe-scalable-training.md) Section 5 for full low-precision recipes.

## The Bottleneck Shift

A recurring theme: fixing one bottleneck exposes another.

```
Unoptimized:       [Memory Wall ████████████████]  ← can't even train
Fix memory:        [Comm Wall ██████████]          ← all-to-all dominates
Fix communication: [CPU Overhead ██████]           ← host can't keep up (GB200)
Fix CPU overhead:  [Kernel Eff ████]               ← small GEMMs remain
```

On H100, the chain typically stops at the communication wall. On GB200 (NVL72), the communication wall is resolved by hardware, exposing CPU overhead as the new bottleneck — hence the emphasis on CUDA Graphs and kernel fusions in the GB200 case study.

## Connections

- [Megatron-Core MoE](megatron-core-moe-scalable-training.md) — full paper summary, Section 9 case study shows this bottleneck shift in practice
- [Memory Wall](moe-memory-wall.md) — memory optimizations (FP8) accelerate compute, amplifying CPU overhead
- [Communication Wall](moe-communication-wall.md) — resolving communication exposes compute bottlenecks
