---
title: "MoE Memory Wall"
type: concept
tags: [llm-training, mixture-of-experts, memory-optimization, megatron, distributed-systems]
created: 2026-05-02
updated: 2026-05-02
sources: [raw/scalable-training-moe-megatron-core.pdf]
status: stable
---

# MoE Memory Wall

**Source**: Megatron-Core MoE Technical Report, Section 4.1 [src](raw/scalable-training-moe-megatron-core.pdf)

## Key Points

- Memory is the **first constraint**: if parameters, optimizer states, and activations exceed GPU capacity, training cannot proceed at all
- DeepSeek-V3's full memory footprint: **199.5 GB/GPU** — must be reduced to fit 80 GB (H100) or 192 GB (GB200) devices
- Seven techniques compose the memory optimization stack: memory-efficient permutation, FP8/FP4 activations, selective recomputation, activation offloading, precision-aware optimizers, optimizer state offloading, and FSDP for MoE
- Memory optimization trades compute for memory — the goal is to use compute efficiently, not eliminate all recomputation

## Memory Anatomy

Training a large MoE model requires storing four categories of data per GPU [src](raw/scalable-training-moe-megatron-core.pdf):

| Component | DeepSeek-V3 (GB/GPU) | Share |
|-----------|---------------------|-------|
| Parameters | 17.9 | 9% |
| Optimizer states | 44.3 | 22% |
| Gradients | 7.7 | 4% |
| Activations | 129.6 | 65% |
| **Total** | **199.5** | 100% |

Activations dominate (65%). This is the primary target for memory optimization.

> [!note] MoE activations are larger than dense models because token dispatch creates additional intermediate tensors: routing maps, dispatch indices, and per-expert token buffers.

## 1. Memory-Efficient Permutation

**Problem**: Token dispatch and combine require permutation operations that create redundant intermediate activations — reshaping and gathering tokens into per-expert buffers essentially doubles the activation memory for a brief moment.

**Solution**: Zero-overhead permutation fuses the permutation with adjacent operations, eliminating the need to store separate input and output buffers. Instead of:
```
input → permute → buffer → compute → buffer → unpermute → output
```
The fused version directly computes on permuted indices without materializing full intermediate tensors.

**Impact**: Reduces MoE-layer activation memory with **zero compute overhead** — it's purely an implementation optimization.

## 2. FP8/FP4 Activation Storage

**Problem**: Activations stored in BF16 consume 2 bytes per element.

**Solution**: Store activations in FP8 (1 byte) or FP4 (0.5 bytes) during the forward pass, converting to higher precision only when needed for the backward pass.

- FP8: ~2× activation memory reduction
- NVFP4: ~4× activation memory reduction

**Trade-off**: Quantization introduces numerical noise. The paper addresses this with selective precision — numerically sensitive components (router, embeddings, optimizer states) stay in BF16 while bulk activations use reduced precision.

See also: [Megatron-Core MoE](megatron-core-moe-scalable-training.md) Section 5 for FP8/FP4 training recipes.

## 3. Selective Recomputation (Activation Checkpointing)

**Problem**: Storing all activations is impossible at scale. Full recomputation (recomputing everything in backward) wastes compute.

**Solution**: Fine-grained, module-level recomputation control. Instead of a binary "recompute everything" flag, select specific modules to recompute:

| Strategy | Memory Saved | Compute Overhead | When to Use |
|----------|-------------|-----------------|-------------|
| Full (none) | Maximum | 2× forward | When memory is the only bottleneck |
| Selective | Moderate | 1.1-1.3× | When compute is also constrained |
| None | 0% | 1× | When memory is abundant |

For DeepSeek-V3 on H100, the optimal recompute set is: `mlp`, `mla_up_proj`, `layernorm`, `moe_act`. On GB200 (more memory), only `mlp` is needed.

**Key insight**: MoE layers have unique recomputation trade-offs because expert computation is localized — you can recompute individual experts rather than the full MoE block.

## 4. Fine-Grained Activation Offloading

**Problem**: Even with selective recomputation, activation memory may still exceed GPU capacity. Traditional offloading moves entire activation tensors to CPU, but the transfer latency is high.

**Solution**: Fine-grained, overlapped offloading with prefetch [src](raw/scalable-training-moe-megatron-core.pdf):

1. **Forward pass**: As each layer completes, its activations are asynchronously copied to CPU using dedicated CUDA streams. The GPU continues computing the next layer without waiting.
2. **Backward pass**: Activations are prefetched from CPU to GPU before the corresponding backward layer begins, overlapping the transfer with computation of other layers.
3. **Paged stashing**: Activations are stored in page-sized chunks, enabling efficient memory management and reducing fragmentation.

**Performance**: Overlapped offloading hides most transfer latency. The paper reports that for DeepSeek-V3, offloading reduces peak GPU memory with <5% throughput overhead compared to full recomputation.

> ⚠️ **Trade-off**: Offloading effectiveness depends on CPU-GPU bandwidth (PCIe or NVLink-C2C). GB200's C2C interconnect makes offloading highly effective; older PCIe connections may not hide enough latency.

## 5. Precision-Aware Optimizer

**Problem**: Standard Adam optimizer stores FP32 master weights + FP32 momentum + FP32 variance = 12 bytes per parameter. For a 685B model, this is ~8.2 TB.

**Solution**: Store optimizer moments (momentum, variance) in BF16 instead of FP32, while keeping master weights in FP32 for numerical stability. This cuts optimizer state from 12 bytes/param to 8 bytes/param (33% reduction).

The optimizer update still uses FP32 arithmetic internally; only storage precision is reduced.

## 6. Optimizer State Offloading

**Problem**: Even with reduced precision, optimizer states for trillion-parameter models exceed GPU memory.

**Solution**: Offload optimizer states to CPU, keeping only the currently-needed states in GPU memory. The optimizer update step:
1. Prefetch states for current parameter group to GPU
2. Compute update in FP32
3. Write updated states back to CPU
4. Proceed to next parameter group

Integrates with ZeRO-style sharding: optimizer states are already sharded across data-parallel ranks, and offloading further reduces per-rank memory.

## 7. FSDP for MoE (Megatron-FSDP)

**Problem**: Standard FSDP (Fully Sharded Data Parallelism) shards parameters across the DP group. But MoE has two distinct parameter types (dense and expert) with different communication patterns.

**Solution**: A dual DeviceMesh architecture [src](raw/scalable-training-moe-megatron-core.pdf):

- **Dense parameters**: Sharded across the DP group (standard FSDP)
- **Expert parameters**: Sharded across the EP + EDP group (Expert FSDP)

Key features:
- **Zero-copy communication**: AllGather and ReduceScatter use direct GPU-to-GPU transfers without CPU staging
- **Computation-communication overlap**: Parameter gathering is overlapped with the forward pass of previous layers
- **EP compatibility**: Works with Expert Parallelism, unlike vanilla FSDP

## Putting It All Together

The memory optimization stack transforms DeepSeek-V3 from 199.5 GB/GPU (impossible) to a feasible configuration:

```
199.5 GB  (raw)
  ↓ FP8 activations
~100 GB
  ↓ Memory-efficient permutation
 ~95 GB
  ↓ Selective recomputation
 ~75 GB
  ↓ Activation offloading
 ~60 GB
  ↓ Precision-aware optimizer
 ~55 GB
  ↓ Optimizer state offloading
 ~45 GB
  ↓ FSDP sharding
 ~35 GB  ← fits in 80 GB with room for micro-batches
```

The exact stack depends on hardware. On GB200 (192 GB), fewer techniques are needed — typically just FP8 + selective recomputation (mlp only) + memory-efficient permutation.

## Connections

- [Megatron-Core MoE](megatron-core-moe-scalable-training.md) — full paper summary
- [Communication Wall](moe-communication-wall.md) — communication optimizations (often trade memory for bandwidth)
- [Compute Efficiency Wall](moe-compute-efficiency-wall.md) — compute optimizations (many memory techniques trade compute for memory)
