---
title: "DeepSeek-V4 Infrastructure: Training and Inference Systems"
type: concept
tags: [llm, deepseek, infrastructure, fp4, kv-cache, t i l e l a n g, ep-overlap, systems]
created: 2026-05-02
updated: 2026-05-02
sources: [raw/DeepSeek-V4.pdf]
status: stable
---

# DeepSeek-V4 Infrastructure

**Source**: DeepSeek-V4 Technical Report, Section 3 [src](raw/DeepSeek-V4.pdf)

## Key Points

- **Fine-grained EP communication overlap** with expert-level pipelining in both forward and backward passes
- **TileLang**: A new kernel development language for flexible, efficient MoE kernel authoring
- **FP4 QAT**: Routed expert weights in FP4 — 1.33x theoretical efficiency on future hardware
- **On-disk KV cache**: Compressed KV entries stored to disk, three SWA caching strategies
- **Quick Instruction**: Special tokens reuse existing KV cache for auxiliary tasks, reducing TTFT

## 1. Fine-Grained EP Communication-Computation Overlap

Expert Parallelism (EP) requires all-to-all communication for token dispatch and combine. V4 introduces **expert-level pipelining** to overlap communication with computation:

**Forward pass**: As soon as tokens for expert `e` are dispatched, that expert begins computation. Meanwhile, dispatch for expert `e+1` proceeds on a separate stream. Similarly, combine for expert `e` overlaps with computation for expert `e+1`.

**Backward pass**: Gradient dispatch and combine are similarly overlapped with expert gradient computation. The key insight: expert computations are independent, so pipelining across experts hides communication latency.

This is a finer granularity than the typical layer-level overlap (e.g., Megatron-Core's 1F1B overlap) — V4 overlaps at the **individual expert level** within a single MoE layer.

## 2. TileLang: Kernel Development Language

**Problem**: MoE models require many specialized GPU kernels (grouped GEMM, sparse attention, compression, routing). Writing and optimizing these in CUDA/CUTLASS for each new architecture is a bottleneck.

**Solution**: TileLang is a new domain-specific language for tile-based GPU kernel development. Key features:

- **Hardware-agnostic**: Write kernels once, compile for Hopper, Blackwell, and future architectures
- **Tile abstraction**: Kernels expressed as operations on tiles — the compiler handles tiling, scheduling, and memory hierarchy
- **Flexible**: Supports the irregular computation patterns common in MoE (variable expert sizes, dynamic token counts, sparse attention)

**Impact**: Accelerated development of the CSA, HCA, and FP4 quantization kernels that are critical to V4's efficiency.

## 3. High-Performance Kernel Libraries

V4 ships with two key kernel libraries:

**Batch-Invariant Libraries**: Kernels that produce identical results regardless of batch size or token distribution. This is critical for deterministic training at scale — debugging distributed training failures requires reproducibility across different parallelism configurations.

**Deterministic Kernels**: All critical operations (attention, MoE routing, communication) are deterministic by design. Non-deterministic operations (e.g., atomic adds in some reduction patterns) are avoided or have deterministic alternatives.

## 4. FP4 Quantization-Aware Training

V4 uses **FP4 precision for routed expert weights**, FP8 for non-expert computation.

**Current hardware**: Peak throughput for FP4 x FP8 operations equals FP8 x FP8 on existing hardware. The benefit today is **memory savings** — smaller weight storage for the massive expert parameter count.

**Future hardware**: The paper explicitly notes that purpose-built hardware could make FP4 roughly **1.33x more efficient** than FP8 — an open lane for future inference gains as hardware catches up.

**QAT approach**: Quantization-aware training means weights are trained with quantization simulated during forward and backward passes, preserving accuracy while enabling low-precision deployment. This is applied specifically to expert weights where the memory savings are most impactful (256 experts x large hidden dims).

## 5. Training Framework

### Muon Implementation

Efficient distributed Muon requires careful handling of the orthogonalization step across data-parallel ranks. V4's implementation shards optimizer states across DP ranks while maintaining correct orthogonalization semantics.

### mHC Implementation

Manifold-Constrained Hyper-Connections are implemented efficiently:
- Dynamic parameter generation uses small projection matrices (n_hc=4 is tiny relative to hidden dim)
- Sinkhorn-Knopp iterations (t_max=20) are batched and run on GPU
- Overall mHC overhead is small (< 1% of total compute)

### Context Parallelism for Long-Context

For 1M-token training, context parallelism partitions sequences across GPUs. V4's CP implementation handles the hybrid attention patterns — CSA compression and HCA compression must be aware of CP boundaries.

### Extended Automatic Differentiation

V4 extends autodiff for **flexible activation checkpointing**. Standard checkpointing is binary (recompute or store). V4's extended AD allows fine-grained control: specific modules can be checkpointed at specific granularities, enabling memory-compute trade-offs tailored to the hybrid attention architecture.

## 6. Inference Framework

### KV Cache Structure

The hybrid attention architecture requires careful KV cache management:

| Attention Type | KV Cache Entries | Compression |
|---------------|-----------------|-------------|
| CSA compressed KV | n/m entries | 1/m |
| CSA SWA KV | n_win entries | None (sliding window) |
| HCA compressed KV | n/m' entries | 1/m' |
| HCA SWA KV | n_win entries | None (sliding window) |

**Key insight**: Only compressed KV entries need to be stored for the full sequence. SWA KV entries are ephemeral — they only need the last n_win tokens.

### On-Disk KV Cache Storage

To support prefix caching across requests, V4 stores KV caches to disk:

**Compressed KV entries**: Stored entirely on disk. On a cache hit, the compressed entries are read directly — no recomputation needed for complete compression blocks. Tail incomplete blocks are recomputed.

**SWA KV entries**: Three strategies, trading storage for computation:

| Strategy | Storage | Recompute | Best For |
|----------|---------|-----------|----------|
| Full SWA Caching | Complete SWA KV | Zero | Low-latency, high-SSD deployments |
| Periodic Checkpointing | Checkpoint every p tokens | Tail recompute | Balanced deployments |
| Zero SWA Caching | None | Last n_win * L tokens | Storage-constrained |

**Zero SWA Caching detail**: Each SWA KV entry depends only on the last n_win tokens from the previous layer. With cached CSA/HCA KV entries, recomputing only the last n_win * L tokens is sufficient. This is practical because n_win is small (128 tokens).

## 7. RL and Agent Infrastructure

### Million-Token RL Framework

The RL training framework was scaled to support 1M-token context rollouts. This required:
- Memory-efficient rollout storage for long trajectories
- Efficient attention computation across 1M-token sequences during RL updates
- Preemption-safe resumption for long-running rollouts

### DSec Sandbox

For agentic AI training (code execution, tool use), V4 introduces DSec:

- **Layered storage**: Base images as readonly EROFS layers; data blocks fetched on demand from 3FS distributed filesystem
- **Fast startup**: File metadata available locally at mount time; millisecond-scale resumption via chainable snapshots
- **Density optimization**: Memory reclamation for safe overcommitment; spinlock contention reduction in container runtime
- **Trajectory logging**: Globally ordered, persistent command/result logs enabling fast-forward resumption and deterministic replay

DSec supports both container-based and microVM-based sandboxes with a unified interface. Switching between them requires only a parameter change.

## Connections

- [DeepSeek-V4 Technical Report](deepseek-v4-technical-report.md) — full paper summary
- [DeepSeek-V4 Architecture](deepseek-v4-architecture.md) — CSA, HCA, mHC, Muon
- [DeepSeek-V4 Post-Training](deepseek-v4-post-training.md) — OPD pipeline uses this infrastructure
- [Megatron-Core MoE](megatron-core-moe-scalable-training.md) — NVIDIA's EP overlap approach (layer-level vs expert-level)
