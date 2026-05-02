---
title: "MoE Communication Wall"
type: concept
tags: [llm-training, mixture-of-experts, communication, all-to-all, megatron, distributed-systems]
created: 2026-05-02
updated: 2026-05-02
sources: [raw/scalable-training-moe-megatron-core.pdf]
status: stable
---

# MoE Communication Wall

**Source**: Megatron-Core MoE Technical Report, Section 4.2 [src](raw/scalable-training-moe-megatron-core.pdf)

## Key Points

- Expert Parallelism (EP) introduces **all-to-all communication** that can consume 30-50% of step time if unoptimized
- Communication volume scales with token count and top-_K_, not expert count — more experts means more GPUs but same per-step communication
- Two dispatcher backends: **DeepEP** (optimized for NVL8/cross-node) and **HybridEP** (optimized for NVL72/Multi-Node NVLink)
- Communication-computation overlap hides all-to-all latency behind expert computation
- On NVL72 systems, the communication wall can be resolved by hardware topology alone

## Communication Anatomy

MoE forward pass requires two all-to-all operations per MoE layer:

```
GPU 0: [T0→E1, T1→E3, T2→E0]     GPU 0: [E0 result, E1 result]
GPU 1: [T3→E2, T4→E0, T5→E1]     GPU 1: [E2 result, E3 result]
         ↓ dispatch                        ↑ combine
GPU 0: E0[T2, T4], E1[T0, T5]     ← expert compute →
GPU 1: E2[T3], E3[T1]
```

**Communication volume per layer**:

| Component | Volume per GPU |
|-----------|---------------|
| Dispatch (forward) | tokens/GPU × hidden_dim × top_K × dtype_bytes |
| Combine (forward) | Same as dispatch |
| Dispatch (backward) | gradients: tokens/GPU × hidden_dim × top_K × dtype_bytes |
| Combine (backward) | Same as dispatch |
| **Total** | **4 × (tokens/GPU × hidden_dim × top_K × dtype_bytes)** |

For DeepSeek-V3 (hidden_dim=7168, top_K=8, BF16): ~18 MB per GPU per MoE layer per iteration. With 256 GPUs and 61 MoE layers, this adds up to tens of GB per iteration.

## 1. DeepEP Dispatcher

**Target**: NVL8 systems where EP spans across nodes (cross-node all-to-all).

**Key innovations**:
- **Fused permute + all-to-all**: Combines token permutation and dispatch into a single kernel, eliminating intermediate buffers
- **NVLink-optimized routing**: Routes traffic through NVLink within a node, then through InfiniBand/RoCE between nodes, using the optimal path for each GPU pair
- **Warp-specialized kernels**: Dedicated warps for communication vs. computation within the dispatch kernel

**Performance**: DeepEP achieves near-peak NVLink + IB bandwidth utilization for MoE all-to-all patterns. Used in the DeepSeek-V3 H100 configuration (1,024 GPUs, EP=64).

## 2. HybridEP Dispatcher

**Target**: NVL72 systems (GB200/GB300) with Multi-Node NVLink — all GPUs in the EP group are connected via NVLink.

**Key innovations**:
- **NVLink-native all-to-all**: Direct GPU-to-GPU transfers over the NVLink fabric, no host memory staging
- **Multi-rail utilization**: Exploits multiple NVLink lanes simultaneously for higher aggregate bandwidth
- **Adaptive routing**: Dynamically selects routes based on link congestion and topology

**Performance**: HybridEP on NVL72 achieves 1.8 TB/s bidirectional bandwidth, making EP all-to-all essentially "free" — communication latency is hidden by the fabric's raw bandwidth. This is why GB200 configurations don't need communication overlap.

## 3. EP Communication Overlap

**Problem**: Even with optimized dispatchers, all-to-all latency can still be significant, especially across nodes.

**Solution**: Overlap EP communication with expert computation using a 1F1B (one-forward-one-backward) pipelining scheme [src](raw/scalable-training-moe-megatron-core.pdf):

**Forward pass overlap**:
```
Layer N:   [dispatch] [expert compute] [combine]
Layer N+1:            [dispatch]        [expert compute] [combine]
```
The dispatch of layer N+1 overlaps with expert compute of layer N (as long as they use different experts).

**Backward pass overlap**:
```
Layer N:   [dispatch grad] [expert grad] [combine grad]
Layer N+1:                 [dispatch grad] [expert grad] ...
```

**Requirements for overlap**:
- Each layer must use different sets of experts (natural for most MoE architectures)
- Extra memory buffers for pipelining dispatch/combine operations (~10-20% additional activation memory)
- This is why FP8 memory savings are critical — they free the memory budget for overlap buffers

**When to use**: Essential on H100/B200 (NVL8) where EP spans nodes. Not needed on GB200 (NVL72) where HybridEP provides sufficient bandwidth.

## Hardware Dependency

The communication wall's severity depends entirely on hardware topology:

| System | NVLink Domain | EP64 Spans | Dominant Strategy |
|--------|--------------|------------|-------------------|
| H100 (NVL8) | 8 GPUs | 8 nodes | DeepEP + overlap |
| GB200 (NVL72) | 72 GPUs | 1 node | HybridEP alone |
| GB300 (NVL72) | 72 GPUs | 1 node | HybridEP alone |

> ⚠️ **Key insight**: The same model (DeepSeek-V3, EP=64) has a communication wall on H100 but not on GB200. Hardware topology dictates the optimization strategy. See [Megatron-Core MoE](megatron-core-moe-scalable-training.md) Section 9.2 for the DeepSeek-V3 case study on both platforms.

## Connections

- [Megatron-Core MoE](megatron-core-moe-scalable-training.md) — full paper summary, Parallel Folding (reduces EP communication need)
- [Memory Wall](moe-memory-wall.md) — memory optimizations free budget for communication overlap buffers
- [Compute Efficiency Wall](moe-compute-efficiency-wall.md) — faster compute amplifies communication bottlenecks
