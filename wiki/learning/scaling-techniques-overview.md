---
title: "LLM Scaling Techniques Overview"
type: concept
tags: [llm-training, distributed-training, scaling, data-parallelism, zero, tensor-parallelism, pipeline-parallelism, context-parallelism, expert-parallelism]
created: 2026-05-02
updated: 2026-05-02
sources: [raw/The_Ultra-Scale_Playbook.pdf]
status: stable
---

# LLM Scaling Techniques Overview

**Source**: The Ultra-Scale Playbook [src](raw/The_Ultra-Scale_Playbook.pdf)

A progressive guide through all major distributed training techniques, following the narrative arc: start on one GPU, hit a limit, add the next technique.

## The Three Pillars

Every technique addresses one or more of:

1. **Memory**: Hard limit — if the training step doesn't fit, training cannot proceed
2. **Compute Efficiency**: Keep hardware computing, not waiting
3. **Communication Overhead**: Minimize data transfer, overlap with compute

Trade-offs between these three are the central challenge of scaling.

## Technique Progression

### 1. Data Parallelism (DP)

**When**: Model fits on one GPU but training is too slow.

**How**: Replicate the model across GPUs. Each processes a different micro-batch. Gradients are averaged via all-reduce before the optimizer step.

**Global batch size**: $gbs = mbs \times grad\_acc \times dp$

**Three DP optimizations**:

| Optimization | What | Why |
|-------------|------|-----|
| Overlap sync with backward | Trigger all-reduce as soon as a layer's gradient is ready | Hide communication behind computation |
| Gradient bucketing | Group gradients into buckets, one all-reduce per bucket | GPU ops more efficient on large tensors |
| Gradient accumulation interplay | Only sync after final micro-batch | Single reduce has same effect, less overhead |

**Limits**: Throughput drops significantly beyond a certain DP degree due to communication overhead (ring latency at 512+ GPU scale). Memory per GPU is constant — DP doesn't reduce memory.

### 2. ZeRO (Zero Redundancy Optimizer)

**When**: Memory redundancy from DP is the bottleneck.

**How**: Partition optimizer states, gradients, and parameters across DP dimension instead of replicating them.

| Stage | What's Sharded | Memory Saved (per GPU) | Communication Change |
|-------|---------------|----------------------|---------------------|
| ZeRO-1 | Optimizer states | 4$\times$ | AllReduce → ReduceScatter (grads) + AllGather (params after optimizer step) |
| ZeRO-2 | + Gradients | 8$\times$ | Same as ZeRO-1, gradients communicated/released on-the-fly |
| ZeRO-3 | + Parameters (FSDP) | Linear with DP degree | AllGather params on demand (fwd+bwd) + ReduceScatter grads |

**ZeRO-1 in detail**: Instead of each GPU storing full optimizer states, shard them across DP ranks. Communication changes from `AllReduce` gradients to `ReduceScatter` (each GPU gets its shard) + `AllGather` updated parameters after optimizer step.

**ZeRO-3 (FSDP) operation**: 
1. **Forward pass**: As we go through layers, `AllGather` the needed parameter shard → compute → immediately discard
2. **Backward pass**: Same pattern reversed — `AllGather` parameter → compute gradient → `ReduceScatter` the gradient shard
3. This adds `2 × num_layers` additional AllGathers compared to ZeRO-2, each with small base latency

**Overlap strategies for ZeRO-1**: The parameter AllGather after optimizer step can be overlapped with the next forward pass on a separate stream. Two approaches: (1) pre-fetch parameters before they're needed, or (2) use CUDA streams to overlap AllGather with computation.

**ZeRO-3 vs PP**: ZeRO-3 communicates weights (AllGather per layer); PP communicates activations (P2P between stages). They solve the same problem (model too large for one GPU) but with different communication patterns. Can be combined but rarely done — inflates global batch size. The playbook notes: DeepSeek-V3 used PP + ZeRO-1 (optimizer state sharding only).

### 3. Tensor Parallelism (TP)

**When**: Even one layer's activation memory exceeds GPU capacity, or ZeRO-3's parameter communication is too heavy.

**How**: Shard weight matrices across GPUs within individual layers. Two modes:

| Mode | How | Communication | When to use |
|------|-----|--------------|-------------|
| Column-wise (column-linear) | Split weight matrix by columns → each GPU computes partial output → AllGather to combine | AllGather after matmul | When output dim is large |
| Row-wise (row-linear) | Split weight by rows → each GPU computes on its shard → AllReduce to sum partial results | AllReduce after matmul | When input dim is large |

**In a transformer block** (Megatron-style):
- **Attention**: Column-wise for QKV projection → split heads across GPUs → row-wise for output projection
- **FFN**: Column-wise for $h \to 4h$ → ReLU/GELU → row-wise for $4h \to h$

**TP trade-off**: Communication operations are part of forward/backward computation — they can't easily be overlapped. **TP throughput is heavily dependent on intra-node NVLink bandwidth**. Higher TP reduces per-GPU throughput but enables larger batch sizes.

**Key limitation**: TP can't scale across nodes effectively — NVLink bandwidth is needed. Beyond 8 GPUs (single node without NVL72), TP communication overhead dominates.

### 4. Sequence Parallelism (SP)

**When**: TP alone still leaves large activation memory because Dropout/LayerNorm need the full hidden dimension.

**How**: Shard Dropout, LayerNorm, and other non-TP operations along the **sequence** dimension. Since LayerNorm computes `(x - mean) / std` per token, splitting along the sequence dimension is valid.

**Memory impact**: Reduces maximum activation size from `b × s × h` to `b × s/TP × h`.

**Transition management**: The challenge is transitioning between TP regions (sharded by hidden dim) and SP regions (sharded by sequence dim) efficiently. Use collective operations (AllGather / ReduceScatter) at the boundaries.

**Note**: Since LayerNorms in the SP region operate on different sequence portions, their gradients differ across TP ranks. An additional AllReduce on LayerNorm weights keeps them synchronized.

### 6. Pipeline Parallelism (PP)

**When**: Model weights alone exceed the memory of a single node (70B+ params).

**How**: Split model layers across GPUs. Data flows sequentially through the pipeline. Each GPU holds a contiguous block of layers.

**Microbatches**: The global batch is split into `m` microbatches flowing through the pipeline. More microbatches = less bubble (but more activation memory).

### PP Schedules and Bubbles

| Schedule | How it Works | Bubble | Memory (activations) |
|----------|-------------|--------|---------------------|
| **AFAB (GPipe)** | All forward passes → all backward passes | `(p-1)/m × (t_f + t_b)` | `m × activations` (stores all!) |
| **1F1B** | Interleave one forward, one backward per stage | $(p-1)/m$ | `p × activations` (much less) |
| **Interleaved (VPP)** | Multiple virtual stages per GPU, depth-first scheduling | `(p-1)/(m × v)` | Lower per stage |
| **Zero Bubble** | Split backward into B (compute grad) + W (update weights); schedule W anywhere after B | ~0 (theoretical) | Depends on variant |
| **DualPipe (DeepSeek-V3)** | Bidirectional pipeline | `(PP/2 - 1)(F&B + B - 3W)` | 2$\times$ PP + 1 |

**Bubble analysis**: At `m = p-1`, bubble is 100% of ideal time — devastating. At `m ≫ p-1`, bubble approaches zero. The playbook benchmarks show: PP=8 with m=32 achieves near-ideal throughput.

**Interleaved scheduling**: With virtual stages `v`, the model is split into `v × PP` chunks. Scheduling becomes complex: depth-first (prioritize early micro-batches through later layers, closing loops fast) vs breadth-first (prioritize keeping all GPUs busy). Interleaved increases communication by `v×` — a trade-off.

**Zero Bubble insight**: By splitting backward into fine-grained B (compute grad) and W (update weights) stages, the W stage can be flexibly scheduled to fill bubbles. The ZB-H2 schedule achieves theoretically zero bubble. DeepSeek-V3's DualPipe extends this with bidirectional scheduling.

### 7. Expert Parallelism (EP)

**When**: Training Mixture-of-Experts models with many experts.

**How**: Distribute experts across GPUs. Each GPU holds a subset of experts. Tokens are routed via all-to-all communication.

**EP is MoE-specific** — applies only to expert layers. Attention layers still need TP/CP/DP/PP.

**EP vs DP**: Many implementations treat EP as a subgroup of DP (same input handling pattern). Key difference: EP uses AllToAll instead of AllReduce. The combination of EP with traditional parallelism creates the dense-sparse mismatch addressed by [Parallel Folding](megatron-core-moe.md).

## 5D Parallelism: Putting It All Together

Modern training combines all dimensions. The playbook visualizes this with a full memory breakdown for each strategy across TP, SP, CP, EP, PP, and FSDP (ZeRO-3) domains.

**Key interaction insights**:
- **ZeRO-3 and PP**: Solve the same problem but rarely combined — requires inflating global batch size to amortize communication
- **ZeRO-1/2 with PP**: Complementary, easily combined. DeepSeek-V3 used PP + ZeRO-1
- **TP with SP**: Always combined (same group, same degree)
- **CP with EP**: Both use all-to-all patterns — can share the same NVLink domain
- **FSDP with TP**: FSDP shards along DP axis, TP shards along TP axis — orthogonal, compatible

## Configuration Workflow

### When You Have Plenty of GPUs

| Model Size | Strategy |
|-----------|----------|
| <10B params | Single technique (TP or ZeRO-3/DP with full recompute) across 8 GPUs |
| 10-100B params | TP + ZeRO-2/DP + selective recompute across 8-16 GPUs. Add PP for multi-node |
| 100B+ params | TP + PP + ZeRO-2 + selective recompute. EP for MoE. CP for long sequences |

**Special cases**:
- Very long sequences → add CP across nodes
- MoE architectures → use EP across nodes
- NVL72 systems → EP and CP can stay within NVLink domain, no cross-node needed

### When GPU-Constrained

- Enable full activation recomputation (trade compute for memory, train slower)
- Increase gradient accumulation (process larger batches with limited memory)
- Use ZeRO-3 with CPU offload (move optimizer states/grads to CPU)

### Benchmark Heatmap

The playbook includes a heatmap visualization showing optimal training configurations across model sizes (1B-80B) and node counts (1-64). The best configurations balance throughput (tokens/sec/GPU, represented by marker size) and GPU utilization (MFU, represented by color). For each combination, the configuration includes DP, TP, PP, and gradient accumulation steps.

## Connections

- [Ultra-Scale Playbook](ultra-scale-playbook.md) — full source note with formulas and benchmarks
- [Parallel Folding](megatron-core-moe.md) — how EP combines with traditional parallelism
- [Memory Wall](megatron-core-moe.md) — detailed memory optimization techniques
- [Communication Wall](megatron-core-moe.md) — EP communication patterns
- [Compute Efficiency Wall](megatron-core-moe.md) — kernel fusion, CUDA Graphs
- [Megatron-Core MoE](megatron-core-moe.md) — production implementation of these techniques
