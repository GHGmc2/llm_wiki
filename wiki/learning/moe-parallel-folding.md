---
title: "MoE Parallel Folding"
type: concept
tags: [llm-training, mixture-of-experts, parallelism, megatron, distributed-systems]
created: 2026-05-02
updated: 2026-05-02
sources: [raw/scalable-training-moe-megatron-core.pdf]
status: stable
---

# MoE Parallel Folding

**Source**: Megatron-Core MoE Technical Report, Section 3 [src](raw/scalable-training-moe-megatron-core.pdf)

## Key Points

- MoE creates a **dense-sparse mismatch**: attention layers want high TP (large QKV matrices), MoE experts want low TP (small per-expert dimensions)
- Prior frameworks forced both to share one parallelism config → suboptimal for at least one layer type
- **Parallel Folding** decouples attention and MoE parallelism mappings, allowing each to use its optimal topology independently
- Breaks the EP ≤ DP constraint: EP can "fold" across TP×CP groups, enabling 8× higher EP with no extra GPUs
- Reduces minimum GPU requirements and keeps communication within NVLink domain

## The Problem: Dense-Sparse Mismatch

### MoE's Parallelism Paradox

Dense models follow a virtuous cycle: more parameters → more GPUs needed for memory → but also more computation per token → communication takes a smaller share → MFU stays stable.

MoE breaks this cycle [src](raw/scalable-training-moe-megatron-core.pdf):

| Model | Total Params | Active Params | Ratio |
|-------|-------------|---------------|-------|
| Llama-70B (Dense) | 70B | 70B | 1:1 |
| DeepSeek-V3 (MoE) | 685B | 37B | 18:1 |

DeepSeek-V3 has 18× more parameters than its active computation suggests. This creates a compounding effect:
1. **Memory grows fast** → forces distribution across many GPUs
2. **More communication** → EP all-to-all scales with token count
3. **Compute stays low** → insufficient computation to overlap with growing communication

### Why Traditional Parallelism Fails

Within a single Transformer block, attention and MoE layers have **conflicting optimal parallelism configurations**:

| Aspect | Attention (Dense) | MoE (Sparse) |
|--------|------------------|--------------|
| Computation | Every token attends to all others | Each token routes to _K_ of _E_ experts |
| TP | Large QKV matrices benefit from high TP | Small per-expert dims make high TP counterproductive |
| CP | Long sequences benefit from high CP | No sequence dependency; CP is irrelevant |
| EP | Not applicable (no experts) | Essential for distributing many experts |

Prior frameworks (GShard, Switch Transformer) forced **one** parallelism configuration for both layer types, treating EP as a sub-dimension of DP:

```
World Size = TP × CP × PP × DP, where EP ⊆ DP
```

This creates three critical challenges:

**Challenge 1: Multiplicative GPU Requirements.** EP=8 forces DP≥8. With CP=8 for long sequences, minimum becomes 1×8×1×8 = 64 GPUs — even though attention and MoE could theoretically share 8 GPUs.

**Challenge 2: Forced Suboptimal Parallelism.** Must choose between high TP (good for attention, fragments experts) or low TP (preserves expert efficiency, underparallelizes attention). Neither works well for both.

**Challenge 3: Cross-Node Communication.** EP within DP often forces all-to-all across node boundaries (5-10× lower bandwidth than NVLink).

## The Solution: Parallel Folding

Parallel Folding **decouples** parallelism mappings for attention and MoE layers [src](raw/scalable-training-moe-megatron-core.pdf):

```
Attention layers: groups over TP × CP × DP × PP
MoE layers:       groups over ETP × EP × EDP × PP
```

The sole constraint: **Pipeline Parallelism (PP) must remain consistent** across both layouts for correct gradient flow.

### How It Works

Instead of EP being constrained within DP, EP can "fold" across arbitrary sub-groups of the attention parallelism configuration:

```
Traditional:  World = TP × CP × (DP) × PP
                          EP ≤ DP

Parallel Folding: World = TP × CP × DP × PP
                   EP can span TP × CP × DP
```

**Concrete example**: Attention configured with TP=4, CP=2, DP=8, PP=4 (256 GPUs)
- Traditional: EP ≤ DP = 8, so max EP = 8
- With Parallel Folding: MoE uses ETP=1, EP=64, EDP=1 (same PP=4)
- EP "folds" across the TP×CP×DP groups → 8× higher expert parallelism
- Attention layers maintain their optimal TP=4, CP=2

### Benefits

1. **Breaks the EP ≤ DP constraint**: EP can exceed DP by folding across TP and CP groups
2. **Reduces minimum GPU requirements**: CP=8, EP=8 needs only 8 GPUs with Folding (vs. 64 in traditional)
3. **Enables independent optimization**: Attention uses high TP for large matrices; MoE uses ETP=1 for full expert width and better GEMM efficiency
4. **Keeps communication in NVLink domain**: Both CP (attention) and EP (MoE) all-to-all stay within NVLink-connected GPUs, avoiding slower cross-node transfers

### The Full Parallelism Stack

With Parallel Folding, Megatron-Core orchestrates **five parallelism dimensions**:

| Dimension | Applies To | Purpose |
|-----------|-----------|---------|
| TP (Tensor) | Attention | Shard large QKV/projection matrices |
| CP (Context) | Attention | Distribute long sequences |
| DP (Data) | Attention | Process different batches |
| PP (Pipeline) | Both | Split model by layers (must be consistent) |
| EP (Expert) | MoE | Distribute experts across GPUs |
| ETP (Expert Tensor) | MoE | Shard within experts (rarely used) |
| EDP (Expert Data) | MoE | Replicate experts for throughput |

**Configuration principles**:
- Attention layers: optimize for large matrices (high TP) and long sequences (high CP)
- MoE layers: optimize for many small experts (high EP, typically ETP=1)
- PP: must remain consistent across both
- DP: used to fill remaining GPUs after TP×CP×EP×PP

### Integration with Memory Optimization

Parallel Folding integrates with Distributed Optimizer and FSDP to further reduce memory:

- **Distributed Optimizer + EP**: Only weights and gradients for local experts reside on each rank; optimizer states are sharded among replicas of the same expert (via EDP)
- **FSDP + EP**: Dual DeviceMesh architecture fully shards parameters, gradients, and optimizer states across data/expert groups, compatible with TP/EP/CP and mixed precision

## Connections

- [Megatron-Core MoE](megatron-core-moe-scalable-training.md) — full paper summary
- [Memory Wall](moe-memory-wall.md) — FSDP for MoE relies on Parallel Folding's group management
- [Communication Wall](moe-communication-wall.md) — Parallel Folding keeps EP communication in NVLink domain
- [Performance Best Practices](moe-performance-best-practices.md) — Guidelines for choosing parallelism configs
