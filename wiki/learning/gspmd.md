---
title: "GSPMD: General and Scalable Parallelization for ML Computation Graphs"
type: source-note
tags: [compiler, parallelization, spmd, gspmd, google, tpu, device-mesh, sharding]
created: 2026-05-02
updated: 2026-05-02
sources: [raw/GSPMD.pdf]
status: stable
---

# GSPMD

**Source**: Google, 2021. 16 pages. Generalized from the GShard backend. Used to scale models to 2048 TPUv3 cores at 50-62% compute utilization.

## Key Points

- **Compiler-based auto-parallelization**: Write single-device code, add sharding annotations, compiler handles the rest
- **Simple, general sharding abstraction**: Replicated, Tiled, Partially Tiled — covers all major parallelism patterns
- **Single API**: `mesh_split(tensor, device_mesh, dims_mapping)` — one call expresses data, model, spatial, and optimizer-state parallelism
- **Automatic sharding propagation**: Infers partitioning for every operator from limited user annotations
- Scaled to 1-trillion-parameter models on 2048 TPUv3 cores

## Core Abstraction: Device Mesh + Sharding Annotations

### Device Mesh

Devices are organized into a logical **multi-dimensional tensor** called a device mesh:

```
device_mesh = [[0, 1, 2, 3],    # dim 0: data parallelism (4 devices)
               [4, 5, 6, 7]]    # dim 1: model parallelism (4 devices)
```

This is a 2D mesh with 16 total devices. The mesh models the physical topology so communication can be optimized.

### Sharding Types

Three sharding types, all represented as tensor-of-device-IDs:

| Type | Description | Example |
|------|-------------|---------|
| **Replicated** | Every device has full data | All tensors in pure data parallelism |
| **Tiled** | Each device has 1/N of data, ordered | Sharding a weight matrix across model-parallel devices |
| **Partially Tiled** | Replicated in subgroups, tiled across subgroups | Combining data + model parallelism |

### mesh_split API

```python
mesh_split(tensor, device_mesh, dims_mapping)
```

- `dims_mapping[i]` = which mesh dimension to shard tensor dimension `i` along
- `-1` = not sharded on this dimension
- Each mesh dimension appears at most once

**This single API expresses ALL parallelism paradigms:**

| Paradigm | How it's expressed |
|----------|-------------------|
| Data parallelism | Shard batch dimension along data mesh axis |
| Model parallelism | Shard weight dimension along model mesh axis |
| Spatial partitioning | Shard image spatial dimensions |
| Weight-update sharding (ZeRO-1) | Shard optimizer states along data mesh axis |
| Pipeline parallelism | Via wrapper that splits graph into stages |

## How It Works

### Annotation → Propagation → Partitioning

1. **User annotates** a few key tensors (typically inputs and weights) with `mesh_split`
2. **GSPMD propagates** sharding annotations through the computation graph using a set of propagation rules for each operator type
3. **GSPMD inserts collectives** (AllReduce, AllGather, ReduceScatter) where sharding changes are needed between operators

### Sharding Propagation

For each operator, GSPMD determines the output sharding from the input sharding:

- **Element-wise ops** (ReLU, Add): Output sharding = input sharding
- **MatMul**: Complex rules depending on which dimensions of operands are sharded
- **Reduce**: Output loses the reduced dimension's sharding
- **Reshape**: Reshaping a tiled dimension may require an AllToAll

### Collective Insertion

When an operator expects a different sharding than what its input provides, GSPMD inserts a collective:

- **Replicated → Tiled**: Slicing (no communication)
- **Tiled → Replicated**: AllGather
- **Tiled(A) → Tiled(B)**: AllToAll (if reshaping between mesh axes)
- **Reduce scattered → summed**: ReduceScatter

### Resharding Conflicts

When GSPMD can't determine a single valid sharding (ambiguous propagation), it uses heuristics to choose, or falls back to the user's explicit annotation. This is the limitation that [PartIR](partir.md) and [TOAST](toast.md) address.

## Use Cases

| Model Type | Parallelism Mix | Devices | Utilization |
|------------|----------------|---------|-------------|
| Dense LLM (1T params) | DP + MP | 2048 TPUv3 | 50-62% |
| MoE (GShard) | DP + EP + MP | 2048 TPUv3 | ~50% |
| Image models | DP + spatial partitioning | 512 TPUv3 | ~55% |
| Speech models | DP + MP | 256 TPUv3 | ~50% |

## Relation to Other Systems

GSPMD is the **foundational system** that PartIR and TOAST build upon:

- **GSPMD**: User provides annotations → compiler propagates. Requires manual trial-and-error for complex sharding.
- **[PartIR](partir.md)**: Composable "tactics" (schedules) that layer sharding decisions incrementally. More predictable, less trial-and-error.
- **[TOAST](toast.md)**: Fully automated — static analysis + MCTS finds optimal sharding without user annotations.

All three share the same core abstraction: device mesh + sharding annotations on tensor dimensions.

## Connections

- [PartIR](partir.md) — Composable SPMD tactics built on GSPMD concepts
- [TOAST](toast.md) — Auto-partitioning using GSPMD's annotation propagation
- [Parallel Folding](moe-parallel-folding.md) — Megatron-Core's parallelism decoupling (manual version of what GSPMD automates)
- [Scaling Techniques Overview](scaling-techniques-overview.md) — The parallelism strategies GSPMD expresses
