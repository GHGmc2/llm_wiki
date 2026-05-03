---
title: "PartIR: Composing SPMD Partitioning Strategies for Machine Learning"
type: source-note
tags: [compiler, spmd, partitioning, mlir, part IR, google, sharding, tactics]
created: 2026-05-02
updated: 2026-05-02
sources: [raw/PartIR.pdf]
status: stable
---

# PartIR

**Source**: Google DeepMind. 36 pages. MLIR-based compiler for composable SPMD partitioning.

## Key Points

- **Composable tactics**: Each parallelism strategy is a "tactic" that incrementally rewrites the ML program IR
- **Schedule-like API**: Tactics compose in any order to form a schedule — each is independent, no undoing previous decisions
- **Hardware and runtime agnostic**: Works with any frontend (TF, PyTorch, JAX) and any backend (XLA, OpenXLA) via MLIR
- **Predictable propagation**: Rewrite rules from algebraic semantics, not cost-based heuristics
- **Formally verified**: The PartIR:Core → PartIR:HLO transformation has a formal proof

## The Problem: GSPMD's Annotation Pain

GSPMD requires users to annotate tensors with sharding specifications. For complex models with mixed parallelism (DP+MP+ZeRO+EP), this becomes trial-and-error:
- Annotations must be placed at the right program points
- Conflicting sharding constraints require manual resolution
- Changing one annotation can cascade into many changes

PartIR solves this through **composable, ordered tactics**. The user writes a schedule of tactics; the compiler handles propagation and conflict resolution deterministically.

## PartIR Architecture

### Two Key Dialects

| Dialect | Purpose | Verification |
|---------|---------|-------------|
| **PartIR:Core** | Functional tiling and reduction loops on top of StableHLO. Abstracts execution semantics for testing and SPMD. | Unit tested |
| **PartIR:HLO** | SPMD collective communication operations (AllReduce, AllGather, ReduceScatter). Makes collectives explicit. | Formally verified (Appendix C) |

### Compiler Pipeline

```
Frontend (TF/PyTorch/JAX)
    ↓
StableHLO (ML program IR)
    ↓
PartIR:Core  ← tactics rewrite the IR
    ↓
PartIR:HLO  ← collectives made explicit, verifiable
    ↓
XLA/OpenXLA → Device code
```

## Tactics: Composable Parallelism Strategies

A **tactic** is a named parallelism strategy that rewrites the program IR. Tactics are applied sequentially and **never undo** previous decisions:

```python
# Example PartIR schedule for a large Transformer
schedule = PartIR.Schedule([
    model_parallelism(axis="m"),     # Megatron-style TP over model axis
    z3(axis="d"),                    # FSDP over data axis
    batch_parallelism(axis="d"),     # Additional batch sharding
])
schedule.apply(stable_hlo_module)
```

### Sharding Propagation: Algebraic Rules, Not Heuristics

Unlike GSPMD's cost-based propagation heuristics, PartIR uses **algebraic rewrite rules** derived from each operation's mathematical semantics:

| Operation | Propagation Rule |
|-----------|-----------------|
| **Element-wise** (ReLU, Add) | Output sharding = input sharding |
| **MatMul** | Sharding of contracting dimension → AllReduce on output; sharding of non-contracting → preserves parallelism |
| **Reduce** | Reduced dimension loses sharding |
| **Reshape** | May require AllToAll if sharded dimension is reshaped |

**Conflict resolution**: When a tensor must be sharded on multiple logical axes along the same dimension, the **order of tactics** in the schedule resolves the ambiguity. The last tactic "wins" for that dimension. This makes the result deterministic and predictable — no trial-and-error annotation placement.

## Combining Strategies: Explicit in the IR

For large model training, multiple strategies combine:

```
Device Mesh: [N devices along "d"] × [M devices along "m"]
           batch axis            model axis

Each device holds: params[m shard] replicated across d
Communication: ReduceScatter along d (gradients), AllReduce along m (activations)
```

PartIR makes these combinations **explicit in the IR** before code generation. This enables:
- **Performance estimation**: Count all collectives, estimate communication cost
- **Simulation**: Verify correctness of the partitioned program
- **Debugging**: Inspect where each collective is inserted and why

### Example: BP + MP Combination

```
BP tactic:  shard batch → AllReduce grads along "d"
MP tactic:  shard weights → AllReduce activations along "m"

After both tactics:
  Forward:  AllGather params along "m" at each layer
  Backward: ReduceScatter grads along "d" + AllReduce activation grads along "m"
```

All collectives are explicit in PartIR:HLO and can be counted, cost-modeled, and verified before running.

## PartIR Architecture

### Two Key Dialects

| Dialect | Purpose |
|---------|---------|
| **PartIR:Core** | Functional tiling and reduction loops on top of array IR (StableHLO). Abstracts execution semantics for testing and SPMD. |
| **PartIR:HLO** | SPMD collective communication operations (AllReduce, AllGather, etc.). Makes collectives explicit in the IR for verification and performance estimation. |

The PartIR:Core → PartIR:HLO transformation is formally verified (Appendix C).

### Compiler Pipeline

```
Frontend (TF/PyTorch/JAX)
    ↓
ML program IR (StableHLO)
    ↓
PartIR:Core  ← tactics applied here as IR rewrites
    ↓
PartIR:HLO  ← collectives made explicit
    ↓
Backend (XLA/OpenXLA) → Device code
```

## Tactics: Composable Parallelism Strategies

A **tactic** is a named parallelism strategy that rewrites the program IR:

| Tactic | What it does | Communication introduced |
|--------|-------------|------------------------|
| `batch_parallelism` | Shard input batch across devices | AllReduce on gradients |
| `model_parallelism` (Megatron) | Shard weight matrices | 2 AllReduce per transformer layer |
| `Z3 / FSDP` | Shard params, grads, optimizer states | AllGather params (fwd+bwd), ReduceScatter grads |
| `Z2` | Shard grads + optimizer states | ReduceScatter grads |
| Custom | User-defined sharding | As specified |

**Key property**: Tactics are **independent** — applying `Z3` after `model_parallelism` produces the same result regardless of order (though the resulting communication pattern differs). Tactics never undo previous sharding decisions.

### Schedule Example

```python
# PartIR schedule for training a large Transformer
schedule = [
    model_parallelism(axis="m"),     # Megatron-style TP
    z3(axis="d"),                    # FSDP over data axis
    batch_parallelism(axis="d"),     # Further batch sharding
]
```

Each tactic is applied sequentially. The compiler propagates shardings and inserts collectives after each tactic, making the result verifiable at each step.

## Sharding Propagation

Unlike GSPMD's cost-based heuristics, PartIR uses **algebraic rewrite rules**:

- Each operation has propagation rules derived from its mathematical semantics
- When sharding a tensor on multiple logical axes along the same dimension, the **order of tactics** in the schedule resolves the conflict
- No trial-and-error annotation placement — the schedule makes decisions explicit

## Combining Strategies

For large model training, multiple strategies are combined:

```
Device Mesh: [N devices along "d"] × [M devices along "m"]
           batch axis            model axis

Each device holds: params[m shard] replicated across d
Gradients: ReduceScatter along d, AllReduce along m
```

PartIR makes these combinations explicit in the IR, enabling:
- **Performance estimation**: Count collectives before running on hardware
- **Simulation**: Verify correctness of partitioned program
- **Debugging**: Inspect where each collective is inserted and why

## MLIR-Based Design

PartIR is built on MLIR (Multi-Level Intermediate Representation):
- Language-agnostic: same IR for TF, PyTorch, JAX
- Hardware-agnostic: backends for XLA, OpenXLA, or custom compilers
- Independently testable: each dialect has its own tests

## Connections

- [GSPMD](gspmd.md) — The foundational system PartIR improves upon
- [TOAST](toast.md) — Auto-partitioning that can generate PartIR schedules automatically
- [Parallel Folding](megatron-core-moe.md) — Megatron-Core's manual parallelism decoupling (PartIR automates this)
- [Scaling Techniques Overview](scaling-techniques-overview.md) — The parallelism strategies that tactics implement
