---
title: "PyTorch Compilation: torch.compile Guide and State"
type: source-note
tags: [pytorch, torch-compile, inductor, dynamo, compilation, performance, triton]
created: 2026-05-04
updated: 2026-05-04
sources: [raw/torch-compile-missing-manual.pdf, raw/pytorch-2.2.pdf, https://blog.ezyang.com/2025/08/state-of-torch-compile-august-2025/]
status: stable
---

# PyTorch Compilation

Three resources covering torch.compile from practical debugging to system architecture.

## 1. torch.compile, the missing manual

**Source**: PyTorch team (Edward Yang), July 2024. 37 pages. Practical debugging guide.

### Key Points

- A hands-on guide for resolving problems when enabling torch.compile on real models
- Focused on technical end users who understand their model but not PyTorch internals
- Covers the three "regimes of enablement" from easy to hard

### The Three Regimes

| Regime | Characteristics | Effort |
|--------|---------------|--------|
| **1. It just works** | Standard model architectures, no exotic ops | Minimal |
| **2. Some work needed** | Custom ops, unusual patterns, graph breaks | Moderate |
| **3. A slog** | Distributed comm, sparsity, data-dependent compute, heavy eager patterns | Significant — expect dozens of bugs |

### What's Compilable

- **Forward pass**: Bread and butter — `@torch.compile` on `nn.Module.forward()`
- **Backward pass**: Automatically compiled with the forward; or use `compiled_autograd` for advanced cases
- **Optimizer**: Horizontal fusion of per-parameter updates. Use `foreach` kernels for best results
- **Distributed wrappers** (FSDP): Use `compiled_autograd` for deferred compilation

### Compilation Modes

| Mode | When to Use |
|------|-----------|
| `default` | General purpose |
| `reduce-overhead` | CUDA Graphs, small batches, graph-friendly loops |
| `max-autotune` | Long-running training, stable shapes |
| `max-autotune-no-cudagraphs` | When CUDA Graphs cause issues |

### Key Trade-offs

- **Compile time vs runtime**: Compiling the full model gives best performance but longest compile time. Compile only Transformer blocks once (regional compilation) to reduce compile time.
- **Static shapes**: Default — fast compilation, fast execution. Recompiles on shape change.
- **Dynamic shapes**: Set `mark_dynamic` to handle varying shapes. Not guaranteed to work for all models.
- **Graph breaks**: Default — transparently bypass uncapturable code. Use `fullgraph=True` to forbid breaks.

## 2. State of torch.compile for Training (August 2025)

**Source**: Edward Yang's blog, August 2025. Comprehensive status update.

### Key Points

- Speedups of **1.5-2×** vs eager are typical
- JIT compilation with local + remote caching; AOT precompile in development
- Compositional with eager: compiled functions compose with autograd, DDP, FSDP
- Graph breaks transparently bypass non-capturable code

### Advanced Parallelism

**DTensor**: "Global tensor" abstraction for sharded tensors over SPMD device mesh. Shard placements are device-mesh-oriented (opposite of JAX). Compilable by torch.compile with eager overhead elimination. Full autograd support with diverging sharding strategies.

**Functional collectives**: Non-mutating collective ops for manual SPMD in compiler-friendly code. No autograd support yet.

**Vision**: Write single-node programs → `SimpleFSDP` inserts collectives → `AutoParallel` discovers optimal sharding (GSPMD-style).

### Optimization Features

| Feature | Status |
|---------|--------|
| **Inductor backend** | Generates Triton kernels, pointwise/reduction fusion, matmul autotuning (cuBLAS/CUTLASS/Triton) |
| **CUDA Graphs** | Built-in; better soundness than manual CUDA Graphs |
| **Activation Checkpointing** | Global memory-compute optimization, better than eager APIs |
| **FP8 support** | Upstreamed to torchao |
| **Flex Attention** | 632 OSS users; supports chunked attention, document masking, CP |
| **Helion** (beta Oct 2025) | Higher-level Triton kernel interface, autotuned structural choices |

### Competition

torch.compile faces competition from:
1. PyTorch native distributed frameworks (e.g., Megatron — hand-optimized eager)
2. Custom compiler stacks on top of FX tracing
3. JAX (XLA-first, years ahead in compile-driven parallelism)

The PyTorch strategy: start with manual patterns → add automatic mechanisms. Opposite of JAX (generic solver → manual escape hatches).

## 3. PyTorch 2 (ASPLOS 2024)

**Source**: Ansel, Yang, He, et al. (Meta / OpenAI / Intel / Quansight), ASPLOS 2024. 18 pages. The official design paper for the PyTorch 2 framework.

### Key Points

- Introduces **TorchDynamo** (graph capture via CPython bytecode transformation) and **TorchInductor** (Triton/C++ backend)
- **2.27× inference and 1.41× training geometric mean speedup** across 180+ models on A100
- Key insight: capture graphs at the **Python bytecode level** using PEP 523 frame evaluation hooks — avoids the limitations of prior approaches (trace, script, symbolic_trace)
- Eager mode flexibility preserved: non-PyTorch code, control flow, and Python constructs all work through graph breaks
- Guards system ensures correctness: recompile when assumptions change (shapes, dtypes, globals)

### The Problem: Prior Graph Capture Approaches

| Approach | Mechanism | Limitation |
|----------|-----------|------------|
| **torch.jit.trace** | Record/replay at C++ dispatcher level | Misses Python control flow; incorrect for data-dependent branches |
| **torch.jit.script** | Subset of Python + type annotations | User must rewrite code; doesn't support all Python |
| **torch.fx.symbolic_trace** | Proxy objects at Python level | Unsound — burns in constants, silently drops side effects (globals, randomness) |
| **ONNX export** | Export to ONNX IR | Limited operator coverage; no training support |

Example of trace failure:
```python
def example(x):
    if len(torch.nonzero(x)) > 1:
        return x + 1
    return x - 1
```
`torch.jit.trace` would capture only one path and produce incorrect results for the other.

Example of symbolic_trace unsoundness:
```python
def example(x):
    global call_count
    call_count += 1           # silently dropped
    return torch.rand(10) + x # burned into graph as constant
```

### TorchDynamo: Bytecode-Level Graph Capture

**Core mechanism**: PEP 523 frame evaluation hook. When CPython calls a function, it creates a `PyFrameObject` and calls the `eval_frame` hook. TorchDynamo replaces the standard CPython interpreter loop with one that:

1. **Analyzes bytecode** of the function being executed
2. **Identifies PyTorch operations** and extracts them into an FX graph
3. **Handles control flow** by treating non-PyTorch code as graph breaks — execute eagerly, resume compilation after
4. **Caches compiled graphs** with guards that check assumptions on subsequent calls

**Guards system**: Before reusing a compiled graph, TorchDynamo checks guards — conditions that must hold for the graph to be valid (shapes, dtypes, global values, etc.). If a guard fails, the function is recompiled. This enables specialization for static shapes while gracefully handling dynamism.

**Graph breaks**: When TorchDynamo encounters code it cannot capture (e.g., non-PyTorch libraries, complex Python), it inserts a graph break — the code before the break is compiled, the uncapturable code runs eagerly, and compilation resumes after. This preserves PyTorch's eager-mode flexibility while still compiling the majority of the computation.

### TorchInductor: The Default Backend

Translates the FX graph into optimized kernels:

- **GPU**: OpenAI Triton kernels (with autotuning for cuBLAS, CUTLASS, Triton backends)
- **CPU**: C++ with OpenMP and vectorization
- **Fusions**: Pointwise operations fused with reductions and matmuls
- **Memory planning**: Reuses buffers across operations to reduce peak memory

### Results (180+ Models on A100)

| Mode | Geometric Mean Speedup |
|------|----------------------|
| Inference | **2.27×** |
| Training | **1.41×** |

### Why Bytecode Level?

The key design choice: operating at the Python **bytecode** level rather than the Python **source** level or the C++ **dispatcher** level. This gives TorchDynamo several advantages:

- **Captures all PyTorch operations** regardless of how they're invoked (direct calls, through wrappers, in loops)
- **Transparent to user code** — no annotations or rewrites needed
- **Handles control flow naturally** — bytecode contains the actual execution path
- **Minimal overhead** — the analysis is cached and guards are checked quickly

## Connections

- [PyTorch Profiling & Tuning](aspe-pytorch-profiling-tuning.md) — Using torch.compile with PyTorch Profiler, Nsight, Triton
- [CUDA Kernel Optimization](aspe-cuda-kernel-optimization.md) — What torch.compile generates under the hood
- [CUDA Graphs & Orchestration](aspe-cuda-graphs-and-orchestration.md) — CUDA Graphs that torch.compile captures
- [AI Systems Performance Engineering](aspe-overview.md) — Full book with torch.compile chapter
