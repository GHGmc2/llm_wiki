---
title: "PyTorch Profiling, Compilation, and Performance Tuning"
type: concept
tags: [pytorch, profiling, torch-compile, nsight, triton, cutlass, performance]
created: 2026-05-02
updated: 2026-05-02
sources: [raw/ai-systems-performance-engineering.pdf]
status: stable
---

# PyTorch Profiling, Compilation, and Performance Tuning

**Source**: AI Systems Performance Engineering, Chapter 13 [src](raw/ai-systems-performance-engineering.pdf)

## Key Points

- **PyTorch Profiler**: Framework-level tracing — identify CPU bottlenecks, GPU idle time, data loading stalls
- **torch.compile**: Automatic kernel fusion and CUDA Graph capture — achieves near-custom-kernel performance
- **Nsight Systems + NVTX**: System-level profiling with annotated PyTorch operations
- **Roofline analysis**: Quantitatively determines if kernel is memory-bound or compute-bound
- **Triton**: Write custom fused kernels in Python — easier than CUDA C++, nearly as fast
- **Profile-guided optimization is iterative**: Measure → optimize → measure — never guess

## PyTorch Profiler

```python
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=1, active=3),
    on_trace_ready=tensorboard_trace_handler('./logs'),
) as prof:
    for step in range(steps):
        train_step()
        prof.step()
```

**What to look for**:
- **GPU idle gaps**: Long periods between kernel launches → CPU bottleneck or data loading
- **CPU utilization**: Main thread stalled → Python overhead, data preprocessing
- **Memory allocation peaks**: OOM risk points, fragmentation
- **NCCL collectives**: Communication time vs compute time

## Nsight Systems + NVTX Markers

NVTX markers annotate PyTorch code so Nsight Systems can show high-level operations:

```python
import torch.cuda.nvtx as nvtx

nvtx.range_push("forward_pass")
output = model(input)
nvtx.range_pop()

nvtx.range_push("backward_pass")
loss.backward()
nvtx.range_pop()
```

**Nsight Systems timeline** shows: GPU kernels, CUDA API calls, NCCL collectives, memory operations — all aligned with NVTX ranges.

## torch.compile: Automatic Optimization

PyTorch 2.0's compiler fuses operations and captures CUDA Graphs:

```python
# Basic usage
model = torch.compile(model)

# Mode options
model = torch.compile(model, mode="default")          # balanced
model = torch.compile(model, mode="reduce-overhead")  # CUDA Graphs, small batches
model = torch.compile(model, mode="max-autotune")     # training, stable shapes
```

**What it does**:
- **Kernel fusion**: Merges multiple operations into single GPU kernel — reduces HBM round-trips
- **CUDA Graph capture**: Record and replay for fixed-shape workloads
- **Autotuning**: Benchmarks multiple kernel variants, picks fastest

**When to use each mode**:
| Mode | Use Case |
|------|----------|
| `default` | General purpose, quick compilation |
| `reduce-overhead` | Small-batch inference, graph-friendly loops |
| `max-autotune` | Long-running training, stable shapes |
| `max-autotune-no-cudagraphs` | When CUDA Graphs cause issues |

## CUTLASS vs Triton vs torch.compile

| Approach | Performance | Effort | When to Use |
|----------|------------|--------|-------------|
| `torch.compile` | Good-excellent | Minimal (one line) | First try — works for most cases |
| **Triton** | Excellent | Moderate (Python DSL) | Custom fused kernels, irregular patterns |
| **CUTLASS** | Peak | High (C++ templates) | Extreme optimization, fixed GEMM shapes |
| **Hand-written CUDA** | Peak | Very high | Only when nothing else works |

## Roofline Analysis Workflow

1. **Run Nsight Compute** on your kernel → get achieved FLOPs and memory throughput
2. **Plot on roofline**: Your kernel's point vs the hardware's theoretical limits
3. **Diagnose**: 
   - **Memory-bound** (below ridge): Reduce data movement — FP8, tiling, fusion
   - **Compute-bound** (below compute roof): Increase FLOPs — Tensor Cores, ILP
   - **Latency-bound** (both low): Increase concurrency — streams, CUDA Graphs
4. **Apply optimization, re-measure**

## Reducing Kernel Launch Overhead

PyTorch can issue many small GPU operations serially. This creates host-side bottlenecks:

```python
# BAD: N separate GPU launches (sequential)
for i in range(N):
    C[i] = A[i] + B[i]

# GOOD: Single vectorized launch
C = A + B

# BEST: torch.compile fuses this automatically
@torch.compile
def fused_op(A, B):
    return A + B
```

**Under the hood**: PyTorch's built-in kernels use launch configurators that pick optimal block sizes and launch multiple blocks per SM — automated occupancy tuning.

## Linux perf for CPU Profiling

When the bottleneck is on the CPU side:

```bash
perf record -g python train.py
perf report
```

Identifies: Python GIL contention, data preprocessing hotspots, I/O wait, thread synchronization overhead.

## Connections

- [CUDA Kernel Optimization](aspe-cuda-kernel-optimization.md) — What torch.compile generates under the hood
- [CUDA Graphs & Orchestration](aspe-cuda-graphs-and-orchestration.md) — CUDA Graphs that torch.compile captures
- [GPU Storage I/O](aspe-gpu-storage-io.md) — Data pipeline profiling
- [AI Systems Performance Engineering](aspe-overview.md) — Full book reference
