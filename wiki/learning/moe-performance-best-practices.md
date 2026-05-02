---
title: "MoE Performance Best Practices"
type: concept
tags: [llm-training, mixture-of-experts, performance-tuning, megatron, gpu, hardware]
created: 2026-05-02
updated: 2026-05-02
sources: [raw/scalable-training-moe-megatron-core.pdf]
status: stable
---

# MoE Performance Best Practices

**Source**: Megatron-Core MoE Technical Report, Section 9 [src](raw/scalable-training-moe-megatron-core.pdf)

## Key Points

- Having many optimization techniques does not automatically yield high performance — each has trade-offs and interactions
- A **systematic three-phase workflow**: (1) establish memory feasibility, (2) select optimal parallelism, (3) profile and target bottlenecks
- The same model requires **completely different optimization strategies** on different hardware (see DeepSeek-V3 case study)
- The process is **iterative**: solving one bottleneck exposes the next — continuous profiling guides each round

## The Three-Phase Workflow

Through tuning Mixtral, DeepSeek-V3, and Qwen3 across GB200 and H100, a repeatable methodology emerged:

### Phase 1: Establish Memory-Feasible Parallelism

Memory is the hard constraint. If the config does not fit in GPU memory, training cannot proceed.

**Parallelism impact on memory**:

| Strategy | Peak Activation | Weight Memory | Optimizer States | Comm Overhead |
|----------|----------------|---------------|-----------------|---------------|
| TP | 1/d | 1/d (with SP) | 1/d | High |
| EP | ~1 (load-dep.) | 1/d (MoE only) | 1/d | Medium |
| PP | 1 (>1 with VPP) | 1/d | 1/d | Medium |
| CP | 1/d | 1 | 1/d (with dist-opt) | Medium |
| DP | 1 | 1 | 1/d (with dist-opt) | Low |

**Quick testing**: Use `--fake-init-process-group` to emulate distributed training on a single GPU, or the interactive memory simulator with web GUI.

**Example**: DeepSeek-V3 BF16 activations alone exceed 130 GB/GPU, ruling out baseline configs on 80 GB devices before any parallelism.

### Phase 2: Select Optimal Parallelism Strategy

Five guidelines for minimizing communication overhead:

**Guideline 1: Minimize Model Parallelism, Maximize Data Parallelism**
- Keep TP/EP/PP/CP as small as possible while avoiding OOM
- Use distributed optimizer to shard states across DP ranks

**Guideline 2: Keep EP and TP Within NVLink Domain**
- EP x TP should fit within NVLink (8 GPUs on NVL8, 72 on NVL72)
- NVLink bandwidth (900 GB/s H100, 1.8 TB/s GB200) far exceeds cross-node
- When scaling beyond NVLink, prefer PP over expanding TP/EP across nodes

> For very large models, EP volume may exceed NVLink capacity even within a node. Enable overlap (see [Communication Wall](moe-communication-wall.md)).

**Guideline 3: Use PP for Multi-Node Scaling**
- Distribute layers across nodes while keeping EPxTP within NVLink
- Enable VPP when PP >= 2 to reduce pipeline bubbles
- Balance workloads across VPP ranks

**Guideline 4: Prefer EP over TP for Expert Layers**
- EP offers larger local matrices, less communication, easier overlap
- When EP = num_experts, local token permutation is eliminated
- Example: Mixtral-8x7B: EP8xTP1 outperforms EP4xTP2

**Guideline 5: Enable CP for Long Sequences (>= 8K tokens)**
- Avoid CP for sequences < 4K (overhead exceeds benefit)
- CP efficiency depends on communication-computation overlap

**Example**: 256-expert MoE on NVL72. Parallel Folding sets ETP=1 (Guideline 4). EP64 fits in NVLink (Guideline 2). Remaining budget determines TP/PP for attention (Guideline 1), DP fills the rest.

### Phase 3: Profile and Optimize Bottlenecks

Profile to identify the dominant wall:

| Bottleneck | Symptom | Solutions |
|-----------|---------|-----------|
| **Memory** | Forced full recompute or excessive parallelism | FP8, selective recompute, precision-aware optimizer, offloading |
| **Communication** | Significant time in collectives | Identify which communication and apply targeted fix |
| **CPU Overhead** | Gaps between GPU kernels | CUDA Graphs, disable Python GC, reduce kernel launches |
| **Computation** | Low SM utilization, no comm/CPU issues | Grouped GEMM, kernel fusions, FP8 precision |

**Iterative nature**: Memory optimizations may enable smaller parallelism (back to Phase 1). Some Phase 3 optimizations have memory costs (EP overlap buffers, CUDA Graphs), requiring revisit.

## Case Study: DeepSeek-V3 on GB200 vs H100

DeepSeek-V3 (685B, 256 experts, top-8, MTP + MLA) shows how the same model demands completely different strategies on different hardware.

### Final Configurations

| Configuration | GB200 | H100 |
|--------------|-------|------|
| Hardware | 256 x GB200 | 1024 x H100 |
| Parallelism (TP/PP/EP) | 1/4/64 | 2/8/64 |
| VPP | 4 | 4 |
| GBS / MBS / SeqLen | 8192 / 1 / 4096 | 8192 / 1 / 4096 |
| Precision | MXFP8 | FP8-Blockwise |
| Dispatcher | HybridEP | DeepEP |
| CUDA Graphs | Enabled | — |
| EP Overlap | — | Enabled |
| **TFLOPS/GPU** | **1,048** | **368** |

### Why They Differ

**Memory capacity drives parallelism**: GB200 (192 GB/GPU) allows TP1/PP4 — shallower pipeline, fewer bubbles. H100 (80 GB/GPU) requires TP2/PP8 — deeper pipeline to fit the model.

Both use EP64: each GPU holds 4 of 256 experts, eliminating local token permutation.

**Memory Wall — different stacks**:

| Technique | GB200 | H100 |
|-----------|-------|------|
| Precision | MXFP8 | Blockwise FP8 |
| Selective recompute | `mlp` only | `mlp`, `mla_up_proj`, `layernorm`, `moe_act` |
| Optimizer offloading | Yes | No |
| FP8 primary weights | Yes | Yes |
| Memory-efficient permutation | Yes | Yes |

On H100, FP8 savings are critical — they free budget for EP communication overlap buffers. GB200's extra memory eliminates that constraint.

**Communication Wall**: On H100/NVL8, EP64 spans 8 nodes — cross-node all-to-all consumes ~50% step time with standard dispatcher. DeepEP + EP overlap are essential. On GB200/NVL72, EP64 stays within NVLink — HybridEP alone suffices, no overlap needed. Hardware topology alone resolves the communication wall.

**Compute Wall — the bottleneck shift**: On GB200, NVL72 eliminates the communication bottleneck, and MXFP8 accelerates GEMMs. But faster GPU computation exposes **CPU overhead** — the host cannot launch kernels fast enough. Now CUDA Graphs, kernel fusions, and CPU/NUMA binding become critical. The bottleneck shifted from communication to CPU.

### Lessons Learned

1. **Hardware topology dictates strategy**: The same model needs different optimization stacks on different hardware
2. **Fixing one bottleneck exposes the next**: The bottleneck shifts memory -> communication -> CPU -> kernel efficiency
3. **FP8 is the highest-leverage optimization**: It attacks all three walls simultaneously
4. **Memory is always the first constraint**: You cannot optimize what you cannot fit
5. **Profile, do not guess**: Nsight Systems reveals the true bottleneck; intuition is often wrong at this scale

## Connections

- [Megatron-Core MoE](megatron-core-moe-scalable-training.md) — full paper summary
- [Parallel Folding](moe-parallel-folding.md) — the parallelism decoupling that enables these configurations
- [Memory Wall](moe-memory-wall.md) — memory optimization techniques
- [Communication Wall](moe-communication-wall.md) — dispatchers and overlap
- [Compute Efficiency Wall](moe-compute-efficiency-wall.md) — CUDA Graphs, Grouped GEMM
- [FP8/FP4 Training](moe-fp8-fp4-training.md) — reduced-precision recipes
