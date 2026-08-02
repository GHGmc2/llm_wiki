---
title: "AI Systems Performance Engineering"
type: source-note
tags: [performance, cuda, gpu, pytorch, profiling, inference, training, systems, book]
created: 2026-05-02
updated: 2026-07-09
sources: [raw/ai-systems-performance-engineering.pdf]
status: stable
---

# AI Systems Performance Engineering

**Source**: Chris Fregly, O'Reilly Media, November 2025. 1061 pages. "The missing reference manual" for AI performance.

**Endorsed by**: Chris Lattner (Modular), Mark Saroufim (Meta/PyTorch), Sebastian Raschka, vLLM maintainers.

## Key Points

- **Full-stack reference**: From CUDA kernel micro-optimizations up to multi-rack cluster orchestration
- **175+ item optimization checklist** covering every layer of the AI stack
- Hands-on with real tools: Nsight Systems/Compute, PyTorch Profiler, CUTLASS, NCCL, vLLM
- Focus on **hardware-software co-design** — "mechanical sympathy" for GPUs
- Covers both training and inference at scale

## Book Scope

The book is organized from low-level (silicon) to high-level (cluster orchestration):

### Part 1: Hardware & Infrastructure (Ch 1-5)

| Chapter | Topic |
|---------|-------|
| 1 | AI System Overview — benchmarking, profiling, DeepSeek scaling story, NVIDIA roadmap |
| 2 | GPU Hardware — H100/B200/GB200, NVLink, NVSwitch, InfiniBand, rack appliances, liquid cooling, Feynman GPU (2028) |
| 3 | OS & Container Tuning — NVIDIA driver stack, Docker, Kubernetes topology manager, MIG slicing, OOM killer avoidance |
| 4 | Distributed Networking — NCCL collective tuning, GPUDirect RDMA, NVSHMEM, RoCE vs IB, congestion control |
| 5 | Storage I/O — NVMe tuning, NVIDIA GDS (GPUDirect Storage), DeepSeek 3FS, NeMo Curator, DALI |

### Part 2: CUDA & GPU Programming (Ch 6-12)

| Chapter | Topic |
|---------|-------|
| 6 | CUDA Programming Model — threads, warps, blocks, grids, streams |
| 7 | GPU Memory Hierarchy — global, shared, registers, L1/L2 cache, tiling, warp shuffle, TMA (Tensor Memory Accelerator) |
| 8 | Occupancy & Warp Efficiency — Nsight Compute roofline analysis, warp stall reasons, register pressure |
| 9 | Kernel Efficiency & Arithmetic Intensity — multilevel microtiling, kernel fusion, structured sparsity, Tensor Cores, CUTLASS, inline PTX/SASS |
| 10 | Thread Block Clusters — cooperative groups, warp specialization, distributed shared memory |
| 11 | Inter-Kernel Pipelining — CUDA streams, stream-ordered memory allocator, events, overlapping compute with transfers |
| 12 | CUDA Graphs & Dynamic Parallelism — graph capture, conditional nodes, multi-GPU graphs with NCCL, NVSHMEM |

### Part 3: PyTorch & Frameworks (Ch 13+)

| Chapter | Topic |
|---------|-------|
| 13 | PyTorch Profiling & Tuning — torch.compile, NVTX markers, Linux perf, kernel roofline, Triton kernels |
| 14+ | Distributed Training — FSDP, TP, PP, EP, activation checkpointing, gradient accumulation, mixed precision |
| Later | Inference Optimization — vLLM, continuous batching, KV cache management, quantization, speculative decoding |

## Key Techniques Covered

### GPU Kernel Optimization
- **Tiling**: Multilevel microtiling for data reuse in shared memory and registers
- **Warp shuffle intrinsics**: Avoid shared memory for warp-level reductions
- **Tensor Cores**: Feeding with TMA (Tensor Memory Accelerator), TMEM management
- **FP8/FP4/INT8**: Mixed precision via Transformer Engine, CUTLASS
- **Structured sparsity**: 2:4 sparsity for 2$\times$ throughput on Tensor Cores
- **Inline PTX/SASS**: Micro-optimizations at the assembly level

### Profiling Methodology
- **Nsight Systems**: Timeline view for system-level bottlenecks
- **Nsight Compute**: Kernel-level roofline analysis, warp stall diagnosis
- **PyTorch Profiler**: Framework-level tracing with NVTX markers
- **Linux perf**: CPU-side profiling
- **Profiler-guided iterative optimization workflow**

### Distributed Training
- NCCL collective tuning (algorithm, protocol, channel selection)
- GPUDirect RDMA and NVSHMEM for GPU-initiated communication
- FSDP, TP, PP, CP, EP combinations
- Activation checkpointing strategies
- Gradient accumulation and overlap

### Inference Optimization
- Continuous batching, PagedAttention
- KV cache quantization and management
- Speculative decoding
- Disaggregated prefill/decode

## Connections to Existing Wiki Pages

This book is the **practical implementation manual** for the concepts in the wiki. Each chapter maps to specific wiki content:

| Wiki Page | Book Chapters |
|-----------|-------------|
| [GPU Memory Hierarchy](aspe-gpu-memory-hierarchy.md) | Ch 6-7 — CUDA model, memory hierarchy, tiling, TMEM, TMA |
| [CUDA Kernel Optimization](aspe-cuda-kernel-optimization.md) | Ch 8-9 — Occupancy, warp efficiency, kernel fusion, ILP, Tensor Cores |
| [CUDA Graphs & Orchestration](aspe-cuda-graphs-and-orchestration.md) | Ch 11-12 — Streams, graphs, atomics, dynamic scheduling, NVSHMEM |
| [Inference Optimization](aspe-inference-optimization-techniques.md) | Ch 17-19 — Disaggregation, autotuning, precision switching, goodput |
| [Compute Efficiency Wall](megatron-core-moe.md) | Ch 7-9, 12 — Kernel optimization, CUDA Graphs for MoE |
| [Communication Wall](megatron-core-moe.md) | Ch 4 — NCCL, GPUDirect, NVSHMEM |
| [Memory Wall](megatron-core-moe.md) | Ch 5, 7 — Storage I/O, memory hierarchy, offloading |
| [NCCL Demystifying](nccl-demystifying.md) | Ch 4, 12 — NCCL internals, CUDA Graphs + NCCL |
| [Scaling Techniques Overview](usp-scaling-techniques-overview.md) | Ch 14+ — Distributed training patterns |
| [Ultra-Scale Playbook](usp-ultra-scale-playbook.md) | Ch 1 — DeepSeek scaling story, Ch 4 — networking |

## DeepSeek Coverage

The book opens with DeepSeek-V3 as a case study in hardware-software co-design under constraints — training a ~680B MoE model on 2,048 H800 GPUs despite US export restrictions. This mirrors the [V3 Insights](deepseek-v3-insights.md) paper.

## NVIDIA Roadmap (from the book)

| GPU | Year | Key Feature |
|-----|------|-------------|
| Blackwell Ultra / Grace Blackwell Ultra | 2025-26 | Higher memory, FP4 support |
| Vera Rubin Superchip | 2026 | Next-gen architecture |
| Rubin Ultra / Vera Rubin Ultra | 2027 | Ultra-scale |
| Feynman GPU | 2028 | "Doubling something every year" |

## Connections

- [PyTorch Compilation](pytorch-compilation.md) — torch.compile deep-dive; covered in the book's compilation chapter
