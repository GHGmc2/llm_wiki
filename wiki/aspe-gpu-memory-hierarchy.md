---
title: "GPU Memory Hierarchy and CUDA Programming Model"
type: concept
tags: [cuda, gpu, memory-hierarchy, shared-memory, registers, tiling, hbm, tensor-cores]
created: 2026-05-02
updated: 2026-08-15
sources: [raw/ai-systems-performance-engineering.pdf]
status: stable
---

# GPU Memory Hierarchy and CUDA Programming

**Source**: AI Systems Performance Engineering, Chapters 6-7 [src](../raw/ai-systems-performance-engineering.pdf)

## Key Points

- GPU memory is deeply hierarchical: from small/fast registers up to large/slow HBM
- **Data locality is everything** — the ratio of compute to memory access (arithmetic intensity) determines performance
- Shared memory (SMEM) is programmer-managed L1 cache — key to tiling strategies
- Tensor Memory Accelerator (TMA) enables async data movement between HBM and SMEM/TMEM
- FP8/FP4 precision reduces memory pressure across the entire hierarchy
- Occupancy = number of active warps per SM — too low = idle hardware, too high = register pressure

## The GPU Memory Hierarchy

From fastest to slowest, smallest to largest:

| Level | Size (per SM) | Latency | Bandwidth | Managed By |
|-------|-------------|---------|-----------|-----------|
| **Registers** | 256 KB (64K × 32-bit) | ~0 cycles | ~8 TB/s | Compiler |
| **SMEM (Shared)** | Up to 228 KB (configurable) | ~20 cycles | ~TB/s | Programmer |
| **L1 Cache** | Shared with SMEM | ~30 cycles | ~TB/s | Hardware |
| **TMEM (Tensor Memory)** | 256 KB per SM | ~0 cycles | ~Tens TB/s | Programmer/TMA |
| **L2 Cache** | Up to 96 MB (device-wide) | ~200 cycles | ~TB/s | Hardware |
| **HBM (Global)** | Up to 288 GB (H200) | ~500+ cycles | ~8 TB/s (H200) | Programmer |

Key insight: **Registers are 256 KB per SM but shared across all threads.** A thread using too many registers limits occupancy — the compiler may spill to local memory (which lives in HBM, very slow).

### SMEM (Shared Memory)

- Programmer-managed on-chip SRAM
- Shared across threads within a thread block
- 32 banks, 4 bytes wide per bank
- **Bank conflicts**: Multiple threads accessing the same bank (different addresses) in the same cycle → serialized. Mitigated by padding or swizzling

**Best practice**: SMEM is the workhorse for tiling. Load a tile from HBM → compute in SMEM → write results back. This is the core pattern in FlashAttention, CUTLASS GEMM, and all high-performance CUDA kernels.

### TMEM (Tensor Memory)

- Dedicated 256 KB per-SM buffer for Tensor Cores
- Transparently communicates with Tensor Cores at tens of TB/s
- Reduces Tensor Core reliance on global memory
- Fed by TMA (Tensor Memory Accelerator) for async tile loads

**Example TMEM usage (C = A × B GEMM)**:
- Operand A: in TMEM
- Operand B: from SMEM
- Accumulator C: in TMEM
- Tiles flow HBM → L2 → SMEM via TMA (`cuda::memcpy_async`)

## Arithmetic Intensity and the Roofline Model

**Arithmetic intensity** = FLOPs / bytes accessed from HBM.

The roofline model tells you whether a kernel is:
- **Memory-bound**: Below the ridge point — limited by HBM bandwidth. Focus on reducing data movement (FP8, tiling, kernel fusion)
- **Compute-bound**: Above the ridge point — limited by compute throughput. Focus on increasing FLOPs utilization (Tensor Cores, ILP)

**H100 ridge point**: ~30-40 FLOPs per byte (depending on precision)

## Tiling: The Universal Optimization

Tiling is the process of breaking large matrices into smaller tiles that fit in on-chip memory:

```
Without tiling (naive matmul):
  for each output element:
    read entire row from HBM → compute → write to HBM
  → $O(N^3)$ HBM accesses

With tiling:
  for each tile:
    load tile from HBM → SMEM (once)
    for each thread block processing that tile:
      compute entirely from SMEM
    write results to HBM
  → O(N³/tile_size) HBM accesses
```

**Multilevel microtiling** (Ch 9): Tiling is applied recursively — tiles from HBM → L2 → SMEM → registers. Each level reduces data movement to the next slower memory.

## Occupancy

**Occupancy** = active warps / maximum warps per SM.

- **Too low**: Hardware sits idle — not enough warps to hide latency
- **Too high**: Register pressure — warps spill registers to local memory, killing performance

**Occupancy tuning factors**:
- Thread block size (too small = low occupancy)
- Register usage per thread (too many = compiler spills)
- Shared memory per block (too much = fewer concurrent blocks)

**Achieved vs theoretical occupancy**: The hardware may not reach theoretical max due to tail effects, wave quantization, and SM-level resource allocation granularity.

## Memory Coalescing

GPUs are optimized for **coalesced memory access**: when threads in a warp access consecutive addresses, the hardware combines them into a single wide memory transaction.

**Coalesced** (good):
```
Thread 0 → addr[0], Thread 1 → addr[1], ..., Thread 31 → addr[31]
→ 1 × 128-byte transaction
```

**Strided** (bad):
```
Thread 0 → addr[0], Thread 1 → addr[32], ..., Thread 31 → addr[992]
→ 32 × transactions, most data unused
```

**Misaligned** (bad):
```
Thread 0 → addr[4], Thread 1 → addr[8], ... (not 128-byte aligned)
→ 2 × transactions with overlap
```

## Async Memory: TMA and cuda::memcpy_async

TMA (Tensor Memory Accelerator, H100+):
- Hardware unit for async tile loads from HBM → SMEM
- Frees up threads for computation during data movement
- Supports multidimensional tile descriptors

```cpp
// Declare a TMA descriptor for 2D tiles
cuda::memcpy_async_tile(tile, desc, SMEM_buffer);
// ... compute on previous tile while this loads ...
cuda::pipeline_memcpy_async_commit(pipe, SMEM_buffer);
// ... later, wait for tile ...
cuda::pipeline_consumer_wait(pipe);
// ... compute on this tile ...
```

## Connections

- [CUDA Kernel Optimization](aspe-cuda-kernel-optimization.md) — Occupancy tuning, warp efficiency, ILP
- [CUDA Graphs & Orchestration](aspe-cuda-graphs-and-orchestration.md) — Streams, graphs, atomics
- [Compute Efficiency Wall](megatron-core-moe.md) — How these techniques apply to MoE
- [AI Systems Performance Engineering](aspe-overview.md) — Full book reference
