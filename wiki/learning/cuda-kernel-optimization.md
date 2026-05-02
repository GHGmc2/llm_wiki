---
title: "CUDA Kernel Optimization: Occupancy, Warp Efficiency, and Arithmetic Intensity"
type: concept
tags: [cuda, gpu, kernel-optimization, occupancy, warp-divergence, ilp, kernel-fusion, tensor-cores]
created: 2026-05-02
updated: 2026-05-02
sources: [raw/AI Systems Performance Engineering.pdf]
status: stable
---

# CUDA Kernel Optimization

**Source**: AI Systems Performance Engineering, Chapters 8-9 [src](raw/AI%20Systems%20Performance%20Engineering.pdf)

## Key Points

- **Occupancy** hides latency: more active warps → hardware can switch during stalls
- **ILP** (Instruction-Level Parallelism) is equally important: independent operations within a thread hide latency without using more registers
- **Warp divergence** from if/else can serialize execution → all threads in a warp execute both branches
- **Kernel fusion** eliminates HBM round-trips: do all operations in on-chip memory before writing back
- **Tensor Cores** provide 2-8× throughput over CUDA cores but require specific data layouts
- **Nsight Compute roofline analysis** tells you exactly whether you're memory-bound or compute-bound

## Latency Hiding: Occupancy vs ILP

GPUs hide memory latency through parallelism. When one warp stalls on a memory load, the warp scheduler switches to another ready warp. Two approaches:

### Occupancy (More Warps)
- Launch more threads/warps per SM
- When warp A stalls on DRAM, warp B runs
- Requires fewer registers per thread (or larger register file)
- Limited by: register file size, shared memory, max warps/SM

### ILP (More Work Per Warp)
- Each thread issues multiple independent loads before computing
- `load A[idx]; load B[idx];` → both loads in flight simultaneously
- While load A waits for DRAM, load B is already in flight
- **Doesn't require more registers** (unlike occupancy)
- Unroll loops to expose ILP

**Best practice**: Combine both. Achieve decent occupancy (50-75%) and maximize ILP within each warp.

## Warp Divergence

In SIMT execution, threads within a warp that take different code paths execute **serially**:

```cpp
// DIVERGENT - 50% of threads idle in each branch
if (threadIdx.x % 2 == 0) {
    heavy_work_A();  // even threads run, odd threads idle
} else {
    heavy_work_B();  // odd threads run, even threads idle
}
// Total time = time(A) + time(B), throughput halved
```

**Impact**: In the ideal 50/50 split case, removing divergence can nearly double throughput. Multiple divergence points compound the loss.

### Mitigation Strategies

1. **Restructure conditions**: Reorder data so threads in the same warp take the same path
2. **Separate into different kernels**: Launch divergent work as separate kernel invocations
3. **Predicated execution** (for small branches): Use `@p` predicates instead of branches
4. **Warp-level primitives**: Use `__shfl_sync`, `__ballot_sync` for intrawarp communication

## Improving Memory Coalescing

**Coalesced access**: Threads in a warp access consecutive addresses → single wide transaction.

**Strided access**: `threads[i]` accesses `addr[i*stride]` → multiple transactions, most data unused.

**Uncoalesced fix**: Change the mapping of threads to data:
```cpp
// BAD: Uncoalesced (column-major layout, row access)
const int x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
const int y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

// GOOD: Coalesced
const int x = blockIdx.x * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);
const int y = blockIdx.y * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
```
This simple coordinate swap can yield **10× speedup** by eliminating uncoalesced accesses.

## Kernel Fusion

The core idea: avoid HBM round-trips by doing multiple operations in a single kernel:

```
WITHOUT fusion:
  Kernel 1: load A → compute → write to HBM
  Kernel 2: load result → compute → write to HBM
  Kernel 3: load result → compute → write to HBM
  → 6 HBM accesses per element

WITH fusion:
  Fused kernel: load A → compute1 → compute2 → compute3 → write to HBM
  → 2 HBM accesses per element
```

**PyTorch**: `torch.compile` automatically fuses operations. Use `mode="max-autotune"` for training, `"reduce-overhead"` for inference.

**Custom kernels**: Triton is the recommended way to write fused kernels in Python. For extreme optimization, write CUDA C++ with CUTLASS.

## Tensor Core Utilization

Tensor Cores provide 2-8× throughput over CUDA cores but require:
- Specific matrix dimensions (128-byte aligned for FP8)
- Data in TMEM or SMEM (not HBM)
- Appropriate precision format (TF32, BF16, FP8, FP4, INT8)

**FP8 on H100**: Tile-wise 1×128 quantization (activations), block-wise 128×128 (weights). GEMM must pad to 128-divisible dimensions.

## Nsight Compute Roofline Analysis

The roofline model guides optimization strategy:

- **Memory-bound** (below ridge): Focus on reducing data movement — FP8 precision, tiling, kernel fusion, mixed precision
- **Compute-bound** (below compute roof): Focus on FLOPs — Tensor Cores, ILP, FP8/FP4 math
- **Latency-bound** (both low): Increase concurrency — more streams, dynamic scheduling, CUDA Graphs
- **In between**: Run multiple kernels/streams in parallel to utilize all hardware units

## Connections

- [GPU Memory Hierarchy](gpu-memory-hierarchy.md) — Memory architecture fundamentals
- [CUDA Graphs & Orchestration](cuda-graphs-and-orchestration.md) — Beyond single-kernel: streams, graphs, atomics
- [Compute Efficiency Wall](moe-compute-efficiency-wall.md) — MoE-specific kernel challenges
- [AI Systems Performance Engineering](ai-systems-performance-engineering.md) — Full book reference
