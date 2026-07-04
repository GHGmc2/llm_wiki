---
title: "Tensor Core Evolution: From Volta to Blackwell"
type: source-note
tags: [gpu, nvidia, tensor-cores, volta, ampere, hopper, blackwell, ptx, sass, mma, wgmma, tcgen05]
created: 2026-05-03
updated: 2026-05-03
sources: [https://newsletter.semianalysis.com/p/nvidia-tensor-core-evolution-from-volta-to-blackwell, https://newsletter.semianalysis.com/p/dissecting-nvidia-blackwell-tensor]
status: stable
---

# Tensor Core Evolution: Volta → Blackwell

**Source**: SemiAnalysis (Dylan Patel, Kimbo Chen), June 2025 / March 2026. Two-part deep dive: architecture evolution + Blackwell microbenchmarking.

## Key Points

- Tensor Cores are the foundation of modern AI — they deliver "Huang's Law" performance improvements by amortizing instruction overhead across many FLOPs per instruction
- Six generations of evolution: from warp-scoped MMA (Volta) to single-thread-initiated `tcgen05` (Blackwell)
- Each generation reduced register pressure and moved data closer to Tensor Cores: registers → shared memory → Tensor Memory (TMEM)
- Blackwell's `tcgen05.mma` removes registers from MMA entirely — operands in SMEM and TMEM only
- 2SM MMA splits operands across CTA pairs for near-perfect weak scaling

## Part 1: Architecture Evolution

### Why Tensor Cores Exist

The motivation: a simple FP32 FMA instruction costs ~1.5pJ in the floating-point unit, but ~30pJ to issue the instruction. That's a **20× instruction overhead**. Tensor Cores solve this by issuing one complex instruction (HMMA/WGMMA/tcgen05) that performs an entire matrix multiply-accumulate — amortizing the instruction cost across many FLOPs.

### Pre-Tensor Core (CUDA Programming Model)

- **PTX**: Virtual ISA abstracting GPU generations. Threads → warps (32 threads) → CTAs → grid
- **SASS**: Architecture-specific assembly (undocumented, reverse-engineered)
- **Memory hierarchy**: per-thread registers, per-CTA shared memory, global memory (all CTAs)

### Volta (V100, 2017) — 1st Gen

**MMA**: `mma.sync` — warp-scoped, quadpair of 8 threads. Shape: 8×8×4. FP16 inputs, FP32 accumulation.

**Why**: Added late in development (months before tapeout) after seeing TPUv1's success. Each SM has 8 Tensor Cores, 1024 FLOPs/cycle/SM.

**Limitation**: Complex interleaved thread-data layout — high register pressure. Data must go through registers (HBM → registers → shared memory → registers → Tensor Core).

```
HMMA.884.F32.F16 → 8×8×4 FP16 MMA
Quadpair: [T0-T3] + [T16-T19] collectively hold matrices
```

### Turing (RTX 20, 2018) — 2nd Gen

Added INT8 and INT4 support. Introduced warp-level synchronous MMA (full 32-thread warp). Enabled DLSS in gaming.

### Ampere (A100, 2020) — 3rd Gen

**Async copy** (`cp.async` / `LDGSTS`): Load from global memory → shared memory directly, bypassing registers. Reduces register pressure significantly.

**Warp-level MMA**: Full 32-thread warp participates (vs Volta's quadpair). Simpler thread layout, lower register pressure. BF16 support — becomes the de facto half-precision standard. 2× throughput over Volta (512 FLOPs/cycle/Tensor Core, 2048 FLOPs/cycle/SM).

**ldmatrix**: Warp-wide vectorized load that matches Tensor Core's data layout — further reduces register usage for address generation.

```
HMMA.16816.F32.BF16 → 16×8×16 BF16 MMA
Full warp of 32 threads collectively hold input/output
```

### Hopper (H100, 2022) — 4th Gen

**Thread Block Cluster**: Group of CTAs co-scheduled on same GPC. DSMEM enables cross-CTA shared memory access through SM-to-SM network.

**TMA** (`cp.async.bulk.tensor` / `UTMALDG`): Dedicated hardware unit for async bulk data transfers. Single thread initiates copy. Address generation, swizzling, bounds-checking handled by hardware. Multicast mode loads same data to multiple SMs in a cluster.

**Warpgroup MMA** (`wgmma.mma_async` / `HGMMA`): 4-warps collectively perform MMA. Operands loaded from shared memory (not registers). Supports wider shapes (m64nNk16, N=8 to 256). FP8 (E4M3, E5M2) with FP32 accumulation.

**Key change**: Tensor Cores read from shared memory directly — registers used only for accumulator, freeing register space for other work.

```
HGMMA.64128.F32.E4M3 → 64×128 FP8 MMA
Warpgroup of 4 warps, operand A in SMEM or registers, B in SMEM
```

### Blackwell (B200, 2024) — 5th Gen

**Tensor Memory (TMEM)**: 256 KB per SM of specialized on-chip memory for Tensor Core operands. 128 lanes × 512 columns × 4 bytes. Restricted access pattern — warpgroup needed to access full TMEM.

**CTA Pair**: Two CTAs differing by last rank bit form a pair mapped to one TPC. Can share input operands — halves SMEM bandwidth for matrix B.

**tcgen05.mma**: Single-thread semantics — one thread initiates MMA for the entire CTA. Removes warps from MMA entirely. Operands in SMEM and TMEM only, **no registers involved**.

**2SM MMA** (`.cta_group::2`): CTA pair collaboratively executes one MMA. Matrix A duplicated, B and D sharded across 2 SMs. Near-perfect 2× weak scaling.

**New precisions**: FP4, FP6 with micro-scaling (MXFP4, MXFP6).

```
tcgen05.mma.cta_group::2 → 2SM MMA
Single thread initiates, TMEM for accumulators
```

## Part 2: Blackwell Deep Dive

### GPC Floorsweeping

Blackwell's GPCs have **yielded SMs** due to manufacturing defects. Software sees logical GPC groupings that may not reflect physical layout:

- B200 has two dies connected via NV-HBI
- L2 latency benchmark reveals ~300 cycle die-to-die crossing penalty
- Not all SMs in a GPC are usable — cluster sizes must account for yielded SMs
- Preferred + fallback cluster sizes enable full GPU utilization despite yields

### Memory Subsystem Benchmarks

**LDGSTS (Async Copy)**:
- Saturates at ~6.6 TB/s at 32 KiB in-flight
- 16-byte loads preferred — same throughput as 8B at half the stages
- Baseline latency ~600ns, doubles after 8 KiB in-flight
- MIO throttle dominates at high bytes-in-flight

**TMA (Tensor Memory Accelerator)**:
- Peak throughput reached at much higher bytes-in-flight than LDGSTS
- Lower throughput at small sizes (< 32 KiB), catches up and surpasses after
- Higher latency than LDGSTS at larger sizes

**Choice**: LDGSTS for small irregular loads (MLA kernels), TMA for large regular loads (MHA kernels, GEMM patterns). FlashInfer uses LDGSTS for MLA and TMA for MHA.

### TMA Multicast

Three scenarios benchmarked:
1. **Baseline**: Each SM loads different data
2. **Explicit multicast**: One CTA issues multicast to all CTAs in cluster
3. **Implicit multicast**: All CTAs issue plain TMA to same data (L2 Request Coalescer handles dedup)

**Finding**: LRC (L2 Request Coalescer) provides effective implicit multicast — nearly matches explicit at < 64 bytes in-flight. Beyond that, explicit multicast needed for full L2 traffic reduction.

### DSMEM vs SMEM

| Access Pattern | Throughput (B/clk) |
|---------------|-------------------|
| Local SMEM (`ld.shared`) | 128 |
| DSMEM (`ld.shared::cluster`) | ~6-8 |
| DSMEM via `UBLKCP` (bulk copy) | ~20-30 |

**Key pitfall**: Using `ld.shared::cluster` for local access emits generic `LD` instead of `LDS` → loses peak local SMEM bandwidth. Use plain `ld.shared` for local, `UBLKCP` for remote.

### MMA Performance Characterization

**Shape dependency**: N dimension (inner dimension) determines whether instruction is SMEM-bound or compute-bound. Below N=128 for SS (both operands in SMEM): SMEM bandwidth limited.

**Roofline analysis for FP8 1SM MMA**:
```
A_bytes = 2*M*K, B_bytes = 2*N*K
SMEM Cycles = (A_bytes + B_bytes) / 128 B/clk
Math Cycles = FLOPs / 16384 FLOPs/clk
→ SMEM-bound for N < 128, compute-bound for N ≥ 128
```

**AB Layout**: TS (A in TMEM, B in SMEM) outperforms SS (both in SMEM) for N < 128. For N ≥ 128, both reach near-peak.

**2SM MMA scaling**: Near-perfect 2× weak scaling. Over 2× for SS mode < N=128 (splits B operand between SMs, reducing per-SM bandwidth).

**Speed-of-Light**: At 4 in-flight MMAs (typical kernel), caps at 78-80% SoL. Only largest N (256) reaches 90% at high in-flight counts.

**Data type latency ordering**:
```
S8 < BF16 ≈ E4M3 ≈ F4 < MXF8 < MXF4
```
Integer fastest; micro-scaling formats add minor scale factor computation overhead.

## Generational Summary

| Generation | GPU | MMA Scope | Operand Location | Key Innovation | FLOPs/clk/SM |
|-----------|-----|-----------|-----------------|----------------|-------------|
| 1st (Volta) | V100 | Warp (quadpair) | Registers → TC | `HMMA` instruction | 1,024 |
| 2nd (Turing) | RTX 20 | Warp (full) | Registers → TC | INT8/INT4 support | 1,024 |
| 3rd (Ampere) | A100 | Warp (full) | Registers → TC | Async copy, BF16 | 2,048 |
| 4th (Hopper) | H100 | Warpgroup (4 warps) | SMEM → TC | TMA, DSMEM, FP8 | 3,072+ |
| 5th (Blackwell) | B200 | Single thread | SMEM/TMEM → TC | TMEM, 2SM MMA, FP4 | 4,096+ |

**The trend**: Each generation moves data closer to compute and reduces the programming abstraction scope. Volta required explicit register management by every thread. Blackwell lets one thread fire the entire MMA while hardware manages data movement through TMEM.

## Connections

- [GPU Memory Hierarchy](aspe-gpu-memory-hierarchy.md) — Registers, SMEM, TMEM details
- [NVIDIA GPU Architecture Microbenchmarks](nvidia-gpu-architecture-microbenchmarks.md) — Ampere/Hopper/Blackwell instruction latency benchmarks
- [CUDA Kernel Optimization](aspe-cuda-kernel-optimization.md) — Occupancy, warp efficiency, Tensor Core utilization
- [GPU Hardware Architecture](aspe-gpu-hardware-architecture.md) — NVL72, NVSwitch, roadmap
- [FP8/FP4 Training](megatron-core-moe.md) — FP8/FP4 on Tensor Cores
- [Compute Efficiency Wall](megatron-core-moe.md) — MoE-specific kernel optimization
