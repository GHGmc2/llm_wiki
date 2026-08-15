---
title: "FlashAttention: Memory-Efficient Exact Attention"
type: concept
tags: [attention, flash-attention, gpu, kernel-fusion, tiling, hbm, online-softmax]
created: 2026-05-02
updated: 2026-08-15
sources: [raw/The_Ultra-Scale_Playbook.pdf, raw/ai-systems-performance-engineering.pdf, raw/from-online-softmax-to-flashattention.pdf]
status: stable
---

# FlashAttention

**Source**: Tri Dao et al. (2022-2024). Integrated into PyTorch, most training frameworks use it by default.

## Key Points

- **Fuses attention computation** into a single GPU kernel — avoids materializing the full N×N attention matrix in HBM
- **Tiling + online softmax**: Computes attention in small blocks, keeping intermediate results in fast SRAM
- **2-4$\times$ speedup** over standard attention, **10-20$\times$ memory reduction** (O(N) vs $O(N^2)$ HBM usage)
- **IO-aware**: Designed around the fact that HBM bandwidth is the bottleneck, not compute
- FlashAttention-2: Better parallelism across sequence length and batch dimensions
- FlashAttention-3: Async execution, FP8 support on Hopper

## The Problem: HBM Round-Trips

Standard attention computes:

```
S = Q @ K^T     (N×N matrix in HBM)
P = softmax(S)  (N×N matrix in HBM)  
O = P @ V       (N×N → N×d)
```

The S and P matrices are $O(N^2)$ in size. For a 64K-token sequence at BF16, S alone is 8 GB. The GPU must write S to HBM, read it back for softmax, write P to HBM, read it back for the V multiplication — each round-trip pays the full HBM latency cost (~500 cycles).

**This is entirely memory-bandwidth-bound.** The actual FLOPs are modest; the GPU sits idle waiting for data.

## How FlashAttention Works

### Tiling

Split Q, K, V into smaller tiles that fit in on-chip SRAM (shared memory):

```
For each tile of Q (on-chip):
  For each tile of K, V (on-chip):
    Compute partial attention for this tile pair entirely in SRAM
    Accumulate results
  Write final output tile to HBM
```

No intermediate S or P matrices ever touch HBM — they stay in SRAM throughout.

### Online Softmax

Standard softmax requires two passes: compute max for numerical stability, then compute exp and normalize. FlashAttention fuses these into a single pass using online statistics:

```
For each K,V tile:
  m_new = max(m_old, row_max(S_tile))
  Update running sum with exp(S_tile - m_new)
  Rescale previous accumulator: acc *= exp(m_old - m_new)
  acc += exp(S_tile - m_new) @ V_tile
```

The rescaling step (`exp(m_old - m_new)`) corrects previous partial results when a new max is discovered — this is the key trick that enables single-pass attention.

### Activation Recomputation

FlashAttention natively recomputes attention scores in the backward pass instead of storing them. This means **anyone using FlashAttention is already doing selective recomputation** — the attention activations are never stored, only recomputed when needed for gradients.

## Versions

| Version | Key Improvement |
|---------|----------------|
| **FlashAttention-1** (2022) | Tiling + online softmax — O(N) memory |
| **FlashAttention-2** (2023) | Better parallelism: split over sequence length AND batch dim; 2$\times$ faster |
| **FlashAttention-3** (2024) | Async execution on Hopper; FP8 support; producer-consumer warp specialization |

## When FlashAttention Helps Most

| Sequence Length | FlashAttention Benefit |
|----------------|----------------------|
| <256 tokens | Minimal — overhead may outweigh benefit |
| 256-2K | Moderate — tiling starts to pay off |
| 2K-16K | Significant — 2-3$\times$ speedup |
| 16K+ | Critical — without it, $O(N^2)$ memory makes training impossible |

## Integration

- **PyTorch**: `torch.nn.functional.scaled_dot_product_attention` uses FlashAttention when available
- **Most training frameworks**: Enabled by default
- **torch.compile**: Automatically selects FlashAttention backend when shapes are compatible

## Connection to Context Parallelism

FlashAttention and Ring Attention (Context Parallelism) share the same core technique: online softmax. FlashAttention tiles the computation on a single GPU; Ring Attention extends tiling across multiple GPUs, each holding a chunk of the sequence. Both avoid materializing the full N×N attention matrix.

## Connections

- [GPU Memory Hierarchy](aspe-gpu-memory-hierarchy.md) — SRAM vs HBM trade-offs that FlashAttention exploits
- [CUDA Kernel Optimization](aspe-cuda-kernel-optimization.md) — Kernel fusion and tiling techniques
- [Ultra-Scale Playbook](usp-ultra-scale-playbook.md) — Context on recomputation and attention memory
- [Scaling Techniques Overview](usp-scaling-techniques-overview.md) — Context Parallelism (Ring Attention) uses the same online softmax
- [Inference Optimization](aspe-inference-optimization-techniques.md) — FlashAttention kernel autotuning for varying sequence lengths

## Mathematical Derivation

**Source**: Zihao Ye (UW), "From Online Softmax to FlashAttention", 2023 [src](../raw/from-online-softmax-to-flashattention.pdf)

This note explains the key mathematical insight that makes FlashAttention possible.

### Step 1: 3-Pass Safe Softmax

Standard softmax overflow problem: `exp(x)` overflows for `x > 11` in FP16. Safe softmax subtracts the max:

```
softmax(x) = exp(x_i - m) / Σ exp(x_j - m)    where m = max(x)
```

But computing `m` requires one full pass over all logits. Then computing `d = Σ exp(x_j - m_N)` requires a second pass. Then `a_i = exp(x_i - m_N)/d` requires a third. **3 passes over Q and K** — I/O inefficient when logits can't fit in SRAM.

### Step 2: 2-Pass Online Softmax

Key insight: create a surrogate sequence `d'_i = Σ_{j=1}^i exp(x_j - m_i)` instead of `d_i = Σ_{j=1}^i exp(x_j - m_N)`. The N-th terms are identical: `d'_N = d_N`.

**Recurrence**:
```
d'_i = d'_{i-1} · exp(m_{i-1} - m_i) + exp(x_i - m_i)
```

This only depends on `m_i` and `m_{i-1}`, so we can compute `m_i` and `d'_i` in the same loop. Still needs a **second pass** for the final softmax values `a_i`.

### Step 3: 1-Pass FlashAttention

**Key insight**: In self-attention, we don't need the attention scores `a_i` themselves — we need `O = A · V`. Can we find a 1-pass recurrence for `O`?

Start with `o_i = Σ a_j · V[j,:]` where `a_j = exp(x_j - m_N) / d'_N`. This depends on `m_N` and `d'_N` (not known until the end). **Surrogate trick again**:

```
o'_i = Σ_{j=1}^i exp(x_j - m_i)/d'_i · V[j,:]
```

The N-th terms match: `o'_N = o_N`. And the recurrence:

```
o'_i = o'_{i-1} · (d'_{i-1}/d'_i) · exp(m_{i-1} - m_i) + exp(x_i - m_i)/d'_i · V[i,:]
```

Only depends on `d'_i, d'_{i-1}, m_i, m_{i-1}, x_i`. **Now everything fits in one loop!**

### Tiled Algorithm

The algorithm is associative, so it's compatible with tiling. Process one tile (block size B) at a time:

```
For each tile i of K^T:
  x_i = Q[k,:] · K^T[(i-1)B : iB, :]
  m_i(local) = max(x_i)
  m_i = max(m_{i-1}, m_i(local))
  d'_i = d'_{i-1} · exp(m_{i-1} - m_i) + Σ exp(x_i[j] - m_i)
  o'_i = o'_{i-1} · d'_{i-1}/d'_i · exp(m_{i-1} - m_i) + Σ exp(x_i[j] - m_i)/d'_i · V[j+(i-1)B, :]
```

SRAM footprint depends only on B and D (head dim), not on L (sequence length). For H100: 228 KB SRAM/SM, typical D=128, B=128 → easily fits. **This is why FlashAttention scales to 64K+ sequences without memory issues.**

### The Rescaling Term

The term `exp(m_{i-1} - m_i)` is the key "fix-up": when a new larger max is found, previous partial results must be down-weighted because they used a smaller max for numerical stability. This rescaling step is what enables single-pass online statistics.
