---
title: "Inference Optimization: Disaggregation, Autotuning, and Precision Switching"
type: concept
tags: [inference, llm, disaggregation, autotuning, precision, kv-cache, speculative-decoding, performance]
created: 2026-05-02
updated: 2026-05-02
sources: [raw/AI Systems Performance Engineering.pdf]
status: stable
---

# Inference Optimization Techniques

**Source**: AI Systems Performance Engineering, Chapters 17-19 [src](raw/AI%20Systems%20Performance%20Engineering.pdf)

## Key Points

- **Prefill-Decode Disaggregation**: Separate GPUs for prompt processing (compute-bound) and token generation (memory-bound) — tighter latency, higher goodput
- **Kernel Autotuning**: Select optimal kernel variant at runtime based on batch size, sequence length — avoid one-size-fits-all performance cliffs
- **Dynamic Precision Switching**: Use FP8 for low-entropy tokens, switch to FP16 for high-entropy — max throughput without accuracy loss
- **Phase-specific optimization**: Prefill = high TP, FP8 compute; Decode = memory-optimized, KV cache compression
- **Goodput**: The metric that matters — throughput under latency SLO constraints

## Prefill-Decode Disaggregation

### The Problem

In monolithic serving, prefill (prompt processing) and decode (token generation) compete for the same GPU resources:
- Prefill is **compute-bound** — processes the entire prompt at once
- Decode is **memory-bound** — generates one token at a time, KV cache access dominates

A single long prompt can block token generation for all other users → **tail latency spikes**.

### The Solution

Split prefill and decode into separate worker pools:

```
[Prefill Workers]          [Decode Workers]
  GPU 0, GPU 1               GPU 2, GPU 3
  (high FLOPS, FP8)          (high memory BW)
       │                           │
       └── KV Cache Transfer ──────┘
```

**Performance impact** (from DistServe paper, cited in book):

| Configuration | Goodput (RPS) |
|--------------|---------------|
| Colocated (1 GPU) | 1.6 RPS |
| 2P1D Disaggregated (3 GPUs) | 3.3 RPS/GPU |

The 2P1D config achieves **2× per-GPU goodput** under SLO constraints (P90 TTFT < 0.4s, P90 TPOT < 0.04s).

### Phase-Specific Optimization

| Phase | Bottleneck | Optimization |
|-------|-----------|-------------|
| Prefill | Compute | High TP, FP8/FP4, high-FLOPS GPUs |
| Decode | Memory bandwidth | KV cache quantization, memory-optimized GPUs |

**Heterogeneous clusters**: Use compute-optimized GPUs for prefill and memory-optimized GPUs for decode → better throughput per dollar.

## Kernel Autotuning

### Attention Kernel Selection

Different sequence lengths need different attention implementations:

```cpp
if (seq_len < 256) {
    attn_kernel = standard_attention;    // simple global-memory kernel
} else {
    attn_kernel = flash_attention;       // tiled, HBM-efficient kernel
}
```

- Short sequences: standard attention — FlashAttention overhead > benefit
- Medium sequences: FlashAttention — tiling amortizes overhead
- Very long: FlashAttention-3 — async, FP8 support

### MLP GEMM Autotuning

For the feed-forward layers `[batch, hidden] × [hidden, 4×hidden]`:

- `batch_size = 1`: Different optimal tile size than `batch_size = 16`
- cuBLAS/cuBLASLt autotunes on first encounter, caches result
- Inference servers can pre-warm with typical batch sizes

### Tile Size Impact

| Tile Size | SMEM (KB) | Occupancy (%) | Throughput (GOPS) |
|-----------|----------|---------------|-------------------|
| 64 × 64 | 48 | 85 | 8.2 |
| 128 × 64 | 64 | 78 | 10.5 |
| 128 × 128 | 96 | 72 | 9.8 |
| 256 × 128 | 128 | 60 | 11.3 |

Bigger tiles = better arithmetic intensity but lower occupancy. The optimal depends on the specific kernel and hardware.

## Dynamic Precision Switching

For autoregressive decoding, precision requirements vary by token:

```python
# Pseudocode for token-level dynamic precision
for token in generate():
    entropy = compute_entropy(logits)
    
    if entropy < 3.0 and use_fp8:
        use_fp8 = False  # switch to FP16 for high-entropy tokens
    elif entropy > 6.0 and not use_fp8:
        use_fp8 = True   # switch to FP8 for low-entropy tokens
    
    next_token = fp8_generate() if use_fp8 else fp16_generate()
```

**Threshold calibration**: Enter FP8 at entropy > 6.0, exit at entropy < 3.0 — determined on validation set. Hysteresis prevents oscillation.

**Typical tokens in FP8**: Punctuation, boilerplate, low-entropy completions (majority of tokens).
**Typical tokens in FP16**: Ambiguous prompts, decision points, high-entropy generation (minority of tokens).

Combined with FP4 weights + FP8 activations, this provides maximum throughput with minimal accuracy impact.

## Goodput: The Metric That Matters

**Goodput** = throughput under latency SLO constraints.

A system may process 1000 tokens/sec but only 200 tokens/sec meet the latency SLO → goodput = 200, not 1000.

**Key SLOs**:
- **TTFT** (Time To First Token): Prefill latency — typically P90 < 0.4s
- **TPOT** (Time Per Output Token): Decode latency — typically P90 < 0.04s
- Both must be satisfied to count as goodput

## Additional Techniques (referenced in book)

- **Speculative decoding**: Draft model generates candidate tokens, target model verifies — reduces TPOT
- **PagedAttention / vLLM**: Dynamic KV cache memory management — eliminates fragmentation
- **Continuous batching**: Add/remove requests from running batch — maximizes GPU utilization
- **KV cache quantization**: FP8 KV cache — halves memory, enables larger batches
- **FlashAttention 1-3**: Tiled attention kernels avoiding HBM round-trips

## Connections

- [CUDA Graphs & Orchestration](cuda-graphs-and-orchestration.md) — CUDA Graphs for inference loops
- [GPU Memory Hierarchy](gpu-memory-hierarchy.md) — KV cache and memory optimization
- [Multi-Head Latent Attention](multi-head-latent-attention.md) — MLA reduces KV cache (complementary)
- [AI Systems Performance Engineering](ai-systems-performance-engineering.md) — Full book reference
