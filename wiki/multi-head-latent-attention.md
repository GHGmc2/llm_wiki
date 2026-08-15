---
title: "Multi-Head Latent Attention (MLA)"
type: concept
tags: [llm, attention, deepseek, kv-cache, memory-optimization, inference, mla]
created: 2026-05-02
updated: 2026-08-15
sources: [raw/insights-into-deepseek-v3.pdf, raw/deepseek-v4.pdf]
status: stable
---

# Multi-Head Latent Attention (MLA)

**Source**: DeepSeek-V3 Insights (ISCA 2025) [src](../raw/insights-into-deepseek-v3.pdf), DeepSeek-V4 report [src](../raw/deepseek-v4.pdf)

## Key Points

- MLA compresses Key and Value vectors into a shared **low-rank latent space**, dramatically reducing KV cache memory
- 7.28$\times$ smaller KV cache than LLaMA-3.1 405B (70 KB vs 516 KB per token)
- During inference, only the compressed latent is cached; full KV is up-projected on-the-fly
- Foundation for DeepSeek's long-context capabilities — enables efficient multi-turn conversations
- Carried forward into DeepSeek-V4, where it complements CSA/HCA hybrid attention

## Problem: KV Cache Explosion

In standard Multi-Head Attention (MHA) or Grouped Query Attention (GQA), the KV cache stores full Key and Value vectors for every token across all layers. For autoregressive decoding, this cache grows linearly with sequence length:

```
KV_cache_size = 2 × num_layers × hidden_dim × seq_len × dtype_bytes
```

For large models at long sequences (e.g., LLaMA-3.1 405B at 128K tokens), this consumes tens of GB just for the KV cache. This is the **primary memory bottleneck** for long-context inference.

**Existing mitigation approaches and their limitations**:
- **GQA (Grouped Query Attention)**: Shares KV heads across query groups — reduces cache but still stores full-dimensional vectors
- **Windowed KV**: Only retains recent tokens in cache — compromises long-context reasoning
- **Quantized KV**: Stores low-bit representations — significant compression but still operates on full dimensions

## MLA Architecture

MLA takes a fundamentally different approach: **compress the KV representation into a low-rank latent space** rather than just reducing the number of heads or bit-width.

![MLA within the DeepSeek-V3 architecture — KV cache compressed to latent c_KV](../raw/assets/v3_architecture.png)

*Figure: MLA compresses Keys and Values into a shared latent vector `c_KV` via low-rank projection. Only the compressed latent is cached; full KV is up-projected on-the-fly during attention. [src](../raw/DeepSeek-V3.pdf)*

### Core Mechanism

Instead of computing and storing full-dimensional Key and Value vectors per head, MLA:

1. **Down-projection**: Projects the input hidden state into a shared latent vector `c_KV` of dimension `d_c` (where `d_c << d`, the hidden dimension)
2. **Caching**: Only the compressed latent `c_KV` is stored in the KV cache
3. **Up-projection (on-the-fly)**: During attention computation, `c_KV` is up-projected to full Keys and Values

```
Standard MHA:
  K = x · W_K  (d → d_head × n_heads)  ← stored
  V = x · W_V  (d → d_head × n_heads)  ← stored

MLA:
  c_KV = x · W_down  (d → d_c)         ← stored (compressed!)
  K = c_KV · W_up_K  (d_c → d_head × n_heads)  ← computed on-the-fly
  V = c_KV · W_up_V  (d_c → d_head × n_heads)  ← computed on-the-fly
```

### Key Design Decisions

**Shared latent for K and V**: Keys and Values share the same latent representation `c_KV`. This is efficient because K and V are derived from the same input and their information is correlated.

**RoPE integration**: Rotary Position Embeddings (RoPE) are applied after up-projection, not on the latent — position information is needed at full dimensionality.

**Query compression**: Queries are also compressed into a separate latent `c_Q` for consistency, though this matters more for training efficiency than KV cache.

### KV Cache Size

**BF16 precision comparison**:

| Model | Attention Type | KV Cache Per Token | vs MLA |
|-------|---------------|-------------------|--------|
| DeepSeek-V3 | MLA | 70.272 KB | 1$\times$ |
| DeepSeek-V2 | MLA | — | ~1$\times$ |
| Qwen-2.5 72B | GQA | 327.680 KB | 4.66$\times$ |
| LLaMA-3.1 405B | GQA | 516.096 KB | 7.28$\times$ |

The compression ratio depends on the latent dimension `d_c`. For DeepSeek-V3, `d_c` is a small fraction of the full KV dimension, achieving the 7$\times$ + compression.

## Why MLA Matters

### Long-Context Inference

KV cache is the dominant memory consumer during long-context autoregressive decoding. MLA's 7$\times$ + compression means:
- A 128K-token conversation that would consume ~66 GB of KV cache with GQA (LLaMA-405B) consumes ~9 GB with MLA
- Multi-turn conversations (where cache persists across turns) become far more practical
- Reasoning models (DeepSeek-R1) that generate long chains of thought benefit enormously

### Foundation for V4's Hybrid Attention

DeepSeek-V4's CSA/HCA attention architecture builds on MLA's compression philosophy but extends it to the attention computation itself — compressing not just the stored representation but also the attention patterns. MLA handles *storage* efficiency; CSA/HCA handle *computation* efficiency. Together they enable V4's 1M-token context at 27% of V3.2's FLOPs.

### Training Efficiency

MLA reduces activation memory during training by storing smaller intermediate representations. This enables larger batch sizes or longer sequences with the same memory budget. The DeepSeek-V3 insights paper notes that MLA's activation savings complement FP8 training's weight savings.

## Limitations and Trade-offs

- **Up-projection compute**: The on-the-fly up-projection adds a small matrix multiplication per attention query — this is cheap compared to the attention computation itself but not zero
- **Latent dimension tuning**: If `d_c` is too small, model quality degrades; if too large, compression benefit is lost
- **Not a replacement for sparse attention at extreme lengths**: At 1M+ tokens, even compressed KV cache grows large — hence V4's hybrid CSA/HCA for additional compression

## Connections

- [DeepSeek-V3 Insights](deepseek-v3-insights.md) — hardware-aware co-design paper introducing MLA in context
- [DeepSeek-V4 Architecture](deepseek-v4.md) — CSA/HCA builds on MLA's compression philosophy
- [Megatron-Core Memory Wall](megatron-core-moe.md) — activation memory optimization from NVIDIA's perspective
- [Ultra-Scale Playbook](usp-ultra-scale-playbook.md) — context on KV cache and attention memory
- [LLM Architecture Comparison](llm-architecture-comparison.md) — comprehensive comparison of attention variants (MHA, GQA, MLA, SWA, DSA) across 23 models
- [Kimi K3](kimi-k3.md) — KDA recurrent attention: fixed-size state extends MLA's latent-space lineage
