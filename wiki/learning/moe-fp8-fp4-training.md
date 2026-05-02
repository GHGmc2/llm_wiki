---
title: "MoE FP8/FP4 Reduced-Precision Training"
type: concept
tags: [llm-training, mixture-of-experts, fp8, fp4, quantization, megatron, gpu]
created: 2026-05-02
updated: 2026-05-02
sources: [raw/scalable-training-moe-megatron-core.pdf]
status: stable
---

# MoE FP8/FP4 Reduced-Precision Training

**Source**: Megatron-Core MoE Technical Report, Section 5 [src](raw/scalable-training-moe-megatron-core.pdf)

## Key Points

- Reduced-precision training **impacts all three walls simultaneously** — lower memory, less communication volume, faster computation
- Four precision recipes supporting multiple hardware generations: per-tensor FP8, blockwise FP8 (Hopper), MXFP8 (Blackwell), NVFP4 (Blackwell)
- MoE introduces unique challenges: dynamic token shapes require **grouped quantization** and padding/unpadding fusion
- FP8/FP4 **primary weights** eliminate redundant FP32/BF16 weight storage, saving 2-4× parameter memory
- Selective precision: numerically sensitive components (router, embeddings) stay in BF16; bulk computation uses reduced precision

## Why It Matters for MoE

Reduced-precision training is uniquely valuable for MoE because it attacks all three walls simultaneously [src](raw/scalable-training-moe-megatron-core.pdf):

| Wall | FP8 Benefit | FP4 Benefit |
|------|------------|------------|
| **Memory** | 2× activation reduction | 4× activation reduction |
| **Communication** | 2× less data per all-to-all | 4× less data per all-to-all |
| **Compute** | 2× Tensor Core throughput | 4× Tensor Core throughput |

This multiplicative effect makes FP8/FP4 the highest-leverage single optimization: it frees memory (enabling longer sequences or less recomputation), reduces communication volume (mitigating the EP bottleneck), and accelerates GEMMs (improving compute efficiency).

## Precision Recipes

Megatron-Core supports four quantization recipes spanning two hardware generations:

### 1. Per-Tensor FP8 (E4M3)

**Hardware**: Hopper (H100), Blackwell (GB200/GB300)

The simplest FP8 scheme: each tensor gets a single scaling factor. Used for operations where tensor-wide statistics are stable (e.g., large GEMMs).

```
x_fp8 = quantize(x_bf16, scale)
y = dequantize(x_fp8, scale)
```

**Pros**: Lowest quantization overhead, simple implementation
**Cons**: Coarse granularity can cause accuracy loss when tensor values have wide dynamic range

### 2. Blockwise FP8

**Hardware**: Hopper (H100) — primary FP8 training recipe

Each tensor is divided into blocks (typically 128×128), and each block gets its own scaling factor. This finer granularity captures local value distributions better than per-tensor scaling.

```
For each block b in tensor:
    scale_b = max(|x_b|) / FP8_MAX
    x_fp8_b = round(x_b / scale_b)
```

**Key advantage on Hopper**: Uses native FP8 Tensor Core instructions with 2× throughput vs. BF16. The blockwise scheme is the recommended FP8 recipe for H100 MoE training.

**DeepSeek-V3 on H100 results**: 368 TFLOPS/GPU with blockwise FP8 (vs. ~180 TFLOPS in BF16).

### 3. MXFP8 (Microscaling FP8)

**Hardware**: Blackwell (GB200/GB300) — native Tensor Core support

An extension of blockwise FP8 with hardware-accelerated microscaling. Each 32-element block shares an 8-bit scaling factor, stored alongside the data. Blackwell Tensor Cores natively apply the scale during computation.

```
Data layout: [e4m3_block_0][scale_0][e4m3_block_1][scale_1]...
```

**Advantage**: Zero-overhead scaling — the hardware handles it, no extra instructions needed. This is the recommended FP8 recipe for Blackwell.

**DeepSeek-V3 on GB200 results**: 1,048 TFLOPS/GPU with MXFP8.

### 4. NVFP4

**Hardware**: Blackwell (GB200/GB300) only

NVIDIA's 4-bit floating point format with native Tensor Core acceleration. Each 16-element block shares a scaling factor. Provides 4× throughput vs. BF16 on supported operations.

**Qwen3-235B on GB300**: 974 TFLOPS/GPU with MXFP8; NVFP4 is available for further acceleration.

> [!note] NVFP4 requires Blackwell Tensor Cores. Hopper GPUs (H100) do not support FP4 natively.

## FP8/FP4 Primary Weights

**Problem**: Standard mixed-precision training stores master weights in FP32 and a BF16 copy for forward/backward. This doubles weight memory.

**Solution**: Store weights directly in FP8 or FP4 as the primary format. The optimizer maintains FP32 master weights only for the update step; forward and backward use the FP8/FP4 copy directly.

| Storage Scheme | Bytes per Param | Memory (685B model) |
|---------------|----------------|---------------------|
| FP32 master + BF16 copy | 6 | 4.1 TB |
| FP32 master + FP8 primary | 5 | 3.4 TB |
| FP32 master + FP4 primary | 4.5 | 3.1 TB |

Combined with optimizer state optimizations (see [Memory Wall](moe-memory-wall.md)), this provides substantial memory savings.

## MoE-Specific Challenges

Reduced-precision training for MoE introduces unique challenges not present in dense models:

### 1. Dynamic Shape Alignment (Padding/Unpadding Fusion)

**Problem**: FP8/FP4 GEMMs require fixed input dimensions, but per-expert token counts are dynamic. Traditional padding to a fixed capacity wastes compute and memory.

**Solution**: Fuse padding and unpadding with quantization into a single kernel. Instead of:

```
pad → quantize → GEMM → dequantize → unpad
```

The fused kernel quantizes only the valid tokens, pads with zeros to the required alignment, issues the GEMM, and discards padded outputs — all in one operation with no intermediate storage of padded values.

### 2. Grouped Quantization

**Problem**: Per-expert activations have different value ranges (one expert might see outlier tokens while another sees typical ones). Per-tensor quantization uses a single scale for all experts, losing precision where ranges differ.

**Solution**: Grouped quantization computes separate scales for each expert's token batch:

```
For each expert e:
    scale_e = max(|tokens_e|) / FP8_MAX
    tokens_fp8_e = quantize(tokens_e, scale_e)
```

Grouped GEMM then applies the per-expert scales during the batched computation. The overhead is minimal — one scale per expert (256 scales for 256 experts) vs. per-tensor (1 scale).

### 3. NVFP4 Quantization Fusion

**Problem**: NVFP4 requires block-level scales (one per 16 elements). Computing scales and applying quantization for thousands of small expert token batches is expensive.

**Solution**: Fuse NVFP4 block-scale computation with the token permutation step. As tokens are permuted into per-expert buffers, block statistics are accumulated and scales computed inline. The combined kernel eliminates a separate quantization pass.

## Selective Precision

Not all operations benefit equally from reduced precision. Megatron-Core uses selective precision:

| Component | Precision | Reason |
|-----------|-----------|--------|
| GEMM (attention, expert FFN) | FP8/FP4 | Bulk computation, stable value ranges |
| Router logits + softmax | BF16 | Small computation, numerical sensitivity |
| Embeddings | BF16 | Sparse updates, wide dynamic range |
| LayerNorm/RMSNorm | BF16 | Small tensors, precision-critical |
| Loss computation | FP32 | Accumulation precision matters |
| Optimizer states | BF16/FP32 | See [Memory Wall](moe-memory-wall.md) |

This selective approach preserves accuracy while maximizing the throughput and memory benefits of reduced precision where they matter most.

## Numerical Stability

The paper reports successful FP8 training at scale (DeepSeek-V3 at 1,024 GPUs, Qwen3-235B at 256 GPUs) with no loss divergence. Key stability techniques:

- Gradient scaling maintained in FP32
- Router and auxiliary loss computed in BF16
- Embedding gradient accumulation in FP32
- Periodic (every N steps) BF16 validation to detect drift

## Connections

- [Megatron-Core MoE](megatron-core-moe-scalable-training.md) — full paper summary
- [Memory Wall](moe-memory-wall.md) — FP8/FP4 activation storage is a key memory optimization
- [Communication Wall](moe-communication-wall.md) — reduced precision halves EP communication volume
- [Compute Efficiency Wall](moe-compute-efficiency-wall.md) — FP8/FP4 accelerate GEMMs but expose CPU overhead
