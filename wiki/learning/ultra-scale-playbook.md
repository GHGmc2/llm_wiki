---
title: "The Ultra-Scale Playbook: Training LLMs on GPU Clusters"
type: source-note
tags: [llm-training, distributed-training, scaling, parallelism, gpu, hf, nanotron, educational]
created: 2026-05-02
updated: 2026-05-02
sources: [raw/The_Ultra-Scale_Playbook.pdf]
status: stable
---

# The Ultra-Scale Playbook

**Source**: HuggingFace/nanotron team. Published Feb 19, 2025. Open-source book. 4,100+ distributed experiments on up to 512 GPUs.

**Authors**: Nouamane Tazi, Ferdinand Mom, Haojun Zhao, Phuc Nguyen, Mohamed Mekkouri, Leandro Werra, Thomas Wolf

## Key Points

- A progressive, story-driven educational book: start with one GPU, hit limits, add techniques one by one
- Covers **all major scaling techniques**: Data Parallelism, ZeRO (1/2/3), Tensor Parallelism, Sequence Parallelism, Context Parallelism (Ring Attention), Pipeline Parallelism, Expert Parallelism
- Three pillars: **Memory** (hard limit), **Compute Efficiency** (keep GPUs busy), **Communication Overhead** (minimize idle time)
- Practical with code (Picotron for education, Nanotron for production), formulas, benchmarks, and profiler traces
- Directly complements the Megatron-Core and DeepSeek-V4 pages already in this wiki

## Structure

The book follows a single narrative arc — each technique is introduced because the previous one hit a limit:

1. **Training on One GPU** — Memory anatomy (params, grads, optimizer states, activations), activation recomputation (full vs selective), gradient accumulation, profiling with PyTorch profiler
2. **Data Parallelism** — Replicate model, all-reduce gradients. Three optimizations: overlap sync with backward pass, gradient bucketing, interplay with gradient accumulation
3. **ZeRO (Zero Redundancy Optimizer)** — ZeRO-1 (optimizer state partitioning), ZeRO-2 (+gradient partitioning), ZeRO-3 (+parameter partitioning = FSDP). Communication patterns: reduce-scatter + all-gather
4. **Tensor Parallelism** — Shard weights/activations within layers. Row vs column sharding. TP in transformer blocks (attention, FFN). TP communication cannot be easily overlapped
5. **Sequence Parallelism** — Shard Dropout/LayerNorm along sequence dim instead of hidden dim. Complements TP; reduces maximum activation size per GPU
6. **Context Parallelism** — Ring Attention for long sequences. AllGather vs P2P implementations. Zig-Zag Ring Attention for balanced compute
7. **Pipeline Parallelism** — Split model by layers. Schedules: AFAB (all-forward-all-backward), 1F1B (one-forward-one-backward), interleaved stages. Bubble formula: bubble = (p-1)/(m) where p=PP degree, m=microbatches. Zero Bubble and DualPipe (DeepSeek-V3)
8. **Expert Parallelism** — For MoE: distribute experts across GPUs, all-to-all for token dispatch/combine
9. **5D Parallelism** — Combining TP+SP+CP+PP+EP+DP+ZeRO. Visual memory savings breakdown for each strategy
10. **Finding the Best Configuration** — Three-step workflow: (1) fit in memory, (2) achieve target global batch size, (3) optimize throughput. Benchmarks across model sizes and node counts
11. **GPU Kernels & Fusion** — GPU architecture primer, writing custom kernels (Triton), tiling, coalesced memory access, kernel fusion
12. **FlashAttention 1-3** — Fused attention kernel avoiding HBM round-trips
13. **Mixed Precision Training** — FP32/BF16/FP8 trade-offs, DeepSeek-V3's FP8 recipe

## Key Formulas

**Parameter count**:
$$
N = h \cdot v + L \cdot (12h^2 + 13h) + 2h
$$

**Memory (BF16 w/ FP32 grad acc)**:
$$
m_{\text{params}} = 2N,\; m_{\text{grad}} = 2N,\; m_{\text{params\_fp32}} = 4N,\; m_{\text{opt}} = 8N
$$
→ $16N$ total per GPU (before sharding)

**Activation memory**:
$$
m_{\text{act}} = L \cdot s \cdot b \cdot h \cdot \left(34 + \frac{5 \cdot n_{\text{heads}} \cdot s}{h}\right)
$$
Scales linearly with batch size $b$, quadratically with sequence length $s$.

**Global batch size**:
$$
gbs = mbs \times grad\_acc \times dp
$$

**Pipeline bubble** (1F1B):
$$
bubble = \frac{(p-1)(t_f + t_b)}{m(t_f + t_b)} = \frac{p-1}{m}
$$
where $p$ = PP degree, $m$ = microbatches.

## Memory Numbers

| Model Size | FP32 | BF16 w/ FP32 grad acc |
|-----------|------|----------------------|
| 1B | 16 GB | 20 GB |
| 7B | 112 GB | 140 GB |
| 70B | 1,120 GB | 1,400 GB |
| 405B | 6,480 GB | 8,100 GB |

**At just 7B params, memory already exceeds a single H100 (80 GB).** Scaling is mandatory.

## Selective Recomputation

The playbook explains HFU (Hardware FLOPs Utilization, includes recomputation) vs MFU (Model FLOPs Utilization, excludes recomputation):

- **Full recomputation**: Checkpoint at each layer boundary, up to 30-40% compute overhead
- **Selective**: Discard attention activations, checkpoint feedforward. For GPT-3 175B: **70% activation memory reduction at 2.7% compute cost**

## Configuration Heuristics

**When you have plenty of GPUs**:
- <10B params: single technique (TP or ZeRO-3/DP with full recompute) across 8 GPUs
- 10-100B params: TP + ZeRO-2/DP + selective recompute across 8-16 GPUs. Add PP for multi-node
- 100B+ params: TP + PP + ZeRO-2 + selective recompute. EP for MoE. CP for long sequences

**When GPU-constrained**:
- Enable full recomputation (trade compute for memory)
- Increase gradient accumulation (process larger batches with same memory)
- Use ZeRO-3 with CPU offload

## Connections to Existing Wiki Pages

This playbook is the **educational foundation** for the techniques implemented in:

- **[Megatron-Core MoE](megatron-core-moe.md)** — Memory Wall (recomputation, offloading), Communication Wall (EP overlap), Compute Wall (kernel fusion, CUDA Graphs), Parallel Folding
- **[Parallel Folding](megatron-core-moe.md)** — Decoupling TP/EP/CP, which the playbook explains why is necessary
- **[DeepSeek-V4 Architecture](deepseek-v4.md)** — CSA/HCA attention is a specialized form of the attention compression context parallelism covers
- **[FP8/FP4 Training](megatron-core-moe.md)** — The playbook's mixed precision section provides the fundamentals
- **[Scaling Techniques Overview](scaling-techniques-overview.md)** — Detailed breakdown of each technique
