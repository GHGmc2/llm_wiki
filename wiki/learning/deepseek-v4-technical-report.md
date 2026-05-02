---
title: "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence"
type: source-note
tags: [llm, mixture-of-experts, deepseek, long-context, hybrid-attention, mhc, muon]
created: 2026-05-02
updated: 2026-05-02
sources: [raw/DeepSeek-V4.pdf]
status: stable
---

# DeepSeek-V4

**Source**: DeepSeek-AI Technical Report, April 2026. 58 pages. Preview release.

## Key Points

- Two models: **V4-Pro** (1.6T params, 49B active) and **V4-Flash** (284B params, 13B active), both with native 1M-token context
- **Hybrid attention** (CSA + HCA) is the headline innovation: at 1M-token context, V4-Pro uses only 27% of V3.2's inference FLOPs and 10% of its KV cache
- **mHC** (Manifold-Constrained Hyper-Connections) replaces residual connections for numerical stability at 1.6T scale
- **Muon optimizer** replaces AdamW for most parameters — faster convergence, better stability
- **On-Policy Distillation** replaces mixed RL: train domain specialists, distill into unified model
- Three reasoning modes: Non-Think, Think High, Think Max
- Open weights, honest about trailing frontier by 3-6 months on general knowledge

## Models at a Glance

| | V4-Pro | V4-Flash |
|---|---|---|
| Total params | 1.6T | 284B |
| Active params | 49B | 13B |
| Layers | 61 | 43 |
| Hidden dim | 7168 | 4096 |
| Routed experts | 256 | 256 |
| Active experts/token | 8 | 6 |
| Shared experts | 1 | 1 |
| Pre-training tokens | 33T | 32T |
| Context length | 1M | 1M |

V4-Flash-Base surpasses V3.2-Base on most benchmarks despite having 42% of V3.2's total parameters — architectural and data-quality improvements drive the gain.

## Architecture Innovations

### Hybrid Attention: CSA + HCA

The central efficiency breakthrough. Two attention types interleaved across layers:

- **CSA (Compressed Sparse Attention)**: Compresses KV cache 1/m, then applies sparse top-k selection via a learned Lightning Indexer. Adds a sliding window of recent uncompressed tokens for local dependencies.
- **HCA (Heavily Compressed Attention)**: Much heavier compression (m' >> m), dense attention over compressed KV. Retains broad, low-resolution view of full context.

CSA gives precise look-up; HCA gives global summary. Interleaving is what makes 1M context economical — neither alone achieves the full efficiency gain.

See: [DeepSeek-V4 Architecture](deepseek-v4-architecture.md)

### mHC (Manifold-Constrained Hyper-Connections)

Upgrades conventional residual connections by constraining the residual mapping to the Birkhoff polytope (doubly stochastic matrices), bounding spectral norm to <= 1. This makes signal propagation non-expansive, enabling stable training at 1.6T scale.

### Muon Optimizer

Replaces AdamW for most parameters (embeddings, prediction head, RMSNorm still use AdamW). Delivers faster convergence and better training stability. Combined with Anticipatory Routing and SwiGLU clamping as stabilizers.

## Pre-Training

Both models trained with FP4 quantization-aware training for routed expert weights, FP8 for non-expert computation. Batch size scheduling from small to large (75.5M for Flash, 94.4M for Pro). Sequence length gradually extended: 4K -> 16K -> 64K -> 1M.

**Training stabilizers**:
- **Anticipatory Routing**: Decouples routing updates from backbone by one step; triggered automatically on loss spikes
- **SwiGLU clamping**: Linear component clamped to [-10, 10], gate capped at 10

**Base model results**: V4-Flash-Base surpasses V3.2-Base on most benchmarks. V4-Pro-Base near-universal dominance — dramatic gains on knowledge (MMLU 90.1 vs 87.8) and long-context (LongBench-V2 51.5 vs 40.2).

## Post-Training: On-Policy Distillation

Replaces V3.2's mixed RL stage. Two-phase pipeline:

1. **Specialist Training**: Train separate domain experts (math, code, agent, instruction) via SFT + GRPO
2. **On-Policy Distillation**: Unify specialists into one model via multi-teacher reverse-KL distillation on student-generated trajectories

See: [DeepSeek-V4 Post-Training](deepseek-v4-post-training.md)

### Three Reasoning Modes

| Mode | Behavior | Use Case |
|------|----------|----------|
| Non-Think | Fast, no CoT | Routine tasks, low-latency |
| Think High | Standard reasoning trace | Complex problem-solving (default) |
| Think Max | Max reasoning — exhaustive decomposition, edge-case stress-testing | Frontier benchmarks, hardest problems |

Think Max prepends a special system prompt and uses larger context window (384K) with reduced length penalties.

## Infrastructure

- **Fine-grained EP communication-computation overlap** with expert-level pipelining
- **TileLang**: A new kernel development language for flexible, efficient MoE kernels
- **FP4 QAT**: Routed experts in FP4; theoretically 1.33x more efficient on future hardware
- **KV cache management**: On-disk storage for compressed KV entries, three SWA caching strategies (full, periodic checkpointing, zero)
- **Quick Instruction**: Special tokens for auxiliary tasks (search queries, intent) using existing KV cache — reduces time-to-first-token

See: [DeepSeek-V4 Infrastructure](deepseek-v4-infrastructure.md)

## Benchmarks: V4-Pro-Max vs Frontier

| Benchmark | V4-Pro-Max | Best Frontier | Leader |
|-----------|-----------|---------------|--------|
| LiveCodeBench (Pass@1) | **93.5** | 91.7 (Gemini) | V4 |
| Codeforces (Rating) | **3206** | 3168 (GPT-5.4) | V4 |
| Apex Shortlist (Pass@1) | **90.2** | 89.1 (Gemini) | V4 |
| Putnam-2025 | **120/120** | — | V4 |
| SWE Verified (Resolved) | 80.6 | 80.8 (Opus) | Tie |
| MMLU-Pro (EM) | 87.5 | **91.0** (Gemini) | Gemini |
| GPQA Diamond (Pass@1) | 90.1 | **94.3** (Gemini) | Gemini |
| SimpleQA-Verified | 57.9 | **75.6** (Gemini) | Gemini |
| MRCR 1M (MMR) | 83.5 | **92.9** (Opus) | Opus |
| HLE (Pass@1) | 37.7 | **44.4** (Gemini) | Gemini |

**DeepSeek's own framing**: Trails absolute frontier by 3-6 months on general knowledge and hardest retrieval. Sets new open-model highs on competitive programming and formal reasoning.

## Connections

- [DeepSeek-V4 Architecture](deepseek-v4-architecture.md) — CSA, HCA, mHC, Muon in detail
- [DeepSeek-V4 Post-Training](deepseek-v4-post-training.md) — OPD, specialist training, reasoning modes
- [DeepSeek-V4 Infrastructure](deepseek-v4-infrastructure.md) — EP overlap, TileLang, FP4 QAT, KV cache
- [Megatron-Core MoE](megatron-core-moe-scalable-training.md) — complementary: NVIDIA's training stack for MoE models
