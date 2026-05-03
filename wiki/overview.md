---
title: "Overview"
type: summary
tags: [meta]
created: 2026-05-02
updated: 2026-05-02
sources: [raw/scalable-training-moe-megatron-core.pdf, raw/DeepSeek-V4.pdf, raw/The_Ultra-Scale_Playbook.pdf, raw/insights-into-deepseek-v3.pdf, raw/DeepSeek-R1.pdf, raw/DeepSeek-V3.pdf, raw/DeepSeek-V3.2.pdf, raw/demystifying-nccl.pdf, raw/gpu-initiated-networking-nccl.pdf, raw/nccl-ep.pdf, raw/GSPMD.pdf, raw/PartIR.pdf, raw/TOAST.pdf]
status: stable
---

# Overview

*The wiki covers the full DeepSeek model lineage (V3 → V3.2 → R1 → V4), NVIDIA's Megatron-Core training stack, and the Ultra-Scale Playbook educational foundation. 18 pages across 7 sources.*

## DeepSeek Model Lineage

| Model | Date | Key Innovation |
|-------|------|---------------|
| [DeepSeek-V3](learning/deepseek-v3.md) | Dec 2024 | MLA, FP8 training, auxiliary-loss-free MoE, MTP, DualPipe — 2.788M H800 GPU hours |
| [V3 Insights (ISCA)](learning/deepseek-v3-insights.md) | Jun 2025 | Hardware-aware co-design on H800, Multi-Plane Network, hardware wishlist |
| [DeepSeek-R1](learning/deepseek-r1.md) | Jan 2025 | Pure RL reasoning via GRPO, emergent self-verification, USAMO qualification |
| [DeepSeek-V3.2](learning/deepseek-v3.2.md) | Dec 2025 | DSA sparse attention, scalable RL (>10% pre-training budget), IMO/IOI gold |
| [DeepSeek-V4](learning/deepseek-v4.md) | Apr 2026 | CSA+HCA hybrid attention, 1M context, mHC, Muon, On-Policy Distillation |

## Training Systems

| Source | Focus |
|--------|-------|
| [Megatron-Core MoE](learning/megatron-core-moe.md) | NVIDIA's production training stack — Three Walls, Parallel Folding, FP8/FP4 |
| [Ultra-Scale Playbook](learning/ultra-scale-playbook.md) | Educational foundation — DP, ZeRO, TP, SP, CP, PP, EP with formulas and benchmarks |

## Communication Infrastructure (NCCL)

| Source | Focus |
|--------|-------|
| [Demystifying NCCL](learning/nccl-demystifying.md) | NCCL internals — Simple/LL/LL128 protocols, ring/tree algorithms, channels |
| [GPU-Initiated Networking](learning/nccl-device-api-gin.md) | NCCL 2.28 Device API — GIN, LSA, Multimem, GPU-initiated RDMA |
| [NCCL EP](learning/nccl-ep.md) | MoE communication library — unified dispatch/combine, LL/HT modes |

## Compiler & Auto-Partitioning Systems

| Source | Focus |
|--------|-------|
| [GSPMD](learning/gspmd.md) | Compiler-based parallelization: `mesh_split` API, automatic sharding propagation |
| [PartIR](learning/partir.md) | Composable SPMD tactics: schedule-like API, MLIR-based, formally verified |
| [TOAST](learning/toast.md) | Auto-partitioning: static analysis + MCTS, zero user annotations needed |

These systems automate the sharding decisions that the [Scaling Techniques Overview](learning/scaling-techniques-overview.md) describes — the compiler stack above NCCL.

## Cross-Cutting Themes

- **Memory / Communication / Compute**: The three constraints appear across all sources
- **FP8/FP4**: Megatron-Core provides recipes, DeepSeek-V3 pioneered FP8 training, V4 uses FP4 QAT
- **EP communication overlap**: Layer-level (Megatron 1F1B) → Expert-level (DeepSeek-V4) → Dual micro-batch (DeepSeek-V3/R1 inference)
- **Attention efficiency**: MLA (V3) → DSA (V3.2) → CSA+HCA (V4) — progressive KV compression/selection
- **Post-training evolution**: SFT+RL (V3) → Pure RL/GRPO (R1) → On-Policy Distillation (V4)
