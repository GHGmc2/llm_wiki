---
title: "Overview"
type: summary
tags: [meta]
created: 2026-05-02
updated: 2026-05-04
sources: [raw/scalable-training-moe-megatron-core.pdf, raw/DeepSeek-V4.pdf, raw/The_Ultra-Scale_Playbook.pdf, raw/DeepSeek-R1.pdf, raw/DeepSeek-V3.pdf, raw/DeepSeek-V3.2.pdf, raw/demystifying-nccl.pdf, raw/gpu-initiated-networking-nccl.pdf, raw/nccl-ep.pdf, raw/GSPMD.pdf, raw/PartIR.pdf, raw/TOAST.pdf, raw/ppo.pdf, raw/stabilizing-rl-llms.pdf, raw/art-of-scaling-rl-compute.pdf, raw/auto-parallelism-survey.pdf, raw/efficient-llm-training-survey.pdf, raw/comet-moe-overlap.pdf, raw/moe-parallel-folding.pdf, raw/scaling-laws-kaplan.pdf, raw/ai-systems-performance-engineering.pdf, raw/the-bitter-lesson.pdf, raw/cross-domain-scaling-laws.pdf, raw/rail-only-network.pdf, raw/muon-optimizer.pdf, raw/communication-overlap-decomposition.pdf]
status: stable
---

# Overview

*51 pages across 26+ sources. Covers LLM architecture, training systems, GPU hardware, scaling laws, RL, inference, and networking.*

## Model Lineage (DeepSeek)

| Model | Date | Key Innovation | Page |
|-------|------|---------------|------|
| DeepSeek-V3 | Dec 2024 | MLA, FP8 training, auxiliary-loss-free MoE, MTP | [v3](learning/deepseek-v3.md) |
| V3 Insights (ISCA) | Jun 2025 | Hardware-aware co-design, Multi-Plane Network | [insights](learning/deepseek-v3-insights.md) |
| DeepSeek-R1 | Jan 2025 | Pure RL (GRPO), emergent reasoning, distillation | [r1](learning/deepseek-r1.md) |
| DeepSeek-V3.2 | Dec 2025 | DSA sparse attention, IMO/IOI gold | [v3.2](learning/deepseek-v3.2.md) |
| DeepSeek-V4 | Apr 2026 | CSA+HCA hybrid attention, 1M context, OPD, Muon optimizer | [v4](learning/deepseek-v4.md) |

## Training Systems

| Source | Focus | Page |
|--------|-------|------|
| Megatron-Core MoE | Production stack: Three Walls, Parallel Folding, FP8/FP4 | [megatron](learning/megatron-core-moe.md) |
| Ultra-Scale Playbook | Educational foundation: DP, ZeRO, TP, SP, CP, PP, EP | [playbook](learning/usp-ultra-scale-playbook.md) |
| Efficient LLM Training Survey | Taxonomy of training systems (380+ refs) | [survey](learning/efficient-llm-training-survey.md) |
| Comet | Fine-grained EP communication overlap | [comet](learning/comet-moe-overlap.md) |
| Muon Optimizer | Matrix orthogonalization, ~2× vs AdamW, Moonlight MoE | [muon](learning/muon-optimizer.md) |
| Comm Overlap via Decomposition | Google ASPLOS 2023 — decompose collectives for TP overlap | [comm-overlap](learning/communication-overlap-decomposition.md) |
| How To Scale Your Model (DeepMind) | Textbook: roofline, TPU, sharding, parallelism — FSDP/TP/PP cost analysis | [scaling-book](learning/scaling-book.md) |

## Scaling Laws

| Source | Focus | Page |
|--------|-------|------|
| Kaplan et al. (2020) | Original scaling laws: $L \propto N^{-\alpha}$, $D \propto N^{0.74}$ | [scaling-laws](learning/llm-scaling-laws.md) |
| Henighan et al. (2020) | Cross-domain: images, video, multimodal, math — universal exponents | [cross-domain](learning/cross-domain-scaling-laws.md) |

## Research Philosophy (in [reading/](reading/))

| Source | Focus | Page |
|--------|-------|------|
| The Bitter Lesson (Sutton) | General methods + computation > human knowledge | [bitter-lesson](reading/the-bitter-lesson.md) |
| You and Your Research (Hamming) | Work on important problems, compound advantage, courage | [hamming](reading/you-and-your-research.md) |
| Don't Teach. Incentivize. (Chung) | Scaling as the ultimate incentive, finding great problems | [dont-teach](reading/dont-teach-incentivize.md) |

## Reinforcement Learning

| Source | Focus | Page |
|--------|-------|------|
| RL Mathematical Foundations (Zhao) | 288-page textbook: MDP → Bellman → TD → Policy Gradient | [rl-foundations](learning/rl-foundations.md) |
| TRPO (Schulman et al.) | Trust region via KL constraint — foundation of PPO/GRPO/GSPO | [trpo](learning/trpo.md) |
| PPO (Schulman et al.) | Clipped policy gradient — simplified TRPO | [ppo](learning/ppo.md) |
| GRPO (DeepSeekMath) | Eliminates critic via group-based advantage — foundation of DeepSeek-R1 | [grpo](learning/grpo.md) |
| GSPO (Qwen) | Sequence-level IS + clipping — stabilizes MoE RL, used in Qwen3 | [gspo](learning/gspo.md) |
| ScaleRL (Meta) | Sigmoid compute-performance curves for RL | [scalerl](learning/scalerl.md) |
| Stabilizing RL (Qwen) | Token-level RL as first-order approx, Routing Replay | [stabilizing](learning/stabilizing-rl-llms.md) |
| Bitter Lesson for RL | Verification as the key to reasoning | [bitter-lesson-rl](learning/bitter-lesson-rl.md) |

## LLM Architecture & Attention

| Source | Focus | Page |
|--------|-------|------|
| Architecture Comparison (Raschka) | 23 models: MHA→GQA→MLA→SWA→DSA evolution | [comparison](learning/llm-architecture-comparison.md) |
| Multi-Head Latent Attention | MLA deep-dive with KV cache analysis | [mla](learning/multi-head-latent-attention.md) |
| FlashAttention | Tiling, online softmax, O(N) HBM attention | [flashattention](learning/flashattention.md) |
| Ring Attention | Distributed attention across GPU ring, zero comm overhead | [ring-attn](learning/ring-attention.md) |

## Inference Systems

| Source | Focus | Page |
|--------|-------|------|
| Inside vLLM (Gordić) | Engine V1 deep-dive: scheduler, paged attention, prefix caching, specdec, disaggregated P/D, MultiProcExecutor | [vllm](learning/vllm-anatomy.md) |

## GPU Hardware & Architecture

| Source | Focus | Page |
|--------|-------|------|
| AI Systems Perf Engineering | 1061-page reference: CUDA → cluster | [ai-systems](learning/aspe-overview.md) |
| GPU Hardware Architecture | NVIDIA roadmap, NVL72, NVLink, HBM | [gpu-hw](learning/aspe-gpu-hardware-architecture.md) |
| GPU Memory Hierarchy | Registers → HBM, SMEM, TMEM, TMA | [gpu-mem](learning/aspe-gpu-memory-hierarchy.md) |
| CUDA Kernel Optimization | Occupancy, warp efficiency, ILP, fusion | [cuda-opt](learning/aspe-cuda-kernel-optimization.md) |
| CUDA Graphs & Orchestration | Streams, graphs, atomics, NVSHMEM | [cuda-graphs](learning/aspe-cuda-graphs-and-orchestration.md) |
| Tensor Core Evolution | Volta → Blackwell MMA, PTX/SASS, TMEM | [tensor-cores](learning/tensor-core-evolution.md) |
| NVIDIA GPU Microbenchmarks | Ampere/Hopper/Blackwell instruction latency | [gpu-bench](learning/nvidia-gpu-architecture-microbenchmarks.md) |
| Thread Block Clusters | DSMEM, warp specialization, persistent kernels | [thread-blocks](learning/aspe-thread-block-clusters.md) |

## Communication & Networking

| Source | Focus | Page |
|--------|-------|------|
| Demystifying NCCL | Protocols, ring/tree, channels | [nccl](learning/nccl-demystifying.md) |
| NCCL Device API / GIN | GPU-initiated networking | [nccl-gin](learning/nccl-device-api-gin.md) |
| NCCL EP | MoE communication library | [nccl-ep](learning/nccl-ep.md) |
| GPU Communication Landscape | Full communication stack survey | [comm-survey](learning/gpu-communication-landscape.md) |
| Distributed Networking | Magnum IO, SHARP, GPUDirect tuning | [networking](learning/aspe-distributed-networking-tuning.md) |
| Rail-only Network | Eliminate spine layer, 38-77% cost reduction | [rail-only](learning/rail-only-network.md) |

## Compiler & Auto-Parallelism

| Source | Focus | Page |
|--------|-------|------|
| GSPMD | Compiler-based auto-parallelization | [gspmd](learning/gspmd.md) |
| PartIR | Composable SPMD tactics, MLIR-based | [partir](learning/partir.md) |
| TOAST | Static analysis + MCTS auto-partitioning | [toast](learning/toast.md) |
| Auto-Parallelism Survey | 30+ systems, communication cost formulas | [auto-par](learning/auto-parallelism-survey.md) |

## Systems & Infrastructure

| Source | Focus | Page |
|--------|-------|------|
| OS/Docker/K8s Tuning | NUMA, THP, MIG, cgroup isolation | [os](learning/aspe-os-docker-k8s-tuning.md) |
| Storage I/O & Data Pipeline | GDS, NVMe, NeMo Curator, 3FS | [storage](learning/aspe-gpu-storage-io.md) |
| PyTorch Profiling | Profiler, torch.compile, Triton, CUTLASS | [pytorch](learning/aspe-pytorch-profiling-tuning.md) |
| Inference Optimization | Disaggregation, autotuning, precision switching | [inference](learning/aspe-inference-optimization-techniques.md) |

## Cross-Cutting Themes

- **Memory / Communication / Compute**: Three constraints appear across all sources
- **FP8/FP4**: From DeepSeek-V3's pioneering FP8 training to Blackwell's NVFP4
- **Attention efficiency**: MLA (V3) → DSA (V3.2) → CSA+HCA (V4) — progressive KV compression; Ring Attention for distributed long context
- **Post-training evolution**: SFT+RL (V3) → Pure RL/GRPO (R1) → On-Policy Distillation (V4)
- **EP communication**: Layer-level 1F1B (Megatron) → Expert-level (DeepSeek-V4) → Sub-kernel (Comet)
- **RL theory**: PPO → GRPO → token-level first-order approximation → sigmoid scaling curves
- **Network design**: Rail-only (sparse patterns don't need full bisection), NCCL algorithms
- **Optimizer evolution**: AdamW → Muon (matrix orthogonalization, ~2× efficiency)
