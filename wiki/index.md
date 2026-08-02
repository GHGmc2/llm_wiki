# Index

*71 pages · Last updated: 2026-07-09*

## Tech Reports

### GPT (OpenAI)

- **GPT-1** (2018) — Semi-supervised pre-training + fine-tuning, 117M · [page](gpt-1.md)
- **GPT-2** (2019) — Zero-shot transfer, 1.5B, task emergence with scale · [page](gpt-2.md)
- **GPT-3** (2020) — In-context few-shot learning, 175B · [page](gpt-3.md)
- **InstructGPT** (2022) — RLHF: SFT → Reward Model → PPO alignment · [page](instruct-gpt.md)
- **GPT-4** (2023) — Multimodal, predictable scaling, undisclosed architecture · [page](gpt-4.md)

### DeepSeek

- **DeepSeek-V3** (Dec 2024) — MLA, FP8 training, auxiliary-loss-free MoE, MTP · [page](deepseek-v3.md)
- **V3 Insights** (ISCA 2025) — Hardware-aware co-design, Multi-Plane Network · [page](deepseek-v3-insights.md)
- **DeepSeek-R1** (Jan 2025) — Pure RL (GRPO), emergent reasoning, distillation · [page](deepseek-r1.md)
- **DeepSeek-V3.2** (Dec 2025) — DSA sparse attention, scalable RL, IMO/IOI gold · [page](deepseek-v3.2.md)
- **DeepSeek-V4** (Apr 2026) — CSA+HCA hybrid attention, 1M context, OPD, Muon · [page](deepseek-v4.md)

### Microsoft

- **MAI-Thinking-1** (2026) — Hill-climbing machine: 35B MoE, system-level optimization, 8K GB200s · [page](mai-thinking-1.md)

## Training Systems & Parallelism

- **Megatron-Core MoE** — Production stack: Three Walls, Parallel Folding, FP8/FP4 · `moe, megatron, nvidia` · [page](megatron-core-moe.md)
- **Ultra-Scale Playbook** — Educational foundation: DP, ZeRO, TP, SP, CP, PP, EP · `scaling, hf` · [page](usp-ultra-scale-playbook.md)
- **How To Scale Your Model** (DeepMind) — Textbook: roofline, TPU, sharding, FSDP/TP/PP · `scaling, tpu, jax` · [page](scaling-book.md)
- **Scaling Techniques Overview** — 5D parallelism systematic overview · `dp, tp, pp, ep, cp` · [page](usp-scaling-techniques-overview.md)
- **Efficient LLM Training Survey** — Taxonomy of training systems, 380+ refs · `survey, training` · [page](efficient-llm-training-survey.md)
- **Comet** — Fine-grained EP communication overlap · `moe, communication` · [page](comet-moe-overlap.md)
- **FlashMoE** (NeurIPS '25) — Single-kernel MoE: dispatch+compute+combine, GPU-initiated RDMA · `moe, gpu-kernel, rdma` · [page](flashmoe.md)
- **Comm Overlap via Decomposition** (ASPLOS '23) — Decompose collectives for TP, 1.14-1.38× · `communication, tp` · [page](communication-overlap-decomposition.md)
- **TSP: Folding TP + SP** — Single-axis TP+SP folding, ring weight circulation + KV exchange · `tp, sp, parallelism` · [page](tsp-folding-parallelism.md)
- **veScale-FSDP** — RaggedShard flexible format, zero-copy FSDP, supports Muon/Shampoo · `fsdp, sharding` · [page](vescale-fsdp.md)
- **Maestro** (Qwen) — Section-centric training, wavefront scheduling, ~40% GPU reduction · `training, scheduling` · [page](maestro.md)
- **Muon Optimizer** — Matrix orthogonalization, ~2× vs AdamW, Moonlight MoE · `optimizer, muon` · [page](muon-optimizer.md)

## Scaling Laws

- **Kaplan et al.** (2020) — $L(N) \propto N^{-\alpha}$, $D \propto N^{0.74}$, 7 orders of magnitude · [page](llm-scaling-laws.md)
- **Henighan et al.** (2020) — Cross-domain: images, video, multimodal, math — universal exponents · [page](cross-domain-scaling-laws.md)

## Reinforcement Learning

### Algorithms

- **RL Mathematical Foundations** (Zhao) — 288-page textbook: MDP → Bellman → TD → Policy Gradient · [page](rl-foundations.md)
- **TRPO** (Schulman et al., 2015) — Trust region via KL constraint, foundation of PPO/GRPO/GSPO · [page](trpo.md)
- **PPO** (Schulman et al., 2017) — Clipped surrogate objective, GAE, simplifies TRPO · [page](ppo.md)
- **GRPO** (DeepSeekMath, 2024) — Eliminates critic via group-based advantage, foundation of R1 · [page](grpo.md)
- **GSPO** (Qwen, 2025) — Sequence-level IS + clipping, used in Qwen3 · [page](gspo.md)
- **ScaleRL** (Meta) — Sigmoid compute-performance curves, 400K+ GPU-hour study · [page](scalerl.md)
- **Stabilizing RL with LLMs** (Qwen) — Token-level first-order approx, Routing Replay · [page](stabilizing-rl-llms.md)
- **Bitter Lesson for RL** — Verification as the key to reasoning LLMs · [page](bitter-lesson-rl.md)
- **KL Estimators in RL** (Wang) — Gradient-correct $k_1, k_2, k_3$: which estimator for which setting · [page](kl-estimators.md)

### Infrastructure

- **Async RL Training Landscape** (HF, 2026) — Survey: 16 open-source libraries across 7 design axes · [page](async-rl-training-landscape.md)
- **RL Systems: Mind the Gap** (SemiAnalysis, 2026) — Matching trainer-generator throughput, case studies at scale · [page](rl-systems-mind-the-gap.md)
- **Predicting Staleness in Async RL** (Applied Compute, 2026) — Closed-form staleness formulas, queue theory, Pareto frontier · [page](staleness-in-async-rl.md)
- **TITO: Token-In, Token-Out** (HF, 2026) — Multi-turn agentic RL invariant: never re-encode decoded tokens · [page](tito-agentic-rl.md)

## LLM Architecture & Attention

- **Architecture Comparison** (Raschka) — 23 models: MHA→GQA→MLA→SWA→DSA evolution · [page](llm-architecture-comparison.md)
- **Multi-Head Latent Attention** — MLA deep-dive: KV compressed to latent space, 10× cache reduction · [page](multi-head-latent-attention.md)
- **FlashAttention** — Tiling, online softmax, O(N) HBM attention · [page](flashattention.md)
- **Ring Attention** — Distributed attention across GPU ring, zero communication overhead · [page](ring-attention.md)

## Inference Systems

- **Inside vLLM** (Gordić, 2025) — Engine V1: scheduler, paged attention, prefix caching, specdec, disaggregated P/D, MultiProcExecutor · [page](vllm-anatomy.md)

## GPU Hardware & Architecture

- **AI Systems Perf Engineering** (Fregly) — 1061-page reference: CUDA → cluster · [page](aspe-overview.md)
- **GPU Hardware Architecture** — NVIDIA roadmap: H100, B200, GB200, NVL72, NVLink · [page](aspe-gpu-hardware-architecture.md)
- **GPU Memory Hierarchy** — Registers → HBM, SMEM, TMEM, TMA · [page](aspe-gpu-memory-hierarchy.md)
- **CUDA Kernel Optimization** — Occupancy, warp efficiency, ILP, kernel fusion · [page](aspe-cuda-kernel-optimization.md)
- **CUDA Graphs & Orchestration** — Streams, CUDA graphs, atomics, NVSHMEM · [page](aspe-cuda-graphs-and-orchestration.md)
- **Tensor Core Evolution** — Volta → Blackwell MMA, PTX/SASS, TMEM · [page](tensor-core-evolution.md)
- **NVIDIA GPU Microbenchmarks** — Ampere/Hopper/Blackwell instruction latency · [page](nvidia-gpu-architecture-microbenchmarks.md)
- **Thread Block Clusters** — DSMEM, warp specialization, persistent kernels · [page](aspe-thread-block-clusters.md)
- **OS/Docker/K8s Tuning** — NUMA, THP, MIG, cgroup isolation · [page](aspe-os-docker-k8s-tuning.md)
- **GPU Storage I/O** — GPUDirect Storage, NVMe, NeMo Curator, 3FS · [page](aspe-gpu-storage-io.md)
- **PyTorch Profiling** — Profiler, torch.compile, Triton, CUTLASS · [page](aspe-pytorch-profiling-tuning.md)
- **Inference Optimization** — Disaggregation, autotuning, precision switching · [page](aspe-inference-optimization-techniques.md)

## Communication & Networking

- **Demystifying NCCL** — Protocols, ring/tree algorithms, channels, PAT · [page](nccl-demystifying.md)
- **Demystifying NVSHMEM** — PGAS symmetric memory, device-initiated put/get, DeepEP case study · [page](nvshmem.md)
- **NCCL Device API / GIN** — GPU-initiated networking, device-side RDMA · [page](nccl-device-api-gin.md)
- **NCCL EP** — Unified expert parallel communication API · [page](nccl-ep.md)
- **GPU Communication Landscape** — Full stack survey: NCCL, NVSHMEM, UCX, MPI · [page](gpu-communication-landscape.md)
- **Distributed Networking** — Magnum IO, SHARP, GPUDirect tuning · [page](aspe-distributed-networking-tuning.md)
- **Rail-only Network** — Eliminate spine layer, 38-77% network cost reduction · [page](rail-only-network.md)
- **Every Microsecond Matters** (ETH/NVIDIA, 2026) — Barrier-free low-latency GPU collectives, 7% from SoL bound · [page](every-microsecond-matters.md)

## Compiler & Auto-Parallelism

- **GSPMD** — Compiler-based auto-parallelization, device mesh + sharding annotations · [page](gspmd.md)
- **PartIR** — Composable SPMD tactics, MLIR-based · [page](partir.md)
- **TOAST** — Static analysis + MCTS auto-partitioning · [page](toast.md)
- **Auto-Parallelism Survey** — 30+ systems, 7 searching methods · [page](auto-parallelism-survey.md)
- **PyTorch Compilation** — torch.compile guide, Inductor, Dynamo · [page](pytorch-compilation.md)

## Research Philosophy

- **The Bitter Lesson** (Sutton) — General methods + computation > human knowledge · [page](the-bitter-lesson.md)
- **You and Your Research** (Hamming) — Work on important problems, compound advantage · [page](you-and-your-research.md)
- **Don't Teach. Incentivize.** (Chung) — Scaling as the ultimate incentive · [page](dont-teach-incentivize.md)

## Cross-Cutting Themes

- **Memory / Communication / Compute** — Three constraints appear across all sources
- **FP8/FP4** — From DeepSeek-V3's pioneering FP8 training to Blackwell's NVFP4
- **Attention efficiency** — MLA (V3) → DSA (V3.2) → CSA+HCA (V4), Ring Attention for distributed long context
- **Post-training evolution** — SFT+RL (V3) → Pure RL/GRPO (R1) → On-Policy Distillation (V4)
- **EP communication** — Layer-level 1F1B (Megatron) → Expert-level (V4) → Sub-kernel (Comet)
- **RL theory** — PPO → GRPO → token-level first-order approx → sigmoid scaling curves
- **Network design** — Rail-only, NCCL ring/tree, GPU-initiated RDMA
- **Optimizer evolution** — AdamW → Muon (matrix orthogonalization, ~2× efficiency)
- **Kernel co-design** — CUDA Graphs → FlashAttention → FlashMoE (single persistent kernel)
