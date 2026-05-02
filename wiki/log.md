# Log

## [2026-05-02] init | Wiki created
- Initial setup: directory structure, schema (AGENTS.md), index, log, overview
- 0 pages touched (fresh wiki)

## [2026-05-02] ingest | Paper: Scalable Training of Mixture-of-Experts Models with Megatron Core (NVIDIA, 2026)
- Created: learning/megatron-core-moe-scalable-training.md
- Created: learning/moe-memory-wall.md
- Created: learning/moe-communication-wall.md
- Created: learning/moe-compute-efficiency-wall.md
- Created: learning/moe-parallel-folding.md
- Created: learning/moe-fp8-fp4-training.md
- Created: learning/moe-performance-best-practices.md
- Updated: wiki/overview.md
- Source: raw/scalable-training-moe-megatron-core.pdf (88 pages)
- User requested all sections: Three Walls, Parallel Folding, FP8/FP4, Performance Best Practices
- 8 pages touched

## [2026-05-02] ingest | Paper: DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence (DeepSeek-AI, April 2026)
- Created: learning/deepseek-v4-technical-report.md
- Created: learning/deepseek-v4-architecture.md
- Created: learning/deepseek-v4-post-training.md
- Created: learning/deepseek-v4-infrastructure.md
- Updated: wiki/overview.md
- Source: raw/DeepSeek-V4.pdf (58 pages)
- User requested all sections: architecture, post-training, infrastructure, benchmarks
- 4 pages touched

## [2026-05-02] ingest | Book: The Ultra-Scale Playbook: Training LLMs on GPU Clusters (HuggingFace/nanotron, Feb 2025)
- Created: learning/ultra-scale-playbook.md
- Created: learning/scaling-techniques-overview.md
- Updated: wiki/overview.md
- Source: raw/The_Ultra-Scale_Playbook.pdf (comprehensive educational book, 4,100+ experiments)
- Covers: DP, ZeRO, TP, SP, CP, PP, EP, 5D parallelism, kernel fusion, FlashAttention, mixed precision
- Serves as educational foundation for existing Megatron-Core and DeepSeek-V4 pages
- 2 pages touched

## [2026-05-02] ingest | Paper: DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (DeepSeek-AI, Jan 2025)
- Created: learning/deepseek-r1.md
- Updated: wiki/overview.md
- Source: raw/DeepSeek-R1.pdf (86 pages)
- Covers: R1-Zero (pure RL → emergent reasoning), GRPO, multi-stage R1 pipeline, distillation, benchmark results
- R1 matches o1 on math/code; USAMO qualification; 96.3 Codeforces percentile
- Core insight: reasoning emerges from RL + verifiable rewards, not from SFT on human demonstrations
- 1 page touched

## [2026-05-02] ingest | Paper: DeepSeek-V3 Technical Report (DeepSeek-AI, Dec 2024)
- Created: learning/deepseek-v3.md
- Updated: wiki/overview.md
- Source: raw/DeepSeek-V3.pdf (53 pages)
- Covers: MLA + DeepSeekMoE, auxiliary-loss-free load balancing, MTP, FP8 training, DualPipe, 14.8T tokens, 2.788M H800 GPU hours
- First open-source large model with FP8 training; zero loss spikes or rollbacks
- 1 page touched

## [2026-05-02] ingest | Paper: DeepSeek-V3.2: Pushing the Frontier of Open LLMs (DeepSeek-AI, Dec 2025)
- Created: learning/deepseek-v3.2.md
- Updated: wiki/overview.md
- Source: raw/DeepSeek-V3.2.pdf (23 pages)
- Covers: DSA sparse attention, scalable RL (>10% pre-training budget), V3.2-Speciale surpasses GPT-5, IMO/IOI gold medals, agent synthesis pipeline
- Connects DSA → V4's CSA architecture evolution
- 1 page touched

## [2026-05-02] ingest | Papers: Compiler/Auto-Partitioning trilogy — GSPMD, PartIR, TOAST
- Created: learning/gspmd.md
- Created: learning/partir.md
- Created: learning/toast.md
- Updated: wiki/overview.md
- Sources: raw/GSPMD.pdf (16pp), raw/PartIR.pdf (36pp), raw/TOAST.pdf (13pp)
- Covers: GSPMD (compiler-based auto-parallelization, mesh_split), PartIR (composable SPMD tactics, MLIR-based), TOAST (static analysis + MCTS auto-partitioning)
- Systems that automate what the wiki's parallelism pages describe — the compiler stack above NCCL
- 3 pages touched

## [2026-05-02] ingest | Article: LLM Scaling Laws: From GPT-3 to o3 (Cameron R. Wolfe, Jan 2025)
- Created: learning/llm-scaling-laws.md
- Updated: wiki/overview.md
- Source: https://cameronrwolfe.substack.com/p/llm-scaling-laws
- Covers: Power laws, Kaplan (2020) vs Chinchilla (2022), compute-optimal training, data wall, RL scaling laws, plateau
- Theoretical foundation for the scaling decisions in all model papers in the wiki
- 1 page touched

## [2026-05-02] ingest | Papers: RL Scaling trilogy — ScaleRL, The Bitter Lesson for RL, Scaling RL Guest Lecture
- Created: learning/scalerl.md
- Created: learning/bitter-lesson-rl.md
- Updated: wiki/overview.md
- Sources: raw/The Art of Scaling Reinforcement Learning Compute for LLMs.pdf (28pp), raw/The Bitter Lesson for RL.pdf (34pp slides), raw/Scaling_RL_Guest_Lecture.pdf (66pp slides)
- ScaleRL: systematic 400K GPU-hr RL scaling study, sigmoid compute-performance curves, CISPO loss, predicted 100K GPU-hr run
- Bitter Lesson: verification as the key to reasoning, generative verifiers, test-time compute scaling
- 2 pages touched

## [2026-05-02] ingest | Book: AI Systems Performance Engineering (Chris Fregly, O'Reilly, Nov 2025)
- Created: learning/ai-systems-performance-engineering.md
- Created: learning/gpu-memory-hierarchy.md
- Created: learning/cuda-kernel-optimization.md
- Created: learning/cuda-graphs-and-orchestration.md
- Created: learning/inference-optimization-techniques.md
- Created: learning/gpu-hardware-architecture.md
- Created: learning/os-docker-k8s-tuning.md
- Created: learning/distributed-networking-tuning.md
- Created: learning/gpu-storage-io.md
- Created: learning/thread-block-clusters.md
- Created: learning/pytorch-profiling-tuning.md
- Updated: wiki/overview.md
- Source: raw/AI Systems Performance Engineering.pdf (1061 pages)
- Full-stack GPU-to-cluster optimization: CUDA, memory hierarchy, kernel opt, CUDA Graphs, streams, inference, hardware, OS/K8s, networking, storage, thread blocks, PyTorch profiling
- 11 pages touched (source-note + 10 chapter pages)
