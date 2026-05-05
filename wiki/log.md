# Log


## [2026-05-04] ingest | Paper: Folding Tensor and Sequence Parallelism — TSP (Shyam et al., Apr 2026)
- Created: learning/tsp-folding-parallelism.md
- Source: raw/tsp-folding-parallelism.pdf (arXiv:2604.26294)
- Folds TP+SP onto single mesh axis — ring-based weight circulation + KV exchange for attention
- 1 page touched
- Created: learning/scaling-book.md
- Short textbook on scaling LLMs on TPUs: roofline analysis, TPU architecture, sharding, parallelism (FSDP/TP/PP/EP), Transformer math, inference, profiling, JAX
- 1 page touched
- Created: learning/cross-domain-scaling-laws.md
- Extends scaling laws to images, video, multimodal, math — nearly universal exponents
- Information-theoretic decomposition: L = S(True) + D_KL(True||Model)
- 1 page touched
- Created: learning/rail-only-network.md, learning/muon-optimizer.md
- Rail-only: eliminates spine layer in GPU clusters, 38-77% network cost reduction, exploits sparse LLM communication patterns
- Muon: matrix orthogonalization optimizer, ~2× compute efficiency vs AdamW, Moonlight 3B/16B MoE model, used in DeepSeek-V4
- 2 pages touched
- Created: learning/vllm-anatomy.md, learning/ring-attention.md, learning/communication-overlap-decomposition.md
- vLLM: deep-dive into V1 engine — scheduler, paged attention, continuous batching, prefix caching, specdec, guided decoding, disaggregated P/D, MultiProcExecutor
- Ring Attention: online safe softmax, outer/inner loop decomposition, ring topology for zero-overhead K,V rotation, communication-compute overlap condition
- Communication Overlap: decompose collectives + dependent computation into finer-grained ops for TP overlap, 1.14-1.38x throughput, 72% FLOPs util on 1024 TPU chips
- 3 pages touched
- Deleted: 73 orphaned figures from 5 papers (trpo, ppo, scalerl leftovers, stabilizing_rl, partir)
- Fixed: 3 formula issues — rl-foundations.md (| |→\lvert\rvert, <→\lt), scalerl.md (|y_g|→\lvert\rvert), gspo.md (3× |y_i|→\lvert\rvert)
- Kept: scalerl_figure3_interpreting_eq1.png (Figure 3 from ScaleRL paper)
- Assets: 198→126 figures
- 4 pages touched

## [2026-05-03] ingest | Articles: Tensor Core Evolution — Volta to Blackwell (SemiAnalysis, 2025-2026)
- Created: learning/tensor-core-evolution.md
- Sources: SemiAnalysis (Dylan Patel, Kimbo Chen) — two-part deep dive
- 5 generations of MMA: warp-scoped (Volta) → warpgroup (Hopper) → single-thread tcgen05 (Blackwell)
- Blackwell deep dive: LDGSTS vs TMA, 2SM MMA, TMEM, floorsweeping, DSMEM throughput, MMA roofline
- 1 page touched

## [2026-05-03] merge | Recovered original pages from git, properly merged without simplification
- Megatron-Core MoE (7→1): recovered all 7 pages (953 lines) from git, merged into megatron-core-moe.md (801 lines), removing only duplicate Key Points/Connections
- DeepSeek-V4 (4→1): recovered all 4 pages (552 lines) from git, merged into deepseek-v4.md (472 lines)
- Wiki at 38 pages. All original detail preserved, only cross-page duplicates removed.
- Created: learning/comet-moe-overlap.md
- Source: raw/comet-moe-overlap.pdf (14 pages)
- Data dependency analysis + thread block specialization for sub-kernel comm-compute overlap
- 1.96× single MoE layer, 1.71× end-to-end speedup; deployed in 10K+ GPU production clusters
- 1 page touched
- Source: raw/auto-parallelism-survey.pdf (22 pages)
- 30+ systems across 7 searching methods: DP, MCTS, RL, ILP, sharding propagation, greedy, grid search
- Includes communication cost formulas for DP/ZeRO/TP variants and inter-layer redistribution table
- 1 page touched
- Sources: raw/gpu-communication-landscape.pdf (survey), raw/pat-algorithm.pdf (NCCL algorithm)
- Landscape: GPU-centric communication stack, vendor APIs, libraries (NCCL, NVSHMEM, UCX, MPI, Gloo), memory management
- PAT: Sylvain Jeaugey's new O(log P) tree algorithm for AllGather/ReduceScatter, Brucks construction, NCCL integration
- 3 figures from PAT LaTeX source
- 1 page touched
- Sources: Two Substack articles — architecture comparison (23 models) and attention variants deep dive
- Covers: MHA→GQA→MLA evolution, SWA, DSA, Gated Attention, normalization placements, MoE trends
- Maps to existing DeepSeek V3/V4/MLA/DSA pages
- 1 page touched
- Sources: raw/2208.11174.pdf (8pp), raw/2402.13499.pdf (12pp), raw/2507.10789.pdf (11pp), raw/2503.20481.pdf (15pp)
- Ampere→Hopper→Blackwell evolution: tensor cores (3rd→5th gen), FP8→FP4, TMA, DSMEM, unified cores, core pipeline reverse-engineering
- 1 page touched

## [2026-05-03] ingest | Essay: The Bitter Lesson (Rich Sutton, 2019)
- Created: reading/the-bitter-lesson.md
- Source: raw/the-bitter-lesson.pdf (short essay)
- Four case studies (chess, Go, speech, vision) showing general methods + computation > human knowledge
- 1 page touched

## [2026-05-03] ingest | Survey: Efficient Training of Large Language Models on Distributed Infrastructures (Duan et al., 2024)
- Created: learning/efficient-llm-training-survey.md
- Source: raw/efficient-llm-training-survey.pdf (42 pages, 380+ refs)
- Comprehensive taxonomy covering hardware, parallelism, computation, memory, communication, fault tolerance
- 1 page touched

## [2026-05-03] lint | Wiki re-lint after figure/formula fixes
- Removed incorrect page-render figures, replaced with original arxiv LaTeX figures
- Fixed all source references (.png→.pdf), corrected figure paths, removed orphans
- Added missing intra-node figure to NCCL page
- 0 broken links, 0 orphan pages, 0 stale

## [2026-05-02] ingest | Book: AI Systems Performance Engineering (Chris Fregly, O'Reilly, Nov 2025)
- Created: learning/aspe-overview.md
- Created: learning/aspe-gpu-memory-hierarchy.md
- Created: learning/aspe-cuda-kernel-optimization.md
- Created: learning/aspe-cuda-graphs-and-orchestration.md
- Created: learning/aspe-inference-optimization-techniques.md
- Created: learning/aspe-gpu-hardware-architecture.md
- Created: learning/aspe-os-docker-k8s-tuning.md
- Created: learning/aspe-distributed-networking-tuning.md
- Created: learning/aspe-gpu-storage-io.md
- Created: learning/aspe-thread-block-clusters.md
- Created: learning/aspe-pytorch-profiling-tuning.md
- Updated: wiki/overview.md
- Source: raw/ai-systems-performance-engineering.pdf (1061 pages)
- Full-stack GPU-to-cluster optimization: CUDA, memory hierarchy, kernel opt, CUDA Graphs, streams, inference, hardware, OS/K8s, networking, storage, thread blocks, PyTorch profiling
- 11 pages touched (source-note + 10 chapter pages)

## [2026-05-02] ingest | Papers: RL Scaling trilogy — ScaleRL, The Bitter Lesson for RL, Scaling RL Guest Lecture
- Created: learning/scalerl.md
- Created: learning/bitter-lesson-rl.md
- Updated: wiki/overview.md
- Sources: raw/art-of-scaling-rl-compute.pdf (28pp), raw/the-bitter-lesson-for-rl.pdf (34pp slides), raw/Scaling_RL_Guest_Lecture.pdf (66pp slides)
- ScaleRL: systematic 400K GPU-hr RL scaling study, sigmoid compute-performance curves, CISPO loss, predicted 100K GPU-hr run
- Bitter Lesson: verification as the key to reasoning, generative verifiers, test-time compute scaling
- 2 pages touched

## [2026-05-02] ingest | Article: LLM Scaling Laws: From GPT-3 to o3 (Cameron R. Wolfe, Jan 2025)
- Created: learning/llm-scaling-laws.md
- Updated: wiki/overview.md
- Source: https://cameronrwolfe.substack.com/p/llm-scaling-laws
- Covers: Power laws, Kaplan (2020) vs Chinchilla (2022), compute-optimal training, data wall, RL scaling laws, plateau
- Theoretical foundation for the scaling decisions in all model papers in the wiki
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

## [2026-05-02] ingest | Paper: DeepSeek-V3.2: Pushing the Frontier of Open LLMs (DeepSeek-AI, Dec 2025)
- Created: learning/deepseek-v3.2.md
- Updated: wiki/overview.md
- Source: raw/DeepSeek-V3.2.pdf (23 pages)
- Covers: DSA sparse attention, scalable RL (>10% pre-training budget), V3.2-Speciale surpasses GPT-5, IMO/IOI gold medals, agent synthesis pipeline
- Connects DSA → V4's CSA architecture evolution
- 1 page touched

## [2026-05-02] ingest | Paper: DeepSeek-V3 Technical Report (DeepSeek-AI, Dec 2024)
- Created: learning/deepseek-v3.md
- Updated: wiki/overview.md
- Source: raw/DeepSeek-V3.pdf (53 pages)
- Covers: MLA + DeepSeekMoE, auxiliary-loss-free load balancing, MTP, FP8 training, DualPipe, 14.8T tokens, 2.788M H800 GPU hours
- First open-source large model with FP8 training; zero loss spikes or rollbacks
- 1 page touched

## [2026-05-02] ingest | Paper: DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (DeepSeek-AI, Jan 2025)
- Created: learning/deepseek-r1.md
- Updated: wiki/overview.md
- Source: raw/DeepSeek-R1.pdf (86 pages)
- Covers: R1-Zero (pure RL → emergent reasoning), GRPO, multi-stage R1 pipeline, distillation, benchmark results
- R1 matches o1 on math/code; USAMO qualification; 96.3 Codeforces percentile
- Core insight: reasoning emerges from RL + verifiable rewards, not from SFT on human demonstrations
- 1 page touched

## [2026-05-02] ingest | Book: The Ultra-Scale Playbook: Training LLMs on GPU Clusters (HuggingFace/nanotron, Feb 2025)
- Created: learning/usp-ultra-scale-playbook.md
- Created: learning/usp-scaling-techniques-overview.md
- Updated: wiki/overview.md
- Source: raw/The_Ultra-Scale_Playbook.pdf (comprehensive educational book, 4,100+ experiments)
- Covers: DP, ZeRO, TP, SP, CP, PP, EP, 5D parallelism, kernel fusion, FlashAttention, mixed precision
- Serves as educational foundation for existing Megatron-Core and DeepSeek-V4 pages
- 2 pages touched

## [2026-05-02] ingest | Paper: DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence (DeepSeek-AI, April 2026)
- Created: learning/deepseek-v4.md
- Created: learning/deepseek-v4.md
- Created: learning/deepseek-v4.md
- Created: learning/deepseek-v4.md
- Updated: wiki/overview.md
- Source: raw/DeepSeek-V4.pdf (58 pages)
- User requested all sections: architecture, post-training, infrastructure, benchmarks
- 4 pages touched

## [2026-05-02] ingest | Paper: Scalable Training of Mixture-of-Experts Models with Megatron Core (NVIDIA, 2026)
- Created: learning/megatron-core-moe.md
- Created: learning/megatron-core-moe.md
- Created: learning/megatron-core-moe.md
- Created: learning/megatron-core-moe.md
- Created: learning/megatron-core-moe.md
- Created: learning/megatron-core-moe.md
- Created: learning/megatron-core-moe.md
- Updated: wiki/overview.md
- Source: raw/scalable-training-moe-megatron-core.pdf (88 pages)
- User requested all sections: Three Walls, Parallel Folding, FP8/FP4, Performance Best Practices
- 8 pages touched

## [2026-05-02] figures | Added rendered figure pages to 7 wiki pages
- Extracted and linked figures from DeepSeek-V3, DeepSeek-R1, DeepSeek-V4, and FlashAttention
- R1: pipeline diagram, accuracy/length curves, aha moment, benchmark comparison
- V3: architecture overview, DualPipe scheduling, FP8 framework
- V4: benchmarks+efficiency, CSA architecture, HCA architecture, reasoning modes table
- FlashAttention: tiled algorithm hardware diagram
- Figures stored in raw/assets/ — rendered at 2x zoom from PDF pages

## [2026-05-02] ingest | Note: From Online Softmax to FlashAttention (Zihao Ye, UW, May 2023)
- Updated: learning/flashattention.md (added full mathematical derivation)
- Source: raw/from-online-softmax-to-flashattention.pdf (6 pages)
- Covers: 3-pass safe softmax → 2-pass online softmax → 1-pass FlashAttention derivation, surrogate sequence trick, tiled algorithm, rescaling term
- 1 page touched
