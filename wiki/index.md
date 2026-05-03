# Index

*Last updated: 2026-05-03*

## Learning (38 pages)
| Page | Summary | Type | Tags | Updated |
|------|---------|------|------|---------|
| [AI Systems Performance Engineering](learning/ai-systems-performance-engineering.md) |  | source-note | performance, cuda, gpu, pytorch, profiling, inference, training, systems, book | 2026-05-03 |
| [A Survey on Auto-Parallelism of Neural Networks Training](learning/auto-parallelism-survey.md) |  | source-note | auto-parallelism, distributed-training, parallelism, survey, dp, tp, pp, mcts, rl | 2026-05-03 |
| [The Bitter Lesson for RL: Verification as the Key to Reasoning LLMs](learning/bitter-lesson-rl.md) |  | source-note | reinforcement-learning, verification, reasoning, bitter-lesson, generative-verifiers, test-time-compute | 2026-05-03 |
| [Comet: Fine-grained Computation-Communication Overlapping for MoE](learning/comet-moe-overlap.md) |  | source-note | moe, communication, overlapping, fine-grained, comet, megatron, gpu | 2026-05-03 |
| [CUDA Graphs, Streams, and GPU Orchestration](learning/cuda-graphs-and-orchestration.md) |  | concept | cuda, gpu, cuda-graphs, streams, atomics, dynamic-scheduling, nvshmem, nccl | 2026-05-03 |
| [CUDA Kernel Optimization: Occupancy, Warp Efficiency, and Arithmetic Intensity](learning/cuda-kernel-optimization.md) |  | concept | cuda, gpu, kernel-optimization, occupancy, warp-divergence, ilp, kernel-fusion, tensor-cores | 2026-05-03 |
| [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](learning/deepseek-r1.md) |  | source-note | llm, deepseek, reinforcement-learning, grpo, reasoning, r1, chain-of-thought, distillation | 2026-05-03 |
| [Insights into DeepSeek-V3: Scaling Challenges and Hardware Reflections](learning/deepseek-v3-insights.md) |  | source-note | llm, deepseek, deepseek-v3, mla, fp8, moe, hardware-codesign, network, isca | 2026-05-03 |
| [DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models](learning/deepseek-v3.2.md) |  | source-note | llm, deepseek, deepseek-v3.2, sparse-attention, dsa, rl, agent, imo | 2026-05-03 |
| [DeepSeek-V3 Technical Report](learning/deepseek-v3.md) |  | source-note | llm, deepseek, deepseek-v3, mla, moe, fp8, multi-token-prediction, dualpipe | 2026-05-03 |
| [DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence](learning/deepseek-v4.md) |  | source-note | llm, mixture-of-experts, deepseek, long-context, hybrid-attention, csa, hca, mhc, muon, post-training, infrastructure | 2026-05-03 |
| [Distributed Networking: NCCL Tuning, GPUDirect, and SHARP](learning/distributed-networking-tuning.md) |  | concept | networking, nccl, gpudirect, rdma, sharp, magnum-io, nixl, nvshmem | 2026-05-03 |
| [Efficient Training of Large Language Models on Distributed Infrastructures: A Survey](learning/efficient-llm-training-survey.md) |  | source-note | llm-training, survey, distributed-systems, parallelism, infrastructure, fault-tolerance | 2026-05-03 |
| [FlashAttention: Memory-Efficient Exact Attention](learning/flashattention.md) |  | concept | attention, flash-attention, gpu, kernel-fusion, tiling, hbm, online-softmax | 2026-05-03 |
| [The Landscape of GPU-Centric Communication](learning/gpu-communication-landscape.md) |  | source-note | gpu, communication, nccl, nvshmem, gpudirect, collective, survey | 2026-05-03 |
| [GPU Hardware Architecture: NVIDIA Roadmap and Rack-Scale Systems](learning/gpu-hardware-architecture.md) |  | concept | gpu, hardware, nvidia, h100, b200, gb200, nvl72, nvlink, nvswitch, roadmap | 2026-05-03 |
| [GPU Memory Hierarchy and CUDA Programming Model](learning/gpu-memory-hierarchy.md) |  | concept | cuda, gpu, memory-hierarchy, shared-memory, registers, tiling, hbm, tensor-cores | 2026-05-03 |
| [GPU Storage I/O and Data Pipeline Optimization](learning/gpu-storage-io.md) |  | concept | storage, gpu, gpudirect-storage, nvme, data-pipeline, deepseek-3fs, nemo-curator, dali | 2026-05-03 |
| [GSPMD: General and Scalable Parallelization for ML Computation Graphs](learning/gspmd.md) |  | source-note | compiler, parallelization, spmd, gspmd, google, tpu, device-mesh, sharding | 2026-05-03 |
| [Inference Optimization: Disaggregation, Autotuning, and Precision Switching](learning/inference-optimization-techniques.md) |  | concept | inference, llm, disaggregation, autotuning, precision, kv-cache, speculative-decoding, performance | 2026-05-03 |
| [LLM Architecture Comparison and Attention Variants](learning/llm-architecture-comparison.md) |  | source-note | llm, architecture, attention, mha, gqa, mla, swa, dsa, moe, comparison | 2026-05-03 |
| [LLM Scaling Laws: From GPT-3 to the Plateau](learning/llm-scaling-laws.md) |  | source-note | scaling-laws, llm, pretraining, kaplan, power-law | 2026-05-03 |
| [Megatron-Core MoE: Scalable Training of Mixture-of-Experts Models](learning/megatron-core-moe.md) |  | source-note | llm-training, mixture-of-experts, distributed-systems, nvidia, parallelism, megatron, memory, communication, compute, fp | 2026-05-03 |
| [Multi-Head Latent Attention (MLA)](learning/multi-head-latent-attention.md) |  | concept | llm, attention, deepseek, kv-cache, memory-optimization, inference, mla | 2026-05-03 |
| [Demystifying NCCL: GPU Communication Protocols and Algorithms](learning/nccl-demystifying.md) |  | source-note | nccl, gpu, communication, collectives, hpc, nvidia, distributed-training | 2026-05-03 |
| [GPU-Initiated Networking for NCCL: Device API and GIN](learning/nccl-device-api-gin.md) |  | source-note | nccl, gpu, networking, rdma, device-api, gin, deep-ep, moe, nvidia | 2026-05-03 |
| [NCCL EP: Unified Expert Parallel Communication API](learning/nccl-ep.md) |  | source-note | nccl, expert-parallelism, moe, gpu, deep-ep, hybrid-ep, vllm, dispatch, combine | 2026-05-03 |
| [NVIDIA GPU Architecture: Ampere, Hopper, and Blackwell Microbenchmark Analysis](learning/nvidia-gpu-architecture-microbenchmarks.md) |  | source-note | gpu, nvidia, ampere, hopper, blackwell, microbenchmarking, architecture, tensor-cores | 2026-05-03 |
| [OS, Docker, and Kubernetes Tuning for GPU Workloads](learning/os-docker-k8s-tuning.md) |  | concept | linux, docker, kubernetes, gpu, numa, hugepages, mig, cgroups, performance | 2026-05-03 |
| [PartIR: Composing SPMD Partitioning Strategies for Machine Learning](learning/partir.md) |  | source-note | compiler, spmd, partitioning, mlir, part IR, google, sharding, tactics | 2026-05-03 |
| [PyTorch Profiling, Compilation, and Performance Tuning](learning/pytorch-profiling-tuning.md) |  | concept | pytorch, profiling, torch-compile, nsight, triton, cutlass, performance | 2026-05-03 |
| [ScaleRL: The Art of Scaling Reinforcement Learning Compute for LLMs](learning/scalerl.md) |  | source-note | reinforcement-learning, scaling-laws, grpo, llm, rl, scalerl, cispo, pipeline-rl | 2026-05-03 |
| [LLM Scaling Techniques Overview](learning/scaling-techniques-overview.md) |  | concept | llm-training, distributed-training, scaling, data-parallelism, zero, tensor-parallelism, pipeline-parallelism, context-p | 2026-05-03 |
| [Tensor Core Evolution: From Volta to Blackwell](learning/tensor-core-evolution.md) |  | source-note | gpu, nvidia, tensor-cores, volta, ampere, hopper, blackwell, ptx, sass, mma, wgmma, tcgen05 | 2026-05-03 |
| [The Bitter Lesson](learning/the-bitter-lesson.md) |  | source-note | ai, scaling, search, learning, philosophy, rich-sutton, bitter-lesson | 2026-05-03 |
| [Thread Block Clusters, DSMEM, and Warp Specialization](learning/thread-block-clusters.md) |  | concept | cuda, gpu, thread-block-clusters, warp-specialization, dsmem, persistent-kernels, cutlass | 2026-05-03 |
| [TOAST: Fast and Scalable Auto-Partitioning via Static Analysis and MCTS](learning/toast.md) |  | source-note | compiler, auto-partitioning, mcts, static-analysis, sharding, google, toast | 2026-05-03 |
| [The Ultra-Scale Playbook: Training LLMs on GPU Clusters](learning/ultra-scale-playbook.md) |  | source-note | llm-training, distributed-training, scaling, parallelism, gpu, hf, nanotron, educational | 2026-05-03 |
