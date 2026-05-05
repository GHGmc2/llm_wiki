# Index

*Last updated: 2026-05-04*

## Learning (52 pages)

### AI Systems Performance Engineering (11 pages)
| Page | Summary | Type | Tags | Updated |
|------|---------|------|------|---------|
| [AI Systems Performance Engineering](learning/aspe-overview.md) |  | source-note | performance, cuda, gpu, pytorch, profiling, inference, training, systems, book | 2026-05-03 |
| [CUDA Graphs, Streams, and GPU Orchestration](learning/aspe-cuda-graphs-and-orchestration.md) |  | concept | cuda, gpu, cuda-graphs, streams, atomics, dynamic-scheduling, nvshmem, nccl | 2026-05-02 |
| [CUDA Kernel Optimization: Occupancy, Warp Efficiency, and Arithmetic Intensity](learning/aspe-cuda-kernel-optimization.md) |  | concept | cuda, gpu, kernel-optimization, occupancy, warp-divergence, ilp, kernel-fusion, tensor-cores | 2026-05-02 |
| [Distributed Networking: NCCL Tuning, GPUDirect, and SHARP](learning/aspe-distributed-networking-tuning.md) |  | concept | networking, nccl, gpudirect, rdma, sharp, magnum-io, nixl, nvshmem | 2026-05-02 |
| [GPU Hardware Architecture: NVIDIA Roadmap and Rack-Scale Systems](learning/aspe-gpu-hardware-architecture.md) |  | concept | gpu, hardware, nvidia, h100, b200, gb200, nvl72, nvlink, nvswitch, roadmap | 2026-05-02 |
| [GPU Memory Hierarchy and CUDA Programming Model](learning/aspe-gpu-memory-hierarchy.md) |  | concept | cuda, gpu, memory-hierarchy, shared-memory, registers, tiling, hbm, tensor-cores | 2026-05-02 |
| [GPU Storage I/O and Data Pipeline Optimization](learning/aspe-gpu-storage-io.md) |  | concept | storage, gpu, gpudirect-storage, nvme, data-pipeline, deepseek-3fs, nemo-curator, dali | 2026-05-02 |
| [Inference Optimization: Disaggregation, Autotuning, and Precision Switching](learning/aspe-inference-optimization-techniques.md) |  | concept | inference, llm, disaggregation, autotuning, precision, kv-cache, speculative-decoding, performance | 2026-05-02 |
| [OS, Docker, and Kubernetes Tuning for GPU Workloads](learning/aspe-os-docker-k8s-tuning.md) |  | concept | linux, docker, kubernetes, gpu, numa, hugepages, mig, cgroups, performance | 2026-05-02 |
| [PyTorch Profiling, Compilation, and Performance Tuning](learning/aspe-pytorch-profiling-tuning.md) |  | concept | pytorch, profiling, torch-compile, nsight, triton, cutlass, performance | 2026-05-02 |
| [Thread Block Clusters, DSMEM, and Warp Specialization](learning/aspe-thread-block-clusters.md) |  | concept | cuda, gpu, thread-block-clusters, warp-specialization, dsmem, persistent-kernels, cutlass | 2026-05-02 |

### Ultra-Scale Playbook (2 pages)
| Page | Summary | Type | Tags | Updated |
|------|---------|------|------|---------|
| [LLM Scaling Techniques Overview](learning/usp-scaling-techniques-overview.md) |  | concept | llm-training, distributed-training, scaling, data-parallelism, zero, tensor-parallelism, pipeline-parallelism, context-parallelism, expert-parallelism | 2026-05-02 |
| [The Ultra-Scale Playbook: Training LLMs on GPU Clusters](learning/usp-ultra-scale-playbook.md) |  | source-note | llm-training, distributed-training, scaling, parallelism, gpu, hf, nanotron, educational | 2026-05-02 |

### Other (39 pages)
| Page | Summary | Type | Tags | Updated |
|------|---------|------|------|---------|
| [A Survey on Auto-Parallelism of Neural Networks Training](learning/auto-parallelism-survey.md) |  | source-note | auto-parallelism, distributed-training, parallelism, survey, dp, tp, pp, mcts, rl | 2026-05-03 |
| [The Bitter Lesson for RL: Verification as the Key to Reasoning LLMs](learning/bitter-lesson-rl.md) |  | source-note | reinforcement-learning, verification, reasoning, bitter-lesson, generative-verifiers, test-time-compute | 2026-05-02 |
| [Comet: Fine-grained Computation-Communication Overlapping for MoE](learning/comet-moe-overlap.md) |  | source-note | moe, communication, overlapping, fine-grained, comet, megatron, gpu | 2026-05-03 |
| [Communication Overlap via Decomposition in Large Deep Learning Models](learning/communication-overlap-decomposition.md) |  | source-note | communication, computation-overlap, intra-layer-parallelism, tp, collective-decomposition, gpu, tpu, google, asplos | 2026-05-04 |
| [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](learning/deepseek-r1.md) |  | source-note | llm, deepseek, reinforcement-learning, grpo, reasoning, r1, chain-of-thought, distillation | 2026-05-02 |
| [Insights into DeepSeek-V3: Scaling Challenges and Hardware Reflections](learning/deepseek-v3-insights.md) |  | source-note | llm, deepseek, deepseek-v3, mla, fp8, moe, hardware-codesign, network, isca | 2026-05-02 |
| [DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models](learning/deepseek-v3.2.md) |  | source-note | llm, deepseek, deepseek-v3.2, sparse-attention, dsa, rl, agent, imo | 2026-05-02 |
| [DeepSeek-V3 Technical Report](learning/deepseek-v3.md) |  | source-note | llm, deepseek, deepseek-v3, mla, moe, fp8, multi-token-prediction, dualpipe | 2026-05-02 |
| [DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence](learning/deepseek-v4.md) |  | source-note | llm, mixture-of-experts, deepseek, long-context, hybrid-attention, csa, hca, mhc, muon, post-training, infrastructure | 2026-05-03 |
| [Efficient Training of Large Language Models on Distributed Infrastructures: A Survey](learning/efficient-llm-training-survey.md) |  | source-note | llm-training, survey, distributed-systems, parallelism, infrastructure, fault-tolerance | 2026-05-03 |
| [FlashAttention: Memory-Efficient Exact Attention](learning/flashattention.md) |  | concept | attention, flash-attention, gpu, kernel-fusion, tiling, hbm, online-softmax | 2026-05-02 |
| [Folding Tensor and Sequence Parallelism for Memory-Efficient Transformer Training & Inference](learning/tsp-folding-parallelism.md) |  | source-note | tensor-parallelism, sequence-parallelism, training, inference, memory-efficiency, long-context, parallelism, attention, mlp | 2026-05-04 |
| [The Landscape of GPU-Centric Communication](learning/gpu-communication-landscape.md) |  | source-note | gpu, communication, nccl, nvshmem, gpudirect, collective, survey | 2026-05-03 |
| [GRPO: Group Relative Policy Optimization](learning/grpo.md) |  | source-note | reinforcement-learning, grpo, ppo, deepseek, policy-gradient, llm | 2026-05-04 |
| [GSPMD: General and Scalable Parallelization for ML Computation Graphs](learning/gspmd.md) |  | source-note | compiler, parallelization, spmd, gspmd, google, tpu, device-mesh, sharding | 2026-05-02 |
| [GSPO: Group Sequence Policy Optimization](learning/gspo.md) |  | source-note | reinforcement-learning, gspo, grpo, sequence-level, policy-gradient, llm, qwen | 2026-05-04 |
| [How To Scale Your Model](learning/scaling-book.md) |  | source-note | scaling, tpu, jax, roofline, training, inference, sharding, parallelism, google-deepmind, book, transformers | 2026-05-04 |
| [Inside vLLM: Anatomy of a High-Throughput LLM Inference System](learning/vllm-anatomy.md) |  | source-note | inference, vllm, paged-attention, continuous-batching, kv-cache, speculative-decoding, prefix-caching, guided-decoding, cuda-graphs, tp, pp, distributed | 2026-05-04 |
| [LLM Architecture Comparison and Attention Variants](learning/llm-architecture-comparison.md) |  | source-note | llm, architecture, attention, mha, gqa, mla, swa, dsa, moe, comparison | 2026-05-03 |
| [LLM Scaling Laws: From GPT-3 to the Plateau](learning/llm-scaling-laws.md) |  | source-note | scaling-laws, llm, pretraining, kaplan, power-law | 2026-05-02 |
| [Megatron-Core MoE: Scalable Training of Mixture-of-Experts Models](learning/megatron-core-moe.md) |  | source-note | llm-training, mixture-of-experts, distributed-systems, nvidia, parallelism, megatron, memory, communication, compute, fp8 | 2026-05-03 |
| [Multi-Head Latent Attention (MLA)](learning/multi-head-latent-attention.md) |  | concept | llm, attention, deepseek, kv-cache, memory-optimization, inference, mla | 2026-05-02 |
| [Muon is Scalable for LLM Training](learning/muon-optimizer.md) |  | source-note | optimizer, muon, adamw, training, scaling-laws, moe, moonlight, deepseek, orthogonalization | 2026-05-04 |
| [Demystifying NCCL: GPU Communication Protocols and Algorithms](learning/nccl-demystifying.md) |  | source-note | nccl, gpu, communication, collectives, hpc, nvidia, distributed-training | 2026-05-02 |
| [GPU-Initiated Networking for NCCL: Device API and GIN](learning/nccl-device-api-gin.md) |  | source-note | nccl, gpu, networking, rdma, device-api, gin, deep-ep, moe, nvidia | 2026-05-02 |
| [NCCL EP: Unified Expert Parallel Communication API](learning/nccl-ep.md) |  | source-note | nccl, expert-parallelism, moe, gpu, deep-ep, hybrid-ep, vllm, dispatch, combine | 2026-05-02 |
| [NVIDIA GPU Architecture: Ampere, Hopper, and Blackwell Microbenchmark Analysis](learning/nvidia-gpu-architecture-microbenchmarks.md) |  | source-note | gpu, nvidia, ampere, hopper, blackwell, microbenchmarking, architecture, tensor-cores | 2026-05-03 |
| [PartIR: Composing SPMD Partitioning Strategies for Machine Learning](learning/partir.md) |  | source-note | compiler, spmd, partitioning, mlir, part IR, google, sharding, tactics | 2026-05-02 |
| [PPO: Proximal Policy Optimization Algorithms](learning/ppo.md) |  | source-note | reinforcement-learning, ppo, policy-gradient, grpo, llm, rl | 2026-05-04 |
| [Mathematical Foundations of Reinforcement Learning](learning/rl-foundations.md) |  | source-note | reinforcement-learning, textbook, bellman, mdp, policy-gradient, td-learning, sgd | 2026-05-04 |
| [PyTorch Compilation: torch.compile Guide and State](learning/pytorch-compilation.md) |  | source-note | pytorch, torch-compile, inductor, dynamo, compilation, performance, triton | 2026-05-04 |
| [Rail-only: A Low-Cost High-Performance Network for Training LLMs with Trillion Parameters](learning/rail-only-network.md) |  | source-note | network, datacenter, llm-training, rail-only, spine-leaf, communication, scaling, gpu-cluster, moe, all-to-all | 2026-05-04 |
| [Ring Attention Explained](learning/ring-attention.md) |  | source-note | attention, ring-attention, long-context, distributed, sequence-parallelism, online-softmax, safe-softmax, gpu, scaling | 2026-05-04 |
| [ScaleRL: The Art of Scaling Reinforcement Learning Compute for LLMs](learning/scalerl.md) |  | source-note | reinforcement-learning, scaling-laws, grpo, llm, rl, scalerl, cispo, pipeline-rl | 2026-05-02 |
| [Scaling Laws for Autoregressive Generative Modeling](learning/cross-domain-scaling-laws.md) |  | source-note | scaling-laws, autoregressive, cross-domain, openai, power-law, kl-divergence, compute-optimal, multimodality | 2026-05-04 |
| [Stabilizing Reinforcement Learning with LLMs: Formulation and Practices](learning/stabilizing-rl-llms.md) |  | source-note | reinforcement-learning, llm, policy-gradient, stability, importance-sampling, routing-replay, moe, qwen | 2026-05-04 |
| [Tensor Core Evolution: From Volta to Blackwell](learning/tensor-core-evolution.md) |  | source-note | gpu, nvidia, tensor-cores, volta, ampere, hopper, blackwell, ptx, sass, mma, wgmma, tcgen05 | 2026-05-03 |
| [TOAST: Fast and Scalable Auto-Partitioning via Static Analysis and MCTS](learning/toast.md) |  | source-note | compiler, auto-partitioning, mcts, static-analysis, sharding, google, toast | 2026-05-02 |
| [TRPO: Trust Region Policy Optimization](learning/trpo.md) |  | source-note | reinforcement-learning, trpo, policy-gradient, trust-region, ppo, foundation | 2026-05-04 |

## Reading (3 pages)
| Page | Summary | Type | Tags | Updated |
|------|---------|------|------|---------|
| [Don't Teach. Incentivize.](reading/dont-teach-incentivize.md) |  | source-note | research-philosophy, rl, scaling, incentives, deepseek, bitter-lesson | 2026-05-04 |
| [The Bitter Lesson](reading/the-bitter-lesson.md) |  | source-note | ai, scaling, search, learning, philosophy, rich-sutton, bitter-lesson | 2026-05-03 |
| [You and Your Research](reading/you-and-your-research.md) |  | source-note | research, philosophy, productivity, hamming, methodology | 2026-05-04 |
