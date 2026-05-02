# Index

*Last updated: 2026-05-02*

## Learning (38 pages)
| Page | Summary | Type | Tags | Updated |
|------|---------|------|------|---------|
| [AI Systems Performance Engineering](learning/ai-systems-performance-engineering.md) | Chris Fregly's 1061-page O'Reilly book: full-stack GPU-to-cluster optimization, 175+ item checklist | source-note | performance, cuda, gpu, pytorch, profiling, inference, training, book | 2026-05-02 |
| [Bitter Lesson for RL](learning/bitter-lesson-rl.md) | Sutton's Bitter Lesson applied to LLMs: verification as the key, generative verifiers, test-time compute scaling | source-note | reinforcement-learning, verification, reasoning, bitter-lesson, generative-verifiers | 2026-05-02 |
| [Compute Efficiency Wall](learning/moe-compute-efficiency-wall.md) | Grouped GEMM, CUDA Graphs, kernel fusions, ECHO, sync-free kernels for MoE compute optimization | concept | llm-training, mixture-of-experts, compute-optimization, cuda, megatron | 2026-05-02 |
| [Communication Wall](learning/moe-communication-wall.md) | EP all-to-all communication, DeepEP/HybridEP dispatchers, communication-computation overlap | concept | llm-training, mixture-of-experts, communication, all-to-all, megatron | 2026-05-02 |
| [CUDA Graphs & Orchestration](learning/cuda-graphs-and-orchestration.md) | CUDA streams, graphs, atomics, dynamic scheduling, multi-GPU overlap, NVSHMEM, roofline-guided orchestration | concept | cuda, gpu, cuda-graphs, streams, atomics, nvshmem, nccl | 2026-05-02 |
| [CUDA Kernel Optimization](learning/cuda-kernel-optimization.md) | Occupancy tuning, warp divergence, ILP, kernel fusion, Tensor Core utilization, Nsight roofline analysis | concept | cuda, gpu, kernel-optimization, occupancy, warp-divergence, ilp, tensor-cores | 2026-05-02 |
| [DeepSeek-R1](learning/deepseek-r1.md) | Pure RL reasoning: GRPO, R1-Zero emergent behaviors, multi-stage pipeline, distillation, USAMO qualification | source-note | llm, deepseek, reinforcement-learning, grpo, reasoning, r1, chain-of-thought | 2026-05-02 |
| [DeepSeek-V3](learning/deepseek-v3.md) | MLA + DeepSeekMoE, auxiliary-loss-free load balancing, MTP, FP8 training, DualPipe, 2.788M H800 GPU hours | source-note | llm, deepseek, deepseek-v3, mla, moe, fp8, multi-token-prediction, dualpipe | 2026-05-02 |
| [DeepSeek-V3.2](learning/deepseek-v3.2.md) | DSA sparse attention, scalable RL (>10% pre-training budget), IMO/IOI gold medals, agent synthesis pipeline | source-note | llm, deepseek, deepseek-v3.2, sparse-attention, dsa, rl, agent, imo | 2026-05-02 |
| [DeepSeek-V3 Insights](learning/deepseek-v3-insights.md) | Hardware-aware co-design: MLA, FP8 training, Multi-Plane Network, H800 constraints, hardware wishlist | source-note | llm, deepseek, deepseek-v3, mla, fp8, moe, hardware-codesign, isca | 2026-05-02 |
| [DeepSeek-V4 Architecture](learning/deepseek-v4-architecture.md) | CSA + HCA hybrid attention, mHC residual connections, Muon optimizer, DeepSeekMoE tweaks | concept | llm, mixture-of-experts, deepseek, hybrid-attention, csa, hca, mhc, muon | 2026-05-02 |
| [DeepSeek-V4 Infrastructure](learning/deepseek-v4-infrastructure.md) | EP overlap, TileLang kernel language, FP4 QAT, on-disk KV cache, DSec sandbox | concept | llm, deepseek, infrastructure, fp4, kv-cache, ep-overlap, systems | 2026-05-02 |
| [DeepSeek-V4 Post-Training](learning/deepseek-v4-post-training.md) | On-Policy Distillation, specialist training, three reasoning modes, interleaved thinking | concept | llm, deepseek, post-training, on-policy-distillation, grpo, reasoning | 2026-05-02 |
| [DeepSeek-V4 Technical Report](learning/deepseek-v4-technical-report.md) | DeepSeek-V4 Preview: 1M-token MoE models with hybrid attention and 1.6T parameters | source-note | llm, mixture-of-experts, deepseek, long-context, hybrid-attention, mhc, muon | 2026-05-02 |
| [Demystifying NCCL](learning/nccl-demystifying.md) | NCCL internals: Simple/LL/LL128 protocols, ring/tree algorithms, channels, ATLAHS simulation | source-note | nccl, gpu, communication, collectives, hpc, nvidia | 2026-05-02 |
| [Distributed Networking Tuning](learning/distributed-networking-tuning.md) | Magnum IO, GPUDirect RDMA, SHARP in-network reduction, NCCL tuning, NVSHMEM, NIXL | concept | networking, nccl, gpudirect, rdma, sharp, magnum-io, nvshmem | 2026-05-02 |
| [FP8/FP4 Training](learning/moe-fp8-fp4-training.md) | Reduced-precision training recipes (per-tensor FP8, blockwise FP8, MXFP8, NVFP4) for MoE | concept | llm-training, mixture-of-experts, fp8, fp4, quantization, megatron | 2026-05-02 |
| [GPU Hardware Architecture](learning/gpu-hardware-architecture.md) | NVIDIA roadmap (H100→Feynman 2028), NVL72, NVLink 5, NVSwitch, HBM evolution, GPUDirect RDMA | concept | gpu, hardware, nvidia, h100, b200, gb200, nvl72, nvlink, roadmap | 2026-05-02 |
| [GPU Memory Hierarchy](learning/gpu-memory-hierarchy.md) | CUDA programming model, memory hierarchy (registers→HBM), tiling, TMEM, TMA, arithmetic intensity, roofline | concept | cuda, gpu, memory-hierarchy, shared-memory, registers, tiling, tensor-cores | 2026-05-02 |
| [GPU-Initiated Networking for NCCL](learning/nccl-device-api-gin.md) | NCCL 2.28 Device API: GIN (GPU-initiated RDMA), LSA, Multimem, GDAKI/Proxy backends | source-note | nccl, gpu, networking, rdma, device-api, gin, deep-ep, moe | 2026-05-02 |
| [GSPMD](learning/gspmd.md) | Compiler-based auto-parallelization: mesh_split API, device mesh, automatic sharding propagation | source-note | compiler, parallelization, spmd, gspmd, google, tpu, device-mesh | 2026-05-02 |
| [Inference Optimization](learning/inference-optimization-techniques.md) | Prefill-decode disaggregation, kernel autotuning, dynamic precision switching, goodput, speculative decoding | concept | inference, llm, disaggregation, autotuning, precision, kv-cache | 2026-05-02 |
| [LLM Scaling Laws](learning/llm-scaling-laws.md) | Power laws, Kaplan vs Chinchilla, compute-optimal training, data wall, RL scaling, the plateau | source-note | scaling-laws, llm, pretraining, chinchilla, kaplan, power-law | 2026-05-02 |
| [Megatron-Core MoE: Scalable Training](learning/megatron-core-moe-scalable-training.md) | NVIDIA technical report on training large-scale MoE models | source-note | llm-training, mixture-of-experts, distributed-systems, nvidia, parallelism | 2026-05-02 |
| [Memory Wall](learning/moe-memory-wall.md) | Activation checkpointing, FP8/FP4, offloading, FSDP for MoE memory optimization | concept | llm-training, mixture-of-experts, memory-optimization, megatron | 2026-05-02 |
| [Multi-Head Latent Attention](learning/multi-head-latent-attention.md) | MLA: low-rank KV cache compression achieving 7.28× smaller cache vs LLaMA-405B GQA | concept | llm, attention, deepseek, kv-cache, memory-optimization, inference, mla | 2026-05-02 |
| [NCCL EP](learning/nccl-ep.md) | Unified MoE communication library: ncclEpDispatch/Combine, LL/HT modes, vLLM integration | source-note | nccl, expert-parallelism, moe, gpu, deep-ep, hybrid-ep, vllm | 2026-05-02 |
| [OS/Docker/K8s Tuning](learning/os-docker-k8s-tuning.md) | NUMA affinity, THP, MIG slicing, cgroup isolation, Kubernetes topology manager, OOM avoidance | concept | linux, docker, kubernetes, gpu, numa, hugepages, mig, cgroups | 2026-05-02 |
| [Parallel Folding](learning/moe-parallel-folding.md) | Decoupling attention and MoE parallelism mappings to break the EP=DP constraint | concept | llm-training, mixture-of-experts, parallelism, megatron, distributed-systems | 2026-05-02 |
| [PartIR](learning/partir.md) | Composable SPMD tactics, schedule-like API, MLIR-based, formally verified core transformation | source-note | compiler, spmd, partitioning, mlir, part IR, google, sharding | 2026-05-02 |
| [Performance Best Practices](learning/moe-performance-best-practices.md) | Three-phase optimization workflow and DeepSeek-V3 case study on GB200 vs H100 | concept | llm-training, mixture-of-experts, performance-tuning, megatron, gpu | 2026-05-02 |
| [PyTorch Profiling & Tuning](learning/pytorch-profiling-tuning.md) | PyTorch Profiler, torch.compile, Nsight+NVTX, roofline analysis, Triton, CUTLASS, Linux perf | concept | pytorch, profiling, torch-compile, nsight, triton, cutlass | 2026-05-02 |
| [ScaleRL](learning/scalerl.md) | Systematic RL compute scaling: sigmoid curves, CISPO loss, PipelineRL, 400K GPU-hr study, 100K extrapolation | source-note | reinforcement-learning, scaling-laws, grpo, llm, rl, scalerl, cispo | 2026-05-02 |
| [Scaling Techniques Overview](learning/scaling-techniques-overview.md) | Progressive guide through DP, ZeRO, TP, SP, CP, PP, EP — with formulas, trade-offs, and configurations | concept | llm-training, distributed-training, scaling, data-parallelism, zero, tensor-parallelism | 2026-05-02 |
| [Storage I/O & Data Pipeline](learning/gpu-storage-io.md) | GPUDirect Storage, NVMe tuning, NeMo Curator, DALI, DeepSeek 3FS, continuous profiling | concept | storage, gpu, gpudirect-storage, nvme, data-pipeline, deepseek-3fs | 2026-05-02 |
| [Thread Block Clusters](learning/thread-block-clusters.md) | Thread block clusters, DSMEM, warp specialization, persistent kernels, CUTLASS automation | concept | cuda, gpu, thread-block-clusters, warp-specialization, dsmem, cutlass | 2026-05-02 |
| [TOAST](learning/toast.md) | Auto-partitioning via static analysis + MCTS search, discovers superior strategies automatically | source-note | compiler, auto-partitioning, mcts, static-analysis, sharding, google | 2026-05-02 |
| [Ultra-Scale Playbook](learning/ultra-scale-playbook.md) | HuggingFace/nanotron open-source book on scaling LLM training — 4,100+ experiments on up to 512 GPUs | source-note | llm-training, distributed-training, scaling, parallelism, gpu, hf, nanotron | 2026-05-02 |

## Health (0 pages)
| Page | Summary | Type | Tags | Updated |
|------|---------|------|------|---------|

## Goals (0 pages)
| Page | Summary | Type | Tags | Updated |
|------|---------|------|------|---------|

## Reading (0 pages)
| Page | Summary | Type | Tags | Updated |
|------|---------|------|------|---------|

## Journal (0 pages)
| Page | Summary | Type | Tags | Updated |
|------|---------|------|------|---------|

## People (0 pages)
| Page | Summary | Type | Tags | Updated |
|------|---------|------|------|---------|

## Ideas (0 pages)
| Page | Summary | Type | Tags | Updated |
|------|---------|------|------|---------|
