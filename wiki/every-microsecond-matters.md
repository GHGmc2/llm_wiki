---
title: "Every Microsecond Matters: Near Speed-of-Light Latency in GPU Collectives"
type: source-note
tags: [gpu, communication, nccl, collective, allreduce, latency, nvlink, inference, llm, hpc, nvidia, gb200, symmetric-memory, barrier-free]
created: 2026-07-09
updated: 2026-07-09
sources: [https://arxiv.org/abs/2607.16100]
status: stable
---

# Every Microsecond Matters: Near Speed-of-Light Latency in GPU Collectives

**Source**: Siyuan Shen, Anton Korzh, John Bachan, Tiancheng Chen, Arnav Goel, Ludwig Schneider, Pouya Kousha, Zhenhao He, Sylvain Jeaugey, Kamil Iskra, Nishank Chandawala, Jeff R. Hammond, Torsten Hoefler (ETH Zürich, NVIDIA, Singapore-ETH Centre), Jul 2026. arXiv:2607.16100. 14 pages. [src](https://arxiv.org/abs/2607.16100)

## Key Points

- **Latency, not bandwidth, is the bottleneck** for LLM decode-heavy inference — small AllReduce collectives lie on the critical path
- Achieves within **7% of the hardware Speed-of-Light (SoL) bound** for AllReduce latency (~1.4 µs on GB200)
- Eliminates global memory barriers via **LL, sentinel synchronization, and double buffering** — barriers alone consume ~40% of small-AllReduce latency
- Introduces a novel **two-shot LL128 atomic AllReduce** algorithm using cache-line atomic additions with embedded flags
- Builds experimental **low-latency APIs** (`ncclLLBuffer`, `send`, `recv`, `bcast`, `recvReduce`) on top of NCCL's device-side APIs
- **Real-world gains**: 7-13% ITL reduction for vLLM inference on Llama/DeepSeek/Qwen; >$11/M tokens savings for DeepSeek-V3; cuSOLVERMp speedups on Alps HPC cluster
- At **trillion-token scale**, each microsecond saved in AllReduce latency compounds to substantial cost reduction (~0.9% per µs for Llama-3.1-70B on GB200)

## Background: Why Latency Matters

### The Problem

GPU collective communication is traditionally optimized for bandwidth. But **long-context, decode-heavy LLM inference** inverts this:
- KV-cache memory grows with sequence length → batch sizes shrink to fit device memory
- Collectives (AllReduce) are invoked frequently with **small message sizes** during decoding
- Latency, not bandwidth, becomes the bottleneck

### The Stakes

In tensor-parallel (TP) LLM inference, many small AllReduce operations lie directly on the critical path of token generation. On 4×GB200, NCCL ring AllReduce takes ~11 µs for small messages. Our kernels reduce this to ~2.37 µs, yielding:

- **8.7% ITL reduction** for Llama-3.1-70B
- ~0.9% cost savings per microsecond saved
- **>$11 per 1M output tokens** saved for DeepSeek-V3 at 8 GPUs

## Toward Near-Speed-of-Light AllReduce

### Speed-of-Light (SoL) Bound

The absolute hardware lower bound for AllReduce latency, ignoring computation overhead:

$$L_{\text{SoL}} = L_{\text{L2\_RTT}} + L_{\text{remote\_store}} + L_{\text{L2\_RTT}}$$

On GB200: $L_{\text{L2\_RTT}} \approx 0.306$ µs, $L_{\text{remote\_store}} \approx 0.792$ µs → **SoL ≈ 1.404 µs**

The SoL is **independent of the number of ranks** — sending to N peers incurs the same latency as sending to 1 under ideal conditions.

### The Cost of Memory Barriers

Global memory barriers for synchronization consume >1 µs each. In a 2-barrier AllReduce taking ~5 µs on 4 GPUs, **barriers alone account for ~40% of latency**. Eliminating them is the primary optimization target.

## Barrier-Free Techniques

| Technique | Mechanism | Trade-offs |
|-----------|-----------|------------|
| **LL** (Low Latency) | Packs 8B flag + 8B data into 16B atomic store; receiver checks flag | Halves payload bandwidth, 2× scratch buffer |
| **Sentinel** | Initializes scratch buffer with sentinel (-NaN); receiver polls until value changes | Preserves full bandwidth, smaller scratch; requires buffer reset + value exclusion |
| **Double Buffering** | Bidirectional exchange with alternating buffers; receive from peer acts as implicit permission for next send | Eliminates barriers between iterations without global coordination |

### LL128 Atomic AllReduce (New)

A two-shot algorithm using **cache-line-level atomic additions** on NVLink:

**ReduceScatter phase**: Threads operate in groups of 8 on 128B (one cache line). Thread 0 relocates its first element, sets it to 1 as a flag, then all 8 threads perform `atomicAdd` to the target rank's scratch buffer.

**AllGather phase**: When the flag element equals N (all ranks contributed), data is confirmed ready. Thread 0 restores the displaced element, all threads write to output.

Properties:
- Scratch space: only $D/N$ (vs $N \cdot D$ for one-shot)
- Bandwidth waste: ~3% for FP32, ~1.5% for FP16
- **NVLink required** (cache-line atomic guarantee); **commutative-only** (addition only); **non-deterministic** (float atomic ordering)
- Limited to FP32/FP16/BF16 (vectorized atomics in CUDA)

## Low-Latency API Design

Built on NCCL's device-side APIs, the experimental API provides thread-level primitives:

| Primitive | Function |
|-----------|----------|
| `ncclLLBuffer` | Wraps symmetric memory; template-selectable LL/sentinel mode; multi-buffering via epochs |
| `send` | Writes to peer buffer slot; LL mode packs flag with data |
| `recv` | Polls buffer slot until valid data (sentinel mode: ≠ sentinel; LL mode: flag == epoch) |
| `recvUnrolled` | Batch receive with compile-time unrolling; supports conditional element ranges |
| `recvReduce` | Receive + reduce from multiple peers with user-defined operator |
| `bcast` | Multicast to all peers (hardware multicast via NVLS when available) |
| `reset` | Clear buffer slot (sentinel restore or zero) |

**Design choice**: Thread-level only (no warp/block APIs) — maximizes user control for lowest possible latency. APIs favor bidirectional patterns to avoid barriers.

![NCCL Low-Latency APIs Overview](raw/assets/emm-low-latency-apis.png)

## Microbenchmark Results

Evaluated on **GB200 NVL72** (4 GPUs/node, 72 GPUs in NVLink domain, 130 TB/s aggregate):

| Implementation | 2 GPUs (128B) | 4 GPUs (128B) |
|---------------|--------------|--------------|
| **Ours (LLBuffer one-shot)** | ~2.37 µs | ~2.5 µs |
| NCCL AGxLL (sym one-shot) | ~3.0 µs | ~3.2 µs |
| NVSHMEM | ~3.5 µs | ~4.0 µs |
| MSCCL++ | ~3.0 µs | — |
| vLLM custom AR | — | ~3.5 µs |
| **SoL bound** | **1.404 µs** | **1.404 µs** |

Key observations:
1. Our LLBuffer one-shot achieves **lowest latency for small messages** across all GPU counts — only ~7% overhead over SoL
2. **LL128 atomic** outperforms standard two-shot at larger scales due to L2 atomic scalability
3. **Multicast** (NVLS `multimem.ld_reduce`) is critical for scalability — offloads reduction to the fabric, especially beneficial at 64 GPUs
4. LL vs sentinel crossover: LL better at very small sizes; sentinel wins as message size and rank count increase

![Microbenchmark latency comparison](raw/assets/emm-poster_child.png)

![One-shot AllReduce API example](raw/assets/emm-one-shot-ar-api-example.png)

## Case Studies

### vLLM LLM Inference (GB200 NVL72)

Tested on Llama-3.1-70B, Llama-4, DeepSeek-V3, Qwen3-Next with 100-200K context, 16K output, batch=8:

| Configuration | ITL Reduction | Throughput Gain |
|--------------|--------------|-----------------|
| TP=4 (best) | 7-13% | 6-12% |
| TP=8 (best) | 9-11% | 8-11% |

Best config = NCCL LL (one-shot) + Sym Mem (enables two-shot + LL128 atomic for larger messages).

Cost estimate: **>$11 per 1M output tokens** saved for DeepSeek-V3 at TP=8.

### cuSOLVERMp (Alps HPC Cluster, GH200)

Single-node distributed eigensolver on 4×GH200:
- Consistent speedups for matrix sizes $m = 32768$ and $m = 65536$
- More pronounced for $m = 32768$ where communication is a larger fraction of runtime
- Only one-shot kernels used (cuSOLVERMp doesn't register symmetric memory)

## Algorithm Comparison

| Algorithm | Sync Rounds | Bandwidth Waste | Scratch Space | Scale-up | Deterministic |
|-----------|------------|-----------------|---------------|----------|---------------|
| One-shot LL | O(1) | 50% | $N \cdot D$ | Poor | Yes |
| One-shot Sentinel | O(1) | 0% | $N \cdot D$ | Poor | Yes |
| Two-shot LL | O(1) | 50% | $D$ | Good | Yes |
| Two-shot Sentinel | O(1) | 0% | $D$ | Good | Yes |
| **LL128 Atomic** | **O(1)** | **~3%** | **$D/N$** | **Best** | **No** |

![Two-shot LL128 atomic algorithm overview](raw/assets/emm-new-algo.png)

## Connections

- [Demystifying NCCL](nccl-demystifying.md) — LL protocol, ring/tree algorithms, NCCL internals
- [NCCL Device API / GIN](nccl-device-api-gin.md) — GPU-initiated networking, the foundation for these device-side APIs
- [NVSHMEM](nvshmem.md) — PGAS symmetric memory, device-initiated communication; this work complements it at the thread level
- [GPU Communication Landscape](gpu-communication-landscape.md) — Full stack context: NCCL, NVSHMEM, UCX, MPI
- [GPU Hardware Architecture](aspe-gpu-hardware-architecture.md) — GB200/NVL72/NVLink domain details
- [Inside vLLM](vllm-anatomy.md) — Inference engine where these kernels are deployed
- [DeepSeek-V3](deepseek-v3.md) — Primary inference target; requires ≥8 GPUs for serving
- [LLM Architecture Comparison](llm-architecture-comparison.md) — Models evaluated: Llama, DeepSeek MoE, Qwen3-Next (hybrid attention)
- [Tensor Core Evolution](tensor-core-evolution.md) — NVLink and NVLS hardware context
- [Rail-only Network](rail-only-network.md) — Scale-out complement; this work focuses on scale-up (NVLink domain)

## Limitations

- **Scale-up only** (NVLink domain) — scale-out collectives are hierarchical and complementary
- LL128 atomic requires **NVLink** (cache-line atomic) and supports only **addition** (commutativity)
- LL128 atomic is **non-deterministic** (floating-point atomic ordering)
- Kernel selection currently driven by empirical measurements, not a performance model
- APIs are **thread-level only** — no warp/block abstractions yet
