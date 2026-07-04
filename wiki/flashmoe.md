---
title: "FlashMoE: Fast Distributed MoE in a Single Kernel"
type: source-note
tags: [moe, gpu-kernel, communication, expert-parallelism, alltoall, persistent-kernel, rdma, dispatch, combine, h100]
created: 2026-05-04
updated: 2026-05-04
sources: [https://arxiv.org/abs/2506.04667, raw/flashmoe.pdf]
status: stable
---

# FlashMoE: Fast Distributed MoE in a Single Kernel

**Source**: Aimuyo, Oh, Singh (Cornell), NeurIPS 2025. arXiv:2506.04667. A fully GPU-resident MoE operator that fuses expert computation and inter-GPU communication into a single persistent kernel, eliminating CPU-managed scheduling and kernel launch overheads.

## Key Points

- **Single persistent GPU kernel** fuses expert dispatch, compute, and combine — no CPU scheduling, no host-initiated communication, no multiple kernel launches
- **One-sided device-initiated (R)DMA**: replaces bulk-synchronous AllToAll collectives with fine-grained, GPU-initiated inter-GPU transfers
- **Fine-grained pipelining**: dispatch, compute, and combine phases interleave at the token level rather than waiting for all tokens
- Payload efficiency: eliminates bloated/redundant network payloads in sparsely activated layers
- On 8×H100 with 128 experts, 16K tokens: **9× GPU utilization, 6× lower latency, 5.7× higher throughput, 4× better overlap** vs baselines — in FP32 vs FP16 baselines
- Code: [github.com/osayamenja/FlashMoE](https://github.com/osayamenja/FlashMoE)

![FlashMoE architecture — single persistent GPU kernel fusing dispatch, compute, and combine with one-sided RDMA](../../raw/assets/flashmoe_architecture.png)

## The Problem: CPU-Bottlenecked MoE

Existing MoE implementations suffer from three sources of overhead:

1. **CPU-managed scheduling**: the CPU decides which tokens go to which experts, adding latency and preventing GPU-side optimization
2. **Host-initiated communication**: NCCL collectives (AllToAll for dispatch/combine) are launched from the CPU, creating gaps between kernels
3. **Frequent kernel launches**: each phase (dispatch, expert compute, combine) is a separate kernel launch, each with ~5-10μs overhead × hundreds of launches

These compound to produce significant GPU idle time — especially for sparse MoE layers where computation per expert is small.

## FlashMoE Design

### Single Persistent Kernel

Instead of multiple kernel launches for dispatch → compute → combine, FlashMoE uses one persistent kernel that loops internally:

```
while (has_work):
    async_recv_tokens()     # one-sided RDMA read from remote GPUs
    compute_expert()         # process the tokens for your local expert shard
    async_send_outputs()     # one-sided RDMA write to destination GPUs
```

This eliminates:
- CPU scheduling latency (no CPU in the loop)
- Kernel launch overheads (one launch, not hundreds)
- Idle gaps between phases (fine-grained interleaving)

### One-Sided Device-Initiated RDMA

Instead of NCCL AllToAll (bulk-synchronous, CPU-launched), FlashMoE uses GPU-initiated (R)DMA:
- **RDMA Read**: pull tokens from remote GPU's memory directly
- **RDMA Write**: push outputs to destination GPU's memory directly
- No CPU involvement in the data path
- Payload efficiency: only send tokens that are actually routed, not all-to-all padding

### Token-Level Pipelining

Conventional MoE: wait for all tokens to be dispatched → compute all experts → combine all outputs.

FlashMoE: dispatch a batch of tokens → immediately start computing → while computing, dispatch the next batch. This overlaps communication and computation at fine granularity, similar to how FlashAttention overlaps HBM loads with compute.

## Performance (8×H100, 128 experts, 16K tokens)

| Metric | Baseline | FlashMoE | Gain |
|--------|----------|----------|------|
| GPU utilization | Baseline | 9× higher | 9× |
| Latency | Baseline | 6× lower | 6× |
| Throughput | Baseline | 5.7× higher | 5.7× |
| Overlap efficiency | Baseline | 4× better | 4× |

Notably, FlashMoE achieves these gains using **FP32** while baselines use FP16 — the efficient overlap and reduced overhead more than compensate for higher-precision compute.

## Connections

- [Megatron-Core MoE](megatron-core-moe.md) — production MoE training stack; FlashMoE's single-kernel approach could replace the EP dispatcher
- [Comet](comet-moe-overlap.md) — fine-grained EP communication overlap via thread block specialization; FlashMoE achieves similar goals with RDMA + persistent kernels
- [NCCL EP](nccl-ep.md) — NCCL's expert parallel collective API; FlashMoE bypasses NCCL entirely with device-initiated RDMA
- [Communication Overlap via Decomposition](communication-overlap-decomposition.md) — Google's TPU collective decomposition; FlashMoE applies similar overlapping philosophy to GPU MoE
- [NCCL Device API / GIN](nccl-device-api-gin.md) — GPU-initiated networking (GIN) enables device-side RDMA; FlashMoE builds on this capability
- [CUDA Graphs & Orchestration](aspe-cuda-graphs-and-orchestration.md) — persistent kernels and GPU-side scheduling connect to CUDA graphs and streams
