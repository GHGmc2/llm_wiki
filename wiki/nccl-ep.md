---
title: "NCCL EP: Unified Expert Parallel Communication API"
type: source-note
tags: [nccl, expert-parallelism, moe, gpu, deep-ep, hybrid-ep, vllm, dispatch, combine]
created: 2026-05-02
updated: 2026-05-02
sources: [raw/nccl-ep.pdf]
status: stable
---

# NCCL EP

**Source**: NVIDIA. 13 pages. Ground-up MoE communication library built on NCCL Device API.

## Key Points

- **Unified API**: Single `ncclEpDispatch` / `ncclEpCombine` interface for both training and inference
- **Two algorithm modes**: LL (Low-Latency) for decoding, HT (High-Throughput) for training/prefill
- **Built on GIN**: Uses NCCL's GPU-Initiated Networking for inter-node RDMA, native NVLink for intra-node
- **Adapts proven kernels**: LL mode from DeepEP, HT mode from Hybrid-EP — replaces their custom backends with NCCL Device API
- **vLLM integration**: Demonstrated end-to-end with production inference framework
- Both C and Python interfaces

## Why NCCL EP?

Modern MoE libraries (DeepEP, Hybrid-EP, TRT-LLM EP) all implement custom communication backends. This fragments the ecosystem:
- Each library reinvents RDMA, NVLink management, topology discovery
- Different APIs for different modes (LL vs HT)
- No integration with NCCL's production infrastructure (fault tolerance, elasticity, monitoring)

NCCL EP addresses this by building MoE communication **natively within NCCL**, reusing its Device API, topology awareness, and ecosystem.

## API Design

### Unified Primitives

```c
ncclEpDispatch(comm, tokens, routing_indices, ...)
ncclEpCombine(comm, expert_outputs, routing_indices, ...)
```

Same interface regardless of:
- **Algorithm mode** (LL or HT — selected at group creation)
- **Hardware** (H100, B200, GB200 — topology abstracted)
- **Scale** (single-node to multi-node)

### Two-Tier Resource Management

| Tier | Resource | Lifetime |
|------|----------|----------|
| **Group** (`ncclEpGroup_t`) | Communicators, topology, memory pools | Application lifetime |
| **Handle** (`ncclEpHandle_t`) | Per-operation state, buffers | Single dispatch/combine |

Groups amortize expensive setup (topology discovery, memory registration) across many operations.

## Algorithm Modes

### Low-Latency (LL) Mode

**Target**: Inference decoding (1-128 tokens per GPU)

**Design**: Adapted from DeepEP low-latency kernel
- Full all-to-all mesh connectivity using RDMA + NVLink
- **Double-buffered communication**: Overlap dispatch and combine phases
- Small batch optimization: minimizes latency at the cost of bandwidth efficiency

**Use case**: Autoregressive decoding where each step processes a small number of tokens — latency directly impacts Time Per Output Token (TPOT).

### High-Throughput (HT) Mode

**Target**: Training and inference prefill (4096+ tokens per GPU)

**Design**: Adapted from Hybrid-EP
- **Hierarchical communication**: Aggregate tokens within NVLink domain first, then inter-node RDMA
- Intra-node: NVLink all-to-all within the node
- Inter-node: RDMA between node representatives

**Use case**: Prefill and training where large batches amortize communication setup costs — bandwidth efficiency matters more than per-token latency.

### Mode Selection

Modes are selected at **group creation time** (`ncclEpCreateGroup`). Future versions will auto-detect based on workload characteristics. Applications switch modes by creating different groups — no code changes to dispatch/combine calls.

## Architecture

NCCL EP adapts existing kernel designs, replacing their communication backends:

```
DeepEP LL kernel  ──→ NCCL EP LL mode  (GIN for RDMA, NVLink native)
Hybrid-EP HT kernel ──→ NCCL EP HT mode (GIN for RDMA, NVLink native)
```

**Key insight**: The compute part of MoE kernels (token permutation, quantization) is preserved. Only the communication backend is replaced with NCCL Device API calls — specifically GIN for inter-node and LSA for intra-node.

## MoE Communication Patterns

NCCL EP handles the unique characteristics of MoE communication:

| Property | NCCL EP Handling |
|----------|-----------------|
| Dynamic routing | Per-step routing indices, no static patterns |
| Irregular message sizes | GIN handles variable-size transfers efficiently |
| Load imbalance | HT mode's hierarchical aggregation masks imbalance |
| Fusion requirements | Dispatch: quantize tokens inline. Combine: custom reduce inline |

### Dispatch
1. Gating network selects top-k experts per token
2. Routing indices map tokens → destination GPUs
3. `ncclEpDispatch`: GPU-initiated RDMA transfers tokens to expert GPUs
4. Optional inline FP8 quantization during transfer

### Combine
1. Experts process tokens, produce outputs
2. `ncclEpCombine`: GPU-initiated RDMA returns outputs to source GPUs
3. Weighted summation (top-k > 1) performed inline during combine

## Performance

Evaluated on H100 clusters across multi-node configurations:
- LL mode: Competitive with standalone DeepEP for inference decoding latency
- HT mode: Competitive with standalone Hybrid-EP for training throughput
- vLLM integration: End-to-end results demonstrating production viability

## Relation to DeepEP and Hybrid-EP

NCCL EP is **not a replacement** — it's a **consolidation**:

| Feature | DeepEP | Hybrid-EP | NCCL EP |
|---------|--------|-----------|---------|
| LL kernels | ✓ (custom RDMA) | — | ✓ (via GIN) |
| HT kernels | — | ✓ (custom RDMA) | ✓ (via GIN) |
| NCCL ecosystem | ✗ | ✗ | ✓ |
| Python API | Limited | Via Megatron | ✓ |
| Auto mode selection | — | — | Planned |

The goal: bring the performance of specialized MoE libraries into NCCL's supported, maintained ecosystem.

## Connections

- [NCCL Device API / GIN](nccl-device-api-gin.md) — GIN is NCCL EP's communication backbone
- [NCCL Demystifying](nccl-demystifying.md) — NCCL internals: protocols, channels, topologies
- [Communication Wall](megatron-core-moe.md) — DeepEP/HybridEP in Megatron-Core training context
- [DeepSeek-V3 Insights](deepseek-v3-insights.md) — EP all-to-all analysis and hardware requirements
- [DeepSeek-V4 Infrastructure](deepseek-v4.md) — DeepEP used in V4 inference
