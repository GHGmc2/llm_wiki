---
title: "GPU-Initiated Networking for NCCL: Device API and GIN"
type: source-note
tags: [nccl, gpu, networking, rdma, device-api, gin, deep-ep, moe, nvidia]
created: 2026-05-02
updated: 2026-05-02
sources: [raw/gpu-initiated-networking-nccl.pdf]
status: stable
---

# GPU-Initiated Networking for NCCL (GIN)

**Source**: NVIDIA. 13 pages. NCCL 2.28 Device API introduction.

## Key Points

- NCCL 2.28 adds **Device API** with three modes: LSA (NVLink/PCIe), Multimem (NVLink SHARP), **GIN** (Network RDMA)
- GIN enables GPU kernels to initiate network communication directly — **no CPU involvement**
- Two backends: **GDAKI** (GPUDirect Async Kernel-Initiated via DOCA GPUNetIO) and **Proxy** (lock-free GPU-to-CPU queues)
- Critical for MoE workloads: eliminates CPU coordination overhead for fine-grained all-to-all
- Integrated with **DeepEP** — demonstrated practical benefits for MoE dispatch/combine

## The Problem: Host-Initiated Communication

Traditional NCCL is **host-initiated**: the CPU orchestrates every communication operation. This is fine for bulk collectives but breaks down for:

- **MoE all-to-all**: Thousands of small, irregular transfers per step — CPU becomes bottleneck
- **Inference decoding**: Latency-sensitive, fine-grained token dispatch
- **Computation-communication fusion**: Cannot interleave compute kernels with network operations when CPU must mediate

GPU-initiated communication eliminates the CPU from the critical path.

## Device API Architecture

Three operation modes in NCCL 2.28:

| Mode | Interconnect | Use Case |
|------|-------------|----------|
| **LSA** (Load/Store Accessible) | NVLink, PCIe | Intra-node direct memory access |
| **Multimem** | NVLink SHARP | In-network reductions within NVSwitch |
| **GIN** (GPU-Initiated Networking) | InfiniBand, RoCE | Inter-node RDMA from GPU kernels |

All modes operate over **collective symmetric memory** — a unified memory region accessible by all GPUs in the communicator.

### GIN: Three-Layer Architecture

**Layer 1: Host-Side API (NCCL Core)**
- Communicator initialization and GIN resource management
- Collective memory window registration
- Device communicator setup

**Layer 2: Device-Side API**
- `put` / `signal` primitives for remote memory operations
- Callable directly from CUDA kernels
- Flexible completion semantics (poll, signal-on-completion)

**Layer 3: Pluggable Network Backend**
- **GDAKI backend**: Direct GPU-to-NIC via DOCA GPUNetIO — GPU writes doorbell registers on NIC
- **Proxy backend**: Lock-free GPU-to-CPU queues — CPU relays to standard RDMA. Enables GIN on hardware without GDAKI support

## GDAKI Backend: GPUDirect Async Kernel-Initiated

The hardware-direct path:

1. GPU kernel writes communication descriptors to GPU memory
2. GPU rings NIC doorbell via memory-mapped I/O (MMIO)
3. NIC DMA reads descriptors and data directly from GPU memory
4. Completion signaled back to GPU (poll or interrupt)

**Zero CPU involvement** — the GPU fully controls network operations. This is what enables MoE dispatch to be fused into compute kernels.

## Proxy Backend: Software Fallback

For hardware without GDAKI support:

1. GPU kernel writes descriptors to a lock-free queue in GPU memory
2. CPU proxy thread polls the queue
3. CPU issues standard RDMA operations on behalf of GPU
4. Completion written back to GPU-visible queue

Adds CPU latency but provides **universal compatibility** across all RDMA-capable hardware.

## Key Design Elements

- **Flexible completion**: Kernels can poll for completion (busy-wait) or use signal-on-completion for event-driven patterns
- **Minimal overhead**: Compile-time optimizations eliminate unnecessary abstraction layers
- **NCCL ecosystem compatibility**: GIN works within NCCL's existing infrastructure (hierarchical communicators, elasticity, fault tolerance)

## DeepEP Integration

The paper demonstrates GIN's practicality through DeepEP integration:

- DeepEP's low-latency MoE dispatch/combine kernels use GIN for inter-node RDMA
- GIN's `put`/`signal` primitives replace DeepEP's custom RDMA backend
- Results: competitive performance with cleaner API, NCCL ecosystem benefits

This is a key connection: **DeepEP** (used in DeepSeek-V3/R1 inference) and **HybridEP** (used in Megatron-Core on GB200) both benefit from GIN's unified NCCL integration.

## Connection to NCCL EP

GIN is the foundation for [NCCL EP](nccl-ep.md), which builds MoE-specific dispatch/combine primitives entirely on NCCL's Device API. NCCL EP adapts DeepEP and Hybrid-EP kernels to use GIN as their communication backend.

## Connections

- [NCCL Demystifying](nccl-demystifying.md) — NCCL internals: protocols, channels, topologies
- [NCCL EP](nccl-ep.md) — MoE communication library built on GIN
- [Communication Wall](megatron-core-moe.md) — DeepEP/HybridEP in Megatron-Core context
- [DeepSeek-V4 Infrastructure](deepseek-v4.md) — DeepEP used in V4 inference
