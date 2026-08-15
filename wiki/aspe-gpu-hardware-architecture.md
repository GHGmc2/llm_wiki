---
title: "GPU Hardware Architecture: NVIDIA Roadmap and Rack-Scale Systems"
type: concept
tags: [gpu, hardware, nvidia, h100, b200, gb200, nvl72, nvlink, nvswitch, roadmap]
created: 2026-05-02
updated: 2026-08-15
sources: [raw/ai-systems-performance-engineering.pdf]
status: stable
---

# GPU Hardware Architecture

**Source**: AI Systems Performance Engineering, Chapter 2 [src](../raw/ai-systems-performance-engineering.pdf)

## Key Points

- **NVL72**: 72 GPUs connected via NVLink 5 + NVSwitch at 130 TB/s aggregate bisection bandwidth — essentially "one big GPU"
- NVLink 5: 1.8 TB/s bidirectional per GPU, single-digit microsecond latency
- GPUDirect RDMA: NICs DMA directly to/from GPU memory, bypassing CPU
- Blackwell B200: 180 GB HBM3e at 8 TB/s, 126 MB L2 cache
- NVIDIA roadmap: Blackwell Ultra → Vera Rubin (2026) → Rubin Ultra (2027) → Feynman (2028)

## HBM Memory Evolution

| GPU | Memory | Bandwidth | L2 Cache |
|-----|--------|-----------|----------|
| H100 (Hopper) | 80 GB HBM3 | ~3.35 TB/s | 50 MB |
| B200 (Blackwell) | 180 GB HBM3e | ~8 TB/s | 126 MB |
| B300 (Blackwell Ultra) | 288 GB HBM3e | Higher | Larger |

Only 180 of 192 GB usable on B200 due to ECC, firmware, manufacturing limitations.

## NVLink and NVSwitch

### NVLink 5 (Blackwell)
- 1.8 TB/s bidirectional aggregate bandwidth per GPU (2$\times$ Hopper's NVLink 4)
- 18 NVLink ports per GPU
- Single-digit microsecond inter-GPU latency

### NVSwitch
- Switching chip purpose-built for NVLink
- Full crossbar: every GPU connected to every NVSwitch
- NVL72: 18 NVSwitch chips, 9 switch trays, 18 compute trays
- One-hop: GPU → NVSwitch → GPU at full bisection bandwidth

### NVL72 as "One Big GPU"
- 72 GPUs fully connected, 130 TB/s aggregate bandwidth
- From programming model: PGAS (Partitioned Global Address Space) via NVSHMEM
- CPU-GPU path (NVLink-C2C) is cache coherent; GPU-GPU is not — software handles synchronization
- Vs 72-GPU H100 cluster over InfiniBand: NVLink bandwidth ~20$\times$ higher, latency ~5$\times$ lower

## Interconnects: NVLink vs InfiniBand

| Metric | NVLink 5 (intra-rack) | InfiniBand NDR400 (inter-rack) |
|--------|----------------------|-------------------------------|
| Bandwidth per GPU | 1.8 TB/s | 50-100 GB/s (per port) |
| Latency | 1-2 μs (small msg) | 5-10 μs+ |
| Use case | TP, EP within node | DP, PP across nodes |

**Practical impact**: For a 1.8T MoE model, NVIDIA reported collective overhead drops from tens of percent to just a few percent when moving from IB to NVL72.

## RDMA: Remote Direct Memory Access

RDMA enables NICs to read/write GPU memory directly:
- **GPUDirect RDMA**: NVIDIA's implementation — NIC registers GPU memory, performs DMA directly
- Bypasses CPU and system RAM entirely
- RDMA on InfiniBand: few μs latency, hundreds of Gbps throughput
- RDMA on TCP/IP: 5-10$\times$ higher latency, limited to ~100 Gbps without RoCE

**Verification checklist**:
```bash
lsmod | grep nvidia_peermem  # must be loaded
NCCL_DEBUG=INFO              # check NET/IB paths
```

## NVIDIA Roadmap

| Year | Architecture | Key Feature |
|------|-------------|-------------|
| 2024 | Blackwell (B200) | 180 GB HBM3e, dual-die, NVLink 5 |
| 2025 | Blackwell Ultra (B300) | 288 GB HBM3e, NVL72 Ultra |
| 2026 | Vera Rubin (VR200) | Next-gen architecture |
| 2027 | Rubin Ultra | Ultra-scale |
| 2028 | Feynman | "Doubling something every year" |

**NVL72 trend**: Each generation packs more compute and memory into a single rack — NVIDIA calls them "AI supercomputers in a rack."

- [OS/Docker/K8s Tuning](aspe-os-docker-k8s-tuning.md) — NUMA, MIG, cgroups for GPU workloads
## Connections

- [GPU Memory Hierarchy](aspe-gpu-memory-hierarchy.md) — On-chip memory details (SMEM, TMEM, registers)
- [CUDA Graphs & Orchestration](aspe-cuda-graphs-and-orchestration.md) — NVSHMEM programming model
- [Communication Wall](megatron-core-moe.md) — Why NVLink bandwidth matters for MoE
- [AI Systems Performance Engineering](aspe-overview.md) — Full book reference
- [Distributed Networking Tuning](aspe-distributed-networking-tuning.md) — NCCL, GPUDirect, SHARP
- [Megatron-Core MoE](megatron-core-moe.md) — GB200/GB300 performance numbers
- [Efficient LLM Training Survey](efficient-llm-training-survey.md) — Hardware references for 380+ training systems surveyed
