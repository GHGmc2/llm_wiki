---
title: "OS, Docker, and Kubernetes Tuning for GPU Workloads"
type: concept
tags: [linux, docker, kubernetes, gpu, numa, hugepages, mig, cgroups, performance]
created: 2026-05-02
updated: 2026-05-02
sources: [raw/ai-systems-performance-engineering.pdf]
status: stable
---

# OS, Docker, and Kubernetes Tuning for GPU Workloads

**Source**: AI Systems Performance Engineering, Chapter 3 [src](raw/ai-systems-performance-engineering.pdf)

## Key Points

- **NUMA affinity**: Bind GPU processes to local CPU cores and memory — avoids cross-socket penalties
- **Transparent Hugepages (THP)**: Enable for training (throughput), disable/madvise for inference (latency)
- **MIG (Multi-Instance GPU)**: Slice one GPU into isolated instances — share without interference
- **Kubernetes Topology Manager**: Aligns GPU, CPU, and NIC NUMA placement
- **cgroup isolation**: Prevent cross-workload interference in production

## NUMA and CPU Affinity

Modern GPU servers have multiple NUMA domains. GPUs 0-3 attached to CPU 0, GPUs 4-7 to CPU 1. Cross-NUMA memory access incurs higher latency and lower bandwidth.

**Binding strategy**:
```bash
# Bind ranks 0-3 to CPU 0's cores, ranks 4-7 to CPU 1's cores
numactl --cpunodebind=0 --membind=0 python train.py  # ranks 0-3
numactl --cpunodebind=1 --membind=1 python train.py  # ranks 4-7

# Let NCCL fine-tune within those cores
export NCCL_IGNORE_CPU_AFFINITY=1
```

**PyTorch**: Launch utilities handle much of this automatically — verify with `numactl --show`.

## Transparent Hugepages (THP)

THP reduces TLB misses by using 2 MB pages instead of 4 KB:

| Workload | THP Setting | Reason |
|----------|------------|--------|
| **Training** | `always` | Throughput matters, large allocations benefit from hugepages |
| **Inference** | `never` or `madvise` | Latency matters, THP compaction can cause jitter |
| **Multi-rank training** | `never` or `madvise` | Many ranks allocating simultaneously → THP compaction contention |

```bash
# Check current setting
cat /sys/kernel/mm/transparent_hugepage/enabled

# Disable
echo never > /sys/kernel/mm/transparent_hugepage/enabled
```

## CPU Isolation

For latency-sensitive threads (data pipeline, inference servers):

- **`isolcpus`**: Reserve cores exclusively — OS scheduler won't place other tasks there
- **`nohz_full`**: Disable timer ticks on isolated cores — reduces jitter
- **cgroup `cpuset`**: Preferred for production — each workload gets its own physical cores and memory regions

```bash
# Kernel boot parameters
isolcpus=8-15 nohz_full=8-15

# cgroup isolation (production)
cset shield --cpu=8-15 --kthreads=on
```

## MIG: Multi-Instance GPU

MIG slices one physical GPU into multiple isolated instances, each with dedicated compute, memory, and bandwidth:

```
A100 80GB → 7 × 10 GB instances (1g.10gb each)
H100 80GB → 7 × 10 GB or combinations of larger slices
```

**Use cases**: Inference serving where multiple tenants share a GPU, or training small models in parallel.

**Limitation**: MIG instances cannot communicate via P2P or NVLink — they're fully isolated.

## Kubernetes Topology Manager

Aligns NUMA placement of GPU, CPU, and NIC:

```yaml
apiVersion: v1
kind: Pod
spec:
  containers:
  - resources:
      limits:
        nvidia.com/gpu: 1
        cpu: 8
        memory: 64Gi
  # Topology manager aligns GPU + CPU + NIC to same NUMA node
```

**Key policies**:
- `none`: No alignment (default)
- `best-effort`: Try to align, don't fail if impossible
- `restricted`: Fail if can't align
- `single-numa-node`: Strictest — all resources on one NUMA node

## Avoiding OOM Killer

GPU workloads allocate large, contiguous memory regions. The OOM killer can terminate processes unexpectedly.

**Mitigations**:
- Set memory limits in cgroups (Kubernetes `resources.limits.memory`)
- Use `cudaMallocAsync` with stream-ordered allocation for predictable memory patterns
- Monitor with `nvidia-smi` and Prometheus exporters
- Reserve headroom: 5-10% of GPU memory for CUDA context, driver overhead

## Common Pitfalls

1. **GPU topology not exposed to container**: Verify `nvidia-smi topo -m` matches host
2. **RDMA in containers**: Container GID assignments must match host — otherwise falls back to CPU-driven copies
3. **NCCL warnings ignored**: "unable to enable P2P, falling back to copy" = severe performance degradation
4. **THP compaction stalls**: Sudden latency spikes during inference — check `/proc/vmstat` for `compact_stall`

## Connections

- [GPU Hardware Architecture](gpu-hardware-architecture.md) — Hardware context for NUMA, NVLink topology
- [Inference Optimization](inference-optimization-techniques.md) — MIG and isolation for serving
- [GPU Hardware Architecture](gpu-hardware-architecture.md) — Hardware context for NUMA, NVLink topology
- [NCCL Demystifying](nccl-demystifying.md) — NCCL environment variables and tuning
