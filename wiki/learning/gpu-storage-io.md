---
title: "GPU Storage I/O and Data Pipeline Optimization"
type: concept
tags: [storage, gpu, gpudirect-storage, nvme, data-pipeline, deepseek-3fs, nemo-curator, dali]
created: 2026-05-02
updated: 2026-05-02
sources: [raw/AI Systems Performance Engineering.pdf]
status: stable
---

# GPU Storage I/O and Data Pipeline Optimization

**Source**: AI Systems Performance Engineering, Chapter 5 [src](raw/AI%20Systems%20Performance%20Engineering.pdf)

## Key Points

- **GPUDirect Storage (GDS)**: GPU reads NVMe directly — bypasses CPU and system RAM
- **Data pipeline is often the bottleneck**: A poorly tuned input pipeline can waste 50% of GPU time
- **Preprocess offline**: Never train with raw text — pre-tokenize, pack into binary formats
- **DeepSeek 3FS**: Fire-Flyer File System — distributed filesystem purpose-built for AI workloads
- **Continuous profiling**: Automate nightly runs to catch regressions

## GPUDirect Storage (GDS)

GDS enables the GPU to read directly from NVMe SSDs via DMA:

```
Without GDS:
  NVMe → CPU RAM → GPU
  (CPU copy, CPU overhead)

With GDS:
  NVMe → GPU (via PCIe DMA)
  (zero-copy, no CPU)
```

**When to use**: Large datasets that don't fit in CPU RAM, streaming training data, checkpoint loading/saving.

**Tools**:
- `gdsio`: Benchmark GDS throughput
- `cuda-checkpoint`: Checkpoint GPU state directly to NVMe

## The Data Pipeline as Bottleneck

> "A poorly tuned input pipeline could waste 50% of your GPU time, whereas algorithmic optimizations might give only a few percent."

**The rule**: The highest-performing GPU is useless if the data pipeline cannot supply inputs fast enough.

### Preprocessing Strategy

| Approach | When | Trade-off |
|----------|------|-----------|
| **Offline preprocessing** | Production training | Disk space, but zero runtime cost |
| **Online (CPU)** | Prototyping | No preprocessing time, but GPU stalls |
| **Online (GPU via DALI)** | Throughput-critical | GPU resources used for preprocessing |

**Offline preprocessing workflow**:
1. Tokenize, clean, filter raw text → binary format
2. Pack into `.bin`/`.idx` memory-mappable files
3. Store N shuffled copies to avoid runtime shuffling cost over N epochs
4. Training loop: memory-map → feed directly to GPU

### NVIDIA DALI

GPU-accelerated data loading and augmentation pipeline:
- Image decoding, resizing, augmentation on GPU
- Avoids CPU bottleneck and host-device transfers
- Use when preprocessing is compute-intensive and GPU-friendly

> Alternative: Fuse preprocessing into GPU compute graph using TorchVision/TensorRT custom kernels — may outperform DALI by avoiding separate kernel launches.

## NVIDIA NeMo Curator

Open-source framework for LLM dataset preparation:
- Cleansing, tokenizing, shuffling at terabyte scale
- Distributed across multiple GPUs/nodes
- Synthetic data generation to augment human datasets
- Output: sharded JSONL/Parquet → converted to `.bin/.idx` for training

## DeepSeek 3FS (Fire-Flyer File System)

DeepSeek's custom distributed filesystem for AI workloads:
- Purpose-built for training and inference I/O patterns
- Open-sourced during DeepSeek Open-Source Week (Feb 2025)
- Designed for high-throughput, low-latency access to training data and checkpoints

## Continuous Profiling Workflow

The book prescribes an iterative methodology:

1. **Establish baseline**: Single GPU → multi-GPU single-node → multi-node. Expect near-linear scaling for simple DP.
2. **Profile multi-GPU**: Nsight Systems for system-level, Nsight Compute for kernel-level
3. **Identify bottleneck**: GPU idle? Check CPU timelines, NCCL collectives, data loading
4. **Apply targeted fix**: Could be OS tuning, NCCL config, data pipeline, kernel optimization
5. **Automate**: Nightly profiling runs, performance dashboards — catch regressions early

**Scaling efficiency check**: If 8 GPUs give only 5× throughput (not ~8×), something is bottlenecked.

## Performance Metrics to Track

| Metric | Tool |
|--------|------|
| Training throughput (samples/sec) | Application-level logging |
| GPU utilization (%) | `nvidia-smi`, Nsight |
| NCCL bandwidth | `NCCL_DEBUG=INFO` |
| Data loading time per step | PyTorch Profiler |
| Storage I/O throughput | `gdsio`, `iostat` |
| Memory usage | `nvidia-smi`, PyTorch memory profiler |

## Connections

- [Distributed Networking](distributed-networking-tuning.md) — NCCL, GPUDirect RDMA
- [GPU Hardware Architecture](gpu-hardware-architecture.md) — NVMe, PCIe topology
- [OS/Docker/K8s Tuning](os-docker-k8s-tuning.md) — Filesystem, hugepages for I/O
- [AI Systems Performance Engineering](ai-systems-performance-engineering.md) — Full book reference
