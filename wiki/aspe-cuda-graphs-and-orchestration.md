---
title: "CUDA Graphs, Streams, and GPU Orchestration"
type: concept
tags: [cuda, gpu, cuda-graphs, streams, atomics, dynamic-scheduling, nvshmem, nccl]
created: 2026-05-02
updated: 2026-07-09
sources: [raw/ai-systems-performance-engineering.pdf]
status: stable
---

# CUDA Graphs, Streams, and GPU Orchestration

**Source**: AI Systems Performance Engineering, Chapters 11-12 [src](raw/ai-systems-performance-engineering.pdf)

## Key Points

- **CUDA Graphs**: Capture a sequence of operations once, replay with one CPU call — 20-30% latency reduction
- **CUDA Streams**: Enable concurrent execution — overlap compute, memory transfers, and communication
- **Atomic work queues**: Dynamic scheduling via L2-cache atomics — balance irregular workloads without CPU
- **Dynamic parallelism**: GPU kernels launch child kernels — eliminate CPU from critical path entirely
- **Multi-GPU overlap**: Pipe NCCL collectives, peer transfers, and compute through separate streams
- **Roofline-guided orchestration**: Choose the right strategy based on whether you're memory-bound or compute-bound

## CUDA Streams: Concurrent Execution

Streams are sequences of operations that execute in order within a stream, but may execute concurrently across streams:

```cpp
cudaStream_t stream0, stream1;
cudaStreamCreate(&stream0);
cudaStreamCreate(&stream1);

// These can overlap:
kernel_A<<<grid, block, 0, stream0>>>();  // compute on stream0
cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream1);  // transfer on stream1

// Stream synchronization:
cudaEvent_t event;
cudaEventRecord(event, stream0);
cudaStreamWaitEvent(stream1, event, 0);  // stream1 waits for stream0 to reach event
```

### Default vs Non-Default Streams

| Stream Type | Behavior |
|------------|----------|
| Legacy default (stream 0) | Implicit synchronization — operations in other streams wait for stream 0 |
| Per-thread default | No implicit sync — each host thread has its own default stream |
| Explicit (non-default) | Full control — use for fine-grained overlapping |

**Best practice**: Use per-thread default streams or explicit streams. Avoid legacy default stream for performance-critical code.

### Stream-Ordered Memory Allocator

`cudaMallocAsync` / `cudaFreeAsync` — memory allocations ordered within a stream, avoiding global device-wide synchronization:

```cpp
float *d_A, *d_B;
cudaMallocAsync(&d_A, N * sizeof(float), stream);
cudaMallocAsync(&d_B, N * sizeof(float), stream);
// ... use buffers ...
cudaFreeAsync(d_A, stream);  // freed when stream reaches this point
```

Enables reuse of memory without explicit synchronization — critical for pipelined multi-GPU workloads.

## CUDA Graphs: Eliminating CPU Launch Overhead

CUDA Graphs capture a sequence of GPU operations (kernels, memcpys, events) as a single replayable graph:

```cpp
cudaGraph_t graph;
cudaGraphExec_t instance;

// 1. Begin capture
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

// 2. Record operations (all captured into graph)
kernel1<<<grid1, block1, 0, stream>>>();
kernel2<<<grid2, block2, 0, stream>>>();
cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream);

// 3. End capture
cudaStreamEndCapture(stream, &graph);

// 4. Instantiate (optimize for repeated launch)
cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);

// 5. Replay (single CPU call)
for (int i = 0; i < iterations; i++) {
    cudaGraphLaunch(instance, stream);
}
```

**Performance**: Reduces per-iteration CPU scheduling overhead by 20-30%, more at larger scale. Critical for:
- MoE inference where many small kernels are launched per step
- Training loops with fixed computation patterns
- Any workload where CPU launch overhead dominates

### Conditional Graph Nodes

Newer GPUs support conditional execution within graphs:

```cpp
cudaGraphConditionalHandle condHandle;
cudaGraphConditionalHandleCreate(&condHandle, graph);

// Set condition (runs on GPU, no CPU round-trip)
__global__ void setHandle(cudaGraphConditionalHandle handle, int* data, int N) {
    unsigned int flag = (reduce_sum(data, N) > threshold) ? 1u : 0u;
    cudaGraphSetConditional(handle, flag);
}

// IF node: body only executes when handle != 0
cudaGraphAddEmptyNode(&ifNode, graph, &setNode, 1);
```

### Multi-GPU Graphs with NCCL

CUDA Graphs can capture NCCL collective operations:

```cpp
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

// NCCL all-reduce captured into graph
ncclAllReduce(sendbuff, recvbuff, count, ncclFloat, ncclSum, comm, stream);

cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
// Replay entire multi-GPU collective pattern with one launch
```

## Dynamic Scheduling with Atomic Work Queues

For irregular workloads (MoE, sparse attention), static work distribution causes load imbalance. Atomic queues enable dynamic, on-GPU scheduling:

```cpp
// Atomic work queue
__device__ int queue_head = 0;

__global__ void dynamic_worker(float* data, int N) {
    while (true) {
        // Batch-grab work to amortize atomic cost
        int start = atomicAdd(&queue_head, BATCH_SIZE);
        if (start >= N) break;
        
        int end = min(start + BATCH_SIZE, N);
        for (int i = start; i < end; i++) {
            process(data[i]);
        }
    }
}
```

**Batching is critical**: One atomic per warp (batch_size=32) instead of per thread → 32$\times$ reduction in atomic contention. L2-cache atomics on modern GPUs are exceptionally fast — even batch_size=8 can eliminate most contention.

**Nsight Compute diagnosis**: Check `atomic_transactions_per_request` — should be ~1.0. Higher means contention.

## Dynamic Parallelism

GPU kernels can launch child kernels — no CPU involvement:

```cpp
__global__ void parent_kernel() {
    // Compute on GPU, then launch child dynamically
    dim3 child_grid(256);
    dim3 child_block(128);
    child_kernel<<<child_grid, child_block>>>();
    cudaDeviceSynchronize();  // wait for child to complete
}
```

**Use case**: When the amount of work depends on GPU-computed values — CPU round-trip would stall the pipeline.

## Multi-GPU Orchestration

Overlap compute, peer transfers, and NCCL collectives across streams:

```cpp
// GPU 0
stream0: compute layers 0-3
stream1: cudaMemcpyPeerAsync (send activations to GPU 1)
stream2: ncclAllReduce (gradients)

// GPU 1  
stream0: compute layers 4-7 (activated by event from GPU 0's stream1)
stream1: cudaMemcpyPeerAsync (send gradients to GPU 0)
```

**NVSHMEM**: Fine-grained GPU-to-GPU memory sharing without explicit transfers — one-sided put/get from GPU kernels.

## Roofline-Guided Orchestration

| Bottleneck | Strategy |
|-----------|----------|
| Memory-bound | Overlap, async memcpy, FP8/FP4, kernel fusion |
| Compute-bound (underachieving) | CUDA Graphs, persistent kernels, fusion |
| In between | Run multiple streams in parallel |

## Connections

- [GPU Memory Hierarchy](aspe-gpu-memory-hierarchy.md) — Foundation for memory optimization
- [CUDA Kernel Optimization](aspe-cuda-kernel-optimization.md) — Single-kernel techniques
- [Thread Block Clusters](aspe-thread-block-clusters.md) — Persistent kernels, warp specialization, DSMEM
- [PyTorch Profiling & Tuning](aspe-pytorch-profiling-tuning.md) — CUDA Graphs via torch.compile
- [NCCL Demystifying](nccl-demystifying.md) — NCCL internals
- [NCCL Device API / GIN](nccl-device-api-gin.md) — GPU-initiated networking (device-side orchestration)
- [AI Systems Performance Engineering](aspe-overview.md) — Full book reference
- [PyTorch Compilation](pytorch-compilation.md) — torch.compile captures CUDA Graphs automatically; 2.27× inference speedup via compilation
