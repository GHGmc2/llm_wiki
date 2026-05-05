---
title: "Inside vLLM: Anatomy of a High-Throughput LLM Inference System"
type: source-note
tags: [inference, vllm, paged-attention, continuous-batching, kv-cache, speculative-decoding, prefix-caching, guided-decoding, cuda-graphs, tp, pp, distributed]
created: 2026-05-04
updated: 2026-05-04
sources: [https://www.aleksagordic.com/blog/vllm]
status: stable
---

# vLLM Anatomy

**Source**: Aleksa Gordić, August 2025. Deep-dive into vLLM V1 engine internals, based on commit 42172ad. The first in a series covering the full inference stack from single-GPU offline to multi-node distributed serving.

## Key Points

- vLLM V1 engine is built around a **scheduler + KV-cache manager** loop: schedule → forward pass → postprocess
- **Paged attention**: KV cache is managed as blocks (default 16 tokens each) in a `free_block_queue` — avoids fragmentation and enables memory sharing
- **Continuous batching**: all sequences are flattened into one "super sequence" with position indices for attention masking — no right-padding needed
- V1 scheduler can mix **prefill and decode** in the same step (V0 couldn't)
- **Chunked prefill**: splits long prompts into chunks to prevent a single request from monopolizing the engine step
- **Prefix caching**: hashes 16-token blocks; reused across requests with shared prefixes — avoids recomputation
- **Speculative decoding**: n-gram/Eagle/Medusa draft models propose tokens; large model verifies via rejection sampling
- **Guided decoding**: FSM-based grammar constraints mask logits via xgrammar bitmask
- **Disaggregated P/D**: separate prefill (compute-bound) and decode (memory-bandwidth-bound) instances with KV-cache transfer connectors
- Scales from `UniProcExecutor` (single GPU) → `MultiProcExecutor` (TP/PP across GPUs) → distributed DP with API server frontend

## Engine Core Architecture

### Constructor Components

- **vLLM Config**: model, cache, parallelism knobs
- **Processor**: raw inputs → `EngineCoreRequests` (validation, tokenization)
- **Engine Core Client**: `InprocClient` (offline) → `DPLBAsyncMPClient` (distributed)
- **Output Processor**: `EngineCoreOutputs` → user-facing `RequestOutput`

Engine Core sub-components:
- **Model Executor**: `UniProcExecutor` (single GPU) or `MultiProcExecutor` (TP/PP)
- **Structured Output Manager**: for guided decoding (FSM)
- **Scheduler**: FCFS or priority policy, `waiting` + `running` queues, KV-cache manager

### Worker Initialization

1. **Init device**: assign CUDA device, verify VRAM, set up distributed settings, instantiate `model_runner` and `InputBatch`
2. **Load model**: instantiate architecture, load weights, `model.eval()`, optional `torch.compile()`
3. **Initialize KV cache**: profiling forward pass → compute available blocks → allocate, reshape, bind KV tensors → capture CUDA graphs

## The Generate Loop: Step by Step

### Request Ingestion

1. Create unique request ID, capture arrival time
2. Tokenize prompt via input preprocessor
3. Pack into `EngineCoreRequest` with priority, sampling params
4. Add to scheduler's `waiting` queue (FCFS append or priority heap-push)

### Engine Step (loop per request)

1. **Schedule**: select decode requests from `running` queue, then prefill from `waiting`. Allocate KV-cache blocks via `allocate_slots`. Preempt if OOM.
2. **Forward pass**: `execute_model` → `Worker` → `model_runner`. Flatten all sequences, custom paged attention kernels, sample tokens.
3. **Postprocess**: append token IDs, detokenize, check stop conditions (max length, EOS, stop tokens, stop strings). Free completed request's KV blocks.

### KV-Cache Block Allocation

Block size = 2 × `block_size`(16) × `num_kv_heads` × `head_size` × `dtype_bytes`

`allocate_slots`:
1. Compute number of new blocks needed (`ceil(new_tokens / 16)`)
2. Check availability in `free_block_queue`; preempt low-priority if needed
3. Allocate blocks from pool, store in `req_to_blocks` map

### Forward Pass Details

Two execution modes:
- **Eager mode**: standard PyTorch forward pass
- **Captured mode**: replay pre-captured CUDA Graphs (latency reduction)

Continuous batching: all sequences flattened into one super-sequence. Position indices + attention masks ensure each sequence only attends to its own tokens — no right-padding.

## Advanced Features

### Chunked Prefill

Cap new tokens per step via `long_prefill_token_threshold`. Long prompts split into chunks to avoid monopolizing the engine step.

### Prefix Caching

1. Split prompt into 16-token chunks, compute hash (built-in or SHA-256) with previous block's hash + current tokens + optional metadata
2. `find_longest_cache_hit`: linear search in `cached_block_hash_to_block`
3. Cache hit → reuse KV blocks directly; cache miss → allocate new blocks, store hash
4. Blocks invalidated only when about to be reallocated from `free_block_queue`

Enabled by default. Disable: `enable_prefix_caching = False`.

### Guided Decoding (FSM)

Grammar-constrained decoding via finite state machines (xgrammar backend):
1. `StructuredOutputManager` compiles grammar asynchronously
2. `_grammar_bitmask` tensor: 32-bit integers encode allowed/disallowed tokens
3. After forward pass, expand bitmask to vocab size, mask disallowed logits to –∞
4. Advance FSM after sampling via `accept_tokens`

### Speculative Decoding

vLLM V1 supports n-gram, Eagle, and Medusa draft methods (no separate small LLM):
- **n-gram**: find matching context in sequence, propose tokens that followed
- **Eagle**: replace transformer stack with lightweight MLP draft
- **Medusa**: auxiliary linear heads predict next k tokens

Process: draft → verify (large model forward pass) → rejection sampling (accept/reject left-to-right). Statistically equivalent to autoregressive decoding, up to k+1 tokens per step.

### Disaggregated P/D

Separate prefill (compute-bound, high TTFT) and decode (memory-bandwidth-bound, latency-sensitive) instances. KV-cache connectors (`SharedStorageConnector`, `LMCache` with NIXL) transfer KV between instances.

## Scaling: UniProcExecutor → MultiProcExecutor

| Executor | GPUs | Parallelism |
|----------|------|-------------|
| `UniProcExecutor` | 1 | None |
| `MultiProcExecutor` | N (same node) | TP (tensor parallel) |
| `MultiProcExecutor` | N×M (multi-node) | TP + PP (pipeline parallel) |

MultiProcExecutor coordination:
1. Spawn daemon worker processes per rank
2. `rpc_broadcast_mq` (shared memory) for work distribution
3. `worker_response_mq` for results
4. Engine calls `execute_model` unchanged — multiprocessing abstracted away

## Connections

- [FlashAttention](flashattention.md) — paged attention builds on tiling/online-softmax techniques
- [Multi-Head Latent Attention](multi-head-latent-attention.md) — MLA reduces KV cache differently (latent compression vs paged allocation)
- [CUDA Graphs & Orchestration](aspe-cuda-graphs-and-orchestration.md) — CUDA graphs used in captured forward pass mode
- [GPU Memory Hierarchy](aspe-gpu-memory-hierarchy.md) — KV cache management depends on HBM capacity and bandwidth
- [Inference Optimization](aspe-inference-optimization-techniques.md) — disaggregated P/D, autotuning, precision switching
- [NCCL EP](nccl-ep.md) — MoE dispatch/combine communication patterns vs vLLM's TP/PP coordination
