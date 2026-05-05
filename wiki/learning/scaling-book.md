---
title: "How To Scale Your Model"
type: source-note
tags: [scaling, tpu, jax, roofline, training, inference, sharding, parallelism, google-deepmind, book, transformers, fsdp, tensor-parallelism, pipeline-parallelism, collective-communication]
created: 2026-05-04
updated: 2026-05-04
sources: [https://jax-ml.github.io/scaling-book/]
status: stable
---

# How To Scale Your Model

**Source**: Austin, Douglas, Frostig, Levskaya, et al. (Google DeepMind), 2025. A short textbook on scaling LLMs on TPUs, covering roofline analysis, hardware architecture, sharding theory, parallelism strategies, and Transformer math. Available at [jax-ml.github.io/scaling-book](https://jax-ml.github.io/scaling-book/).

## Key Points

- **Goal**: strong scaling — increase chip count N× and get N× throughput. Bottlenecked by **compute, communication, and memory**.
- **Roofline analysis**: the unifying framework — every operation is bounded by one of three constraints. Arithmetic intensity = FLOPs / bytes.
- **TPU architecture**: systolic arrays (128×128 MXU, 256×256 on v6e) optimized for matmul (N FLOPs/byte), 2D/3D torus ICI interconnect, HBM → VMEM → MXU pipeline.
- **Sharding theory**: 4 cases for distributed matmul — AllGather, ReduceScatter, AllReduce, AllToAll — each with precise cost formulas based on ICI bandwidth.
- **Transformer math**: $6 \cdot \text{num_params} \cdot \text{num_tokens}$ FLOPs in training, $3DF$ params per MLP layer, $4DNH$ per attention layer.
- **Parallelism strategies**: DP ($B/X > C/W$), FSDP (same comms, less memory), TP ($F > Y \cdot C/W$), PP (bubble overhead), combined FSDP+TP (optimal $X = \sqrt{BF \cdot N}$).
- **Applied**: training and serving LLaMA 3 on TPU v5p/v5e with cost estimates.

## Chapter 1: Roofline Analysis

Every algorithm is bounded by compute (FLOPs/s), communication (inter-chip bandwidth), or memory (HBM bandwidth).

### Three Time Components

$$T_{\text{math}} = \frac{\text{Computation FLOPs}}{\text{Accelerator FLOPs/s}} \qquad T_{\text{comms}} = \frac{\text{Communication Bytes}}{\text{Bandwidth Bytes/s}}$$

Bounds (assuming overlapped communication):
$$T_{\text{lower}} = \max(T_{\text{math}}, T_{\text{comms}}) \qquad T_{\text{upper}} = T_{\text{math}} + T_{\text{comms}}$$

### Arithmetic Intensity

$$\text{Arithmetic Intensity} = \frac{\text{Computation FLOPs}}{\text{Communication Bytes}}$$

When AI > accelerator's critical intensity → compute-bound; otherwise → memory/communication-bound.

For TPU v5e MXU: critical intensity ≈ 240 FLOPs/byte (1.97e14 FLOPs/s ÷ 8.2e11 bytes/s HBM bandwidth).

### Matmul Roofline

For $X[B, D] \cdot Y[D, F] \rightarrow Z[B, F]$ in bfloat16:

$$\text{Intensity}(\text{matmul}) = \frac{2BDF}{2BD + 2DF + 2BF}$$

When $B \ll D, F$ (typical for Transformers):

$$\frac{2BDF}{2BD + 2DF + 2BF} \approx \frac{2BDF}{2DF} = B$$

Partitioned across 2 chips (sharded along D):

$$T_{\text{math}} = \frac{BDF}{C} \qquad T_{\text{comms}} = \frac{2BF}{W_{\text{ici}}}$$

Become compute-bound when $\frac{D}{2} > \frac{C}{W_{\text{ici}}} = 4377$, or $D > 8755$. Note this depends on D, not B — completely different from the single-chip case.

### Network Communication Rooflines

When communication is between chips (not HBM→compute), use ICI/DCN bandwidth:

| Link | TPU v5p BW | TPU v5e BW |
|------|-----------|-----------|
| HBM → MXU | 2.8 TB/s | 0.81 TB/s |
| ICI (per axis) | 90 GB/s bidi | 45 GB/s bidi |
| DCN (egress/chip) | 6.25 GB/s | 3.125 GB/s |
| PCIe | 16 GB/s | 16 GB/s |

### Quantization Impact

int8 weight quantization (activations in bf16) lowers the critical batch size:

$$B_{\text{crit}} = \frac{C_{\text{bf16}}}{2 \cdot W_{\text{hbm}}} \approx 120 \quad \text{(vs 240 for full bf16)}$$

## Chapter 2: TPU Architecture

### Chip Components

A TPU is a matrix multiplication machine attached to HBM:

- **MXU** (Matrix Multiply Unit): systolic array performing `bf16[8,128] × bf16[128,128] → f32[8,128]` every 8 cycles on v5e (256×256 on v6e). 1.97e14 bf16 FLOPs/s on v5e.
- **VPU** (Vector Processing Unit): 2D SIMD machine (8×128 ALUs) for elementwise ops, ~1.4e13 FLOPs/s on v5p (10× smaller than MXU).
- **VMEM** (Vector Memory): on-chip scratchpad, ~22× faster than HBM. 128 MiB on v5e. Data must be in VMEM before MXU/VPU can use it.
- **HBM**: 16–96 GiB per chip, bandwidth 0.81–2.8 TB/s.

### Systolic Array

A 128×128 grid of multiply-add ALUs. Weights flow in from above (RHS), activations from the left (LHS). Results accumulate downward. After initial pipeline bubble, new inputs stream in without additional bubbles. Matrices should be padded to ≥128 in both dimensions (256 on v6e).

### TPU Specs

| Model | HBM/chip | HBM BW | bf16 FLOPs/s | int8 FLOPs/s | Pod size |
|-------|----------|--------|-------------|-------------|----------|
| TPU v4p | 32 GB | 1.2 TB/s | 2.75e14 | 2.75e14 | 16×16×16 |
| TPU v5p | 96 GB | 2.8 TB/s | 4.59e14 | 9.18e14 | 16×20×28 |
| TPU v5e | 16 GB | 0.81 TB/s | 1.97e14 | 3.94e14 | 16×16 |
| TPU v6e | 32 GB | 1.6 TB/s | 9.20e14 | 1.84e15 | 16×16 |

### Networking Topology

- **ICI**: nearest-neighbor interconnect forming 2D torus (v5e/v6e — 4 neighbors) or 3D torus (v4p/v5p — 6 neighbors). Wraparound links require axis size = 16 (v5e/v6e) or multiple of 4 (v5p).
- **DCN**: connects hosts across pods, much slower than ICI (6.25 GB/s vs 90 GB/s).
- **PCIe**: connects TPU trays to host CPU — 16 GB/s per chip.
- GPUs differ: use switched topology (NVLink/NVSwitch) approximating any-to-any, but more expensive and limited scale.

## Chapter 3: Sharded Matmuls

### Sharding Notation

Arrays named with subscript mesh axes: $A[I_X, J_Y]$ means dimension I sharded over mesh axis X, J over Y. Fully replicated: $A[I, J]$.

### Collective Operations

**AllGather**: reassembles shards → $A[I_X] \rightarrow A[I]$. Cost: $T = V / W_{\text{ici}}$ (bidirectional ring), independent of shard count.

**ReduceScatter**: sums partial sums, scatters result → $A[I]\{U_X\} \rightarrow A[I_X]$. Same cost as AllGather.

**AllReduce** = ReduceScatter + AllGather. Cost: $T = 2V / W_{\text{ici}}$.

**AllToAll**: transposes sharding → $A[I_X, J] \rightarrow A[I, J_X]$. Cost: $T = V / (4 \cdot W_{\text{ici}})$ for 1D ring — 1/4 of AllGather.

Latency-bound regime: when per-shard bytes < 45 kB (TPU v5e), hop latency (~1 μs) dominates.

### Four Cases for Sharded Matmul

1. **Neither sharded on contracting dim**: multiply locally, zero comms. $A[I_X, J] \cdot B[J, K_Y] \rightarrow C[I_X, K_Y]$.

2. **One sharded on contracting dim**: AllGather the sharded input first. $A[I, J_X] \cdot B[J, K] \rightarrow \text{AllGather}_X \rightarrow C[I, K]$.

3. **Both sharded on contracting dim**: local multiply → AllReduce partial sums. $A[I, J_X] \cdot_{\text{LOCAL}} B[J_X, K] \rightarrow C[I, K]\{U_X\} \xrightarrow{\text{AllReduce}} C[I, K]$.

4. **Both have non-contracting dim sharded on same axis**: invalid. Must AllGather one input first. $A[I_X, J] \cdot B[J, K_X]$ is impossible.

## Chapter 4: Transformer Math

### Parameter and FLOP Counts

| Component | Params per layer | Training FLOPs per layer |
|-----------|-----------------|------------------------|
| MLP (gated, 3 matmuls) | 3DF | 18BTDF |
| Attention (QKVO) | 4DNH | 24BTDNH + 12BT²NH |
| Other (layernorm) | D | O(BTD) |
| Unembedding | DV (total) | 12BTDV (total) |

Where B = tokens, T = sequence length, D = d_model, F = d_ff, N = num_heads, H = head_dim, V = vocab_size, L = num_layers.

### Training FLOPs Rule of Thumb

Ignoring attention (dominant only when $T > 8D$):

$$6 \cdot \text{num_params} \cdot \text{num_tokens}$$

Forward pass: 2× params × tokens. Backward pass: 4× params × tokens (gradients w.r.t. weights + inputs). Total: 6×.

### Memory: KV Cache

Per sequence in int8: $2 \cdot S \cdot L \cdot K \cdot H$ bytes. For $D = 8192$, $L = 64$, $S = 8192$ with MHA: $2 \cdot 8192 \cdot 64 \cdot 8192 = 8$ GiB.

### Gradient Checkpointing

Trade compute for memory:
- **Block remat**: save 1 checkpoint per layer → ~8ND FLOPs (vs 6ND), but only 4.2TB activations saved
- **Big matmuls only**: save 7 checkpoints per layer → less recomputation, more memory

### Flash Attention

Computes attention in K/V chunks using running max (M), output (O), and softmax sum (L) — never materializes the full $S \times S$ attention matrix:

$$L^{\text{combined}} = e^{M^1 - \max(M^1, M^2)} \cdot L^1 + e^{M^2 - \max(M^1, M^2)} \cdot L^2$$

## Chapter 5: Training Parallelism

### Four Strategies

| Strategy | Sharding | What moves | Critical batch size |
|----------|----------|-----------|-------------------|
| **Data Parallelism** | $B$ sharded, weights replicated | Gradients (AllReduce in backward) | $B/X > C/W$ |
| **FSDP (ZeRO-3)** | $B$ + $D_{\text{weights}}$ sharded | Weights (AllGather in forward) | Same as DP |
| **Tensor Parallelism** | $D$ + $F$ sharded | Activations (AllGather + ReduceScatter) | $F > Y \cdot C/W$ |
| **Pipeline Parallelism** | Layers sharded | Activations between stages | P2P, minimal |

### FSDP Communication Cost

Same as pure DP since AllReduce ≡ AllGather + ReduceScatter. Per layer:

$$T_{\text{math}} = \frac{4 \cdot B \cdot D \cdot F}{X \cdot C} \qquad T_{\text{comms}} = \frac{4 \cdot D \cdot F}{W_{\text{ici}}}$$

Compute-bound when $B/X > C / W_{\text{ici}} \approx 2550$ (TPU v5p, bf16).

### Tensor Parallelism Cost

AllGather activations before first matmul, ReduceScatter after second:

$$T_{\text{math}} = \frac{4 \cdot B \cdot D \cdot F}{Y \cdot C} \qquad T_{\text{comms}} = \frac{4 \cdot B \cdot D}{W_{\text{ici}}}$$

Compute-bound when $F > Y \cdot C / W_{\text{ici}}$. For TPU v5p: $F > 2550 \cdot Y$. With $F \approx 30\text{k}$ (LLaMA 70B): max 8-way TP comfortably.

### Combined FSDP + TP

Shard $B$ over X (FSDP) and $F$ over Y (TP). Total comms:

$$T_{\text{comms}} = \frac{4D}{W_{\text{ici}}} \max\left(\frac{F \cdot X}{N \cdot M_X}, \frac{B}{X \cdot M_Y}\right)$$

Optimal FSDP ratio: $X_{\text{opt}} = \sqrt{\frac{B}{F} \cdot \frac{M_X}{M_Y} \cdot N}$.

FSDP moves weights, TP moves activations — as batch shrinks, TP gets cheaper (smaller AllGathers); as model grows, FSDP gets cheaper (more FLOPs to hide comms).

### Pipeline Parallelism

Layers split across stages, activations microbatched. Bubble overhead: $\frac{Z - 1}{Z + \mu - 1}$ where Z = stages, μ = microbatches. More microbatches → less bubble, more memory.

### Scaling Across Pods (DCN)

DCN bandwidth is ~14× slower than ICI (6.25 vs 90 GB/s). Data parallelism over DCN feasible when $B/X > C / W_{\text{dcn}} \approx 73,400$ — requires very large batch sizes. Prefer ICI for FSDP/TP, DCN only for data parallelism with large batches.

## Practical Example: LLaMA 3 70B on TPU v5p

From the applied training chapter:
- $D = 8192$, $F \approx 28,672$, $L = 80$, total params: 70B
- Batch size: ~16M tokens
- Required FLOPs: $6 \cdot 70 \cdot 10^9 \cdot 15 \cdot 10^{12} = 6.3 \cdot 10^{24}$
- With FSDP + TP on ~18,823 chips (2 pods) at 50% MFU: ~17 days

## Connections

- [AI Systems Performance Engineering](aspe-overview.md) — GPU-centric counterpart, covers CUDA, memory hierarchy, kernel optimization
- [Ultra-Scale Playbook](usp-ultra-scale-playbook.md) — similar educational scope, GPU-focused with 4,100+ experiments
- [GSPMD](gspmd.md) — Google's compiler-based auto-parallelization; the sharding notation here maps directly to GSPMD's device mesh
- [Scaling Techniques Overview](usp-scaling-techniques-overview.md) — the parallelism strategies covered in depth
- [Megatron-Core MoE](megatron-core-moe.md) — production training stack; this book provides the theoretical foundation
- [Ring Attention](ring-attention.md) — extends the sharding theory to attention with online softmax
- [FlashAttention](flashattention.md) — attention optimization that changes the roofline in Ch 4
- [GPU Hardware Architecture](aspe-gpu-hardware-architecture.md) — complements Ch 12 on GPU differences from TPUs
- [Tensor Core Evolution](tensor-core-evolution.md) — GPU counterpart to TPU systolic arrays
- [NCCL Demystifying](nccl-demystifying.md) — GPU collective communication primitives mirroring TPU's ICI collectives
- [LLM Scaling Laws](llm-scaling-laws.md) — why scaling efficiency matters; this book shows how to achieve it
- [Muon Optimizer](muon-optimizer.md) — alternative optimizer that changes the memory/compute tradeoff
