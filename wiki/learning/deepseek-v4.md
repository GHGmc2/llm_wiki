---
title: "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence"
type: source-note
tags: [llm, mixture-of-experts, deepseek, long-context, hybrid-attention, csa, hca, mhc, muon, post-training, infrastructure]
created: 2026-05-02
updated: 2026-05-03
sources: [raw/DeepSeek-V4.pdf]
status: stable
---

# DeepSeek-V4

**Source**: DeepSeek-AI Technical Report, April 2026. 58 pages. Preview release.

## Key Points

- Two models: **V4-Pro** (1.6T params, 49B active) and **V4-Flash** (284B params, 13B active), both with native 1M-token context
- **Hybrid attention** (CSA + HCA) is the headline innovation: at 1M-token context, V4-Pro uses only 27% of V3.2's inference FLOPs and 10% of its KV cache
- **mHC** (Manifold-Constrained Hyper-Connections) replaces residual connections for numerical stability at 1.6T scale
- **Muon optimizer** replaces AdamW for most parameters — faster convergence, better stability
- **On-Policy Distillation** replaces mixed RL: train domain specialists, distill into unified model
- Three reasoning modes: Non-Think, Think High, Think Max
- Open weights, honest about trailing frontier by 3-6 months on general knowledge

## Models at a Glance

| | V4-Pro | V4-Flash |
|---|---|---|
| Total params | 1.6T | 284B |
| Active params | 49B | 13B |
| Layers | 61 | 43 |
| Hidden dim | 7168 | 4096 |
| Routed experts | 256 | 256 |
| Active experts/token | 8 | 6 |
| Shared experts | 1 | 1 |
| Pre-training tokens | 33T | 32T |
| Context length | 1M | 1M |

V4-Flash-Base surpasses V3.2-Base on most benchmarks despite having 42% of V3.2's total parameters — architectural and data-quality improvements drive the gain.

## Architecture Innovations

### Hybrid Attention: CSA + HCA

The central efficiency breakthrough. Two attention types interleaved across layers:

- **CSA (Compressed Sparse Attention)**: Compresses KV cache 1/m, then applies sparse top-k selection via a learned Lightning Indexer. Adds a sliding window of recent uncompressed tokens for local dependencies.
- **HCA (Heavily Compressed Attention)**: Much heavier compression (m' >> m), dense attention over compressed KV. Retains broad, low-resolution view of full context.

CSA gives precise look-up; HCA gives global summary. Interleaving is what makes 1M context economical — neither alone achieves the full efficiency gain.

See: [DeepSeek-V4 Architecture]#architecture

### mHC (Manifold-Constrained Hyper-Connections)

Upgrades conventional residual connections by constraining the residual mapping to the Birkhoff polytope (doubly stochastic matrices), bounding spectral norm to <= 1. This makes signal propagation non-expansive, enabling stable training at 1.6T scale.

### Muon Optimizer

Replaces AdamW for most parameters (embeddings, prediction head, RMSNorm still use AdamW). Delivers faster convergence and better training stability. Combined with Anticipatory Routing and SwiGLU clamping as stabilizers.

## Pre-Training

Both models trained with FP4 quantization-aware training for routed expert weights, FP8 for non-expert computation. Batch size scheduling from small to large (75.5M for Flash, 94.4M for Pro). Sequence length gradually extended: 4K -> 16K -> 64K -> 1M.

**Training stabilizers**:
- **Anticipatory Routing**: Decouples routing updates from backbone by one step; triggered automatically on loss spikes
- **SwiGLU clamping**: Linear component clamped to [-10, 10], gate capped at 10

**Base model results**: V4-Flash-Base surpasses V3.2-Base on most benchmarks. V4-Pro-Base near-universal dominance — dramatic gains on knowledge (MMLU 90.1 vs 87.8) and long-context (LongBench-V2 51.5 vs 40.2).

## Post-Training: On-Policy Distillation

Replaces V3.2's mixed RL stage. Two-phase pipeline:

1. **Specialist Training**: Train separate domain experts (math, code, agent, instruction) via SFT + GRPO
2. **On-Policy Distillation**: Unify specialists into one model via multi-teacher reverse-KL distillation on student-generated trajectories

See: [DeepSeek-V4 Post-Training]#post-training

### Three Reasoning Modes

| Mode | Behavior | Use Case |
|------|----------|----------|
| Non-Think | Fast, no CoT | Routine tasks, low-latency |
| Think High | Standard reasoning trace | Complex problem-solving (default) |
| Think Max | Max reasoning — exhaustive decomposition, edge-case stress-testing | Frontier benchmarks, hardest problems |

Think Max prepends a special system prompt and uses larger context window (384K) with reduced length penalties.

## Infrastructure

- **Fine-grained EP communication-computation overlap** with expert-level pipelining
- **TileLang**: A new kernel development language for flexible, efficient MoE kernels
- **FP4 QAT**: Routed experts in FP4; theoretically 1.33x more efficient on future hardware
- **KV cache management**: On-disk storage for compressed KV entries, three SWA caching strategies (full, periodic checkpointing, zero)
- **Quick Instruction**: Special tokens for auxiliary tasks (search queries, intent) using existing KV cache — reduces time-to-first-token

See: [DeepSeek-V4 Infrastructure]#infrastructure

## Benchmarks: V4-Pro-Max vs Frontier

| Benchmark | V4-Pro-Max | Best Frontier | Leader |
|-----------|-----------|---------------|--------|
| LiveCodeBench (Pass@1) | **93.5** | 91.7 (Gemini) | V4 |
| Codeforces (Rating) | **3206** | 3168 (GPT-5.4) | V4 |
| Apex Shortlist (Pass@1) | **90.2** | 89.1 (Gemini) | V4 |
| Putnam-2025 | **120/120** | — | V4 |
| SWE Verified (Resolved) | 80.6 | 80.8 (Opus) | Tie |
| MMLU-Pro (EM) | 87.5 | **91.0** (Gemini) | Gemini |
| GPQA Diamond (Pass@1) | 90.1 | **94.3** (Gemini) | Gemini |
| SimpleQA-Verified | 57.9 | **75.6** (Gemini) | Gemini |
| MRCR 1M (MMR) | 83.5 | **92.9** (Opus) | Opus |
| HLE (Pass@1) | 37.7 | **44.4** (Gemini) | Gemini |

**DeepSeek's own framing**: Trails absolute frontier by 3-6 months on general knowledge and hardest retrieval. Sets new open-model highs on competitive programming and formal reasoning.

## Connections

- [Parallel Folding](megatron-core-moe.md) — how EP decoupling works in Megatron-Core
- [Multi-Head Latent Attention](multi-head-latent-attention.md) — MLA, the KV cache compression foundation
- [DeepSeek-V3](deepseek-v3.md) — base architecture that V4 extends
- [DeepSeek-V3.2](deepseek-v3.2.md) — DSA sparse attention (predecessor to CSA)
- [DeepSeek-R1](deepseek-r1.md) — GRPO algorithm and reasoning paradigm
- [FlashAttention](flashattention.md) — attention kernel optimization

# DeepSeek-V4 Architecture

**Source**: DeepSeek-V4 Technical Report, Section 2 [src](raw/DeepSeek-V4.pdf)

## Inherited from DeepSeek-V3

V4 retains the DeepSeekMoE framework with minor tweaks:

- **Activation function**: Changed from Sigmoid to `Sqrt(Softplus(.))` for affinity scores
- **Load balancing**: Auxiliary-loss-free strategy retained, augmented with slight sequence-wise balance loss
- **Routing**: Removed constraint on number of routing target nodes; redesigned parallelism to maintain efficiency
- **Hash routing**: First 3 MoE layers use Hash routing (deterministic expert assignment by token ID) instead of dense FFN

Multi-Token Prediction (MTP) carried over unchanged from V3.

## Hybrid Attention: CSA and HCA

The core efficiency breakthrough. Two attention types are **interleaved across layers** — neither alone achieves the full efficiency gain.

### Compressed Sparse Attention (CSA)

**Compression + sparsity in two stages**:

**Stage 1: KV Compression**. Every `m` KV entries are compressed into one via a learned weighted sum. Two compression branches (a and b) with overlapping windows produce compressed entries at 1/m the sequence length. Compression weights are computed via softmax over 2m elements (m from each branch).

**Stage 2: Sparse Selection**. A learned **Lightning Indexer** scores each compressed block against the query token using low-rank indexer queries. Top-k selection picks only the most relevant compressed blocks for core attention:

```
Indexer Queries: cQ = h_t * W_DQ
                 qI = cQ * W_IUQ (low-rank decompression)
Index Scores:    I_{t,s} = sum_h(wI_{t,h} * ReLU(qI_{t,h} · KIComp_s))
Selection:       CSprsComp_t = {CComp_s | I_{t,s} in Top-k}
```

**Core Attention**: Shared Key-Value Multi-Query Attention (MQA) on the selected sparse KV entries. Grouped output projection splits outputs into g groups, each projected to a smaller intermediate dimension before final projection — reducing the O(c * n_h * d) output projection cost.

**Sliding window**: A small set of recent uncompressed KV entries (window size `n_win`) is concatenated to sparse entries for local fine-grained dependencies.

**V4-Pro config**: m=4, top-k=1024, n_Ih=64, c_I=128, n_h=64, c=512, d_c=1024, g=8, n_win=128

### Heavily Compressed Attention (HCA)

**Extreme compression, dense attention**:

- Compression rate `m'` is much larger than CSA's `m` (m'=128 for both models)
- Uses a single compression branch (no overlap between blocks)
- After compression, attention is **dense** over the compressed KV (no top-k selection)
- Same MQA and grouped output projection as CSA
- Adds sliding window for local dependencies

**Trade-off**: HCA retains a broad, low-resolution view of the full context but loses fine-grained selectivity. Interleaving HCA layers with CSA layers gives the model both modes.

### Efficiency Analysis

At 1M-token context, the hybrid attention delivers dramatic savings vs V3.2:

| Metric | V4-Pro | V4-Flash |
|--------|--------|----------|
| Single-token FLOPs (vs V3.2) | 27% | 10% |
| KV cache size (vs V3.2) | 10% | 7% |

The savings come from: (1) compressed KV entries reducing both storage and attention computation, (2) sparse top-k selection in CSA further reducing attention FLOPs, (3) FP4 precision for expert weights reducing memory.

**V4-Pro has MORE active parameters than V3.2 (49B vs 37B) but uses 3.7x LESS inference FLOPs at 1M context.** This is the entire thesis: efficiency, not capability, is the bottleneck for long-context models.

### Layer Interleaving

The first two layers use pure sliding window attention (Flash) or HCA (Pro). Subsequent layers interleave CSA and HCA. The exact interleaving pattern is not specified but the paper notes that the hybrid configuration was validated through ablations — neither CSA-only nor HCA-only configurations matched the hybrid's quality-efficiency trade-off.

## mHC: Manifold-Constrained Hyper-Connections

**Problem**: Standard Hyper-Connections (HC) decouple residual width from hidden size, offering a complementary scaling axis. But training frequently exhibits numerical instability when stacking many layers — the residual mapping can amplify signals uncontrollably.

**Solution**: mHC constrains the residual mapping matrix B_l to the **Birkhoff polytope** — the manifold of doubly stochastic matrices:

```
B_l in M = {M in R^{n×n} | M·1_n = 1_n, 1_n^T·M = 1_n^T, M >= 0}
```

This ensures the spectral norm ||B_l||_2 <= 1, making the residual transformation **non-expansive** — signals cannot blow up across layers. The manifold is closed under multiplication, guaranteeing stability in deep stacks.

**Dynamic parameterization**: Three linear mappings (input A_l, residual B_l, output C_l) are generated dynamically from the input:
1. Input is flattened and RMS-normalized
2. Raw parameters computed via learned projections + static biases
3. Constraints applied: Sigmoid for A_l and C_l (non-negative, bounded); Birkhoff projection via Sinkhorn-Knopp for B_l (doubly stochastic)
4. Gating factors initialized close to zero for stable training warm-up

**V4 config**: n_hc = 4 (expansion factor), t_max = 20 (Sinkhorn-Knopp iterations)

## Muon Optimizer

Replaces AdamW for the **majority of parameters**. Embedding module, prediction head, and RMSNorm weights still use AdamW.

**Key properties**:
- Momentum = 0.95, weight_decay = 0.1
- RMS of each update matrix rescaled to 0.18 to reuse AdamW learning rates
- Matrix-aware orthogonalization improves conditioning of optimization trajectories

**Training stability**: Despite Muon, additional stabilizers were still required:
- **Anticipatory Routing**: Decouples routing from backbone by one step during loss spikes
- **SwiGLU clamping**: Linear component clamped to [-10, 10], gate capped at 10

## Model Configurations

| Parameter | V4-Flash | V4-Pro |
|-----------|----------|--------|
| Layers | 43 | 61 |
| Hidden dim | 4096 | 7168 |
| CSA compression rate (m) | 4 | 4 |
| HCA compression rate (m') | 128 | 128 |
| CSA top-k | 512 | 1024 |
| Query heads (n_h) | 64 | 64 |
| Head dim (c) | 512 | 512 |
| Query compression dim (d_c) | 1024 | 1024 |
| Output groups (g) | 8 | 8 |
| Intermediate output dim (d_g) | 1024 | 1024 |
| SWA window (n_win) | 128 | 128 |
| Routed experts | 256 | 256 |
| Active experts/token | 6 | 8 |
| Expert intermediate dim | 2048 | 2048 |
| mHC n_hc | 4 | 4 |
| Sinkhorn-Knopp iterations | 20 | 20 |

# DeepSeek-V4 Post-Training

**Source**: DeepSeek-V4 Technical Report, Section 5 [src](raw/DeepSeek-V4.pdf)

## Pipeline Overview

```
Base Model
    |
    v
SFT (domain-specific data)
    |
    v
GRPO (reinforcement learning per domain)
    |
    +---> Math Specialist
    +---> Code Specialist
    +---> Agent Specialist
    +---> Instruction Specialist
    |
    v
On-Policy Distillation (multi-teacher)
    |
    v
Unified DeepSeek-V4 Model
```

## Phase 1: Specialist Training

For each target domain, a separate expert model is trained independently:

1. **SFT**: Fine-tune the base model on high-quality, domain-specific data to establish foundational capabilities
2. **RL with GRPO**: Apply Group Relative Policy Optimization guided by domain-specific prompts and reward signals

**Reward models**: Easy-to-verify tasks (math, code) use rule-based verifiers and test cases. Hard-to-verify tasks use **generative reward models** — the model leverages its own logic to generalize across complex tasks.

**Reasoning effort calibration**: Specialists are trained under different RL configurations to produce models optimized for varying reasoning depths. This directly enables the three reasoning modes:

| Mode | Characteristics | RL Config |
|------|----------------|-----------|
| Non-Think | Fast, intuitive, low latency | High length penalty, small context |
| Think High | Conscious logical analysis | Moderate length penalty, medium context |
| Think Max | Push reasoning to fullest extent | Low length penalty, large context (384K) |

Each mode uses distinct `<think>` / `</think>` response formats. Think Max additionally prepends a system prompt instruction (see below).

### Think Max System Prompt

```
Reasoning Effort: Absolute maximum with no shortcuts permitted.
You MUST be very thorough in your thinking and comprehensively decompose the
problem to resolve the root cause, rigorously stress-testing your logic against
all potential paths, edge cases, and adversarial scenarios.
Explicitly write out your entire deliberation process, documenting every
intermediate step, considered alternative, and rejected hypothesis to ensure
absolutely no assumption is left unchecked.
```

## Phase 2: On-Policy Distillation (OPD)

OPD replaces V3.2's mixed RL consolidation stage. Given N expert models (teachers) and a student model:

**Formally**: The student optimizes a reverse-KL loss against teacher output distributions on its **own generated trajectories**:

```
L_OPD = sum_i(w_i * KL(p_student || p_teacher_i))
```

The student generates tokens, and each teacher provides its probability distribution for those tokens. The student learns to match the teachers' distributions on its own outputs — not on training data from the teachers.

**Key advantage over mixed RL**: OPD preserves each specialist's domain expertise without the interference that mixed RL can cause when reward signals from different domains conflict.

### Infrastructure for OPD

- **Full-vocabulary OPD**: Efficient teacher scheduling required for computing reverse-KL across the full 128K vocabulary
- **Preemptible rollout service**: Training tasks can be preempted and resumed without data loss
- **Million-token RL framework**: RL infrastructure scaled to support 1M-token context rollouts
- **DSec sandbox**: Sandbox infrastructure for agentic AI training (see Infrastructure page)

## Tool-Call Schema

V4 introduces a new **XML-based tool-call schema** using `<|DSML|>` tokens:

```
<|DSML|tool_calls>
<|DSML|invoke name="tool_name">
<|DSML|parameter name="param" string="true">value</|DSML|parameter>
</|DSML|invoke>
</|DSML|tool_calls>
```

The XML format reduces escaping failures and tool-call errors compared to V3.2's format. Agent scaffolding calling V4 should expect this format.

## Interleaved Thinking

A refinement of V3.2's context management for reasoning traces:

**Tool-calling scenarios (Figure 7a)**: All reasoning content (inside `<think>` tags) is preserved across the entire conversation — including across user message boundaries. The model maintains a coherent, cumulative chain of thought over long-horizon agent tasks.

**General conversation (Figure 7b)**: Original V3.2 strategy preserved — reasoning from previous turns is discarded on new user messages, keeping context concise.

> For agent frameworks that simulate tool interactions via user messages (e.g., Terminus), the tool-calling path may not trigger. DeepSeek recommends non-Think models for such architectures.

## Quick Instruction

A latency optimization for chatbot scenarios. Auxiliary tasks (web search trigger, intent recognition, etc.) are conventionally handled by a separate small model requiring redundant prefill.

**V4's approach**: Append dedicated special tokens to the input sequence, each corresponding to a specific auxiliary task. By directly reusing the already-computed KV cache, Quick Instruction:
- Eliminates redundant prefilling
- Enables parallel execution of tasks (search queries, authority check, domain classification)
- Reduces time-to-first-token (TTFT)
- Eliminates the engineering overhead of maintaining a separate small model

## Post-Training Infrastructure

- **FP4 Quantization Integration**: FP4 models used throughout RL and OPD stages for memory efficiency
- **Efficient Teacher Scheduling**: Multi-teacher OPD requires scheduling N teacher distributions per student token
- **Preemptible Rollout Service**: Fault-tolerant, globally ordered trajectory logging with fast-forward resumption
- **DSec Sandbox**: Layered container/VM storage, density optimizations for massive concurrency, trajectory logging for agent training

# DeepSeek-V4 Infrastructure

**Source**: DeepSeek-V4 Technical Report, Section 3 [src](raw/DeepSeek-V4.pdf)

## 1. Fine-Grained EP Communication-Computation Overlap

Expert Parallelism (EP) requires all-to-all communication for token dispatch and combine. V4 introduces **expert-level pipelining** to overlap communication with computation:

**Forward pass**: As soon as tokens for expert `e` are dispatched, that expert begins computation. Meanwhile, dispatch for expert `e+1` proceeds on a separate stream. Similarly, combine for expert `e` overlaps with computation for expert `e+1`.

**Backward pass**: Gradient dispatch and combine are similarly overlapped with expert gradient computation. The key insight: expert computations are independent, so pipelining across experts hides communication latency.

This is a finer granularity than the typical layer-level overlap (e.g., Megatron-Core's 1F1B overlap) — V4 overlaps at the **individual expert level** within a single MoE layer.

## 2. TileLang: Kernel Development Language

**Problem**: MoE models require many specialized GPU kernels (grouped GEMM, sparse attention, compression, routing). Writing and optimizing these in CUDA/CUTLASS for each new architecture is a bottleneck.

**Solution**: TileLang is a new domain-specific language for tile-based GPU kernel development. Key features:

- **Hardware-agnostic**: Write kernels once, compile for Hopper, Blackwell, and future architectures
- **Tile abstraction**: Kernels expressed as operations on tiles — the compiler handles tiling, scheduling, and memory hierarchy
- **Flexible**: Supports the irregular computation patterns common in MoE (variable expert sizes, dynamic token counts, sparse attention)

**Impact**: Accelerated development of the CSA, HCA, and FP4 quantization kernels that are critical to V4's efficiency.

## 3. High-Performance Kernel Libraries

V4 ships with two key kernel libraries:

**Batch-Invariant Libraries**: Kernels that produce identical results regardless of batch size or token distribution. This is critical for deterministic training at scale — debugging distributed training failures requires reproducibility across different parallelism configurations.

**Deterministic Kernels**: All critical operations (attention, MoE routing, communication) are deterministic by design. Non-deterministic operations (e.g., atomic adds in some reduction patterns) are avoided or have deterministic alternatives.

## 4. FP4 Quantization-Aware Training

V4 uses **FP4 precision for routed expert weights**, FP8 for non-expert computation.

**Current hardware**: Peak throughput for FP4 x FP8 operations equals FP8 x FP8 on existing hardware. The benefit today is **memory savings** — smaller weight storage for the massive expert parameter count.

**Future hardware**: The paper explicitly notes that purpose-built hardware could make FP4 roughly **1.33x more efficient** than FP8 — an open lane for future inference gains as hardware catches up.

**QAT approach**: Quantization-aware training means weights are trained with quantization simulated during forward and backward passes, preserving accuracy while enabling low-precision deployment. This is applied specifically to expert weights where the memory savings are most impactful (256 experts x large hidden dims).

## 5. Training Framework

### Muon Implementation

Efficient distributed Muon requires careful handling of the orthogonalization step across data-parallel ranks. V4's implementation shards optimizer states across DP ranks while maintaining correct orthogonalization semantics.

### mHC Implementation

Manifold-Constrained Hyper-Connections are implemented efficiently:
- Dynamic parameter generation uses small projection matrices (n_hc=4 is tiny relative to hidden dim)
- Sinkhorn-Knopp iterations (t_max=20) are batched and run on GPU
- Overall mHC overhead is small (< 1% of total compute)

### Context Parallelism for Long-Context

For 1M-token training, context parallelism partitions sequences across GPUs. V4's CP implementation handles the hybrid attention patterns — CSA compression and HCA compression must be aware of CP boundaries.

### Extended Automatic Differentiation

V4 extends autodiff for **flexible activation checkpointing**. Standard checkpointing is binary (recompute or store). V4's extended AD allows fine-grained control: specific modules can be checkpointed at specific granularities, enabling memory-compute trade-offs tailored to the hybrid attention architecture.

## 6. Inference Framework

### KV Cache Structure

The hybrid attention architecture requires careful KV cache management:

| Attention Type | KV Cache Entries | Compression |
|---------------|-----------------|-------------|
| CSA compressed KV | n/m entries | 1/m |
| CSA SWA KV | n_win entries | None (sliding window) |
| HCA compressed KV | n/m' entries | 1/m' |
| HCA SWA KV | n_win entries | None (sliding window) |

**Key insight**: Only compressed KV entries need to be stored for the full sequence. SWA KV entries are ephemeral — they only need the last n_win tokens.

### On-Disk KV Cache Storage

To support prefix caching across requests, V4 stores KV caches to disk:

**Compressed KV entries**: Stored entirely on disk. On a cache hit, the compressed entries are read directly — no recomputation needed for complete compression blocks. Tail incomplete blocks are recomputed.

**SWA KV entries**: Three strategies, trading storage for computation:

| Strategy | Storage | Recompute | Best For |
|----------|---------|-----------|----------|
| Full SWA Caching | Complete SWA KV | Zero | Low-latency, high-SSD deployments |
| Periodic Checkpointing | Checkpoint every p tokens | Tail recompute | Balanced deployments |
| Zero SWA Caching | None | Last n_win * L tokens | Storage-constrained |

**Zero SWA Caching detail**: Each SWA KV entry depends only on the last n_win tokens from the previous layer. With cached CSA/HCA KV entries, recomputing only the last n_win * L tokens is sufficient. This is practical because n_win is small (128 tokens).

## 7. RL and Agent Infrastructure

### Million-Token RL Framework

The RL training framework was scaled to support 1M-token context rollouts. This required:
- Memory-efficient rollout storage for long trajectories
- Efficient attention computation across 1M-token sequences during RL updates
- Preemption-safe resumption for long-running rollouts

### DSec Sandbox

For agentic AI training (code execution, tool use), V4 introduces DSec:

- **Layered storage**: Base images as readonly EROFS layers; data blocks fetched on demand from 3FS distributed filesystem
- **Fast startup**: File metadata available locally at mount time; millisecond-scale resumption via chainable snapshots
- **Density optimization**: Memory reclamation for safe overcommitment; spinlock contention reduction in container runtime
- **Trajectory logging**: Globally ordered, persistent command/result logs enabling fast-forward resumption and deterministic replay

DSec supports both container-based and microVM-based sandboxes with a unified interface. Switching between them requires only a parameter change.
