---
title: "DeepSeek-V4 Architecture: CSA, HCA, mHC, and Muon"
type: concept
tags: [llm, mixture-of-experts, deepseek, hybrid-attention, csa, hca, mhc, muon, architecture]
created: 2026-05-02
updated: 2026-05-02
sources: [raw/DeepSeek-V4.pdf]
status: stable
---

# DeepSeek-V4 Architecture

**Source**: DeepSeek-V4 Technical Report, Section 2 [src](raw/DeepSeek-V4.pdf)

## Key Points

- V4 retains Transformer + Multi-Token Prediction + DeepSeekMoE from V3, adding three major upgrades
- **Hybrid Attention** (CSA + HCA) interleaves two compression strategies — the key to 1M-token efficiency
- **mHC** constrains residual mappings to doubly stochastic matrices for stability at 1.6T scale
- **Muon** replaces AdamW for most parameters, delivering faster convergence

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

## Connections

- [DeepSeek-V4 Technical Report](deepseek-v4-technical-report.md) — full paper summary and benchmarks
- [DeepSeek-V4 Post-Training](deepseek-v4-post-training.md) — On-Policy Distillation and reasoning modes
- [DeepSeek-V4 Infrastructure](deepseek-v4-infrastructure.md) — training/inference systems
- [Megatron-Core MoE](megatron-core-moe-scalable-training.md) — contrast with NVIDIA's MoE architecture approach
