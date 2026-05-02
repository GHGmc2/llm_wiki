---
title: "ScaleRL: The Art of Scaling Reinforcement Learning Compute for LLMs"
type: source-note
tags: [reinforcement-learning, scaling-laws, grpo, llm, rl, scalerl, cispo, pipeline-rl]
created: 2026-05-02
updated: 2026-05-02
sources: [raw/The Art of Scaling Reinforcement Learning Compute for LLMs.pdf]
status: stable
---

# ScaleRL: The Art of Scaling RL Compute

**Source**: Meta / UT Austin / UCL / UC Berkeley / Harvard / Periodic Labs. 28 pages. 400,000+ GPU-hours of systematic RL scaling experiments.

## Key Points

- **First large-scale systematic study** of RL compute scaling for LLMs — previously an "art not science"
- **Sigmoidal compute-performance curves**: `RC - R0 = (A - R0) × 1/(1 + (Cmid/C)^B)` — parameters for asymptote (A), compute efficiency (B), midpoint (Cmid)
- **ScaleRL recipe**: Combines existing methods into a predictable, scalable RL pipeline — achieves SOTA asymptotic performance
- **Predicted 100K GPU-hour run** from 50K GPU-hour fit — demonstrated on 8B dense and 17B×16 MoE (Scout) models
- **CISPO > GSPO > DAPO/GRPO**: CISPO (truncated IS + vanilla policy gradient) is most stable and hyperparameter-robust
- **FP32 at LM head is critical**: +0.09 asymptotic pass rate (0.52 → 0.61)

## The Problem: RL Scaling is Ad-Hoc

While pretraining has well-established scaling laws (Kaplan, Chinchilla), RL scaling for LLMs remains **an art**:

- RL compute has grown massively: DeepSeek-R1-Zero used 100K H800 GPU-hours for RL (3.75% of pretraining), o1→o3 shows >10× RL compute increase
- No principled way to evaluate algorithmic improvements for scaling
- Most RL papers provide ad-hoc solutions for specific contexts, not general scaling methodology
- Academic researchers are sidelined — can't run large-scale RL experiments

## The Sigmoidal Scaling Framework

ScaleRL models RL performance vs compute as a sigmoid:

```
RC - R0 = (A - R0) × 1 / (1 + (Cmid / C)^B)
```

| Parameter | Meaning | What affects it |
|-----------|---------|----------------|
| **A** | Asymptotic pass rate (performance ceiling) | Algorithm quality, precision, reward design |
| **B** | Scaling exponent (compute efficiency) | Loss aggregation, normalization, curriculum, off-policyness |
| **Cmid** | Midpoint (how much compute to get halfway to A) | Pipeline efficiency, batch size |
| **R0** | Initial performance (before RL) | Base model quality |

**Key insight**: A (asymptote) and B (efficiency) are **decoupled** — different design choices affect different parameters. This means you can separately optimize for "how good" (A) and "how fast" (B).

## Three Key Findings

### 1. Not All Recipes Yield the Same Asymptote

ScaleRL achieves A=0.61 asymptotic pass rate on math, significantly higher than:
- GRPO/DeepSeek-style: lower A
- DAPO/Qwen-style: moderate A
- MiniMax recipe: ~0.50

The asymptote is **not** determined by compute alone — algorithm quality matters for the ceiling.

### 2. Compute Efficiency is Modulated by "Details"

Design choices like loss aggregation, normalization, curriculum, and off-policy algorithm primarily affect **B** (compute efficiency), not A. They determine how fast you reach the asymptote, not how high it is.

### 3. Stable Recipes Follow Predictable Trajectories

ScaleRL's sigmoid fits from 50K GPU-hours accurately predicted performance at 100K GPU-hours. This enables the same kind of extrapolation that pretraining scaling laws provide.

## The Full ScaleRL Recipe

ScaleRL combines existing methods into a single predictable, scalable objective:

```
J_ScaleRL(θ) = E[ 1/Σ|yg| · Σ_i Σ_t sg(min(ρ_i,t, ε)) · Ânorm_i · log π_train(y_i,t) ]

Constraints: 0 < mean({r_j}) < 1, pass_rate(x) < 0.9
```

Where:
- `ρ_i,t = π_train(y_i,t) / π_oldgen(y_i,t)` — token-level importance sampling ratio
- `Ânorm_i = Â_i / Â_std` — batch-level normalized advantage
- `sg(min(ρ, ε))` — stop-gradient truncated IS weight (CISPO)
- `pass_rate(x) < 0.9` — No-Positive-Resampling filter
- Forced interruption phrase: "Okay, time is up. Let me stop thinking and formulate a final answer now.<｜end▁of▁thinking｜>"

| Component | Choice | Why |
|-----------|--------|-----|
| **RL setup** | PipelineRL-8 (async off-policy, k=8) | Tight on-policy feedback loop, higher compute efficiency |
| **Loss function** | CISPO (truncated IS + vanilla PG) | Most stable, hyperparameter-robust |
| **Loss aggregation** | Prompt-average | Highest asymptotic performance |
| **Advantage normalization** | Batch-level | Theoretically sound, marginally best |
| **Precision** | **FP32 at LM head** | +0.09 asymptote — largest single improvement |
| **Length control** | Forced interruption at think budget | Prevents explosion, maintains stability |
| **Filtering** | Zero-variance filtering | Removes samples with no policy gradient signal |
| **Resampling** | No-Positive-Resampling (pass_rate ≥ 0.9 removed) | Prevents overfitting to easy prompts, improves asymptote A |

## Leave-One-Out (LOO) Validation

To validate each component's contribution, the paper runs LOO experiments at 16K GPU-hours: start from ScaleRL, revert one axis at a time to baseline, re-train.

**Key finding**: Most LOO variants reach a similar asymptotic reward A (within ±0.02 error margin). The main difference is in **compute efficiency B** — the cumulative effect of all components is robust. No single component is critical to the asymptote in isolation, but together they provide stability and faster convergence.

To visualize efficiency differences, the paper rearranges the sigmoid equation: `F(Rc) = C^B_mid / (A-R0)/(Rc-R0) - 1`, then plots `log F(Rc)` vs `log C`. This makes the slope B directly visible — ScaleRL achieves the steepest slope.

### Three Independent Runs Variance

To estimate fitting variability, 3 independent ScaleRL runs show ±0.02 error margin for asymptotic performance A.

## Scaling Across Compute Axes

The paper tests whether the sigmoid framework predicts scaling across multiple dimensions:

### Generation Length

| Config | Cmid | B | A |
|--------|------|---|---|
| ScaleRL (12K think budget) | 2542 | 1.92 | 0.610 |
| ScaleRL-32K (longer think) | 11272 | 1.89 | **0.645** |

Longer generation length **slows early progress** (higher Cmid) but **lifts the asymptote A** significantly. This validates long-context RL as a **ceiling-raising knob**, not just an efficiency trade-off. Downstream evaluations confirm the same pattern — longer thinking eventually surpasses shorter thinking.

### Batch Size

| Config | Cmid | B | A |
|--------|------|---|---|
| ScaleRL-bs512 | 2818 | 1.77 | 0.605 |
| ScaleRL (bs768) | 2542 | 1.92 | 0.610 |
| ScaleRL-bs2048 | 10909 | 1.70 | **0.645** |

Larger batch sizes are slower initially (higher Cmid) but reach a higher asymptote. At moderate batch sizes, allocation between bs512 and bs2048 is a **second-order choice** — both A and B differ modestly. Clearer differences may emerge at much larger batches (2K+).

### Number of Generations per Prompt

| Config | A |
|--------|---|
| 8 gen/prompt | 0.585 |
| 16 gen/prompt (default) | 0.610 |
| 24 gen/prompt | 0.590 |
| 32 gen/prompt | 0.595 |

16 generations per prompt appears optimal — more generations don't improve the asymptote significantly and cost more compute.

### Model Scale: MoE

| Config | Cmid | B | A |
|--------|------|---|---|
| ScaleRL 8B dense | 2542 | 1.92 | 0.610 |
| ScaleRL Scout (17B×16 MoE) | 4242 | 1.65 | **0.710** |

The MoE model achieves a much higher asymptote (0.710 vs 0.610) while using only 1/6 of the RL training compute to match the 8B's performance. ScaleRL remains stable on MoE — low truncation rates (<2%), no instability pathologies.

### Multi-Task RL (Math + Code)

Joint training on math and code yields clean, parallel power-law trends for each domain. The math curve tracks closely with the math-only run, while the code curve shows its own scaling trajectory (B=1.09, A=0.615). Extended runs remain aligned with extrapolated curves — the sigmoid framework generalizes beyond single-domain training.

## PipelineRL: Why It Outperforms PPO-Off-Policy

PipelineRL consistently outperforms classic PPO-off-policy. The key difference:

| Approach | Generator-Trainer Relationship |
|----------|------------------------------|
| **PPO-off-policy** | Alternating phases — trainer operates on stale rollouts (k steps behind) |
| **PipelineRL** | Streaming — batches flow immediately to trainer, model updates flow back to generators in real-time |

This tight feedback loop keeps training closer to the on-policy regime, reducing the mismatch between generator and trainer distributions. **This affects the asymptote A**, not just efficiency B — very few design axes shift the asymptote this way.

Off-policyness k=8 found optimal (both k=4 and k=8 performed equally on 8B dense; k=8 adopted as default).

## Training Stability and Truncations

**Truncation thresholds**: At batch size 768, truncations in the 10-15% range typically destabilize training. Extended GRPO runs correlated instability with rising truncation rates.

**ScaleRL stability**:
- 8B model: truncations <5% for >90% of training
- Batch 2048: slightly higher (~7%) due to longer generation lengths — but effective batch size remains large after excluding truncated samples
- 34K generation length (batch 768): truncations briefly spiked to ~4% then fell below 2%
- Scout MoE: truncations <2% consistently, <1% for >90% of steps

Larger models are more robust — likely due to better length regulation and stronger instruction-following.

## Entropy Is Not a Reliable Predictor

An unexpected finding: entropy curves for batch sizes 768 and 2048 were nearly identical per step, despite the 2048-batch run achieving much stronger downstream performance at every stage. This challenges the common practice of using entropy as a proxy for exploration quality. Simply maintaining higher entropy does not translate into better generalization.

## Generalization Insights

While the paper focuses on in-distribution validation for predictive scaling, downstream generalization correlates with certain design choices:

| Choice | Effect on Generalization |
|--------|------------------------|
| Larger batch size | + (significant) |
| Reducing truncations | + (stability → better generalization) |
| Longer generation lengths | + (ceiling-raising) |
| Larger model scale | + (strongest effect) |
| Multi-task training | Generalizes across domains |

## Connections

- [DeepSeek-R1](deepseek-r1.md) — R1-Zero used GRPO (100K H800 GPU-hours). ScaleRL explains why GRPO works (rule-based rewards, group baselines) and where to improve (CISPO, FP32 head, PipelineRL, No-Positive-Resampling)
- [The Bitter Lesson for RL](bitter-lesson-rl.md) — Verification as the key; ScaleRL provides the systematic methodology for scaling verifier-based RL
- [DeepSeek-V3.2](deepseek-v3.2.md) — V3.2's scalable RL framework (>10% pretraining budget) maps to ScaleRL's compute scaling analysis
- [LLM Scaling Laws](llm-scaling-laws.md) — Pretraining scaling laws context — ScaleRL brings similar predictability to RL
- [DeepSeek-V4 Post-Training](deepseek-v4-post-training.md) — V4's GRPO usage and OPD pipeline
