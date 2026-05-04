---
title: "Stabilizing Reinforcement Learning with LLMs: Formulation and Practices"
type: source-note
tags: [reinforcement-learning, llm, policy-gradient, stability, importance-sampling, routing-replay, moe, qwen]
created: 2026-05-04
updated: 2026-05-04
sources: [raw/stabilizing-rl-llms.pdf]
status: stable
---

# Stabilizing RL with LLMs

**Source**: Qwen Team (Alibaba), Dec 2025. 13 pages. arXiv:2512.01374. 30B MoE experiments across hundreds of thousands of GPU hours.

## Key Points

- **Theoretical framework**: Token-level RL objective is a **first-order approximation** to the true sequence-level objective
- Two conditions for this approximation to be valid: minimize **training-inference discrepancy** AND **policy staleness**
- **On-policy**: Basic REINFORCE + importance sampling is the most stable recipe
- **Off-policy**: Clipping + **Routing Replay** (for MoE) are essential to mitigate staleness
- Once training is stabilized, prolonged optimization yields comparable performance regardless of cold-start initialization

## The Theoretical Framework

### Sequence-Level vs Token-Level

The true objective is sequence-level: optimize the full response's reward:

$$
J_{\text{seq}}(\theta) = \mathbb{E}_{x \sim D, y \sim \mu_{\theta_{\text{old}}}} \left[ \frac{\pi_\theta(y \mid x)}{\mu_{\theta_{\text{old}}}(y \mid x)} R(x, y) \right]
$$

The gradient involves the full sequence likelihood ratio — numerically unstable due to large range and high variance.

### The Token-Level Surrogate

The surrogate token-level objective (REINFORCE + token-level IS):

$$
J_{\text{token}}(\theta) = \mathbb{E}_{x, y \sim \mu_{\theta_{\text{old}}}} \left[ \sum_{t=1}^{|y|} \frac{\pi_\theta(y_t \mid x, y_{1:t-1})}{\mu_{\theta_{\text{old}}}(y_t \mid x, y_{1:t-1})} R(x, y) \right]
$$

**Key insight**: When $\pi_\theta$ is close to $\mu_{\theta_{\text{old}}}$, the sequence-level likelihood ratio decomposes:

$$
\frac{\pi_\theta(y \mid x)}{\mu_{\theta_{\text{old}}}(y \mid x)} = \prod_{t=1}^{\lvert y \rvert} (1 + \delta_t) \approx 1 + \sum_{t=1}^{\lvert y \rvert} \delta_t
$$

This shows the token-level objective is a **first-order approximation** to the sequence-level objective. The approximation is valid only when $\delta_t$ are small — i.e., the policy hasn't moved far from the generation policy.

### Two Conditions for Validity

1. **Training-inference discrepancy must be minimized**: The policy used during training ($\pi_\theta$) and inference ($\mu_{\theta_{\text{old}}}$) must produce similar token distributions. Any discrepancy (e.g., different precision, different decoding) breaks the approximation.

2. **Policy staleness must be low**: When doing multiple gradient updates on the same rollout data (off-policy), $\pi_\theta$ moves further from $\mu_{\theta_{\text{old}}}$, making $\delta_t$ larger and breaking the approximation.

## Practical Recipes

### On-Policy Training (Most Stable)

For on-policy (one gradient update per rollout):
- **Basic REINFORCE + importance sampling correction**: This alone achieves highest training stability
- No clipping needed when staleness is low
- Importance sampling IS inherently part of the token-level objective, not an add-on

### Off-Policy Training (Needs Extra Stabilization)

When splitting a large batch into mini-batches for multiple updates (accelerating convergence):
- **Clipping**: Restrains policy staleness by preventing aggressive updates — keeps $\delta_t$ small
- **Routing Replay** (MoE-specific): Fixes the routed experts during optimization — reduces both training-inference discrepancy AND policy staleness

**Why Routing Replay matters for MoE**: In MoE models, the router determines which experts process each token. If the router changes between generation and training, the expert assignments change, creating a training-inference discrepancy that breaks the first-order approximation. Routing Replay fixes the experts to those used during generation, eliminating this source of instability.

### Cold-Start Initialization

A practical finding: once RL training is stabilized, models with different cold-start initializations (SFT vs no SFT, different SFT data, etc.) **consistently achieve comparable final performance** given prolonged RL training. This motivates focusing more on RL stability than on cold-start quality.

## Empirical Validation

Experiments with a 30B MoE model across hundreds of thousands of GPU hours:

| Setting | Key Finding |
|---------|------------|
| **On-policy** | REINFORCE + IS alone is most stable; clipping and Routing Replay help but aren't necessary |
| **Off-policy** | Clipping + Routing Replay are **necessary** — without them, training diverges |
| **Cold-start** | Different initialization methods converge to comparable final performance with prolonged stable RL |
| **MoE vs Dense** | MoE models require Routing Replay for stability under off-policy updates; dense models don't have this issue |

## Connections

- [PPO](ppo.md) — PPO's clipping mechanism formalized as staleness mitigation
- [GRPO / DeepSeek-R1](deepseek-r1.md) — GRPO uses group-based advantage instead of token-level IS; this paper explains why token-level IS is already part of the objective
- [ScaleRL](scalerl.md) — CISPO loss, PipelineRL, and the asymptotic performance framework — this paper provides theoretical grounding for why those design choices matter
- [CISPO](scalerl.md) — Truncated importance sampling directly addresses policy staleness per this paper's framework
