---
title: "PPO: Proximal Policy Optimization Algorithms"
type: source-note
tags: [reinforcement-learning, ppo, policy-gradient, grpo, llm, rl]
created: 2026-05-04
updated: 2026-05-04
sources: [raw/ppo.pdf]
status: stable
---

# PPO: Proximal Policy Optimization

**Source**: Schulman et al. (OpenAI), 2017. arXiv:1707.06347. The foundational policy gradient algorithm behind modern LLM RL training.

## Key Points

- PPO is a policy gradient method that balances **simplicity, sample efficiency, and stability**
- Introduces **clipped surrogate objective** to prevent destructively large policy updates
- PPO-Clip is the dominant variant — simpler than TRPO, competitive performance, widely adopted
- Foundation for GRPO, DAPO, CISPO, and all modern LLM RL algorithms
- Used in DeepSeek-R1, ScaleRL, GPT-4 RLHF, and virtually all production RL pipelines

## The Problem PPO Solves

Policy gradient methods have high variance and can take destructively large steps, collapsing performance. Trust Region Policy Optimization (TRPO) solved this with a constrained optimization problem (KL divergence constraint), but TRPO is complex — requires second-order optimization, conjugate gradient, line search.

PPO simplifies TRPO's constraint into the **loss function itself** via clipping.

## PPO-Clip: The Dominant Variant

The core idea: clip the probability ratio so the policy doesn't change too much in a single update.

**Probability ratio** (importance sampling):
$$
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}
$$

**Clipped surrogate objective**:
$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) A_t,\; \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon) A_t \right) \right]
$$

Where $A_t$ is the advantage estimate and $\varepsilon$ is the clipping parameter (typically 0.2).

**Two cases**:
- **Positive advantage** ($A_t > 0$): The action was good. Increase its probability, but cap the ratio at $1 + \varepsilon$ (don't push too hard).
- **Negative advantage** ($A_t < 0$): The action was bad. Decrease its probability, but floor the ratio at $1 - \varepsilon$ (don't punish too hard).

The $\min$ ensures we never benefit from moving the ratio outside the clip range — removing the incentive for destructively large updates.

## PPO-Penalty (Alternative)

Uses a KL penalty instead of clipping:
$$
L^{\text{KLPEN}}(\theta) = \mathbb{E}_t \left[ r_t(\theta) A_t - \beta \cdot \text{KL}(\pi_{\theta_{\text{old}}} \parallel \pi_\theta) \right]
$$

Where $\beta$ is adaptively adjusted based on the observed KL divergence. Simpler conceptually but less popular in practice — PPO-Clip dominates.

## Advantage Estimation

PPO uses **Generalized Advantage Estimation (GAE)**:
$$
A_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^\infty (\gamma\lambda)^l \delta_{t+l}
$$
where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$.

GAE balances bias-variance trade-off: $\lambda = 0$ = TD(0) (low variance, high bias), $\lambda = 1$ = Monte Carlo (high variance, low bias). Typical: $\lambda = 0.95$.

## PPO in LLM Training

PPO is the foundation for RLHF (Reinforcement Learning from Human Feedback) and reasoning RL:

| RLHF Component | PPO Role |
|---------------|----------|
| **Policy model** | The LLM being optimized (actor) |
| **Value model** | Critic predicting expected reward (used for advantage) |
| **Reward model** | Provides scalar reward signal |
| **Reference model** | KL baseline to prevent reward hacking |

### GRPO: PPO Without the Critic

**Group Relative Policy Optimization (GRPO)** — used in DeepSeek-R1 — replaces PPO's learned value function (critic) with group-based advantage estimation:

Instead of $A_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ (PPO with critic), GRPO computes:
$$A_i = \frac{r_i - \text{mean}(\{r_1, \dots, r_G\})}{\text{std}(\{r_1, \dots, r_G\})}$$

This eliminates the critic model entirely — removing a major source of complexity and potential reward hacking.

### Key PPO Parameters for LLMs

| Parameter | Typical Range | Effect |
|-----------|--------------|--------|
| $\varepsilon$ (clip) | 0.1 — 0.3 | Larger = more aggressive updates |
| $\beta$ (KL penalty) | 0.01 — 0.1 | Larger = stay closer to reference |
| $\gamma$ (discount) | 0.95 — 0.99 | Higher = longer horizon |
| $\lambda$ (GAE) | 0.9 — 0.98 | Higher = more Monte Carlo |
| Batch size | 64 — 512 prompts | Larger = more stable, more compute |
| Mini-batch size | 4 — 32 | Smaller = more updates per rollout |

## Connections

- [DeepSeek-R1](deepseek-r1.md) — GRPO is PPO without the critic
- [ScaleRL](scalerl.md) — Systematic study of PPO variants for LLM RL scaling
- [The Bitter Lesson for RL](bitter-lesson-rl.md) — Why RL with verifiable rewards works
- [DeepSeek-V4 Post-Training](deepseek-v4.md) — GRPO used in specialist training
- [LLM Scaling Laws](llm-scaling-laws.md) — Why scaling compute matters for RL too
