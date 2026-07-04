---
title: "TRPO: Trust Region Policy Optimization"
type: source-note
tags: [reinforcement-learning, trpo, policy-gradient, trust-region, ppo, foundation]
created: 2026-05-04
updated: 2026-05-04
sources: [raw/trpo.pdf]
status: stable
---

# TRPO: Trust Region Policy Optimization

**Source**: Schulman, Levine, Abbeel, Jordan, Moritz (Berkeley), 2015. arXiv:1502.05477. The paper that introduced trust region constraints to policy gradients — the foundation PPO simplified.

## Key Points

- Introduces **trust region constraint** for policy updates: ensures each update doesn't move the policy too far
- Uses **KL divergence** as the trust region metric between old and new policies
- Formulated as a **constrained optimization** problem, solved via conjugate gradient + line search
- Foundation for PPO (which replaced the complex constrained optimization with simple clipping)
- The first algorithm to reliably stabilize policy gradient training at scale

## The Problem TRPO Solves

Vanilla policy gradient methods can take **destructively large steps** — a single bad update can permanently degrade performance. The core issue: policy gradient steps in parameter space don't correspond to predictable changes in policy behavior. A small parameter change can cause a large change in action probabilities.

TRPO's solution: constrain the **policy change** (not the parameter change) using KL divergence:

$$
\text{maximize}_\theta \; \mathbb{E}\left[ \frac{\pi_\theta(a \mid s)}{\pi_{\theta_{\text{old}}}(a \mid s)} A(s, a) \right]
$$

subject to:

$$
\mathbb{E}\left[ \text{KL}(\pi_{\theta_{\text{old}}} \parallel \pi_\theta) \right] \leq \delta
$$

Where $\delta$ is the trust region size — typically a small constant.

### The KL Constraint

KL divergence measures how much the new policy differs from the old one. By bounding this, TRPO ensures each update is a **local improvement** — the policy cannot change so much that it collapses.

Key properties:
- KL is invariant to parameter reparameterization — fixes the "small parameter change = large policy change" problem
- Provides theoretical monotonic improvement guarantees under certain conditions
- But: requires second-order optimization (Fisher information matrix, conjugate gradient, line search)

## Why TRPO Was Replaced

TRPO is theoretically elegant but **practically complex**:

1. **Conjugate gradient**: Must solve a linear system at each iteration — expensive for large models
2. **Line search**: Must verify the KL constraint after each proposed step — adds overhead
3. **Second-order information**: Requires computing (or approximating) the Fisher information matrix
4. **Implementation complexity**: Difficult to implement correctly, hard to debug

**PPO** (2017) replaced all of this with a **clipped surrogate objective** — achieving similar stability with a fraction of the complexity. PPO turned TRPO's constrained optimization into an unconstrained optimization with a clever clipping trick.

**GRPO** (2024) further simplified by eliminating the value model entirely, using group-based advantage estimation.

## The Evolution

| Algorithm | Year | Key Innovation | Complexity |
|-----------|------|---------------|------------|
| **TRPO** | 2015 | KL-constrained policy updates, monotonic improvement theory | High (conjugate gradient + line search) |
| **PPO** | 2017 | Clipped surrogate objective — simplifies TRPO's constraint | Medium (value model needed) |
| **GRPO** | 2024 | Group-based advantage — eliminates value model | Low (no critic, token-level IS) |
| **GSPO** | 2025 | Sequence-level IS + clipping — fixes GRPO's token-level instability | Low (no critic, sequence-level) |

**The trend**: Each generation removes complexity while preserving (or improving) stability. TRPO → PPO: replace KL constraint with clipping. PPO → GRPO: remove value model. GRPO → GSPO: fix token-level mismatch with sequence-level IS.

## Relevance to LLM RL

While TRPO itself is not used in LLM training (too computationally expensive at LLM scale), its core insight — **constrain policy updates to a trust region** — lives on in every modern RL algorithm:

- PPO's clipping is TRPO's KL constraint in loss-function form
- GRPO's group-based advantage is TRPO's baseline estimation without a value model
- GSPO's sequence-level clipping extends TRPO's idea of matching the constraint scope to the optimization unit

## Connections

- [PPO](ppo.md) — TRPO simplified: clipping replaces constrained optimization
- [GRPO](grpo.md) — Further simplified: group advantage replaces value model
- [GSPO](gspo.md) — Sequence-level: matching optimization unit to reward unit
