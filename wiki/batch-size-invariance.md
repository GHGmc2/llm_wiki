---
title: "Batch Size-Invariance for Policy Optimization"
type: source-note
tags: [reinforcement-learning, ppo, batch-size, policy-gradient, importance-sampling, ewma, off-policy, trpo, openai, staleness]
created: 2026-08-06
updated: 2026-08-15
sources: [raw/batch-size-invariance-policy-optimization.pdf, https://arxiv.org/abs/2110.00641]
status: stable
---

# Batch Size-Invariance for Policy Optimization

**Source**: Jacob Hilton, Karl Cobbe, John Schulman (OpenAI), Oct 2021 / NeurIPS 2022. arXiv:2110.00641. 32 pages. [src](../raw/batch-size-invariance-policy-optimization.pdf)

## Key Points

- **Batch size-invariance** ("perfect scaling"): doubling batch size halves the steps needed — behavior preserved as a function of examples processed via formulaic hyperparameter compensation
- **SGD is batch size-invariant up to a critical batch size** (gradient SNR ≈ 1): scale learning rate inversely with batch size
- **PPO is NOT batch size-invariant** — its update control couples batch size to policy age
- **Core insight**: the "old" policy serves TWO independent purposes — off-policy correction (must be the **behavior policy** $\pi_{\theta_{\text{behav}}}$) and update-size control (can be any recent policy — the **proximal policy** $\pi_{\theta_{\text{prox}}}$)
- **Decoupled objectives**: $L^{\text{CLIP}}_{\text{decoupled}}$ separates IS ratio (behav) from clipping (prox)
- **PPO-EWMA / PPG-EWMA**: EWMA of policy weights as proximal policy — batch size-invariant at small batch sizes
- **4 compensating adjustments** when batch size ÷ c: Adam step size ÷ $\sqrt{c}$, EWMA center of mass × c, advantage normalization sample size × c, PPG phases $N_\pi$ × c
- **Practical payoff**: batch size freely chosen by compute constraints (GPU memory); efficient stale-data reuse
- **Experiments (Procgen)**: final-return spread 0.052 across 256× batch range (0.019 without outlier env); artificial staleness shows decoupling is load-bearing

## The Core Insight

PPO's KL-penalized objective uses $\pi_{\theta_{\text{old}}}$ in two roles:

$$L^{\text{KLPEN}}(\theta) = \hat{\mathbb{E}}_t \left[ \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)} \hat{A}_t - \beta \operatorname{KL}\left[\pi_{\theta_{\text{old}}}(\cdot \mid s_t), \pi_\theta(\cdot \mid s_t)\right] \right]$$

1. **Behavior policy** (IS ratio): must be the sampling policy for unbiased gradients
2. **Proximal policy** (KL/update control): only needs to be *recent* — **age matters, identity doesn't**

> ⚠️ The standard justification for trust regions — "don't move far from the behavior policy" — is subtly wrong. The right justification: control update speed, i.e., approximate the natural policy gradient (Kakade 2001).

## Decoupled Objectives

KL-penalized form:

$$L^{\text{KLPEN}}_{\text{decoupled}}(\theta) = \hat{\mathbb{E}}_t \left[ \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{behav}}}(a_t \mid s_t)} \hat{A}_t - \beta \operatorname{KL}\left[\pi_{\theta_{\text{prox}}}(\cdot \mid s_t), \pi_\theta(\cdot \mid s_t)\right] \right]$$

Clipped form (the key trick — rewrite $L^{\text{CLIP}}$ so $\pi_{\theta_{\text{old}}}$'s uses separate):

$$L^{\text{CLIP}}_{\text{decoupled}}(\theta) = \hat{\mathbb{E}}_t \left[ \frac{\pi_{\theta_{\text{prox}}}(a_t \mid s_t)}{\pi_{\theta_{\text{behav}}}(a_t \mid s_t)} \min\left( r_t(\theta) \hat{A}_t,\ \operatorname{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

with $r_t(\theta) := \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{prox}}}(a_t \mid s_t)}$.

Sanity check: $\beta = 0$ or $\epsilon = \infty$ → vanilla IS policy gradient (proximal policy dependence vanishes).

## Why PPO Breaks Batch Size-Invariance

The critical batch size: gradient SNR ≈ 1. Below it, SGD scales via learning rate (Mandt 2017, Goyal 2017, Shallue 2018, McCandlish 2018...).

PPO's problem: changing iteration batch size changes the **age of the behavior AND proximal policies** (coupled in vanilla PPO). Behavior age only matters if too large; but **proximal age controls KL-penalty strength = how fast the policy can change**. Batch-size invariance requires keeping proximal age constant in environment steps.

## PPO-EWMA / PPG-EWMA

Mechanism: maintain an **exponentially-weighted moving average of policy weights**, updated after every gradient step with decay $\beta_{\text{prox}}$; use it as the proximal policy.

- Why EWMA (not a lagged snapshot): storing every intermediate policy is prohibitive; EWMA approximates "policy from K steps ago" cheaply; averaging in parameter space may even improve it (Izmailov 2018); age controlled via $\beta_{\text{prox}}$
- Center of mass of the EWMA (in gradient steps): $\frac{1}{1-\beta_{\text{prox}}} - 1$

### Batch Size-Invariance Recipe (batch size ÷ c)

1. **Adam step size ÷ $\sqrt{c}$** (vanilla SGD: lr ÷ c) — most important adjustment; without it training is highly unstable at small batches
2. **EWMA: multiply $\frac{1}{1-\beta_{\text{prox}}} - 1$ by $c$** — keeps proximal age constant in environment steps
3. **Advantage normalization: multiply EWMA effective sample size by $c$** — prevents standard error blowup; matters most in environments with noisy advantage-variance estimates
4. **PPG: multiply $N_\pi$ (policy iterations per phase) by $c$** — holds phase batch size constant, preserving policy/auxiliary phase dynamics

Constraints: optimization batch size sufficiently small; **number of policy epochs = 1** (multiple epochs at tiny batches = redundant re-training on same data).

## Experiments (Procgen Benchmark)

### Artificial Staleness

Data buffered for a fixed number of iterations before use (mimics async training where staleness breaks on-policy algorithms):
- Vanilla PPO underspecified under staleness: two natural $\pi_{\theta_{\text{old}}}$ choices ($\pi_{\theta_{\text{recent}}}$ vs $\pi_{\theta_{\text{behav}}}$)
- Decoupled objective: proximal = recent, IS = behavior — **performance degrades quickly with stale data unless decoupled**
- Explains why PPO works and enables efficient stale-data reuse

### Batch Size-Invariance Results

- Batch sizes from 256 parallel envs down to 1 (÷4 each step; ÷256 total)
- Final mean-normalized-return spread across full range: **0.052** (0.019 excluding outlier env Heist)
- Ablations: Adam step-size adjustment most important at every batch size; advantage-normalization matters at smallest batch sizes; EWMA adjustment matters little everywhere (PPG robust to KL penalty changes)

## Connections

- [PPO](ppo.md) — the algorithm made batch size-invariant; clipping as update control vs staleness mitigation
- [TRPO](trpo.md) — trust-region lineage; the paper reframes trust regions as update-speed control, not behavior-policy proximity
- [Stabilizing RL with LLMs](stabilizing-rl-llms.md) — stale-data reuse + off-policy corrections; EWMA relates to staleness management
- [Predicting Staleness in Async RL](staleness-in-async-rl.md) — async RL makes behavior/proximal distinction sharper; EWMA is a staleness-robust proximal choice
- [ScaleRL](scalerl.md) — batch size is a key compute lever in LLM RL; invariance vs asymptote-shift tension (see batch-size-invariance.md)
- [GSPO](gspo.md) — sequence-level IS; the behavior/proximal decomposition applies at sequence granularity
- [KL Estimators](kl-estimators.md) — KL against a proximal (EWMA) reference vs behavior policy changes estimator choice
- [GRPO](grpo.md) — GRPO's group/advantage design interacts with batch sizing and update control
- [Async RL Training Landscape](async-rl-training-landscape.md) — staleness management axes; EWMA-style aging in production systems
- [Kimi k1.5](kimi-k1.5.md) — online mirror descent: another way to control update size independent of behavior policy
