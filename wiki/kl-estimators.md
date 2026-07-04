---
title: "Choosing KL Estimators in RL: From Value Unbiasedness to Gradient Correctness"
type: source-note
tags: [reinforcement-learning, kl-divergence, ppo, grpo, rlhf, importance-sampling, gradient-analysis, off-policy]
created: 2026-05-04
updated: 2026-05-04
sources: [https://xihuai18.github.io/reinforcement-learning/2025/12/01/kl-estimators-en.html]
status: stable
---

# KL Estimators in RL

**Source**: Xihuai Leo Wang, Dec 2025. A rigorous analysis of three KL divergence estimators ($k_1, k_2, k_3$) in reinforcement learning, focusing on gradient correctness rather than value estimation accuracy.

## Key Points

- Three KL estimators: $k_1 = -\log\frac{p}{q}$, $k_2 = \frac{1}{2}(\log\frac{p}{q})^2$, $k_3 = \frac{p}{q} - 1 - \log\frac{p}{q}$
- **Value unbiasedness ≠ gradient correctness**: an estimator that accurately estimates KL values may optimize the wrong objective when backpropagated

![DeepSeek-V3.2 technical report — using $\rho k_3 = \frac{q_\theta}{\mu} k_3$ for unbiased KL estimation with correct reverse-KL gradient](../../raw/assets/kl_est_deepseek.png)

![Reverse KL (mode-seeking) — the policy concentrates on high-probability regions of the reference, sacrificing diversity](../../raw/assets/kl_est_reverse.png)

![Forward KL (mass-covering) — the policy tries to cover the entire support of the reference distribution](../../raw/assets/kl_est_forward.png)

- **On-policy naive implementation**: only $k_2$ gives correct reverse-KL gradients; $k_1$ has zero gradient, $k_3$ optimizes forward KL
- **With explicit $\rho$ construction**: $\rho k_3$ and $\text{sg}(\rho) k_2$ are equivalent, both lower variance than $\rho k_1$
- **As reward shaping (detached)**: only $k_1$ keeps the policy-gradient term aligned with reverse-KL regularization
- DeepSeek-V3.2 uses $\frac{q_\theta}{\mu} k_3$ for unbiased KL estimation with correct gradient

## The Three Estimators

| Estimator | Definition | Design Principle |
|-----------|-----------|-----------------|
| $k_1$ | $-\log\frac{p}{q_\theta}$ | Naive log-ratio |
| $k_2$ | $\frac{1}{2}(\log\frac{p}{q_\theta})^2$ | Local second-order KL surrogate |
| $k_3$ | $\frac{p}{q_\theta} - 1 - \log\frac{p}{q_\theta}$ | Control variate + Bregman divergence |

**Value estimation properties** (on-policy, estimating reverse KL $D_{\text{KL}}(q_\theta \parallel p)$):

| Estimator | Bias | Variance | Notes |
|-----------|------|----------|-------|
| $k_1$ | Unbiased | High (can be ±) | Direct definition |
| $k_2$ | Biased (small near KL=0) | Low (always +) | Square always non-negative |
| $k_3$ | Unbiased | Low (always +) | Best at value estimation |

## Gradient Correctness: The Key Insight

$$\text{KL value estimator} \neq \text{KL optimization objective} \neq \text{gradient returned by code}$$

### Unified $\rho$ Framework

Define $\rho(x) = \frac{q_\theta(x)}{\text{sg}(\mu(x))}$ where $\mu$ is the sampling policy:
- On-policy: $\rho \equiv 1$ numerically, but $\nabla_\theta \rho = s_\theta \neq 0$
- Off-policy: $\rho = \frac{q_\theta}{\mu}$, $\nabla_\theta \rho = \rho \cdot s_\theta$

### Gradient Expectations

Basic gradients of the estimators:
- $\nabla_\theta k_1 = s_\theta$
- $\nabla_\theta k_2 = -\left(\log\frac{p}{q_\theta}\right) s_\theta = k_1 \cdot s_\theta$
- $\nabla_\theta k_3 = \left(1 - \frac{p}{q_\theta}\right) s_\theta$

Under loss $L = \rho \cdot k$:

| Loss | Gradient Random Variable | Expected Gradient | Reverse KL? |
|------|-------------------------|-------------------|-------------|
| $\rho k_1$ | $\rho s_\theta (k_1+1)$ | $\nabla D_{\text{KL}}(q\|p)$ | ✓ (higher variance) |
| $\rho k_2$ | $\rho s_\theta (k_2 + k_1)$ | $\nabla_\theta \mathbb{E}_{q_\theta}[k_2]$ | ✗ (surrogate, not KL) |
| $\text{sg}(\rho) k_2$ | $\rho s_\theta k_1$ | $\nabla D_{\text{KL}}(q\|p)$ | ✓✓ (recommended) |
| $\rho k_3$ | $\rho s_\theta k_1$ | $\nabla D_{\text{KL}}(q\|p)$ | ✓✓ (recommended) |

**Key finding**: $\text{sg}(\rho) k_2$ and $\rho k_3$ produce identical gradient random variables — same mean, variance, and higher-order statistics.

### Naive On-Policy (no explicit $\rho$)

Without constructing $\rho = \frac{q_\theta}{\text{sg}(q_\theta)}$, autograd directly backpropagates through the sample mean:
- $k_1$: $\mathbb{E}[\nabla k_1] = 0$ — **completely ineffective**
- $k_2$: $\mathbb{E}[\nabla k_2] = \nabla D_{\text{KL}}(q\|p)$ — **only correct choice** ✓
- $k_3$: $\mathbb{E}[\nabla k_3] = \nabla D_{\text{KL}}(p\|q)$ — **wrong direction** (forward KL) ✗

### KL as Reward Shaping (detached)

When KL is `.detach()`ed and used as reward shaping:
$$\tilde{R} = R - \beta \cdot \text{sg}(\hat{k})$$

The unbiasedness condition is $\mathbb{E}_{q_\theta}[s_\theta \cdot \hat{k}] = \mathbb{E}_{q_\theta}[s_\theta \cdot k_1]$.

- $k_1$: ✓ unbiased — policy gradient correctly implements reverse-KL regularization
- $k_2$: ✗ biased — $\mathbb{E}[s_\theta \cdot k_2] \neq \mathbb{E}[s_\theta \cdot k_1]$
- $k_3$: ✗ biased — mixes forward-KL gradient bias: $\mathbb{E}[s_\theta \cdot k_3] = -\nabla D_{\text{KL}}(p\|q_\theta) + \nabla D_{\text{KL}}(q_\theta\|p)$

## Quick Reference

| Setting | Recommended | Avoid |
|---------|------------|-------|
| On-policy, KL as loss (naive) | $k_2$ ✓ | $k_1$, $k_3$ |
| On-policy, KL as loss (explicit $\rho$) | $\rho k_3$ or $\text{sg}(\rho) k_2$ | $\rho k_2$ |
| Off-policy, KL as loss | $\rho k_3$ or $\text{sg}(\rho) k_2$ | $\rho k_2$ |
| KL as reward shaping (detached) | $k_1$ only | $k_2$, $k_3$ |

## Connections

- [PPO](ppo.md) — PPO's KL penalty uses these estimators; the clipped surrogate objective was designed to avoid explicit KL computation
- [GRPO / DeepSeek-R1](deepseek-r1.md) — DeepSeek-V3.2 uses $\frac{q_\theta}{\mu} k_3$ for unbiased KL estimation
- [Stabilizing RL with LLMs](stabilizing-rl-llms.md) — token-level importance sampling is deeply connected to which KL estimator to use
- [TRPO](trpo.md) — TRPO's original KL constraint was exactly what $k_2$ approximates locally
