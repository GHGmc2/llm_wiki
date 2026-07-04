---
title: "Scaling Laws for Autoregressive Generative Modeling"
type: source-note
tags: [scaling-laws, autoregressive, cross-domain, openai, power-law, kl-divergence, compute-optimal, multimodality]
created: 2026-05-04
updated: 2026-05-04
sources: [https://arxiv.org/abs/2010.14701, raw/cross-domain-scaling-laws.pdf]
status: stable
---

# Cross-Domain Scaling Laws

**Source**: Henighan, Kaplan, Katz, Chen, et al. (OpenAI), Oct 2020. arXiv:2010.14701. Extends scaling laws beyond text to images, video, multimodal, and mathematical problem solving. Finds nearly universal power-law exponents across domains.

## Key Points

- Scaling laws hold across **four domains**: images, video, multimodal (image↔text), and math — not just language
- **Nearly universal exponents**: the power-law exponents for optimal model size vs compute are similar across all domains
- Cross-entropy decomposition: $L = S(\text{True}) + D_{\text{KL}}(\text{True} \parallel \text{Model})$ — separates irreducible entropy from model inadequacy
- Billion-parameter Transformers are **nearly perfect models** of YFCC100M at 8×8 resolution
- Mutual information scaling: quantifies "how much is a picture worth in words"
- Math: identifies scaling laws for **out-of-distribution** extrapolation — performance continues to improve beyond training distribution
- Classification fine-tuning scales smoothly even as generative loss plateaus
- 33 figures, covering scaling across all major modalities at the time

![Image VQ16 full loss decomposition — test loss (colored) with fitted curve showing irreducible loss asymptote at 4.09 nats](../../raw/assets/cross_scaling_ImageVQ16ComputeFullLoss.png)

## The Universal Scaling Law

Across all domains, test loss follows:
$$L(C) = \left(\frac{C_0}{C}\right)^\alpha + L_\infty$$

Where:
- $C$ = compute (PF-days)
- $C_0$, $\alpha$ = fitted constants
- $L_\infty$ = irreducible loss (entropy of the true distribution)
- $\alpha$ is **nearly universal** across domains

## Domain-Specific Findings

### Images (YFCC100M)

- At resolution 8×8, a 1B-parameter Transformer nearly achieves the irreducible loss (KL ≈ 0)
- At 32×32, models need to be significantly larger to approach irreducibility
- Can forecast model size needed for any target KL divergence at any resolution
- Cross-entropy loss tracks human-perceived image quality

### Video

- Similar power-law scaling as images
- Longer videos require more compute but follow the same functional form

### Multimodal (Image↔Text)

- Mutual information between captions and images follows a scaling law
- Quantifies "Is a picture worth a thousand words?" — computes the information-theoretic answer
- Joint text-image models improve predictably with scale

### Mathematical Problem Solving

- Scaling laws hold even when **extrapolating beyond the training distribution**
- Math problem accuracy improves smoothly with compute despite training only on specific problem distributions
- Suggests generalization ability scales predictably

## Information-Theoretic Decomposition

The cross-entropy loss decomposes as:
$$L = \underbrace{S(\text{True})}_{\text{irreducible entropy}} + \underbrace{D_{\text{KL}}(\text{True} \parallel \text{Model})}_{\text{reducible (model inadequacy)}}$$

This gives a principled way to:
- **Estimate the entropy** of the true data distribution (even without access to it)
- **Forecast** how large a model must be to reach any given KL divergence
- **Separate** data-limited from model-limited regimes

### Practical Implication

For YFCC100M at 8×8: $D_{\text{KL}} \approx 0$ at 1B params → the model has "solved" this resolution. At higher resolutions, more model capacity (or more data) is needed. This framework predicts exactly how much.

## Relationship to Kaplan et al. (2020)

| Aspect | Kaplan (2020) | Henighan (2020) |
|--------|--------------|-----------------|
| Domain | Text only | Text + Images + Video + Multimodal + Math |
| Key finding | $L \propto N^{-\alpha}$ for text | Same functional form holds across all domains |
| Universality | — | Nearly universal exponents across domains |
| Decomposition | No formal decomposition | Information-theoretic: $L = S + D_{KL}$ |
| Downstream tasks | Loss only | Shows classification fine-tuning also scales |

These two papers together establish scaling laws as a **general phenomenon** of autoregressive modeling, not specific to language.

## Connections

- [LLM Scaling Laws](llm-scaling-laws.md) — Kaplan et al. (2020), the text-only precursor; this paper generalizes those findings
- [ScaleRL](scalerl.md) — extends scaling law methodology to RL; the sigmoidal curve replaces the power law for bounded metrics
- [Muon Optimizer](muon-optimizer.md) — the ~2× efficiency claim is based on scaling law methodology pioneered here
- [DeepSeek-V4](deepseek-v4.md) — V4's training decisions are informed by scaling law predictions
