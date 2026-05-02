---
title: "LLM Scaling Laws: From GPT-3 to the Plateau"
type: source-note
tags: [scaling-laws, llm, pretraining, chinchilla, kaplan, power-law, compute-optimal]
created: 2026-05-02
updated: 2026-05-02
sources: [https://cameronrwolfe.substack.com/p/llm-scaling-laws]
status: stable
---

# LLM Scaling Laws

**Source**: Cameron R. Wolfe, Ph.D. — Deep (Learning) Focus Substack, January 2025. Comprehensive overview article.

## Key Points

- **Power laws are the foundation**: LLM test loss decreases as a power law with respect to compute, model size, and data size
- **Kaplan et al. (2020)**: OpenAI's original scaling laws — prioritize increasing model size over data (N^0.74 for data)
- **Chinchilla (2022)**: Compute-optimal training — scale model size and data **equally**. Most models before Chinchilla were undertrained
- **Scaling is exponential decay, not exponential growth**: Each unit of improvement costs more than the last — diminishing returns
- **Data wall**: The internet is finite — web-scraped data may not suffice for continued scaling
- **RL scaling laws** are emerging but less standardized than pretraining scaling laws
- **"Plateau"**: Recent models show diminishing benefits from pure pretraining scale — shifting focus to reasoning, RL, and test-time compute

## The Mechanics of Scaling Laws

### Power Law Foundation

A scaling law is a power-law relationship between an LLM's test loss (L) and some quantity of interest (X):

```
L(X) = a · X^(-b) + L_∞
```

Where `a` and `b` are fitted constants, and `L_∞` is the irreducible loss (entropy of the data distribution).

**Key quantities**:
| Symbol | Quantity |
|--------|----------|
| N | Number of model parameters |
| D | Dataset size (tokens) |
| C | Training compute (FLOPs) |

The relationship: `C ≈ 6ND` — compute is approximately proportional to parameters × data.

### Two Landmark Papers

**Kaplan et al. (2020)** — OpenAI's original scaling laws:
- For a fixed compute budget, prioritize increasing model size
- Data should scale as `D ∝ N^0.74` (slower than parameters)
- Led to models like GPT-3 (175B params on ~300B tokens)
- Model size is the primary driver of performance

**Chinchilla / Hoffmann et al. (2022)** — DeepMind's correction:
- **Compute-optimal**: Model size and data should scale **equally** (`N ∝ D`)
- Most models were significantly undertrained — needed more data, not more parameters
- A 70B model trained on 1.4T tokens matches a 175B model on 300B tokens
- Chinchilla (70B) matched Gopher (280B) with 4× fewer parameters but 4× more data

**The Chinchilla insight reshaped the field**: post-2022, models use more data relative to their size (e.g., Llama 3 8B on 15T tokens, DeepSeek-V3 671B on 14.8T tokens).

## Common Misconceptions

1. **"Exponential improvement from logarithmic compute"**: FALSE. Scaling laws are exponential decay — each unit of improvement requires more compute than the previous one. Diminishing returns are baked in.

2. **"Bigger models are always better"**: FALSE. A model must be matched to its data budget. Overtrained small models can beat undertrained large ones (Chinchilla vs Gopher).

3. **"Scaling laws predict capabilities"**: PARTIALLY FALSE. They predict test loss, not emergent capabilities. Test loss correlates with downstream performance but doesn't guarantee specific abilities.

## Data: The Limiting Factor

The Chinchilla laws imply we need exponentially more data for each improvement. But:

- Most pretraining data comes from web scraping
- The internet is finite (~100-200T tokens of usable text)
- Synthetic data can help but has quality and diversity challenges
- This "data wall" is cited as one reason for the scaling plateau

## The Scaling Plateau and Beyond

Recent observations (2024-2025):
- Pure pretraining scaling shows diminishing returns
- Next-token prediction loss improvements are smaller per unit of compute
- Shifting focus to: reasoning (R1, o1), RL post-training, test-time compute scaling, data quality over quantity

**Ilya Sutskever's framing**: "The age of pretraining is ending." The field is transitioning from scaling pretraining compute to scaling inference-time compute and RL training.

## RL Scaling Laws

A newer, less standardized area:
- RL scaling laws are more bespoke than pretraining — depend heavily on reward design, prompt distribution, algorithm choice
- Emerging work shows smooth capability improvements with RL compute scale (R1, V3.2)
- Fundamental difference: pretraining scaling predicts loss; RL scaling predicts task performance directly
- Still early — no single standardized approach like Chinchilla for RL

## Connections to Wiki Content

This article provides the theoretical foundation for the scaling decisions documented in:

- **[DeepSeek-V3](deepseek-v3.md)**: 671B params, 14.8T tokens — roughly Chinchilla-optimal (parameters:tokens ≈ 1:22)
- **[DeepSeek-V4](deepseek-v4-technical-report.md)**: 1.6T params, 33T tokens — pushing data scaling further
- **[DeepSeek-R1](deepseek-r1.md)**: Example of RL scaling laws in practice — emergent reasoning from RL compute
- **[Ultra-Scale Playbook](ultra-scale-playbook.md)**: The engineering techniques needed to train at Chinchilla scale
- **[Megatron-Core MoE](megatron-core-moe-scalable-training.md)**: Training infrastructure for scaling to trillions of params
