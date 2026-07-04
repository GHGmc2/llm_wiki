---
title: "Training Language Models to Follow Instructions with Human Feedback (InstructGPT)"
type: source-note
tags: [gpt, openai, rlhf, alignment, instruction-tuning, reinforcement-learning, ppo]
created: 2026-05-04
updated: 2026-05-04
sources: [https://arxiv.org/abs/2203.02155, raw/instruct-gpt.pdf]
status: stable
---

# InstructGPT: Aligning Language Models with Human Preferences

**Source**: Ouyang, Wu, Jiang, Almeida, Wainwright, Mishkin, et al. (OpenAI), 2022. arXiv:2203.02155. Introduces Reinforcement Learning from Human Feedback (RLHF) to align GPT-3 with user instructions — the technique behind ChatGPT.

## Key Points

- **RLHF pipeline**: supervised fine-tuning (SFT) → reward model training → PPO-based RL fine-tuning
- **1.3B InstructGPT** outperforms **175B GPT-3** in human preference evaluations
- Human labelers strongly prefer InstructGPT outputs: 85% vs GPT-3 on helpfulness, truthfulness, harmlessness
- **Truthfulness** (TruthfulQA): InstructGPT shows significantly reduced hallucination vs GPT-3
- **Generalization**: improves performance on instructions from held-out labelers (not just the labelers who provided training data)
- **"Alignment tax"**: minimal degradation on public NLP benchmarks despite strong alignment improvements
- Cost: only ~20,000 human preference comparisons needed for effective reward model

![InstructGPT RLHF pipeline — SFT → Reward Model training → PPO optimization loop](../../raw/assets/instructgpt_InstructGPT_Diagram3.1.png)

## The RLHF Pipeline

### Stage 1: Supervised Fine-Tuning (SFT)

- Collect human-written demonstrations of desired behavior (prompt → ideal response)
- Fine-tune GPT-3 on these demonstrations
- Produces the **SFT model** — already better at following instructions than base GPT-3

### Stage 2: Reward Model (RM) Training

- For each prompt, the SFT model generates multiple responses (e.g., 4-9)
- Human labelers **rank** the responses from best to worst
- Train a reward model to predict human preference scores from rankings
- Reward model: 6B parameter GPT-3, trained with pairwise ranking loss:
$$L(\theta) = -\frac{1}{\binom{K}{2}} \mathbb{E}_{(x,y_w,y_l)}[\log \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))]$$
where $y_w$ is the preferred response and $y_l$ is the dispreferred response.

### Stage 3: PPO Fine-Tuning

- Optimize the SFT model against the reward model using Proximal Policy Optimization (PPO)
- Add a **KL penalty** to prevent the policy from diverging too far from the SFT distribution:
$$R(x,y) = r_\theta(x,y) - \beta \cdot \text{KL}(\pi^{\text{RL}}(y \mid x) \parallel \pi^{\text{SFT}}(y \mid x))$$
- This prevents reward over-optimization ("reward hacking") where the model exploits reward model weaknesses

### PPO in RLHF

PPO for language model alignment:
1. Generate completions from the current policy
2. Score completions with the reward model
3. Compute advantages relative to a value baseline
4. Update policy with PPO's clipped surrogate objective
5. Mix in pretraining gradient (PTX loss) to prevent catastrophic forgetting:
$$L_{\text{total}} = L_{\text{PPO}} + \gamma \cdot L_{\text{pretrain}}$$

## Results

| Model | Helpfulness | Truthfulness | Harmlessness |
|-------|-------------|-------------|--------------|
| GPT-3 175B (base) | Baseline | Poor | Poor |
| GPT-3 175B + prompting | Better | Moderate | Moderate |
| SFT 175B | Better | Good | Good |
| PPO 175B | Good | Good | Good |
| PPO-ptx 175B | **Best** | **Best** | **Best** |

Key finding: **1.3B PPO-ptx model preferred over 175B GPT-3** — alignment matters more than raw scale for user satisfaction.

## Significance

InstructGPT established the RLHF methodology that powers ChatGPT, Claude, Gemini, and most modern aligned LLMs. It demonstrated that human preferences can be encoded into a reward model and optimized against, producing models that are both more capable (at following instructions) and more aligned (truthful, harmless).

## Connections

- [GPT-3](gpt-3.md) — the base model InstructGPT aligns
- [GPT-4](gpt-4.md) — incorporates RLHF learnings from InstructGPT
- [PPO](ppo.md) — the RL algorithm at the core of RLHF
- [GRPO / DeepSeek-R1](deepseek-r1.md) — GRPO eliminates the critic model from PPO, a direct evolution of this pipeline
- [Stabilizing RL with LLMs](stabilizing-rl-llms.md) — theoretical framework for RL stability in LLMs
- [The Bitter Lesson for RL](bitter-lesson-rl.md) — RL with verifiable rewards as a general principle
