---
title: "The Bitter Lesson for RL: Verification as the Key to Reasoning LLMs"
type: source-note
tags: [reinforcement-learning, verification, reasoning, bitter-lesson, generative-verifiers, test-time-compute]
created: 2026-05-02
updated: 2026-05-02
sources: [raw/the-bitter-lesson-for-rl.pdf]
status: stable
---

# The Bitter Lesson for RL

**Source**: Rishabh Agarwal (Periodic Labs). 34-page slide deck. Applies Sutton's "Bitter Lesson" to RL for LLMs.

## Key Points

- **Sutton's Bitter Lesson applied to RL**: General methods that leverage computation are ultimately most effective — **verification** is that general method for reasoning
- **Verification scales**: Rule-based (math, code) and generative verifiers both improve with more compute
- **Generative verifiers** pose verification as a reasoning problem — the verifier is itself an LLM that checks the answer
- **Test-time compute** is the new scaling frontier: more thinking = better answers
- **Easy-to-hard generalization**: Generative verifiers trained on easy problems generalize to harder ones

## The Bitter Lesson

Richard Sutton's 2019 essay: "The biggest lesson from 70 years of AI research is that **general methods that leverage computation** are ultimately the most effective, and by a large margin."

The two methods that scale arbitrarily: **search** and **learning**.

Applied to LLM reasoning:
- **Search** → test-time compute, chain-of-thought, tree search
- **Learning** → RL with verifiable rewards, self-play

### Compute Growth

- Training compute of frontier models grows 4-5$\times$ per year
- Roughly 10$\times$ compute for fixed cost every 5 years
- Methods that scale with compute are the methods that win

## Verification: The Scalable Method for Reasoning

### Rule-Based Verifiers

For domains with ground truth:
- **Math**: boxed answer verification (DeepSeek-R1, ScaleRL)
- **Code**: test case pass/fail (Codeforces, SWE-bench)
- **Formal proofs**: Lean/Isabelle proof checking

These are the simplest, most reliable verifiers. They're what enabled DeepSeek-R1-Zero's emergent reasoning.

### Generative Verifiers

For domains without ground truth, train an LLM to verify:

```
Input: (question, candidate_answer)
Output: "correct" or "incorrect" + reasoning trace
```

**Key insight**: Generative verifiers generalize from **easy to hard** problems. A verifier trained on easy math problems can verify solutions to competition-level problems.

**Examples of generative verifiers**:
- OpenAI's deliberative alignment: reasoning enables safer models
- Seed1.5-Thinking: verifier-guided RL training
- Generative Verifiers (Zhang et al., ICLR 2025): next-token prediction as reward modeling

### The Bottleneck

The quality of the generative verifier limits everything. A poor verifier provides noisy reward signals → RL doesn't converge well. Improving verifier quality is the highest-leverage investment.

### Challenges

- **Reward underspecification**: Getting the right answer with wrong reasoning — the verifier may approve incorrect reasoning that happens to produce the correct output
- **Generalization beyond training domains**: Verifiers trained on math may not transfer to coding or writing

## Test-Time Compute Scaling

The bitter lesson's search component: more thinking = better answers.

- OpenAI o1/o3: scaled chain-of-thought during inference
- DeepSeek-R1: "aha moment" — model learns to think longer spontaneously
- DeepSeek-V4 Think Max: explicit system prompt demanding exhaustive reasoning

Test-time compute is the **fastest-growing** compute allocation in frontier models: o1→o3 shows >10$\times$ increase.

## Summary: What Scales

| Method | Scales With | Maturity |
|--------|------------|----------|
| Rule-based verification | Compute (more verifier queries) | Mature |
| Generative verification | Compute (larger verifier models) | Emerging |
| Test-time compute (CoT) | Compute (longer thinking) | Mainstream |
| RL from verifiable rewards | Compute (more RL steps) | Maturing |
| Self-play | Compute (more games) | Early |

## Connections

- [ScaleRL](scalerl.md) — Systematic framework for RL compute scaling (the "learning" side)
- [DeepSeek-R1](deepseek-r1.md) — Pure RL with rule-based verifiers (the canonical example)
- [LLM Scaling Laws](llm-scaling-laws.md) — Pretraining scaling context and the plateau
- [DeepSeek-V4 Post-Training](deepseek-v4.md) — V4's verifier-dependent post-training pipeline
