---
title: "Don't Teach. Incentivize."
type: source-note
tags: [research-philosophy, rl, scaling, incentives, deepseek, bitter-lesson]
created: 2026-05-04
updated: 2026-05-04
sources: [raw/dont-teach-incentivize.pdf]
status: stable
---

# Don't Teach. Incentivize.

**Source**: Hyung Won Chung (OpenAI), MIT EI Seminar. 59 slides. A talk on research philosophy through the lens of AI scaling.

## Key Points

- **Don't teach, incentivize**: Rather than explicitly programming or teaching desired behaviors, create the right incentives and let the system discover them
- Focus on finding **great problems**, not just solving existing ones
- **Scaling is the ultimate incentive**: more compute → more data → more parameters → emergent capabilities
- The bitter lesson in practice: general methods that leverage computation win over human-crafted domain knowledge
- Directly connected to the DeepSeek-R1 philosophy: "the key is hard questions, a reliable verifier, and sufficient compute"

## The Core Philosophy

> "We, the technical people, focus too much on problem solving itself. In my view, more attention should go to finding great problems to solve. Great researchers are good at finding impactful problems. I think this ability comes from having the right perspective."

The talk argues that the most impactful AI advances come not from teaching systems explicit rules, but from creating environments where desired behaviors emerge naturally through scale and incentives:

| Approach | Example | Outcome |
|----------|---------|---------|
| **Teach** | Hand-crafted features, explicit rules, SFT on human demonstrations | Limited by human knowledge |
| **Incentivize** | RL with verifiable rewards, scaling compute, emergent behaviors | Can surpass human capabilities |

### The Incentivize Approach in Practice

- **GPT series**: Instead of teaching language rules, incentivize next-token prediction at scale → emergent reasoning, translation, coding
- **DeepSeek-R1-Zero**: Instead of SFT on human reasoning traces, incentivize correct answers via rule-based rewards → emergent self-verification, reflection, "aha moments"
- **AlphaGo/Zero**: Instead of teaching Go strategy, incentivize winning via self-play → discovered moves no human had considered

### Finding Great Problems

The talk emphasizes problem selection over problem solving:

- Work on problems where scaling is the primary bottleneck (not cleverness)
- Look for problems where the right incentive structure exists (a verifiable reward signal)
- Avoid problems where human domain knowledge is required for each improvement

## Connection to Scaling Laws

The "incentivize" philosophy is the operational principle behind scaling laws: if performance improves as a power law with compute, the right strategy is to scale compute, not to hand-engineer better algorithms. This directly echoes Sutton's Bitter Lesson and Hamming's "work on important problems."

## Connections

- [The Bitter Lesson](the-bitter-lesson.md) — Sutton's essay: general methods + computation > human knowledge
- [DeepSeek-R1](deepseek-r1.md) — Pure RL (no SFT) → emergent reasoning behaviors
- [LLM Scaling Laws](llm-scaling-laws.md) — Performance improves predictably with compute
- [You and Your Research](you-and-your-research.md) — Hamming on finding important problems
- [Stabilizing RL with LLMs](stabilizing-rl-llms.md) — Theoretical framework for why token-level RL works
