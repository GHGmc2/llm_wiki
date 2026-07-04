---
title: "The Bitter Lesson"
type: source-note
tags: [ai, scaling, search, learning, philosophy, rich-sutton, bitter-lesson]
created: 2026-05-03
updated: 2026-05-03
sources: [raw/the-bitter-lesson.pdf]
status: stable
---

# The Bitter Lesson

**Source**: Rich Sutton, March 13, 2019. [Original](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) [pdf](raw/the-bitter-lesson.pdf)

## Key Points

> "The biggest lesson that can be read from 70 years of AI research is that **general methods that leverage computation are ultimately the most effective**, and by a large margin."

- **Search and learning** are the two methods that scale arbitrarily with computation
- Building human knowledge into agents helps short-term but plateaus and inhibits long-term progress
- Moore's law means computation is the only thing that consistently improves over time
- Four case studies: computer chess, Go, speech recognition, computer vision — all followed the same pattern

## The Core Argument

The fundamental tension in AI research: **leveraging human knowledge** vs **leveraging computation**.

Researchers naturally want to encode what they know into their systems — it works in the short term and is personally satisfying. But over time, as computation becomes exponentially cheaper (Moore's law), general methods that simply use more computation consistently win.

> "Seeking an improvement that makes a difference in the shorter term, researchers seek to leverage their human knowledge of the domain, but the only thing that matters in the long run is the leveraging of computation."

## Four Case Studies

### 1. Computer Chess

**1997**: Deep Blue defeats Kasparov using massive deep search. The human-knowledge-based chess community was dismayed — they wanted elegant, human-like play to win. "Brute force" search won instead.

**Pattern**: Early efforts focused on encoding chess expertise. Later, pure search + special hardware proved vastly superior.

### 2. Computer Go

Same pattern, delayed 20 years. Enormous effort went into human-knowledge approaches. All proved irrelevant once search + self-play learning (AlphaGo) was applied at scale.

**Key addition**: Self-play learning amplifies the search paradigm — learning is itself a way to leverage massive computation.

### 3. Speech Recognition

1970s DARPA competition: human-knowledge methods (phonemes, vocal tract models) vs statistical methods (Hidden Markov Models). Statistics won. Decades later, deep learning doubled down: even less human knowledge, even more computation, dramatically better results.

### 4. Computer Vision

Early: edge detection, generalized cylinders, SIFT features — all human-designed. Today: convolutional neural networks using only the notion of convolution. All the hand-crafted features are discarded.

## The Pattern

1. AI researchers try to build knowledge into their agents
2. This always helps in the short term and is personally satisfying
3. In the long run it plateaus and inhibits further progress
4. Breakthrough progress arrives by scaling computation through search and learning

## What to Build Instead

> "We should stop trying to find simple ways to think about the contents of minds... we should build in only the meta-methods that can find and capture this arbitrary complexity."

The actual contents of minds (concepts of space, objects, symmetries) are arbitrarily complex. Don't build them in. Build the methods that can **discover** them. Want AI agents that can discover like we can, not which contain what we have discovered.

## Two Scalable Methods

| Method | How It Leverages Computation |
|--------|------------------------------|
| **Search** | More compute = deeper search trees, more positions evaluated |
| **Learning** | More compute = larger models, more data, more training steps |

These are the **only two methods** that continue to improve as computation increases. Everything else eventually plateaus.

## Connections to Modern AI

- **[The Bitter Lesson for RL](bitter-lesson-rl.md)** — Applies this principle to LLM reasoning: verification + RL scale, human-crafted reasoning patterns don't
- **[LLM Scaling Laws](llm-scaling-laws.md)** — The mathematical formalization: test loss follows a power law with compute
- **[DeepSeek-R1](deepseek-r1.md)** — Pure RL with rule-based verifiers (no human reasoning traces) → emergent reasoning behaviors
- **[ScaleRL](scalerl.md)** — Systematic framework for scaling RL compute predictably

The bitter lesson explains *why* the scaling approach in these papers works. Human knowledge plateaus. Computation doesn't.
