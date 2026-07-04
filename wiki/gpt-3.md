---
title: "Language Models are Few-Shot Learners (GPT-3)"
type: source-note
tags: [gpt, openai, pretraining, transformer, language-model, few-shot, scaling, in-context-learning]
created: 2026-05-04
updated: 2026-05-04
sources: [https://arxiv.org/abs/2005.14165, raw/gpt-3.pdf]
status: stable
---

# GPT-3: Few-Shot Learners

**Source**: Brown, Mann, Ryder, Subbiah, Kaplan, Dhariwal, et al. (OpenAI), 2020. arXiv:2005.14165. The paper that demonstrated 175B-parameter language models can perform tasks via in-context few-shot learning — no gradient updates, just examples in the prompt.

## Key Points

- **175B parameter** Transformer, trained on ~300B tokens from Common Crawl, WebText2, Books, Wikipedia
- Introduces **in-context learning**: provide a few examples (shots) in the prompt, the model completes the task without weight updates
- Evaluated in three settings: **zero-shot** (task description only), **one-shot** (one example), **few-shot** (10-100 examples)
- Performance improves smoothly with model size — few-shot GPT-3 approaches or beats fine-tuned SOTA on many tasks
- **96 layers**, d_model=12288, 96 heads, context=2048

![GPT-3 training curves — loss decreases predictably with compute across 8 model sizes from 125M to 175B](../../raw/assets/gpt3_training_curves.png)
- Strong on: translation, QA, cloze tasks, common sense reasoning, reading comprehension
- Weak on: natural language inference (comparing two sentences), some reading comprehension datasets
- Demonstrates that scaling model size is a viable path to general-purpose AI

## Architecture

| Component | Specification |
|-----------|--------------|
| Layers | 96 |
| d_model | 12,288 |
| Attention heads | 96 |
| FFN dimension | 49,152 |
| Parameters | 175B |
| Context window | 2048 |
| Training tokens | ~300B (filtered) |
| Training data | Common Crawl (60%), WebText2 (22%), Books (8%), Wikipedia (3%) |
| Batch size | 3.2M tokens |

### Sparse Attention Variants

GPT-3 introduces alternating dense and sparse attention patterns (similar to Sparse Transformers):
- Alternating dense and locally banded attention layers
- Matches dense performance while reducing compute by ~70% for long sequences

## In-Context Learning

Unlike fine-tuning which requires gradient updates, GPT-3 learns from examples provided directly in the prompt:

```
Translate English to French:
sea otter → loutre de mer
cheese → fromage
hello → [model completes: bonjour]
```

This is **in-context few-shot learning** — the model uses the provided examples to infer the task without parameter updates. The "learning" happens during the forward pass through the attention mechanism, which effectively implements a form of meta-learning.

## Scaling Behavior

GPT-3 was evaluated across 8 model sizes (125M → 175B). Key observations:
- **Smooth, predictable improvement** with scale on most tasks
- Many tasks show a **phase transition** where capabilities emerge suddenly at larger scales
- **Few-shot outperforms one-shot, which outperforms zero-shot** — the gap narrows with scale
- For some tasks (TriviaQA), the few-shot gap to fine-tuned SOTA nearly closes at 175B

## Limitations

- Weak at text synthesis requiring logical coherence across long passages
- Repetition and contradiction at length
- Natural language inference (ANLI) remains near random even at 175B
- Biases inherited from web-scale training data

## Connections

- [GPT-1](gpt-1.md) — introduced generative pre-training (117M, required fine-tuning)
- [GPT-2](gpt-2.md) — zero-shot transfer without fine-tuning (1.5B)
- [GPT-4](gpt-4.md) — multimodal, improved reasoning, RLHF post-training
- [InstructGPT](instruct-gpt.md) — RLHF to align GPT-3 with human instructions
- [LLM Scaling Laws](llm-scaling-laws.md) — the scaling principles GPT-3 embodies; Kaplan et al. (same authors) formalized the theory
- [The Bitter Lesson](the-bitter-lesson.md) — scale + general methods beats hand-crafted domain knowledge
