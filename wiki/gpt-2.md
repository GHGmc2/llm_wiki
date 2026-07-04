---
title: "Language Models are Unsupervised Multitask Learners (GPT-2)"
type: source-note
tags: [gpt, openai, pretraining, transformer, language-model, zero-shot, scaling]
created: 2026-05-04
updated: 2026-05-04
sources: [raw/gpt-2.pdf]
status: stable
---

# GPT-2: Unsupervised Multitask Learning

**Source**: Radford, Wu, Child, Luan, Amodei, Sutskever (OpenAI), 2019. Demonstrated that a sufficiently large language model can perform tasks zero-shot — without any task-specific fine-tuning — simply by conditioning on natural language instructions.

## Key Points

- **1.5B parameter** Transformer, trained on WebText (8 million web pages, ~40GB)
- Core claim: language modeling is a form of unsupervised multitask learning — diverse tasks (translation, summarization, QA) emerge from a single objective
- **Zero-shot transfer** without fine-tuning: the model reads a task description in natural language and produces the answer
- Achieved SOTA on 7 of 8 tested language modeling datasets in zero-shot
- Established that **scale enables generalization** — qualitative task capabilities emerge with model size
- Released in stages due to concerns about misuse (the "GPT-2 controversy")
- Architecture changes from GPT-1: LayerNorm moved to input of each sub-block (pre-norm), additional LayerNorm after final self-attention block, deeper (48 layers), modified initialization

## Architecture

| Component | Specification |
|-----------|--------------|
| Layers | 48 |
| d_model | 1600 |
| Attention heads | 25 |
| Parameters | 1.5B |
| Context window | 1024 tokens |
| Tokenizer | BPE (byte-level) |
| Vocabulary | 50,257 |

## Zero-Shot Task Transfer

GPT-2 demonstrated that conditioning on natural language allows task execution without fine-tuning:

```
Translate to French: "Hello, how are you?"  →  "Bonjour, comment allez-vous?"
```

The model learns the pattern of question → answer from its pretraining data, generalizing across tasks that appear in natural language format on the web.

## Scaling Trends

GPT-2 released 4 sizes demonstrating smooth performance improvement with scale:
- Small: 124M (12L/768d)
- Medium: 355M (24L/1024d)
- Large: 774M (36L/1280d)
- XL: 1.5B (48L/1600d)

Each increase in model size improved zero-shot performance across all tasks — foreshadowing the scaling laws that Kaplan et al. would formalize the following year.

## Connections

- [GPT-1](gpt-1.md) — predecessor (117M, required fine-tuning)
- [GPT-3](gpt-3.md) — scaled to 175B, in-context few-shot learning
- [LLM Scaling Laws](llm-scaling-laws.md) — the theoretical framework GPT-2's scaling trends motivated
- [The Bitter Lesson](the-bitter-lesson.md) — general methods + scale > task-specific engineering
