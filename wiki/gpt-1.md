---
title: "Improving Language Understanding by Generative Pre-Training (GPT-1)"
type: source-note
tags: [gpt, openai, pretraining, transformer, language-model, decoder-only, foundation]
created: 2026-05-04
updated: 2026-05-04
sources: [raw/gpt-1.pdf]
status: stable
---

# GPT-1: Generative Pre-Training

**Source**: Radford, Narasimhan, Salimans, Sutskever (OpenAI), 2018. The paper that introduced the GPT architecture — generative pre-training followed by discriminative fine-tuning. The origin of the entire GPT lineage.

## Key Points

- Introduces **semi-supervised** approach: unsupervised generative pre-training on large text corpus → supervised fine-tuning on specific tasks
- **12-layer Transformer decoder** (117M parameters), trained on BooksCorpus (7,000 unpublished books)
- Pre-training objective: standard language modeling (predict next token)
- Fine-tuning: add a linear classification head, fine-tune all parameters end-to-end
- Achieved **SOTA on 9 of 12 NLP benchmarks** (natural language inference, question answering, semantic similarity, text classification)
- Key discovery: language model pre-training provides a universal representation that transfers across tasks
- **Task-specific input transformations**: introduced traversal-style approaches to adapt a single model to diverse tasks (textual entailment, similarity, QA, classification)

## Architecture

| Component | Specification |
|-----------|--------------|
| Layers | 12 |
| d_model | 768 |
| Attention heads | 12 |
| FFN dimension | 3072 |
| Parameters | 117M |
| Tokenizer | BPE (40,000 merges) |
| Context window | 512 tokens |
| Training data | BooksCorpus (~5GB) |

### Pre-Training

Standard left-to-right language modeling:
$$L_1(\mathcal{U}) = \sum_{i} \log P(u_i \mid u_{i-k}, \dots, u_{i-1}; \Theta)$$

### Fine-Tuning

For a labeled dataset $\mathcal{C}$ with inputs $x$ and labels $y$:
$$L_2(\mathcal{C}) = \sum_{(x,y)} \log P(y \mid x^1, \dots, x^m)$$

Combined objective (language modeling auxiliary loss improves generalization):
$$L_3(\mathcal{C}) = L_2(\mathcal{C}) + \lambda \cdot L_1(\mathcal{C})$$

## Task-Specific Input Transformations

GPT-1 introduced traversal-style input representations to handle diverse NLP tasks with a single architecture:
- **Textual entailment**: concatenate premise and hypothesis with delimiter
- **Similarity**: encode both sentences, use element-wise operations on their representations
- **QA & commonsense**: concatenate context, question, and answer choices

## Significance

This paper established the **pre-train then fine-tune** paradigm that dominated NLP for years. It demonstrated that a single Transformer architecture, pre-trained generatively, could transfer to diverse tasks — setting the stage for GPT-2, GPT-3, BERT, and eventually the modern LLM era.

## Connections

- [GPT-2](gpt-2.md) — scaled to 1.5B parameters, removed fine-tuning in favor of zero-shot transfer
- [GPT-3](gpt-3.md) — 175B parameters, in-context learning without gradient updates
- [LLM Scaling Laws](llm-scaling-laws.md) — Kaplan et al. formalized the scaling principles GPT-1 hinted at
- [LLM Architecture Comparison](llm-architecture-comparison.md) — where GPT fits in the architecture landscape
