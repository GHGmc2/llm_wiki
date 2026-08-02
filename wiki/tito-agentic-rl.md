---
title: "TITO: Token-In, Token-Out for Agentic RL"
type: source-note
tags: [reinforcement-learning, agentic-rl, tito, multi-turn, tokenization, chat-template, grpo, ppo, tool-use, huggingface]
created: 2026-07-09
updated: 2026-07-09
sources: [https://huggingface.co/blog/huggingface/tito]
status: stable
---

# TITO: Token-In, Token-Out for Agentic RL

**Source**: Quentin Gallouédec, Kashif Rasul (Hugging Face), May 29, 2026. Blog post. [src](https://huggingface.co/blog/huggingface/tito)

## Key Points

- **TITO (Token-In, Token-Out)** is an invariant for multi-turn agentic RL: keep sampled tokens in a buffer and never re-encode them
- The natural loop (MITO: Messages-In, Tokens-Out) re-renders and re-tokenizes the conversation each turn, causing token drift — `encode(decode(tokens)) ≠ tokens` — and silently broken gradients
- The fix: never re-encode decoded tokens; use a running buffer as source of truth; parse for tool dispatch only; add tool responses via a "delta" trick (diff two template renders)
- The only chat-template requirement is **prefix preservation for tool messages** — 18 of 19 major model families already satisfy it by default
- Qwen3 is the lone exception (one-line Jinja fix: `{% if loop.last or ... %}` → `{% if true %}`)
- Renderers (per-model hand-coded objects) are a heavier alternative; TITO is simpler when you control the token stream
- History rewriting (compaction, sub-agent summarization) genuinely breaks the math — freeze pre-rewrite as prompt with zero loss mask
- Truncation is a non-event under TITO but creates real problems for renderers

## Core Problem: The MITO Loop

The natural way to implement multi-turn agentic RL:

1. Keep conversation as a list of messages
2. Each turn: render → generate → parse → append tool result → repeat
3. At the end: tokenize the entire conversation and backprop

This breaks in two ways:

### Token Drift

**Decoding isn't injective.** Multiple distinct BPE token sequences can decode to the same text. When you decode the model's generated tokens, parse the text, and re-encode: you may land on different token IDs. The gradient then targets tokens the policy never sampled.

Causes:
- BPE merges aren't stable across token boundaries — greedy canonical segmentation exists but many non-canonical ones do too
- JSON whitespace, argument ordering, boolean casing (`false` vs `False`)
- Special token re-rendering after parse

### Lost Turn Boundaries

Re-tokenizing the full conversation loses the per-turn assistant/tool boundaries. The loss mask must be reconstructed after the fact by parsing role markers from the rendered string — fragile and template-specific.

## The TITO Fix

One rule: **never re-encode tokens you've decoded.**

Implementation:
1. Model's sampled tokens go into a running buffer — the **source of truth**
2. Parse decoded text only for routing (detect tool calls)
3. Throw away parsed text — it never feeds back to the tokenizer
4. Append tool responses via `compute_delta`: render the conversation with and without the tool message, subtract the prefix, append the suffix tokens by ID

```python
prefix = tok.apply_chat_template(messages_without_tool, return_dict=False)
full = tok.apply_chat_template(messages_with_tool, return_dict=False, add_generation_prompt=True)
delta = full[len(prefix):]
buffer.extend(delta)
```

Key consequence: **per-turn boundaries are built as you go** — the buffer records which tokens came from prompt, model, tool, etc. No post-hoc reconstruction needed.

## Prefix Preservation Property

The `compute_delta` trick requires one property of the chat template:

```
render([user, asst_with_tool_call, tool_result]) starts with render([user, asst_with_tool_call])
```

That is: appending a tool result extends the render verbatim, token-for-token. Required **only** for tool messages.

Tested across 19 open-weights model families:

| Family | Prefix-Preserving? |
|--------|-------------------|
| Qwen2.5 / Qwen2.5-Coder | Yes |
| Qwen3 | **No** (fix below) |
| Qwen3 Instruct (2507) | Yes |
| Qwen3-VL / Qwen3.5 / Qwen3.6 | Yes |
| DeepSeek-V3.1 / R1 / R1-0528 | Yes |
| Llama 3.1 / 3.2 / 4 | Yes |
| Gemma 4 / Function Gemma | Yes |
| gpt-oss | Yes |
| GLM-4.5 / GLM-5 | Yes |
| MiniMax-M2.1 | Yes |

The property is weak and narrowly-scoped — modern templates satisfy it almost by accident.

### Qwen3 Fix

Qwen3 renders an empty `<think></think>` block only on the **last** assistant turn. Appending a tool result demotes that turn from being last, the block disappears, and the prefix breaks.

The one-line Jinja fix:
```diff
- {%- if loop.last or (not loop.last and reasoning_content) %}
+ {%- if true %}
```

## Renderers vs TITO

**Renderers** (per-model hand-coded objects) are the heavier alternative:
- `renderers` library (PrimeIntellect) and `tinker-cookbook` ship hand-coded bridges per model family
- Benefits: unified API, `message_indices` for loss masking, fails loud on drift, works without controlling the token stream
- Costs: every new model needs a hand-coded file, template changes propagate as maintenance

For RL specifically, renderers guard against problems TITO never has:
- BPE retokenization drift only bites pipelines that re-encode decoded strings
- TITO never does — even non-canonical samples stay verbatim in the buffer
- The lone requirement (prefix preservation) is checked at the token level, not text level

When you train against an endpoint that only speaks messages (not tokens), a renderer is the right tool.

## Edge Cases

### History Rewriting

Agents that edit their own past: `clear_thinking`, conversation compaction, sub-agent summaries. Tokens the model produced are no longer the tokens going back in — the objective itself is undefined.

**Workaround**: Pick the **last** rewrite point, freeze everything before it as prompt (zero loss mask). Everything after is genuine sampled tokens under TITO. The cost: a long trajectory with periodic rewrites may leave only the final few hundred tokens with loss.

### Truncation

Under TITO: a non-event. Generation hits the limit, buffer ends, rollout terminates. A dangling `<think>` never needs to close.

Under renderers: the bridge anchors on the close token to extend byte-for-byte. Without it, the bridge returns `None` and the caller falls back to full re-render, triggering drift. The fix requires teaching each renderer to synthesize missing close tokens.

## Connections

- **GRPO/PPO correctness** ([grpo](grpo.md), [ppo](ppo.md)) — TITO ensures importance ratio is computed on the tokens the policy actually produced
- **Async RL Training Landscape** ([async-rl-training-landscape.md](async-rl-training-landscape.md)) — same authors; TRL's async trainer incorporates TITO principles
- **Stabilizing RL with LLMs** ([stabilizing-rl-llms.md](stabilizing-rl-llms.md)) — token-level approximation validity depends on training-inference consistency; TITO prevents silent inconsistency
- **RL Systems: Mind the Gap** ([rl-systems-mind-the-gap.md](rl-systems-mind-the-gap.md)) — trainer-generator throughput considerations when buffering tokens per TITO
- **DeepSeek-V4** ([deepseek-v4.md](deepseek-v4.md)) — on-policy distillation uses multi-turn agentic rollouts where TITO applies
- **Multi-Head Latent Attention** ([multi-head-latent-attention.md](multi-head-latent-attention.md)) — KV cache compression becomes critical in long multi-turn agentic trajectories

## Key Figure

The fundamental problem: round-tripping through decode-then-encode can land on different tokens:

![Non-injective tokenization round-trip](raw/assets/tito-round-trip.png)

![Qwen3 prefix break](raw/assets/tito-qwen3-prefix-break.png)
