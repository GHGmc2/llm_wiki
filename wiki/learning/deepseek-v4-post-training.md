---
title: "DeepSeek-V4 Post-Training: On-Policy Distillation and Reasoning Modes"
type: concept
tags: [llm, deepseek, post-training, on-policy-distillation, grpo, reasoning, rl]
created: 2026-05-02
updated: 2026-05-02
sources: [raw/DeepSeek-V4.pdf]
status: stable
---

# DeepSeek-V4 Post-Training

**Source**: DeepSeek-V4 Technical Report, Section 5 [src](raw/DeepSeek-V4.pdf)

## Key Points

- V4 replaces V3.2's mixed RL stage entirely with **On-Policy Distillation (OPD)**
- Two-phase pipeline: train domain specialists via SFT+GRPO, then distill into unified model
- Three explicit reasoning modes replace one-shot inference: Non-Think, Think High, Think Max
- **Interleaved Thinking** retains reasoning across tool calls and user turns in agentic scenarios
- **Quick Instruction** uses special tokens for auxiliary tasks, reusing existing KV cache

## Pipeline Overview

```
Base Model
    |
    v
SFT (domain-specific data)
    |
    v
GRPO (reinforcement learning per domain)
    |
    +---> Math Specialist
    +---> Code Specialist
    +---> Agent Specialist
    +---> Instruction Specialist
    |
    v
On-Policy Distillation (multi-teacher)
    |
    v
Unified DeepSeek-V4 Model
```

## Phase 1: Specialist Training

For each target domain, a separate expert model is trained independently:

1. **SFT**: Fine-tune the base model on high-quality, domain-specific data to establish foundational capabilities
2. **RL with GRPO**: Apply Group Relative Policy Optimization guided by domain-specific prompts and reward signals

**Reward models**: Easy-to-verify tasks (math, code) use rule-based verifiers and test cases. Hard-to-verify tasks use **generative reward models** — the model leverages its own logic to generalize across complex tasks.

**Reasoning effort calibration**: Specialists are trained under different RL configurations to produce models optimized for varying reasoning depths. This directly enables the three reasoning modes:

| Mode | Characteristics | RL Config |
|------|----------------|-----------|
| Non-Think | Fast, intuitive, low latency | High length penalty, small context |
| Think High | Conscious logical analysis | Moderate length penalty, medium context |
| Think Max | Push reasoning to fullest extent | Low length penalty, large context (384K) |

Each mode uses distinct `<think>` / `</think>` response formats. Think Max additionally prepends a system prompt instruction (see below).

### Think Max System Prompt

```
Reasoning Effort: Absolute maximum with no shortcuts permitted.
You MUST be very thorough in your thinking and comprehensively decompose the
problem to resolve the root cause, rigorously stress-testing your logic against
all potential paths, edge cases, and adversarial scenarios.
Explicitly write out your entire deliberation process, documenting every
intermediate step, considered alternative, and rejected hypothesis to ensure
absolutely no assumption is left unchecked.
```

## Phase 2: On-Policy Distillation (OPD)

OPD replaces V3.2's mixed RL consolidation stage. Given N expert models (teachers) and a student model:

**Formally**: The student optimizes a reverse-KL loss against teacher output distributions on its **own generated trajectories**:

```
L_OPD = sum_i(w_i * KL(p_student || p_teacher_i))
```

The student generates tokens, and each teacher provides its probability distribution for those tokens. The student learns to match the teachers' distributions on its own outputs — not on training data from the teachers.

**Key advantage over mixed RL**: OPD preserves each specialist's domain expertise without the interference that mixed RL can cause when reward signals from different domains conflict.

### Infrastructure for OPD

- **Full-vocabulary OPD**: Efficient teacher scheduling required for computing reverse-KL across the full 128K vocabulary
- **Preemptible rollout service**: Training tasks can be preempted and resumed without data loss
- **Million-token RL framework**: RL infrastructure scaled to support 1M-token context rollouts
- **DSec sandbox**: Sandbox infrastructure for agentic AI training (see Infrastructure page)

## Tool-Call Schema

V4 introduces a new **XML-based tool-call schema** using `<|DSML|>` tokens:

```
<|DSML|tool_calls>
<|DSML|invoke name="tool_name">
<|DSML|parameter name="param" string="true">value</|DSML|parameter>
</|DSML|invoke>
</|DSML|tool_calls>
```

The XML format reduces escaping failures and tool-call errors compared to V3.2's format. Agent scaffolding calling V4 should expect this format.

## Interleaved Thinking

A refinement of V3.2's context management for reasoning traces:

**Tool-calling scenarios (Figure 7a)**: All reasoning content (inside `<think>` tags) is preserved across the entire conversation — including across user message boundaries. The model maintains a coherent, cumulative chain of thought over long-horizon agent tasks.

**General conversation (Figure 7b)**: Original V3.2 strategy preserved — reasoning from previous turns is discarded on new user messages, keeping context concise.

> For agent frameworks that simulate tool interactions via user messages (e.g., Terminus), the tool-calling path may not trigger. DeepSeek recommends non-Think models for such architectures.

## Quick Instruction

A latency optimization for chatbot scenarios. Auxiliary tasks (web search trigger, intent recognition, etc.) are conventionally handled by a separate small model requiring redundant prefill.

**V4's approach**: Append dedicated special tokens to the input sequence, each corresponding to a specific auxiliary task. By directly reusing the already-computed KV cache, Quick Instruction:
- Eliminates redundant prefilling
- Enables parallel execution of tasks (search queries, authority check, domain classification)
- Reduces time-to-first-token (TTFT)
- Eliminates the engineering overhead of maintaining a separate small model

## Post-Training Infrastructure

- **FP4 Quantization Integration**: FP4 models used throughout RL and OPD stages for memory efficiency
- **Efficient Teacher Scheduling**: Multi-teacher OPD requires scheduling N teacher distributions per student token
- **Preemptible Rollout Service**: Fault-tolerant, globally ordered trajectory logging with fast-forward resumption
- **DSec Sandbox**: Layered container/VM storage, density optimizations for massive concurrency, trajectory logging for agent training

## Connections

- [DeepSeek-V4 Technical Report](deepseek-v4-technical-report.md) — full paper summary
- [DeepSeek-V4 Architecture](deepseek-v4-architecture.md) — hybrid attention, mHC, Muon
- [DeepSeek-V4 Infrastructure](deepseek-v4-infrastructure.md) — DSec sandbox, RL framework
