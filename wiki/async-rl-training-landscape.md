---
title: "Keep the Tokens Flowing: Lessons from 16 Open-Source RL Libraries"
type: source-note
tags: [reinforcement-learning, async-training, rl, grpo, ppo, distributed-training, survey, huggingface, trl, verl, open-source]
created: 2026-05-04
updated: 2026-08-06
sources: [https://huggingface.co/blog/async-rl-training-landscape]
status: stable
---

# Async RL Training Landscape

**Source**: Dirhoussi, Gallouédec, Rasul, Tunstall, Beeching, et al. (Hugging Face), March 2026. A comprehensive survey of 16 open-source RL libraries for LLM post-training, analyzing their architectures across 7 design axes.

## Key Points

> [!note] SLIME vs slime: the SLIME in this survey is **Mozilla's** open-source RL library — a different project from Zhipu's in-house `slime` framework described in [GLM-4.5](glm-4.5.md) / [GLM-5](glm-5.md). Name collision only.

- 16 libraries surveyed: TRL, verl, SkyRL, SLIME, PipelineRL, NeMo-RL, PRIME-RL, AReaL, ART, MILES, ROLL, OAT, open-instruct, TorchForge, Tunix, Atropos
- **7 architectural axes**: Orchestration, Rollout Buffer, Weight Sync, Staleness Management, Partial Rollout, LoRA Support, Training Backend
- **Ray dominates**: 8 of 16 libraries use Ray's distributed actor model; Monarch (Meta) emerging as PyTorch-native alternative
- **Critic-free trend**: GRPO, REINFORCE++, Online DPO free ~50% GPU memory but increase sync pressure (larger group sizes)
- **MoE mismatch**: DeepSeek-V3.2 discovered router inconsistency between inference and training frameworks — a correctness issue, not performance
- **Hybrid staleness**: production systems converge on depth bounding + optional IS correction + per-sample version tagging
- TRL's async trainer: bounded queue with per-token model_version, NCCL bucketed weight sync, partial rollout support for agents

## The 7 Axes

### Axis 1: Orchestration & Concurrency Primitive

Four orchestration types, ordered by abstraction level:

| Type | Key Libraries | Trade-off |
|------|--------------|-----------|
| **Distributed Actor Model (Ray)** | verl, SkyRL, NeMo-RL, SLIME, MILES, ROLL, OAT, open-instruct | Richest abstraction, built-in scheduling/fault tolerance; non-trivial runtime dependency |
| **Native Python Concurrency** | verifiers-rl, PipelineRL, ART, AReaL, PRIME-RL | Minimal deps, easy debug; limited to single-node without additional IPC |
| **Pub/Sub Message Bus** | PipelineRL (inter-pool) | Clean cross-pool decoupling; no lifecycle management |
| **HTTP Microservices** | Any inference server | Language-agnostic, maximum decoupling; highest latency |

8 of 16 libraries use Ray — not coincidence. RL training involves fundamentally heterogeneous components (inference, training, reward models, environments) that map naturally to Ray's actor model with isolated state, async RPC, and shared-memory object store.

### Axis 2: Rollout Buffer Design

| Buffer Type | Libraries | Characteristics |
|------------|-----------|----------------|
| No buffer (synchronous) | TRL, ART | Zero staleness, maximum idle time |
| Double-buffer (one-step-ahead) | verifiers-rl, SLIME, MILES, OAT | Exactly one batch overlap |
| Bounded async queue | SkyRL, verl, NeMo-RL, ROLL, PRIME-RL, TorchForge, Tunix, open-instruct, AReaL | Multiple batches in flight; staleness bounded by capacity |
| Unbounded / stream | PipelineRL, SLIME, Atropos | Continuous generation; staleness via explicit version control |

### Axis 3: Weight Synchronisation

The most architecturally consequential axis. Interrupt granularity determines how much generation is wasted:

| Interrupt Model | Libraries | What Happens |
|----------------|-----------|-------------|
| **Never** (per-forward-pass swap) | PipelineRL, open-instruct (opt-in) | Weight swap between token decode steps (~1-10ms); sequences never stop |
| Per HTTP Request (abort + resync) | SkyRL, SLIME, MILES | In-flight requests aborted, partial tokens resubmitted |
| Soft Pause (drain in-flight) | PRIME-RL, AReaL, verl (async) | New requests blocked; in-progress complete naturally |
| Per Training Step (blocking) | NeMo-RL, ROLL, OAT, TorchForge, Tunix, verifiers-rl | Full stop between generation and training |

Transport mechanisms: NCCL Broadcast (most common), NCCL + Bucketing (vLLM's packed=True), CUDA IPC (Zero-copy), Filesystem + HTTP, JAX Cross-mesh reshard.

### Axis 4: Staleness Management

Three orthogonal strategies, often combined:

| Strategy | How It Works | Libraries |
|----------|-------------|-----------|
| **Per-sample version rejection** | Tag each sample with policy version; drop if too old | PRIME-RL, open-instruct, MILES |
| **Depth bounding** | Limit queue/buffer capacity to bound staleness architecturally | verl, SLIME, ROLL, verifiers-rl |
| **IS-weighted loss correction** | Reweight stale samples by $\pi_\theta / \pi_{\text{old}}$ (clipped TIS) | PRIME-RL, SkyRL, NeMo-RL, OAT |

Production trend: hybrid approaches combining depth bounding + optional IS correction + per-sample version tagging.

### Axis 5: Partial Rollout Handling

For long-context tasks where a single rollout can take minutes:

| Strategy | Libraries |
|----------|-----------|
| Implicit continuation (swap between forward passes) | PipelineRL, AReaL |
| Abort + retry with prefix | SkyRL, SLIME |
| Explicit save/resume | verl |
| Soft pause, in-flight complete | PRIME-RL, AReaL, open-instruct |
| No support / batch boundary only | verifiers-rl, OAT, TorchForge, Atropos, Tunix |

### Axis 6: LoRA Training

LoRA reduces trainable params 99%+, enables adapter-only sync (~50MB vs full model broadcast):

- 8 of 16 libraries support adapter-only sync to vLLM/SGLang
- Three implementation families: HuggingFace peft, Megatron-Bridge, custom
- **MoE LoRA** is emerging: per-expert adapters × E experts, requires EP-aware gather before sync
- With adapter-only sync, transfer is so fast that nearly any interrupt model delivers equivalent throughput

### Axis 7: Training Backend & Parallelism

| Backend | Libraries | Distinguishing Feature |
|---------|-----------|----------------------|
| Megatron-LM | verl, SLIME, MILES, ROLL, NeMo-RL | TP + PP + CP + EP + ETP; full MoE support |
| DeepSpeed ZeRO | PipelineRL, verifiers-rl, OAT, verifiers-rl | ZeRO-2/3 DP; simpler, lighter |
| FSDP2 (PyTorch native) | PRIME-RL, TorchForge, TRL | HSDP + TP + CP; growing ecosystem |
| JAX/XLA | Tunix | TPU-native, 2D mesh |

Weight sync speed is a direct function of training backend — Megatron requires AllGathering from every TP/PP/EP rank before broadcast.

## Library Comparison Table

| Library | Orchestration | Inference | Weight Sync | Staleness | Partial Rollout | Training Backend | LoRA |
|---------|--------------|-----------|-------------|-----------|----------------|-----------------|------|
| **TRL** (HF) | Native Python | vLLM, SGLang | NCCL packed | Depth + IS (opt) | Soft pause | FSDP2 | ❌ |
| **verl** (ByteDance) | Ray | vLLM, SGLang | NCCL + checkpoint buckets | IS correction | Save/resume | FSDP/FSDP2 or Megatron | ✅ (adapter-only) |
| **SkyRL** (NovaSky) | Ray | vLLM, SGLang, Megatron | NCCL or CUDA IPC | Version rejection | Abort + retry | Megatron or FSDP2 | ✅ (adapter-only) |
| **SLIME** (Mozilla) | Ray | vLLM, SGLang | NCCL or CUDA IPC | IS correction | Abort + recycle | Megatron or FSDP2 | ✅ (adapter-only) |
| **PipelineRL** (Meta) | Native + Pub/Sub | vLLM | NCCL pg + HTTP notify | Version rejection | **Implicit continuation** ✅ | DeepSpeed ZeRO-3 | ✅ |
| **NeMo-RL** (NVIDIA) | Ray | vLLM, SGLang | NCCL, bucketed | IS correction | Abort + recycle | Megatron-LM | 🟧 |
| **PRIME-RL** (PrimeIntellect) | Native Python | vLLM, SGLang | Filesystem or NCCL | **Full hybrid** | Group cancellation | FSDP2 | ✅ (adapter-only) |
| **AReaL** (inclusionAI) | Native Python | vLLM, SGLang | NCCL chunked or filesystem | Depth + IS (opt) | Soft pause | FSDP2, Megatron, Archon | ✅ (adapter-only) |
| **ART** (NousResearch) | Native Python | vLLM | LoRA adapter swap | Synchronous | — | Unsloth or Megatron | ✅ (adapter-only) |
| **MILES** (NVIDIA) | Ray | vLLM, SGLang, Megatron | NCCL or CUDA IPC | Version rejection | **Implicit continuation** ✅ | DTensor or Megatron-Bridge | 🟧 |
| **ROLL** (NousResearch) | HTTP Microservices | vLLM, SGLang, OpenAI | FS checkpoint + restart | Depth bounding | — | PyTorch, TRL | ✅ (adapter-only) |
| **OAT** (OpenRLHF) | Ray | vLLM, SGLang | NCCL + ZeRO-3 gather | IS correction | — | DeepSpeed ZeRO-2/3 | ✅ |
| **open-instruct** (AI2) | Ray | vLLM, SGLang | NCCL broadcast | Depth + IS (opt) | Drain-or-inflight | DeepSpeed ZeRO | ⚠️ |
| **TorchForge** (Meta) | Monarch | vLLM | torchstore + shmem | Version rejection | — | FSDP2 via TorchTitan | ❌ |
| **Tunix** (Google) | JAX-native | vLLM, SGLang, JAX | Cross-mesh reshard | Depth bounding | — | JAX/XLA 2D mesh | ✅ (adapter-only) |
| **Atropos** | Ray | vLLM, SGLang | NCCL packed | IS correction | Abort + recycle | Megatron-LM | ❌ |

✅ = supported · 🟧 = partial/custom · ❌ = not · ⚠️ = experimental

## Emerging Trends

### Critic-Free Algorithms

Eliminating the value network frees ~50% GPU memory but increases sync pressure: GRPO requires G=8-32 completions per prompt, more rollouts per step, faster policy drift. Asymmetric filtering (keep positives, discard negatives) creates intra-batch version spread that batch-level staleness tracking cannot detect.

### MoE Training-Inference Mismatch

DeepSeek-V3.2's production experience revealed two structural correctness issues:
1. **Router inconsistency**: different floating-point rounding in inference vs training gating functions → different expert selections for identical inputs
2. **Sampling mask mismatch**: top-p/top-k truncates vocabulary at generation; IS ratio is undefined for truncated tokens

Solutions: Keep Routing (record and enforce inference routing during training), Keep Sampling Mask (record truncation mask, apply to both policies). No open-source library implements either.

### Process Rewards

PRM-based credit assignment (scoring intermediate reasoning steps) breaks the assumption that rewards are cheap. A PRM forward pass over 32K tokens from a 7B model can be significant. Requires async reward pipelines (PRIME-RL's Orchestrator-Trainer pattern) and dedicated preprocessor GPU tier.

### Distillation as Async RL

On-policy distillation (student generates, teacher scores) is structurally identical to GRPO's async coordination. Libraries treating the scoring phase as a pluggable component (verl, SLIME, PRIME-RL, AReaL, NeMo-RL) support both GRPO and distillation without architectural changes.

## TRL's Async Trainer Design

TRL's synchronous baseline interleaves training steps and generation with idle periods — the async trainer eliminates this idle time:

![Synchronous TRL training timeline](../raw/assets/async-rl-trl-sync-timeline.png)

*Figure: TRL's synchronous training timeline — training steps and generation alternate, with no overlap. [src](https://huggingface.co/blog/async-rl-training-landscape)*

In colocate mode, TRL runs vLLM in-process, reducing the weight sync path to a shared-memory exchange:

![TRL with vLLM in colocate mode](../raw/assets/async-rl-trl-vllm-colocate.png)

*Figure: TRL with vLLM in colocate mode — weight sync via CUDA IPC / shared memory instead of NCCL broadcast. [src](https://huggingface.co/blog/async-rl-training-landscape)*

TRL's announced async GRPO trainer will use:
1. **Bounded queue with per-token model_version** — token-level provenance from the start
2. **NCCL bucketed weight sync** — vLLM's packed=True broadcast, exploring Awex and Mooncake for cross-engine transfer
3. **Partial rollout support** — prefix-resume and abort-and-retry for agentic workloads
4. **Lightweight orchestration** — native Python, no heavyweight runtime dependency

## Connections

- [PPO](ppo.md) — the algorithm TRL's async trainer builds on
- [GRPO / DeepSeek-R1](deepseek-r1.md) — critic-free RL at scale; the survey identifies critic-free methods as increasing sync pressure
- [ScaleRL](scalerl.md) — async off-policy RL with PipelineRL; directly analyzed in Axis 1 and Axis 5
- [Stabilizing RL with LLMs](stabilizing-rl-llms.md) — IS correction and staleness theory (Ch 3-4 directly reference these strategies)
- [DeepSeek-V3.2](deepseek-v3.2.md) — MoE router inconsistency and Keep Routing as a correctness requirement
- [Megatron-Core MoE](megatron-core-moe.md) — training backend used by 5 of 16 surveyed libraries
- [Maestro](maestro.md) — Qwen's section-centric compound workload training; structurally similar to async RL's heterogeneous component problem
- [NCCL Demystifying](nccl-demystifying.md) — NCCL broadcast and bucketing as the primary weight sync transport
- [TITO: Agentic RL](tito-agentic-rl.md) — Token-In, Token-Out invariant for multi-turn tool-use rollouts (same authors); TRL's async trainer incorporates these principles
- [Predicting Staleness in Async RL](staleness-in-async-rl.md) — Closed-form staleness formulas; the mathematical framework for Axis 4 (staleness management)
- [Single-Rollout Async Optimization](single-rollout-async-agentic-rl.md) — SAO: effectiveness-focused async RL design (vs throughput-focused libraries surveyed)
