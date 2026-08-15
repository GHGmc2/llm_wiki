---
title: "Kimi K2.5: Visual Agentic Intelligence"
type: source-note
tags: [kimi, moonshot, multimodal, agentic, vision, agent-swarm, rl, zero-vision-sft, joint-pretraining, parl]
created: 2026-08-06
updated: 2026-08-15
sources: [raw/kimi-k2.5.pdf, https://arxiv.org/abs/2602.02276]
status: stable
---

# Kimi K2.5: Visual Agentic Intelligence

**Source**: Kimi Team (Moonshot AI), Feb 2026. arXiv:2602.02276. 30 pages. [src](../raw/kimi-k2.5.pdf)

## Key Points

- **Joint text-vision optimization**: text and vision trained together so modalities enhance each other — early vision fusion with constant token ratio beats late fusion
- **Zero-vision SFT**: SFT with ONLY text data (image manipulation proxied via IPython programs) is sufficient to activate vision+agentic capabilities — better than text-vision SFT (lacks high-quality vision data)
- **Visual RL improves text performance**: MMLU-Pro +1.7, GPQA-Diamond +2.1, LongBench v2 +2.2 after vision RL (cross-modal transfer)
- **Joint multimodal RL**: domains organized by *ability* (knowledge, reasoning, coding, agentic) not modality — maximize cross-modal transfer
- **Agent Swarm + PARL**: trainable orchestrator + frozen subagents; dynamic task decomposition + parallel scheduling; up to **4.5× latency reduction** vs single-agent
- **PARL reward**: instantiation reward (anti serial-collapse) + subagent finish rate (anti spurious parallelism) + task outcome; $\lambda_1, \lambda_2$ annealed to zero
- **Critical Steps metric**: stage duration = longest subagent in parallel group — incentivizes *effective* parallelization, not mere concurrency
- **MoonViT-3D**: patch-n-pack generalized to temporal dimension — up to 4 frames as spatiotemporal volume, fully shared parameters, 4× temporal pooling compression; video capability from image pretraining without bifurcation
- **Toggle token-efficient RL**: alternates budget-limited phase (conditional on accuracy > λ) and standard scaling phase every m iterations; **25-30% output token reduction** with negligible performance loss; fixes length-overfitting under rigid budgets
- **DEP (Decoupled Encoder Process)**: vision encoder decoupled from PP stage-0 — reuses text-only parallel strategies, no load imbalance

## Joint Optimization of Text and Vision

Key insight: joint optimization of text and vision **enhances both modalities and avoids conflict**:
- Conventional: add visual tokens to text backbone late. K2.5: **early vision fusion, text+vision tokens at constant ratio** through pre-training
- Foundation: Kimi K2 base (1.04T/32B MoE, MuonClip, 384 experts sparsity 48)

## Zero-Vision SFT

- SFT uses **only text data**; all image manipulations proxied through programmatic operations in IPython — a generalization of vision tool-use
- Activates pixel-level operations (object size via binarization/counting) and generalizes to localization, counting, OCR
- RL curves show zero-vision SFT is sufficient when paired with long-running RL; text-vision SFT performed *worse* (lack of high-quality vision data)
- Likely enabled by joint text-vision pretraining

## Joint Multimodal RL

### Outcome-Based Visual RL (three domains)

1. Visual grounding and counting
2. Chart and document understanding
3. Vision-critical STEM problems (filtered to require visual input)

Trajectories extracted for rejection-sampling fine-tuning (RFT) → self-improving data pipeline.

### Cross-Modal Transfer

| Text benchmark | Before Vision-RL | After Vision-RL | Δ |
|----------------|-----------------|-----------------|---|
| MMLU-Pro | 84.7 | 86.4 | +1.7 |
| GPQA-Diamond | 84.3 | 86.4 | +2.1 |
| LongBench v2 | 56.7 | 58.9 | +2.2 |

Analysis: visual RL improves calibration in structured information extraction (counting, OCR-like reasoning).

### Joint RL Paradigm

Domains organized by **ability** (knowledge, reasoning, coding, agentic), not modality; Generative Reward Models (GRMs) optimize across heterogeneous traces without modality barriers.

## Agent Swarm & Parallel Agent RL (PARL)

### Architecture

Trainable **orchestrator** + **frozen subagents** (fixed intermediate policy checkpoints):
- Deliberately avoids end-to-end co-optimization: credit assignment ambiguity + training instability
- Subagent outputs treated as *environmental observations*, not differentiable decision points
- Training: small-size subagents first, then larger; dynamic inference instance ratios between subagents/orchestrator

### PARL Reward

$$r_{\text{PARL}}(x,y) = \lambda_1 \cdot r_{\text{parallel}} + \lambda_2 \cdot r_{\text{finish}} + r_{\text{perf}}(x,y)$$

- $r_{\text{parallel}}$ (instantiation): mitigates **serial collapse** (orchestrator defaulting to single-agent)
- $r_{\text{finish}}$ (subagent finish rate): prevents **spurious parallelism** (spawning many subagents without meaningful decomposition)
- $r_{\text{perf}}$: task-level outcome
- $\lambda_1, \lambda_2$ annealed to zero — final policy optimizes the primary objective

### Critical Steps as Resource Constraint

$$\text{CriticalSteps} = \sum_{t=1}^{T} \left( S^{(t)}_{\text{main}} + \max_i S^{(t)}_{\text{sub},i} \right)$$

Stage duration governed by the longest-running subagent in the parallel cohort. Constraining on critical steps (not total steps) incentivizes **well-balanced decomposition** that shortens the longest branch — not mere concurrency.

### Prompt Construction

Synthetic prompts stressing sequential limits: **wide search** (many independent sources), **deep search** (multiple reasoning branches with delayed aggregation), real-world (long-context document analysis, large-scale downloads). Prompts never explicitly instruct parallelization — task distribution naturally favors it. Parallelism decisions are **learned** through RL (accuracy and parallelism both increase smoothly during training).

## MoonViT-3D (Vision Encoder)

- Init from SigLIP-SO-400M; NaViT patch-packing: images → flattened 1D patch sequences (native resolution, no sub-image splitting)
- **Temporal generalization**: up to 4 consecutive frames as a spatiotemporal volume — 2D patches jointly flattened and packed into one 1D sequence; identical attention across space and time
- Lightweight temporal pooling before MLP projector: 4× temporal compression for longer videos
- Continual pretraining with **caption-only cross-entropy loss** (no contrastive loss) on alt text, synthetic captions, grounding bboxes, OCR texts
- Two-stage alignment: (1) align MoonViT-3D with Moonlight-16B-A3B (~1T tokens, few FLOPs), (2) short stage updating only MLP projector to bridge to the 1T LLM

## Pre-training Pipeline (~15T tokens, 3 stages)

| Stage | Data | Seq len | Tokens | Trainable |
|-------|------|---------|--------|-----------|
| ViT Training | alt text, synthetic captions, grounding, OCR | 4096 | 1T | ViT |
| Joint Pre-training | video+text, knowledge, interleaving, OS screenshots | 4096 | 15T | ViT & LLM |
| Joint Long-context Mid-training | long text/video, reasoning, long-CoT | 32768 → 262144 | 500B → 200B | ViT & LLM |

Joint stage extends K2's distribution with unique tokens, more coding weight, per-source epoch caps. Long-context via **YaRN**.

## Post-Training

### SFT
Synthesized candidates from K2, K2 Thinking, in-house experts; domain-specialized pipelines with human annotation + multi-stage verification.

### RL Objective (token-level clipping on log-ratio)

$$L_{\text{RL}}(\theta) = \mathbb{E}_{x\sim D}\left[\frac{1}{N}\sum_{j=1}^{K}\sum_{i=1}^{|y_j|} \operatorname{Clip}\left(\frac{\pi_\theta(y_j^i|x,y_j^{0:i})}{\pi_{\text{old}}(y_j^i|x,y_j^{0:i})}, \alpha, \beta\right)(r(x,y_j)-\bar{r}(x)) - \tau\left(\log\frac{\pi_\theta(y_j^i|x,y_j^{0:i})}{\pi_{\text{old}}(y_j^i|x,y_j^{0:i})}\right)^2\right]$$

- Departs from k1.5's OMD by adding **token-level clipping on log-ratio regardless of advantage sign** (vs PPO which clips only selected tokens) — bounds off-policy drift from training/inference framework discrepancies; essential for long-horizon multi-step tool use
- MuonClip optimizer

### Reward Function

- Rule-based outcome rewards for verifiable tasks (reasoning, agentic) + **budget-control reward** for token efficiency
- **Task-specific visual rewards**: F1 + soft matching (grounding via IoU; points via Gaussian-weighted distance); polygon rasterization → segmentation IoU; OCR via normalized edit distance; counting via absolute difference; visual puzzles verified by LLM (Kimi K2)
- **Generative Reward Models (GRMs)**: fine-grained evaluators aligned with Kimi values (helpfulness, readiness, relevance, detail, aesthetics, instruction following) applied across chat/coding/search/artifact agents; multiple alternative rubrics mitigate reward hacking

### Toggle (Token-Efficient RL)

Alternates every m iterations:
- **Phase 0 (budget-limited)**: enforce task-dependent token budget — only when mean accuracy > λ (conditional, avoids premature quality sacrifice)
- **Phase 1 (standard scaling)**: full max-token generation for inference-time scaling

Budget: ρ-th percentile of correct-response lengths, estimated once at training start.

Results (on K2 Thinking): **25-30% output token reduction** with negligible performance impact; fewer redundant patterns (repeated verifications, mechanical calculations); generalizes to domains not trained on (GPQA, MMLU-Pro) with marginal degradation.

## DEP (Decoupled Encoder Process)

Problem: vision encoder in PP stage-0 causes drastic load/memory fluctuation from variable image counts/resolutions; forces custom PP configs (e.g., manual layer rebalancing) and blocks reuse of text-optimized parallel strategies.

DEP (3 sub-steps per training step, exploiting encoder's position — start of forward, end of backward):
1. **Balanced vision forward** for all visual data in the global batch
2. ... then LLM forward/backward with vision features precomputed
3. Encoder backward decoupled

→ negligible additional overhead; reuse of K2's text-optimized infra with minimal modification.

## Results

- SOTA across coding, vision, reasoning, agentic tasks
- **Agent Swarm: up to 4.5× latency reduction** vs single-agent baselines
- Computer-use capability, video understanding, ZeroBench (w/ tools), MMMU-Pro

## Connections

- [Kimi K2](kimi-k2.md) — base model (1.04T, MuonClip); RL objective lineage (k1.5 + clipping)
- [Kimi K3](kimi-k3.md) — successor: native vision via MoonViT-V2, KDA attention
- [Kimi k1.5](kimi-k1.5.md) — policy optimization origin; length penalty lineage → Toggle
- [GLM-5](glm-5.md) — comparable agentic-era foundation model
- [DeepSeek-V3.2](deepseek-v3.2.md) — training/inference framework mismatch (K2.5's log-ratio clipping motivated by the same concern)
- [Stabilizing RL with LLMs](stabilizing-rl-llms.md) — off-policy clipping theory
- [Muon Optimizer](muon-optimizer.md) — MuonClip in RL
- [ScaleRL](scalerl.md) — RL scaling; token efficiency trade-offs
- [TITO: Agentic RL](tito-agentic-rl.md) — multi-turn tool-use token fidelity
- [LLM Architecture Comparison](llm-architecture-comparison.md) — multimodal positioning
- [Ring Attention](ring-attention.md) — long-context activation (262K)
