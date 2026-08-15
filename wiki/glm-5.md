---
title: "GLM-5: from Vibe Coding to Agentic Engineering"
type: source-note
tags: [glm, zhipu, moe, agentic, reasoning, coding, dsa, async-rl, post-training, distillation, agent, long-context, chinese-chip]
created: 2026-08-06
updated: 2026-08-15
sources: [raw/glm-5.pdf, https://arxiv.org/abs/2602.15763]
status: stable
---

# GLM-5: from Vibe Coding to Agentic Engineering

**Source**: GLM-5 Team (Zhipu AI & Tsinghua University), Feb 2026. arXiv:2602.15763. 40 pages. [src](../raw/glm-5.pdf)

## Key Points

- **744B total / 40B active MoE**, 256 experts, 80 layers — doubled total size of GLM-4.5; 28.5T training tokens
- **DSA (DeepSeek Sparse Attention)**: ~90% of long-context attention entries redundant; 1.5-2× attention computation reduction; 128K contexts at ~half GPU cost
- **DSA adaptation is cheap**: 20B-token sparse adaptation matches original MLA model (vs DeepSeek-V3.2's 943.7B)
- **Muon Split**: per-head matrix orthogonalization fixes MLA's gap vs GQA-8 under Muon — stable attention logits without clipping
- **MLA-256 variant**: head dim 192→256 with 1/3 fewer heads — same training compute, less decoding compute
- **Search-based SWA pattern** (beam search on RULER@16K) beats fixed SWA interleave; Gated DeltaNet also evaluated
- **Sequential RL**: Reasoning → Agentic → General RL with **On-Policy Cross-Stage Distillation** to prevent catastrophic forgetting
- **Asynchronous RL infrastructure** (slime): generation fully decoupled from training; Token-in-Token-out; double-sided IS token masking (a.k.a. DIS, see [SAO](single-rollout-async-agentic-rl.md)); drop off-policy/noisy samples; DP-aware routing
- **Full-stack adaptation to 7 Chinese chip platforms** (Ascend, Moore Threads, Hygon, Cambricon, Kunlunxin, MetaX, Enflame) from day 1; W4A8 mixed-precision
- **Intelligence Index v4.0 = 50** — first open-weights model at 50; #1 open on LMArena Text + Code; ~20% avg improvement over GLM-4.7; comparable to Claude Opus 4.5 / GPT-5.2 (xhigh)

## Architecture

### Model Scaling

256 experts, 80 layers (vs 160/89 in GLM-4.5) — fewer layers minimizes expert-parallel communication overhead. 744B total, 40B active.

### Multi-Latent Attention (MLA) Under Muon

Key experiment: with the **Muon optimizer**, MLA (576-dim latent KV) **underperforms GQA-8** (2048-dim KV):

| Dataset | GQA-8 | MLA | MLA + Muon Split | MLA-256 + Muon Split |
|---------|-------|-----|------------------|----------------------|
| Hellaswag | 77.3 | 77.3 | 77.8 | 77.4 |
| MMLU | 61.2 | 61.5 | 62.5 | 62.0 |
| C-Eval | 60.0 | 59.7 | 62.1 | 59.9 |
| RACE | 79.6 | 77.8 | 79.9 | 79.6 |
| BBH | 53.3 | 48.9 | 51.8 | 51.3 |
| GSM8K | 47.6 | 46.2 | 45.0 | 47.5 |
| HumanEval | 38.5 | 33.5 | 36.7 | 36.6 |

**Muon Split fix**: instead of orthogonalizing $W_{UQ}, W_{UK}, W_{UV}$ as whole matrices, split per-head and orthogonalize independently → projection weights update at different scales per head → MLA matches GQA-8, and attention logit scale stays stable **without any clipping** at GLM-5's scale (cf. Kimi's MuonClip; note [Kimi K3](kimi-k3.md) uses the same per-head idea — Per-Head Muon — but at 2.8T still required QK-Clip-style weight clipping).

**MLA-256**: MLA decoding does 576-dim dot product vs GQA's 128. Since MLA trains/prefills like MHA, increase head dim 192→256, reduce head count by 1/3 — same training compute/params, lower decoding compute.

### DSA (DeepSeek Sparse Attention)

- ~90% of attention entries in long contexts are redundant — DSA model matches its dense predecessor on benchmarks
- 1.5-2× attention computation reduction for long sequences; 128K contexts at ~half GPU cost — critical for reasoning-heavy agents
- **Training**: warm-up 1000 steps (14 sequences × 202,752 tokens each, lr 5e-3), then sparse adaptation on mid-training data/hyperparameters for **20B tokens** — tiny vs DeepSeek-V3.2's 943.7B; SFT-ties MLA on loss and benchmarks

### Ablation: Efficient Attention Variants (GLM-9B)

Baseline: GQA across 40 layers, fine-tuned at 128K context. Evaluated:
- **SWA Interleave**: fixed alternating full/windowed layers
- **Gated DeltaNet (GDN)**: linear attention via gated linear recurrence (quadratic → linear in seq len)
- **Search-based SWA Pattern** (PostNAS-inspired): beam search (beam 8, 2 layers/step, ~10 steps) over layer subsets for SWA conversion, evaluated on **RULER@16K**, generalized to all lengths
  - Found pattern: `SFSSFFSSSFFFFSSFSFFFFFFSFSFSSFSSFSFSSFSSS` (S=SWA, F=full)
  - Significantly outperforms fixed interleave

## Training Pipeline

1. **Pre-training**: 27T token corpus (code + reasoning prioritized early), extending to 28.5T total across all stages (incl. mid-training)
2. **Mid-training**: context 4K → 200K progressively; long-context agentic data for workflow stability
3. **Post-training**: sequential RL — **Reasoning RL → Agentic RL → General RL** — with **On-Policy Cross-Stage Distillation** between stages to prevent catastrophic forgetting (retain reasoning edge while becoming generalist)

## Asynchronous RL for Agentic Tasks

### Infrastructure (slime framework)

- Further decouples generation from training to maximize GPU utilization (built on GLM-4.5's decoupled rollout engines)
- Massive-scale trajectory exploration without synchronization bottlenecks
- **Scaling out**: flexible training via highly customizable rollouts
- **Scaling up**: tail-latency optimization for RL rollouts
- **Rollout robustness**: heartbeat-driven fault tolerance
- **Server-based multi-task training design**

### Async Stability Techniques

- **Token-in-Token-out vs Text-in-Text-out** — token-level fidelity in training loops (see [TITO](tito-agentic-rl.md))
- **Direct double-sided importance sampling (DIS) token masking** — tokens outside the trust region are masked (gradient = 0), not clipped; bounds policy drift under lag (see [SAO](single-rollout-async-agentic-rl.md))
- **Dropping off-policy and noisy samples**
- **DP-aware routing for acceleration**

## Environment Scaling for Agents

- **SWE environments**: real software engineering tasks
- **Terminal environments**: synthesized from seed data + web corpus
- **Search tasks** with context management for search agents
- **Slide generation**: rejection sampling + masking-based refinement

## Chinese Chip Adaptation (Day-1)

Full-stack optimization across 7 domestic platforms: Huawei Ascend, Moore Threads, Hygon, Cambricon, Kunlunxin, MetaX, Enflame:
- Mixed-precision **W4A8 quantization**
- High-performance fusion kernels
- Specialized inference engine optimizations

## Evaluation

- **Intelligence Index v4.0 = 50** (10 evals: GDPval-AA, τ²-Bench Telecom, Terminal-Bench Hard, SciCode, AA-LCR, AA-Omniscience, IFBench, HLE, GPQA Diamond, CritPt) — first open model at 50; up from GLM-4.7's 42
- **LMArena**: #1 open model in both Text Arena and Code Arena; on par with Claude Opus 4.5 and Gemini 3 Pro
- **8 ARC benchmarks** (HLE, SWE-bench Verified, SWE-bench Multilingual, Terminal-Bench 2.0, BrowseComp, MCP-Atlas, τ²-Bench, Vending Bench 2): ~20% avg improvement over GLM-4.7; comparable to Claude Opus 4.5 / GPT-5.2 (xhigh); better than Gemini 3 Pro
- **Vending-Bench 2** (1-year simulated business): #1 open, $4,432 final balance — approaches Claude Opus 4.5; strong long-term planning
- **CC-Bench-V2** (internal): significantly outperforms GLM-4.7 on frontend/backend/long-horizon

## Connections

- [GLM-4.5](glm-4.5.md) — predecessor; expert-model iteration + self-distillation; slime infra origins
- [DeepSeek-V3.2](deepseek-v3.2.md) — DSA introduced there; GLM-5 adapts with 50× less training budget
- [Single-Rollout Async Optimization](single-rollout-async-agentic-rl.md) — SAO deployed in GLM-5.2's agentic RL pipeline (same team lineage)
- [Async RL Training Landscape](async-rl-training-landscape.md) — async RL infra survey; slime is a production example
- [Stabilizing RL with LLMs](stabilizing-rl-llms.md) — DIS token masking relates to token-level stabilization
- [TITO: Agentic RL](tito-agentic-rl.md) — Token-in-Token-out fidelity in async agentic loops
- [DeepSeek-V3](deepseek-v3.md) — MoE + MLA lineage; DeepSeek-V3 head dim chosen by H800 roofline (contrast)
- [Multi-Head Latent Attention](multi-head-latent-attention.md) — MLA basis; Muon Split interaction finding
- [Muon Optimizer](muon-optimizer.md) — Muon used in GLM-5 training; Muon Split is a per-head variant
- [Kimi K2](kimi-k2.md) — MuonClip (QK-clip) vs GLM-5's Muon Split — two solutions to Muon instability
- [LLM Architecture Comparison](llm-architecture-comparison.md) — MLA/DSA variants across models
