---
title: "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"
type: source-note
tags: [llm, deepseek, reinforcement-learning, grpo, reasoning, r1, chain-of-thought, distillation]
created: 2026-05-02
updated: 2026-05-02
sources: [raw/DeepSeek-R1.pdf]
status: stable
---

# DeepSeek-R1

**Source**: DeepSeek-AI, January 2025 (revised January 2026). 86 pages. Built on DeepSeek-V3-Base (671B MoE, 37B active).

## Key Points

- **DeepSeek-R1-Zero**: Pure RL (no SFT) on V3-Base — reasoning behaviors (self-verification, reflection, "aha moment") emerge spontaneously
- **GRPO**: Group Relative Policy Optimization — group-based advantage, no critic model needed
- **DeepSeek-R1**: Multi-stage pipeline: cold-start SFT → RL → rejection sampling (600K) → SFT → RL with preference rewards
- **Rule-based rewards only**: Accuracy (verifiable) + format (think/answer tags) — no neural reward models for reasoning
- **Distillation > RL for small models**: Distilled 1.5B model beats GPT-4o on math; RL on small models can't match distillation
- **Training cost**: $294K total ($202K R1-Zero + $10K SFT data + $82K R1) on H800 at $2/hr
- **Results**: AIME 2024 79.8%, Codeforces 96.3% (2029 rating), MATH-500 97.3%, USAMO qualification

## DeepSeek-R1-Zero: Pure RL Without SFT

The key experiment: skip SFT entirely and apply RL directly to DeepSeek-V3-Base. The hypothesis: human-defined reasoning patterns may limit model exploration, while unrestricted RL can better incentivize emergent reasoning.

### GRPO: Group Relative Policy Optimization

GRPO replaces the critic model (used in PPO) with group-based advantage estimation. The full objective:

```
J_GRPO(θ) = E[ 1/G Σ_i min(π_θ/π_old · A_i, clip(π_θ/π_old, 1-ε, 1+ε) · A_i) - β·D_KL(π_θ||π_ref) ]
```

Where:
- `A_i = (r_i - mean({r_1,...,r_G})) / std({r_1,...,r_G})` — group-normalized advantage
- `D_KL` estimates KL divergence without requiring a separate critic
- `π_ref` is updated every 400 steps to the latest policy

1. For each question, sample a **group** of G=16 outputs from the current policy
2. Score each output using rule-based rewards
3. Compute advantage as normalized reward within the group
4. Optimize policy to maximize clipped advantage while staying close to reference policy

**Key advantage over PPO**: No critic model needed — eliminates a major source of complexity and potential reward hacking.

**Training config**: learning rate 3e-6, KL coefficient 0.001, temperature 1, 16 outputs per question, max 32K tokens (later 65K), 10,400 total steps, batch size 512. Each rollout generates 8,192 outputs → split into 16 mini-batches → single inner epoch. Reference model updated every 400 steps.

### Reward Design

Only **rule-based rewards** — two components with equal weight:

| Reward | What it measures |
|--------|-----------------|
| **Accuracy** | Final answer correctness (math: boxed answer verification; code: test case pass/fail) |
| **Format** | Reasoning in `<think>...</think>`, answer in `<answer>...</answer>` tags |

**Deliberately no neural reward models** for reasoning — the authors observed that neural RMs are susceptible to reward hacking at scale and retraining them adds complexity.

### Emergent Behaviors

During RL training, R1-Zero spontaneously developed:

- **Self-verification**: The model checks its own work mid-reasoning
- **Reflection**: "Wait, let me reconsider..." moments
- **Alternative exploration**: Trying multiple approaches within a single response
- **The "aha moment"**: A distinct shift where the model starts using "wait" to flag re-evaluation — characterized by a sudden increase in reflection behavior (Figure 9b in paper)

**Quantitative trajectory**: AIME 2024 pass@1 jumped from 15.6% → 77.9%. Cons@16 reached 86.7%, surpassing average human competitors.

**Thinking time**: Response length steadily increased throughout training — the model autonomously learned that more thinking = better results.

### Limitations of R1-Zero

- **Poor readability**: Responses can be hard to follow
- **Language mixing**: English/Chinese interleaved in a single chain-of-thought
- **Limited non-reasoning ability**: Pure RL reasoning training doesn't transfer to writing, open QA

## DeepSeek-R1: Multi-Stage Pipeline

To address R1-Zero's limitations, R1 uses a four-stage pipeline:

### Stage 1: Cold-Start SFT
- Collect thousands of high-quality, diverse reasoning prompts
- For each prompt, generate multiple trajectories from R1-Zero at temperature 1.0
- Filter: keep only correct answers + readable format (sympy for math, repetition/language-mixing detection)
- **Human annotators** convert reasoning traces into natural conversational style
- These human examples prompt an LLM to rewrite additional data in similar style
- All LLM outputs undergo a second round of human verification
- Result: high-quality, human-aligned reasoning trajectories for SFT warm-start

### Stage 2: Reasoning-Oriented RL (→ R1-Dev1, Dev2)
- Apply GRPO with rule-based rewards on reasoning tasks
- Add **language consistency reward**: Penalize language mixing in CoT
- R1-Dev2: Extends RL training further — AIME 74.0%, Codeforces 90.5 percentile

### Stage 3: Rejection Sampling + SFT (→ R1-Dev3)
- Generate many outputs from R1-Dev2, filter using rule-based rewards
- Combine reasoning data (600K curated samples) with non-reasoning data (200K writing, QA, etc.)
- SFT on this mixed dataset to restore general capabilities

### Stage 4: RL with Diverse Rewards (→ R1)
- Reasoning data: rule-based rewards (same as R1-Zero)
- General data: **helpful reward model** (66K preference pairs, trained on DeepSeek-V3 judgments) + **safety reward model** (106K safe/unsafe annotations)
- Language consistency reward applied to all data
- Final 400 steps incorporate general instruction data

**Total reward**: `Reward = Reward_reasoning + Reward_general + Reward_language`

### Reward Models

| Model | Training Data | Architecture |
|-------|--------------|-------------|
| Helpful RM | 66K preference pairs (DeepSeek-V3 judged, Δ > 1) | DeepSeek-R1 + reward head |
| Safety RM | 106K safe/unsafe annotations | DeepSeek-R1 + reward head (pointwise) |

Length bias mitigation: chosen and rejected responses have comparable lengths.

## Distillation

To enable broader access, DeepSeek distilled R1 into smaller models from 800K R1-generated reasoning samples:

| Model | Based on | AIME 2024 | Codeforces Rating |
|-------|---------|-----------|-------------------|
| DeepSeek-R1-Distill-Qwen-1.5B | Qwen2.5-Math-1.5B | 28.9 | 954 |
| DeepSeek-R1-Distill-Qwen-7B | Qwen2.5-Math-7B | 55.5 | 1189 |
| DeepSeek-R1-Distill-Llama-8B | Llama-3.1-8B | 50.4 | 1205 |
| DeepSeek-R1-Distill-Qwen-14B | Qwen2.5-14B | 69.7 | 1481 |
| DeepSeek-R1-Distill-Qwen-32B | Qwen2.5-32B | 72.6 | 1691 |
| DeepSeek-R1-Distill-Llama-70B | Llama-3.3-70B-Instruct | 70.0 | 1633 |

**Comparison baselines**: GPT-4o-0513: AIME 9.3, Codeforces 759. Claude 3.5 Sonnet: AIME 16.0, Codeforces 717.

**The 1.5B model surpasses GPT-4o on AIME** — a model with 1.5B parameters beating the best closed-source models.

### Distillation vs RL on Small Models

A critical experiment (Appendix F): compare distilling from R1 vs applying large-scale RL directly to small models.

| Approach | Qwen2.5-32B AIME 2024 |
|----------|----------------------|
| Large-scale RL (10K+ steps) on 32B base | On par with QwQ-32B-Preview |
| Distilled from DeepSeek-R1 | **Significantly better** |

**Two conclusions**:
1. Distilling powerful models into smaller ones yields excellent results — more economical and effective than RL on small models
2. Advancing beyond human intelligence still requires more powerful base models and larger-scale RL

**Historical validation**: Qwen2-Math-7B-Zero (trained pre-o1 release, August 2024, ~10K GRPO steps) significantly outperformed GPT-4o and Qwen2-Math-7B-Instruct on AIME 2024 (22.3% vs 9.3% vs 7.9%), proving that RL-driven reasoning emergence works even before reasoning models existed publicly.

## Training Cost

| Stage | H800 GPU Hours | Cost (@ $2/hr) |
|-------|---------------|----------------|
| DeepSeek-R1-Zero | 101K | $202K |
| SFT data creation | 5K | $10K |
| DeepSeek-R1 | 41K | $82K |
| **Total** | **147K** | **$294K** |

Remarkably economical for a frontier reasoning model — about 5% of typical pretraining cost.

## Self-Evolution Analysis

### MATH Difficulty Breakdown

The paper analyzes R1-Zero's performance on MATH stratified by difficulty (1-5):

| Difficulty | Early Training | Late Training | Improvement |
|-----------|---------------|---------------|-------------|
| Level 1-2 | ~0.90-0.95 | Stable | Small |
| Level 3 | ~0.78 → 0.95 | | +0.17 |
| Level 4 | ~0.78 → 0.95 | | +0.17 |
| Level 5 | ~0.55 → 0.90 | | **+0.35** |

Hard problems (level 5) show the most dramatic improvement — the model doesn't just get better at easy tasks, it genuinely learns to tackle harder reasoning.

### Reflective Word Analysis

Three human experts selected representative reflective words: "wait", "mistake", "however", "but", "retry", "error", "verify", "wrong", "evaluate", "check". Across training:

- Reflective word frequency increased **5-7×** from start to end
- The word **"wait"** shows a distinct pattern: nearly absent early → occasional (steps 4000-7000) → **significant spikes after step 8000**
- Different reflection patterns emerge at different stages — the model learns different forms of self-correction sequentially

### The "Aha Moment"

At step ~8000, R1-Zero exhibits a qualitative shift: the model pauses mid-solution, says "Wait, wait. Wait. That's an aha moment I can flag here. Let's reevaluate this step-by-step...", and then re-derives the solution. This is not prompted — it's an emergent behavior from RL optimization of verifiable rewards.

## Language Consistency (LC) Reward

Without LC reward, language mixing gradually increases during RL (the base model was trained on English+Chinese). The LC reward directly penalizes mixed-language CoT outputs:

- **With LC reward**: Stable language consistency, slight degradation on code benchmarks (0.46→0.38 LiveCodeBench), minimal impact on math
- **Without LC reward**: Language consistency deteriorates; better code performance but less readable

The trade-off: human readability vs raw benchmark performance. R1 chooses readability.

## RL Scaling: The Longer You Train, The Better It Gets

R1-Zero's response length steadily increases throughout training (Figure 1b), from ~2K tokens to ~15K+ tokens. This is purely intrinsic — no external adjustment to length penalties. The model autonomously discovers that longer thinking produces better answers.

**Training dynamics**:
- Steps 0-2K: Short responses, low accuracy
- Steps 2K-8K: Gradual length increase, steady accuracy improvement
- Step 8.2K: **Discontinuous jump** — max output length increased from 32K to 65K tokens → significant performance jump
- Steps 8K-10.4K: Continued improvement, reflective behaviors intensify

## Safety Evaluation

The paper includes comprehensive safety analysis across 4 categories (Discrimination, Illegal, Harmful, Ethical) and 50 languages:

| Model (with risk control) | Overall Unsafe Rate |
|--------------------------|-------------------|
| DeepSeek-V3 + risk control | 5.3% (best) |
| Claude-3.7-Sonnet | 10.7% |
| DeepSeek-R1 + risk control | 8.5% |
| o1 (2024-12-17) | 9.0% |

Across 50 languages with risk control: DeepSeek-V3 86.5%, DeepSeek-R1 85.9%, Claude-3.7-Sonnet 88.3% — approaching state-of-the-art.

## Limitations

- **Structure output & tool use**: No native structured output; tool-calling not optimized
- **Software engineering**: RL not applied at scale to SWE tasks due to long eval times → R1 doesn't show huge improvement over V3 on SWE
- **Reward hacking**: Observed in Figure 6 — reward score increases while Codeforces performance decreases when using neural reward models. Rule-based rewards are essential
- **Few-shot degradation**: Few-shot prompting consistently degrades R1's performance — zero-shot recommended
- **Language mixing**: LC reward mitigates but doesn't eliminate
- **Model capacity matters**: RL from base models works best with sufficiently large models; small models benefit more from distillation

## Key Findings from Discussion (Appendix G)

1. **RL effectiveness depends on base model capacity**: Larger base models show greater performance gains from pure RL
2. **Verifiers are critical**: Rule-based RMs and LLM-as-judge (for concise-answer tasks) are robust; open-ended generation lacks reliable verifiers
3. **Both RL and SFT are indispensable**: RL discovers reasoning patterns human annotation can't capture; SFT handles tasks without reliable reward signals
4. **Distillation is the pragmatic path**: For smaller models, distilling from R1 >>> applying RL directly

## The Core Insight

> "We believe that the key to unlocking reasoning potential lies not in large-scale human annotation but in the provision of hard reasoning questions, a reliable verifier, and sufficient computational resources for reinforcement learning."

### Generalization to Unseen Data

R1 was evaluated on AIME 2025 (released after training) to test generalization:

| Model | AMC 12 2024 | AIME 2025 | USAMO Index |
|-------|------------|-----------|-------------|
| GPT-4o | 84.0/150 | 2.0/15 | 104.0 |
| DeepSeek-V3 | 98.3/150 | 3.3/15 | 131.3 |
| OpenAI o1 | 141.0/150 | 12.0/15 | 261.0 |
| **DeepSeek-R1** | **143.7/150** | **11.3/15** | **256.7** |

USAMO qualification threshold: 251.5. R1 qualifies — positions among top-tier US high school math students. R1 scores 143.7 on AMC 12 (nearly perfect) and approaches o1 on AIME 2025 (75% solve rate vs 80%).

### Human Comparison

| Benchmark | R1 | R1-Zero | Human |
|-----------|-----|---------|-------|
| AIME 2024 | 79.8% | 77.9% | ~50% (avg competitor) |
| Codeforces | 96.3%ile | 80.4%ile | 50%ile |
| GPQA Diamond | 71.5% | 75.8% | ~80% (PhD w/ web access) |

R1 surpasses average human competitors on math and coding. Still trails PhD-level humans on GPQA — but this is with web access for humans.

This directly challenges the SFT-first paradigm. The paper shows that pre-trained checkpoints inherently possess substantial reasoning potential — the job of post-training is to **incentivize**, not to teach.

## Connections

- [DeepSeek-V4 Post-Training](deepseek-v4-post-training.md) — V4's OPD pipeline uses the same GRPO algorithm and specialist training philosophy
- [DeepSeek-V3 Insights](deepseek-v3-insights.md) — V3 base architecture that R1 builds on
- [DeepSeek-V4 Technical Report](deepseek-v4-technical-report.md) — V4's Think Max mode is a direct descendant of R1's reasoning paradigm
- [Megatron-Core MoE](megatron-core-moe-scalable-training.md) — Training infrastructure for the base model
