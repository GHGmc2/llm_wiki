---
title: "Overview"
type: summary
tags: [meta]
created: 2026-05-02
updated: 2026-05-02
sources: [raw/scalable-training-moe-megatron-core.pdf]
status: stable
---

# Overview

*This wiki is just getting started. The overview will grow into a high-level synthesis of everything in the wiki — the big picture, key themes, and how everything connects.*

## Current State

The wiki has its first content: a deep dive into NVIDIA's Megatron-Core MoE training stack and the "Three Walls" framework for scaling Mixture-of-Experts models.

### Key Knowledge

**Megatron-Core MoE** is NVIDIA's open-source stack for training large-scale MoE models. The wiki covers the full paper across 7 interconnected pages:

- **[Megatron-Core MoE](learning/megatron-core-moe-scalable-training.md)** — Complete paper summary: architecture, performance numbers, all sections
- **[Parallel Folding](learning/moe-parallel-folding.md)** — How attention and MoE layer parallelism are decoupled to break the EP <= DP constraint
- **[Memory Wall](learning/moe-memory-wall.md)** — 7 memory optimization techniques reducing DeepSeek-V3 from 199.5 GB/GPU to feasible levels
- **[Communication Wall](learning/moe-communication-wall.md)** — EP all-to-all, DeepEP vs HybridEP dispatchers, hardware dependency
- **[Compute Efficiency Wall](learning/moe-compute-efficiency-wall.md)** — Grouped GEMM, CUDA Graphs, ECHO, the bottleneck shift
- **[FP8/FP4 Training](learning/moe-fp8-fp4-training.md)** — Four precision recipes (per-tensor FP8, blockwise FP8, MXFP8, NVFP4), MoE-specific challenges
- **[Performance Best Practices](learning/moe-performance-best-practices.md)** — Three-phase optimization workflow, DeepSeek-V3 case study on GB200 vs H100

1. **Add more sources** — drop articles, PDFs, or notes into `raw/` and ask the LLM to process them
2. **Ask a question** — the LLM will search the wiki, synthesize an answer, and offer to file it
3. **Share an insight** — anything mentioned in conversation can be filed into the wiki

## Wiki Structure

| Category | Purpose |
|----------|---------|
| [Health](health/) | Physical health, mental health, habits, routines |
| [Goals](goals/) | Short and long-term goals, progress tracking |
| [Learning](learning/) | Topics being studied, courses, skills, notes |
| [Reading](reading/) | Book notes, article summaries, reading lists |
| [Journal](journal/) | Time-based reflections and personal entries |
| [People](people/) | Important people, relationships, networks |
| [Ideas](ideas/) | Brainstorms, project ideas, things to explore |
