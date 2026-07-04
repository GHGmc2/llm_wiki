---
title: "A Survey on Auto-Parallelism of Neural Networks Training"
type: source-note
tags: [auto-parallelism, distributed-training, parallelism, survey, dp, tp, pp, mcts, rl]
created: 2026-05-03
updated: 2026-05-03
sources: [raw/auto-parallelism-survey.pdf]
status: stable
---

# Auto-Parallelism Survey

**Source**: Liang et al., 2023. 22 pages. TechRxiv. Comprehensive taxonomy of auto-parallelism systems.

## Key Points

- **Auto-parallelism**: Automatically generates optimal parallelism strategies for a given model and cluster, replacing manual tuning
- **Formal definition**: Input = computation graph G + device topology D → Output = partition set P, sub-graphs, pipeline schedules
- Survey covers 30+ systems across 7 searching methods: DP, MCTS, RL, ILP, Sharding Propagation, Greedy, Grid Search
- Key challenge: exponential search space — model size N, tensor dimensions K, devices D → O(D^(N×K)) possibilities

## Formal Model

**Computation graph**: DAG `G = (V, E)` where nodes `v_i` are operators (matmul, conv, etc.) and edges `e_ij` are data dependencies.

**Device topology**: Undirected graph `D = (V_D, E_D)` where nodes are devices (GPU, CPU) and edges are hardware connections (NVLink, PCIe, IB) with bandwidth labels.

**Partition**: `p_i = (index, g)` where `index` is a 2D array of split indices and `g` is a device group. Applying `P` to `G` generates per-device sub-graphs with inserted communication operators.

## Parallelism Schemes Taxonomy

### Intra-Operator (Tensor Partitioning)

Communication cost per training step for a matrix multiplication $T_{out} = T_{in} W$, where $b$ = batch size, $w_{in}, w_{out}$ = matrix dimensions, $p$ = parallelism degree:

| Scheme | What's Partitioned | Collective Pattern | Communication Cost |
|--------|-------------------|-------------------|---------------------|
| **Vanilla DP** | Batch dimension | AllReduce (gradients) | $\frac{2(p-1)}{p} \cdot w_{in} w_{out}$ |
| **ZeRO-1** | Optimizer states | ReduceScatter + AllGather | $\frac{2(p-1)}{p} \cdot w_{in} w_{out}$ |
| **ZeRO-2** | + Gradients | ReduceScatter + AllGather | $\frac{2(p-1)}{p} \cdot w_{in} w_{out}$ |
| **ZeRO-3** | + Parameters | ReduceScatter + AllGather | $\frac{3(p-1)}{p} \cdot w_{in} w_{out}$ |
| **Row-TP** | Output activation | AllReduce ($T_{out}$) | $\frac{2(p-1)}{p} \cdot b \cdot w_{out}$ |
| **Column-TP** | Input activation | AllReduce ($E_{in}$) | $\frac{2(p-1)}{p} \cdot b \cdot w_{in}$ |
| **2D-TP** | Weight, grads, input, output | Broadcast + Reduce | $\frac{3\log p}{2\sqrt{p}} \cdot \bigl(b w_{in} + w_{in} w_{out}\bigr)$ |
| **3D-TP** | All tensors | ReduceScatter + AllGather | $\frac{3(p^{1/3}-1)}{p} \cdot \bigl(b w_{in} + w_{in} w_{out} + b w_{out}\bigr)$ |

### Inter-Layer Tensor Redistribution

When layers use different parallelism strategies, tensors must be redistributed between layers (communication volume as a function of batch size $b$, output width $w_{out}$, and parallelism degree $p$):

| Layer $l$ → $l+1$ | DP | Row-TP | Column-TP | 2D-TP | 3D-TP |
|-------------------|-----|--------|-----------|-------|-------|
| **DP** | $0$ | $\frac{2(p-1)}{p}bw_{out}$ | $\frac{2(p-1)}{p^2}bw_{out}$ | $\frac{2(1-p^{-1/2})}{p}bw_{out}$ | $\frac{2(1-p^{-1/3})}{p}bw_{out}$ |
| **Row-TP** | $\frac{p-1}{p}bw_{out}$ | $\frac{p-1}{p}bw_{out}$ | $0$ | $\frac{p-1}{p}bw_{out}$ | $\frac{p-1}{p}bw_{out}$ |
| **Column-TP** | $\frac{2(p-1)}{p}bw_{out}$ | $0$ | $\frac{p-1}{p}bw_{out}$ | $\frac{2(1-p^{-1/2})}{p}bw_{out}$ | $\frac{2(1-p^{-1/3})}{p}bw_{out}$ |
| **2D-TP** | $\frac{2(1-p^{-1/2})}{p}bw_{out}$ | $\frac{2(1-p^{-1/2})}{p}bw_{out}$ | $\frac{p-1}{p}bw_{out}$ | $0$ | $\frac{2}{p}bw_{out}$ |
| **3D-TP** | $\frac{2(1-p^{-1/3})}{p}bw_{out}$ | $\frac{2(1-p^{-1/3})}{p}bw_{out}$ | $\frac{p-1}{p}bw_{out}$ | $\frac{2}{p}bw_{out}$ | $0$ |

Redistribution uses AllGather or AllToAll depending on the source and target partition patterns. Values represent communication volume — lower is better. The diagonal ($0$) means no redistribution needed for same-strategy adjacent layers.

### Inter-Operator (Graph Partitioning)

- **Inter-layer MP**: Splits model layers across devices. Only activations communicated between successive layers on different devices.
- **Pipeline Parallelism (PP)**: Well-scheduled inter-layer MP with micro-batching to overlap computation. Key difference from MP: PP uses pipelining to hide communication latency.
- **Hybrid**: Combines intra-operator and inter-operator — optimal strategies often mix both.

### Communication Analysis Insight

From the original OWT (One Weird Trick) paper finding: **DP is better for CNN** (small weights, large activations), **TP is better for FC** (large weights, small activations). This inspired auto-parallelism to search for per-layer optimal strategies rather than one-size-fits-all.

## Strategy Searching Methods (30+ Systems)

Let $N = |V|$ be the number of operators in the computation graph, $K$ the maximum tensor dimension, and $D$ the number of devices.

| Method | Representative Systems | Worst-Case Complexity | Approach |
|--------|----------------------|----------------------|----------|
| **Dynamic Programming** | OptCNN, Tofu, TensorOpt, AccPar, PaSE, DAPPLE, PipeDream, RaNNC, Piper | $\mathcal{O}(N K^3)$ to $\mathcal{O}(N^5)$ | DP over layers or operators |
| **MCMC** | FlexFlow | $\mathcal{O}(\text{iterations} \cdot \text{samples})$ | Markov Chain Monte Carlo over SOAP space |
| **MCTS + Interaction Net** | Automap | Minutes (neural cost model) | Neural network predicts performance, guides tree search |
| **Reinforcement Learning** | ColocRL, HDP, GDP, Spotlight, REGAL | Hours (RL training) | RL agent learns device placement |
| **ILP** | Alpa, Pesto | $\mathcal{O}(N^5)$ (simplified) | Two-level: inter-op ILP + intra-op DP |
| **Sharding Propagation** | GSPMD | $\mathcal{O}(N)$ | Compiler propagates user annotations |
| **Greedy / Heuristic** | Neo | Not given analytically | Karmarkar-Karp + greedy selection |
| **Grid Search** | Chimera, DistIR | Exponential (exhaustive) | Brute force over config space |

### Key System Details

**OptCNN / Tofu**: DP + TP via graph elimination and regeneration. $\mathcal{O}(N K^3)$ complexity. Foundation for many later systems.

**FlexFlow**: MCMC (Markov Chain Monte Carlo) search over SOAP space (Sample, Operator, Attribute, Parameter). Handles arbitrary parallelism combinations.

**Alpa**: Two-level hierarchy: inter-operator (PP + DP) uses ILP, intra-operator (TP) uses DP. $\mathcal{O}(N^5)$ but simplified for practical use.

**GSPMD**: Compiler-based sharding propagation — user annotates key tensors, compiler propagates. $\mathcal{O}(N)$ linear complexity but requires manual annotation.

**Automap**: MCTS with interaction network cost model. Neural network predicts performance of candidate strategies, guiding tree search. A few minutes for large models.

**PipeDream**: DP + PP with dynamic programming. Models pipeline bubbles in cost function. $\mathcal{O}(N^3 m_k^2)$ per hierarchy level, where $m_k$ is the device count at that level.

**D-Rec (Double Recursive)**: $\mathcal{O}(N)$ — sub-linear complexity by exploiting repeated substructures in neural networks.

### Heterogeneous Topology

Load balancing across different device types (CPU + GPU of multiple generations). Key systems:
- **AccPar**: Computes partition ratios for each device type, then partitions layer-by-layer
- **DeepSpeed**: Heuristically offloads optimizer updates to CPU
- **HeterPS**: RL-based device selection per layer, supports DP + PP only

## Challenges and Future Directions

1. **Expanding search space**: Most systems only support a subset of parallelism schemes. The full space (DP + ZeRO + TP (1D/2D/3D) + PP + SP + CP + EP) remains unexplored
2. **Heterogeneous topology**: Few systems handle mixed device types with load balancing
3. **Communication topology awareness**: Intra-node vs inter-node bandwidth must inform strategy selection
4. **Dynamic adaptation**: Strategies that adapt to changing cluster conditions during training
5. **Cost model accuracy**: Profiling-based models are accurate but slow; symbolic models are fast but approximate

## Connections

- **[GSPMD](gspmd.md)** — Compiler-based sharding propagation (covered in detail)
- **[PartIR](partir.md)** — Composable SPMD tactics (scheduling-based)
- **[TOAST](toast.md)** — Static analysis + MCTS auto-partitioning
- **[Scaling Techniques Overview](usp-scaling-techniques-overview.md)** — DP, ZeRO, TP, PP fundamentals
- **[Parallel Folding](megatron-core-moe.md)** — Megatron-Core's EP+TP decoupling
