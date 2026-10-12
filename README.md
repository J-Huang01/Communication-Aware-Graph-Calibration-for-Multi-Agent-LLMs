# CAGE-Cal

Communication-Aware Graph Calibration for Multi-Agent LLMs

## Overview

**CAGE-Cal (Communication-Aware Agent-Graph Calibration)** estimates the
probability that a heterogeneous multi-agent panel's plurality answer is
correct across diverse communication topologies. Agreement alone can be a
misleading confidence signal. The paper identifies two opposing calibration
failures:

- **Diversity-induced under-confidence (DUC):** incorrect agents spread
  across different wrong answers, so vote share underestimates panel reliability.
- **Communication-induced over-confidence (COC):** agents converge to a wrong
  answer through influence, so vote share overestimates panel reliability.

For each query, CAGE-Cal compares an observed post-communication agent graph
with a matched IID graph in which the same agents answer without
communication. Here, the counterfactual is a matched no-communication
reference execution. Pairwise and group-level dependencies help the
calibrator distinguish benign disagreement from false consensus.

The method has three components:

**Agent Dependency Graph Construction** — Build observed and IID reference
graphs over matched agent identities for the same query and rollout.
Pairwise edges capture communication adjacency and query-dependent
correctness correlations estimated from nearby training queries. Hyperedges
capture group-level dependencies through shared model families, prompting
roles, answer clusters, and communication exposure.

**Counterfactual Shift Calibration** — A shared calibration agent graph
encoder combines pairwise message passing with hypergraph convolution.
The calibration head uses the observed and reference graph embeddings,
their difference, and benchmark identity to estimate plurality-answer
correctness. Training combines smoothed binary cross-entropy with a Brier
penalty, followed by per-benchmark post-hoc calibration.

**Confidence-Guided Answer Selection (CAGE-Select)** — Generate candidate
panels under all evaluated topologies and select the plurality answer with
the highest CAGE-Cal confidence. Selection operates on completed panel
outputs, and its inference cost includes generating all candidate panels.

Experiments use five benchmarks (TriviaQA, TruthfulQA, MMLU-Pro, GSM8K, and
BIG-Bench Hard) and five communication topologies (IID, debate, chain,
hub-spoke, and tree). Panels are drawn from a canonical pool of 20 agents
(4 backbone LLMs x 5 prompting roles) and graded by an answer-equivalence
judge to produce correctness labels.

## Setup

```bash
conda create -n cagecal python=3.12 -y && conda activate cagecal
pip install -r requirements.txt
```

Set environment variables for LLM access:

```bash
# panel backbones served by local vLLM
export VLLM_QWEN_URL="http://localhost:8001/v1"
export VLLM_LLAMA_URL="http://localhost:8002/v1"
export VLLM_GEMMA_URL="http://localhost:8003/v1"
export VLLM_PHI_URL="http://localhost:8004/v1"
# answer-equivalence judge for grading
export JUDGE_PROVIDER="openrouter"
export OPENROUTER_API_KEY="your-key"
# panel size: natural (N=20) or matched (N=10)
export MAS_UQ_AGENT_CONFIG="natural"
```

## Quick Start

### Generate Panels

```bash
python scripts/run_main_rollout.py \
  --out-dir results/panels \
  --n-per-bench 500 --n-rollouts 3 --temperature 0.7
```

### Grade and Deduplicate

```bash
python scripts/grade_main_rollout.py --root results/panels --workers 8
python scripts/dedup_rollouts.py results/panels
```

### Train CAGE-Cal

```bash
python scripts/run_cage_cal.py \
  --n_seeds 10 --epochs 15 --device cuda:0
```

### Leave-One-Topology-Out Generalization

```bash
python scripts/cage_cal_loto.py \
  --n_seeds 5 --epochs 15 --device cuda:0
```

### Confidence-Guided Answer Selection with CAGE-Select

```bash
python scripts/cage_select.py \
  --n_seeds 10 --epochs 15 --device cuda:0
```

Results are written under `results/panels/`. Per-query correlation matrices
and sentence embeddings are cached under `cache/` and reused across runs.

## Project Structure

```
Communication-Aware-Graph-Calibration-for-Multi-Agent-LLMs/
├── cage_cal/                     # Core library
│   ├── agent_population.py       # Canonical agents (4 backbones x 5 roles)
│   ├── topology.py               # iid / debate / chain / hub-spoke / tree
│   ├── roles.py, roles_v2.py     # Atomic prompting roles
│   ├── benchmarks/               # TriviaQA, TruthfulQA, MMLU-Pro, GSM8K, BBH
│   ├── judge.py, scoring.py      # Answer-equivalence grading
│   ├── per_query_W.py            # Per-query failure-correlation W(x)
│   ├── cage_cal_features.py      # Communication adjacency + panel features
│   ├── calibration.py            # ECE / AUROC / AUARC metrics
│   └── inference/vllm_client.py  # OpenAI-compatible LLM client
└── scripts/
    ├── run_main_rollout.py       # Panel generation
    ├── grade_main_rollout.py     # Grading
    ├── dedup_rollouts.py         # Deduplication
    ├── cage_gnn_hypergraph.py     # Agent dependency graphs + shared encoder
    ├── cage_gnn_hyper_selfcal.py  # Counterfactual shift calibration learning
    ├── run_cage_cal.py           # CAGE-Cal training + post-hoc calibration
    ├── cage_cal_loto.py          # Leave-one-topology-out evaluation
    └── cage_select.py            # Confidence-guided answer selection
```
