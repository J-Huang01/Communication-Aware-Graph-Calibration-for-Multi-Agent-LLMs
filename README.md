# CAGE-Cal
Counterfactual Graph for Multi-Agent LLM Calibration

## Overview

CAGE-Cal is a calibration framework for multi-agent LLM systems. Existing
methods treat agreement as evidence: when many agents in a panel give the
same answer, that answer is assumed reliable. This breaks once agents
communicate, because communication induces correlated failures and false
consensus, so the same vote share can mean reliable agreement under one
topology and over-confidence under another. CAGE-Cal goes beyond counting
agreement: for each query it compares an observed post-communication agent
graph with a matched counterfactual no-communication graph and calibrates
confidence from the difference.

The framework consists of three stages:

**Panel Generation** — A canonical population of 20 agents (4 backbone
LLMs x 5 prompting roles) answers each query under five communication
topologies (iid, debate, chain, hub-spoke, tree). Panels are graded by an
answer-equivalence judge to produce correctness labels.

**Counterfactual Graph Calibration** — A shared two-tower relational
encoder encodes the
observed graph and the matched iid counterfactual graph. The calibration
head maps the panel embeddings and their difference to the probability that
the plurality answer is correct, refined by a per-benchmark post-hoc step.

**Confidence-Routed Selection** — CAGE-Select runs each query under
multiple topologies and returns the answer with the highest CAGE-Cal
confidence, turning topology choice into a per-query decision.

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

### Train CAGE-Cal (Stage 2)

```bash
python scripts/run_cage_cal.py \
  --n_seeds 10 --epochs 15 --device cuda:0
```

### Leave-One-Topology-Out Generalization

```bash
python scripts/cage_cal_loto.py \
  --n_seeds 5 --epochs 15 --device cuda:0
```

### CAGE-Select Inference (Stage 3)

```bash
python scripts/cage_select.py \
  --n_seeds 10 --epochs 15 --device cuda:0
```

Results are written under `results/panels/`. Per-query correlation matrices
and sentence embeddings are cached under `cache/` and reused across runs.

## Project Structure

```
cage-cal/
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
    ├── cage_gnn_hypergraph.py    # CAGE-Cal model + graph construction
    ├── cage_gnn_hyper_selfcal.py # Calibration-aware training loop
    ├── run_cage_cal.py           # Stage 2: CAGE-Cal training
    ├── cage_cal_loto.py          # Leave-one-topology-out evaluation
    └── cage_select.py            # Stage 3: confidence-routed selection
```
