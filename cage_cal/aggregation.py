from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


# ----- Helpers -------------------------------------------------------------

def _normalize(s: str) -> str:
    return (s or "").strip().lower()


def _vote(rollouts: list[dict[str, Any]]) -> tuple[str, float]:
    """Aggregate one rollout (multiple agents) by majority vote.

    rollouts here = list of agent outputs from one MAS run, NOT multiple rollouts.
    Returns (majority_answer, vote_share).
    """
    if not rollouts:
        return "", 0.0
    answers = [_normalize(r.get("answer", "")) for r in rollouts]
    counter = Counter(answers)
    top, top_count = counter.most_common(1)[0]
    return top, top_count / len(answers)


# ----- 1. Majority vote ----------------------------------------------------

def majority_vote(rollout: dict[str, dict]) -> tuple[str, float]:
    """rollout = {agent_id: {'answer': str, ...}}.
    Returns (winning_answer, vote_share ∈ [0,1])."""
    return _vote(list(rollout.values()))


# ----- 2. Avg-logprob (Kadavath 2022 style) -------------------------------

def avg_logprob(rollout: dict[str, dict]) -> tuple[str, float]:
    """Confidence = exp(mean of agents' mean_logprob), answer = highest-logprob agent."""
    items = list(rollout.values())
    if not items:
        return "", 0.0
    # pick highest-logprob agent's answer
    best = max(items, key=lambda r: r.get("mean_logprob", -np.inf))
    answer = best.get("answer", "")
    # confidence = exp(mean across agents); if no logprob default to vote share
    lps = [r.get("mean_logprob", None) for r in items]
    lps = [lp for lp in lps if lp is not None]
    if not lps:
        return _vote(items)
    return answer, float(np.exp(np.mean(lps)))


def max_logprob(rollout: dict[str, dict]) -> tuple[str, float]:
    """Confidence = exp(max of agents' mean_logprob), answer = highest-logprob agent.

    Intuition: "at least one agent is confidently saying X" — different from
    avg_logprob which averages confidence across all agents.
    """
    items = list(rollout.values())
    if not items:
        return "", 0.0
    best = max(items, key=lambda r: r.get("mean_logprob", -np.inf))
    answer = best.get("answer", "")
    lps = [r.get("mean_logprob", None) for r in items]
    lps = [lp for lp in lps if lp is not None]
    if not lps:
        return _vote(items)
    return answer, float(np.exp(np.max(lps)))


# ----- 3. DiverseAgentEntropy (Feng EMNLP 2025 style) ---------------------

def diverse_agent_entropy(rollout: dict[str, dict]) -> tuple[str, float]:
    """Weighted entropy over answer distribution.

    Confidence = 1 - normalized_entropy(answer_dist). Answer = majority.
    """
    items = list(rollout.values())
    if not items:
        return "", 0.0
    answers = [_normalize(r.get("answer", "")) for r in items]
    counter = Counter(answers)
    n = len(answers)
    probs = np.array([c / n for c in counter.values()])
    H = -np.sum(probs * np.log(probs + 1e-12))
    H_max = np.log(len(counter)) if len(counter) > 1 else 1.0
    norm_H = H / max(H_max, 1e-12)
    confidence = float(1.0 - norm_H)
    answer = counter.most_common(1)[0][0]
    return answer, confidence


# ----- 4. DiscoUQ-LLM (Jiang 2026 style) ----------------------------------
#   Trained LR on disagreement features. Features per rollout:
#     - vote_share
#     - n_unique_answers / n_agents
#     - max(answer_length) / mean(answer_length) (length ratio)
#     - mean answer-embedding pairwise cosine
#     - mean (mean_logprob) (proxy for self-confidence)

def disco_uq_features(rollout: dict[str, dict],
                      embedder=None) -> np.ndarray:
    """Extract DiscoUQ-style features for one rollout. Returns (5,) vector."""
    items = list(rollout.values())
    if not items:
        return np.zeros(5, dtype=np.float32)
    answers = [_normalize(r.get("answer", "")) for r in items]
    counter = Counter(answers)
    n = len(items)

    f_vote_share = counter.most_common(1)[0][1] / n
    f_unique_ratio = len(counter) / n
    lengths = [len(a) for a in answers]
    f_length_ratio = max(lengths) / max(np.mean(lengths), 1e-9)
    # Embedding cos avg (skip if embedder not provided -> use string similarity proxy)
    if embedder is not None and len(answers) > 1:
        embeds = embedder.encode(answers, normalize_embeddings=True, show_progress_bar=False)
        embeds = np.asarray(embeds, dtype=np.float32)
        cos_sum = 0.0
        n_pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                cos_sum += float(np.dot(embeds[i], embeds[j]))
                n_pairs += 1
        f_emb_cos = cos_sum / max(n_pairs, 1)
    else:
        f_emb_cos = 0.0

    lps = [r.get("mean_logprob", 0.0) for r in items]
    f_mean_logprob = float(np.mean(lps)) if lps else 0.0

    return np.array([f_vote_share, f_unique_ratio, f_length_ratio,
                     f_emb_cos, f_mean_logprob], dtype=np.float32)


class DiscoUQLLM:
    """Trained DiscoUQ-LLM aggregator."""
    def __init__(self):
        self.lr = LogisticRegression(max_iter=1000)
        self.fitted = False

    def fit(self, train_features: np.ndarray, train_labels: np.ndarray):
        if len(np.unique(train_labels)) < 2:
            # all-same labels: predict majority
            self._const = float(np.mean(train_labels))
            self.fitted = True
            return self
        self.lr.fit(train_features, train_labels.astype(int))
        self._const = None
        self.fitted = True
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        assert self.fitted
        if self._const is not None:
            return np.full(len(features), self._const, dtype=np.float32)
        return self.lr.predict_proba(features)[:, 1].astype(np.float32)


def disco_uq_aggregate(rollout: dict[str, dict], model: DiscoUQLLM,
                       embedder=None) -> tuple[str, float]:
    """Apply trained DiscoUQ to one rollout."""
    answer, _ = majority_vote(rollout)
    feats = disco_uq_features(rollout, embedder=embedder).reshape(1, -1)
    conf = float(model.predict_proba(feats)[0])
    return answer, conf


# ----- 6. GNN-CC (Li et al. 2411.02454 style, lite reimplementation) -------
#   GNN over same-LLM sample similarity graph; we approximate by training a
#   small GNN on a per-rollout text-similarity graph (not heterogeneous).
#   Lite version: use simple GraphSAGE-equivalent via sklearn LR on graph
#   summary stats (this is a fair-baseline reproduction, not the full paper).

def gnn_cc_features(rollout: dict[str, dict], embedder=None) -> np.ndarray:
    """GNN-CC summary features: (mean_pairwise_cos, std_pairwise_cos, mean_self_sim, vote_share)."""
    items = list(rollout.values())
    if len(items) < 2:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    answers = [str(r.get("answer", "")) for r in items]
    if embedder is None:
        # crude proxy: edit-distance-ish via length similarity
        lens = np.array([len(a) for a in answers])
        sim = 1.0 - np.abs(lens[:, None] - lens[None, :]) / (np.maximum(lens[:, None], lens[None, :]) + 1e-9)
        np.fill_diagonal(sim, 0)
    else:
        E = np.asarray(embedder.encode(answers, normalize_embeddings=True, show_progress_bar=False),
                       dtype=np.float32)
        sim = E @ E.T
        np.fill_diagonal(sim, 0)
    triu = sim[np.triu_indices_from(sim, k=1)]
    mean_pair = float(triu.mean())
    std_pair = float(triu.std())
    mean_self = 1.0   # always 1 since normalized
    _, vs = majority_vote(rollout)
    return np.array([mean_pair, std_pair, mean_self, vs], dtype=np.float32)


class GNNCC:
    """GNN-CC lite-baseline (LR on graph summary stats; faithful to spirit)."""
    def __init__(self):
        self.lr = LogisticRegression(max_iter=1000)
        self.fitted = False

    def fit(self, train_features: np.ndarray, train_labels: np.ndarray):
        if len(np.unique(train_labels)) < 2:
            self._const = float(np.mean(train_labels))
            self.fitted = True
            return self
        self.lr.fit(train_features, train_labels.astype(int))
        self._const = None
        self.fitted = True
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        assert self.fitted
        if self._const is not None:
            return np.full(len(features), self._const, dtype=np.float32)
        return self.lr.predict_proba(features)[:, 1].astype(np.float32)


def gnn_cc_aggregate(rollout: dict[str, dict], model: GNNCC,
                     embedder=None) -> tuple[str, float]:
    answer, _ = majority_vote(rollout)
    feats = gnn_cc_features(rollout, embedder=embedder).reshape(1, -1)
    conf = float(model.predict_proba(feats)[0])
    return answer, conf


# ----- Oracle isotonic upper bound ----------------------------------------

def fit_oracle_isotonic(raw_confidence: np.ndarray, correctness: np.ndarray) -> IsotonicRegression:
    """Fit isotonic on the same data we'll evaluate on. Upper bound."""
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_confidence, correctness.astype(float))
    return iso


# ----- Registry / convenience ---------------------------------------------

UNTRAINED_BASELINES = {
    "majority": majority_vote,
    "avg_logprob": avg_logprob,
    "diverse_agent_entropy": diverse_agent_entropy,
}
TRAINED_BASELINES = {
    "discouq": (DiscoUQLLM, disco_uq_features, disco_uq_aggregate),
    "gnn_cc": (GNNCC, gnn_cc_features, gnn_cc_aggregate),
}
