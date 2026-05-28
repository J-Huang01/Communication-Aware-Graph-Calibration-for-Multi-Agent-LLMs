from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _strip_tree(aid: str) -> str:
    """Normalize tree's `::tree*` suffix so agent_ids align with iid."""
    parts = aid.split("::")
    if len(parts) == 3 and parts[2].startswith("tree"):
        return f"{parts[0]}::{parts[1]}"
    return aid


def _embed_cache_path(cache_dir: Path, bench: str, model_name: str) -> Path:
    safe = model_name.replace("/", "_")
    return cache_dir / f"sbert__{safe}__{bench}.npz"


def _load_or_compute_embeddings(
    questions: list[str],
    qids: list[str],
    bench: str,
    cache_dir: Path,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> np.ndarray:
    """Returns (n_questions, dim) float32 embeddings, cached on disk."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _embed_cache_path(cache_dir, bench, model_name)
    if cache_path.exists():
        npz = np.load(cache_path, allow_pickle=True)
        cached_qids = npz["qids"].tolist()
        if cached_qids == qids:
            return npz["emb"].astype(np.float32)
    # Compute fresh
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    emb = model.encode(questions, batch_size=64, show_progress_bar=False,
                        normalize_embeddings=True).astype(np.float32)
    np.savez_compressed(cache_path, emb=emb, qids=np.array(qids, dtype=object))
    return emb


def _load_cell_labels(root: Path, topo: str, bench: str) -> dict:
    """Returns {(qid, rollout): {agent_id (stripped): bool correct}}."""
    lfn = root / topo / bench / "labels.jsonl"
    if not lfn.exists():
        return {}
    out: dict = collections.defaultdict(dict)
    for line in lfn.open():
        l = json.loads(line)
        aid = _strip_tree(l["agent_id"])
        out[(l["qid"], l["rollout"])][aid] = bool(l["correct"])
    return dict(out)


def _correctness_matrix(
    panels: dict, qids: list[str], agents: list[str]
) -> np.ndarray:
    """Build (n_qids * n_rollouts, n_agents) binary correctness array.

    Skips (qid, rollout) panels where any agent is missing — usually fine.
    Returns float32 for downstream Pearson.
    """
    rows = []
    for qid in qids:
        for ro in (1, 2, 3):
            agents_dict = panels.get((qid, ro))
            if not agents_dict:
                continue
            if not all(a in agents_dict for a in agents):
                continue
            rows.append([1.0 if agents_dict[a] else 0.0 for a in agents])
    if not rows:
        return np.zeros((0, len(agents)), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def _pearson_W(E: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Standard Pearson correlation matrix over (rows, agents) → (agents, agents).
    No per-row residualization. Diagonal = 1.0; if any agent has zero
    variance in this neighborhood (e.g., always correct), that pair's
    correlation is set to 0 to avoid division-by-zero.
    """
    if E.shape[0] < 2:
        N = E.shape[1]
        out = np.zeros((N, N), dtype=np.float32)
        np.fill_diagonal(out, 1.0)
        return out
    X = E - E.mean(axis=0, keepdims=True)
    cov = X.T @ X / max(X.shape[0] - 1, 1)
    var = np.diag(cov)
    # Where variance is essentially 0, treat as constant: correlation 0 except diagonal
    std = np.sqrt(np.clip(var, eps, None))
    W = cov / (std[:, None] * std[None, :] + eps)
    bad = var < eps
    if bad.any():
        W[bad, :] = 0.0
        W[:, bad] = 0.0
    np.fill_diagonal(W, 1.0)
    return W.astype(np.float32)


class PerQueryWEstimator:
    """K-NN estimator for W(x) per (qid, topology).

    Caches:
      - SBERT embeddings per benchmark (one .npz per bench)
      - Per-cell W(x) for all test qids (one .npz per (topo, bench))

    Embedding model is small (MiniLM-L6-v2, 22M params) — runs CPU fast.
    """

    def __init__(
        self,
        root: Path | str,
        cache_dir: Path | str = "cache/per_query_W",
        sbert_cache_dir: Path | str = "cache/sbert",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        k: int = 20,
    ):
        self.root = Path(root)
        self.cache_dir = Path(cache_dir)
        self.sbert_cache_dir = Path(sbert_cache_dir)
        self.embedding_model = embedding_model
        self.k = k
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _per_cell_cache_path(self, topo: str, bench: str) -> Path:
        return self.cache_dir / f"W__{topo}__{bench}__k{self.k}.npz"

    def precompute(
        self,
        bench: str,
        questions_all: list[str],
        qids_all: list[str],
        topos: list[str],
        train_qids: set[str] | None = None,
    ) -> None:
        """Compute and cache W(x) per (qid, topology) for one benchmark.

        For each qid, the neighborhood used to estimate W is the k
        nearest neighbors **among train_qids only** (excluding self if
        the qid is itself a train qid). If train_qids is None, all qids
        (excluding self) are used as candidate neighbors — convenient
        for fully descriptive use but not a clean train/test separation.

        Stores cache as a dict per topology: arr of shape
        (n_qids, n_agents, n_agents), aligned to qids_all order.
        """
        # 1. SBERT embeddings (cached)
        emb_all = _load_or_compute_embeddings(
            questions_all, qids_all, f"{bench}",
            self.sbert_cache_dir, self.embedding_model)

        # 2. Identify candidate-neighbor pool (train qids only)
        if train_qids is None:
            train_qids = set(qids_all)
        train_mask = np.array([q in train_qids for q in qids_all])
        train_idx = np.where(train_mask)[0]
        emb_train = emb_all[train_idx]
        train_qids_ordered = [qids_all[i] for i in train_idx]

        # 3. K-NN: for each qid, find k nearest TRAIN qids (excluding
        # self if self is itself a train qid).
        sims = emb_all @ emb_train.T  # (n_all, n_train)
        # Mask out self-match (set sim = -inf if same qid)
        for i, q in enumerate(qids_all):
            if q in train_qids:
                j = train_qids_ordered.index(q)
                sims[i, j] = -np.inf
        knn_idx = np.argsort(-sims, axis=1)[:, : self.k]

        # 4. Per topology, compute W(x) for each qid
        for topo in topos:
            cache_path = self._per_cell_cache_path(topo, bench)
            if cache_path.exists():
                continue
            panels = _load_cell_labels(self.root, topo, bench)
            if not panels:
                continue
            agent_set: set = set()
            for d in panels.values():
                agent_set.update(d.keys())
            agents = sorted(agent_set)
            N = len(agents)
            n_qids = len(qids_all)
            W_arr = np.zeros((n_qids, N, N), dtype=np.float32)
            for ti in range(n_qids):
                neighbor_qids = [train_qids_ordered[j] for j in knn_idx[ti]]
                E = _correctness_matrix(panels, neighbor_qids, agents)
                W_arr[ti] = _pearson_W(E)
            np.savez_compressed(
                cache_path,
                W=W_arr,
                agents=np.array(agents, dtype=object),
                qids=np.array(qids_all, dtype=object),
            )

    def load(self, topo: str, bench: str) -> tuple[np.ndarray, list[str], list[str]]:
        """Returns (W_arr (n_qids, N, N), agents, qids) for one cell."""
        cache_path = self._per_cell_cache_path(topo, bench)
        if not cache_path.exists():
            raise FileNotFoundError(
                f"per-query W not cached for {topo}/{bench}; "
                f"call precompute() first"
            )
        npz = np.load(cache_path, allow_pickle=True)
        # Backward-compat: older caches used 'qids_test' key
        qids_key = "qids" if "qids" in npz.files else "qids_test"
        return (
            npz["W"].astype(np.float32),
            npz["agents"].tolist(),
            npz[qids_key].tolist(),
        )
