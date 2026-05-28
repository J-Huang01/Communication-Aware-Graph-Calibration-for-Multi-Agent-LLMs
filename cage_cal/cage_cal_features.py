from __future__ import annotations

import collections
from collections import Counter
from typing import Any

import numpy as np

from cage_cal.aggregation import disco_uq_features, majority_vote


# -- Backbone family lookup ----------------------------------------------
_FAMILY = {
    "qwen3-8b": "qwen",
    "llama-3.1-8b": "llama",
    "gemma-3-12b": "gemma",
    "phi-4": "phi",
}

def _agent_family(aid: str) -> str:
    bb = aid.split("::", 1)[0]
    return _FAMILY.get(bb, bb)


# -- Communication adjacency per topology --------------------------------
def comm_adjacency(topo: str, agents: list[str]) -> np.ndarray:
    """Approximate W_comm for in-degree variance feature.

    iid:        zero matrix
    debate:     J - I (full mesh)
    chain:      upper-triangular shift (i sees agents 0..i-1)
    hub_spoke:  last agent (hub) sees all others; others see no one
    tree:       coarse approximation: zero (tree's structural per-node
                graph is in node_id metadata, not in this matrix)
    """
    N = len(agents)
    M = np.zeros((N, N), dtype=np.float32)
    if topo == "debate":
        M = (np.ones((N, N), dtype=np.float32) - np.eye(N, dtype=np.float32))
    elif topo == "chain":
        for i in range(N):
            for j in range(i):
                M[i, j] = 1.0
    elif topo == "hub_spoke":
        # Last agent is hub, sees all spokes
        M[-1, :-1] = 1.0
    # iid, tree: leave as zero
    return M


# -- Graph metrics --------------------------------------------------------
def _off_diag(W: np.ndarray) -> np.ndarray:
    """Return the off-diagonal entries of W as a flat array."""
    N = W.shape[0]
    if N < 2: return np.zeros(0, dtype=np.float32)
    mask = ~np.eye(N, dtype=bool)
    return W[mask]


def _rho_bar(W: np.ndarray) -> float:
    od = _off_diag(W)
    return float(od.mean()) if od.size else 0.0


def _rho_max_laplacian(W: np.ndarray) -> float:
    """Largest eigenvalue of L = D - W. Uses |W| for degree."""
    N = W.shape[0]
    if N < 2: return 0.0
    A = np.abs(W).copy()
    np.fill_diagonal(A, 0.0)
    D = np.diag(A.sum(axis=1))
    L = D - A
    try:
        ev = np.linalg.eigvalsh(L)
        return float(ev.max())
    except np.linalg.LinAlgError:
        return 0.0


def _freeman_centralization(W: np.ndarray) -> float:
    """Freeman degree centralization on |W|: range [0, 1]."""
    N = W.shape[0]
    if N < 3: return 0.0
    A = np.abs(W).copy()
    np.fill_diagonal(A, 0.0)
    deg = A.sum(axis=1)
    d_max = deg.max()
    denom = (N - 1) * (N - 2)
    if denom == 0: return 0.0
    return float((d_max - deg).sum() / denom)


def _same_family_cluster_fraction(W: np.ndarray, agents: list[str]) -> float:
    """Mean off-diagonal correlation restricted to same-family pairs."""
    N = len(agents)
    if N < 2: return 0.0
    fams = [_agent_family(a) for a in agents]
    vals = []
    for i in range(N):
        for j in range(i + 1, N):
            if fams[i] == fams[j]:
                vals.append(W[i, j])
    return float(np.mean(vals)) if vals else 0.0


def _density(W: np.ndarray, thresh: float = 0.2) -> float:
    """Fraction of off-diagonal |W_ij| > thresh."""
    od = _off_diag(W)
    if od.size == 0: return 0.0
    return float((np.abs(od) > thresh).mean())


def _spectral_modularity(W: np.ndarray, agents: list[str]) -> float:
    """Modularity-like quantity using the same-family partition."""
    N = len(agents)
    if N < 2: return 0.0
    fams = [_agent_family(a) for a in agents]
    A = np.abs(W).copy()
    np.fill_diagonal(A, 0.0)
    m = A.sum() / 2.0 + 1e-9
    deg = A.sum(axis=1)
    Q = 0.0
    for i in range(N):
        for j in range(N):
            if i == j: continue
            same = 1.0 if fams[i] == fams[j] else 0.0
            Q += (A[i, j] - deg[i] * deg[j] / (2.0 * m)) * same
    return float(Q / (2.0 * m))


def _n_eff(W: np.ndarray) -> float:
    """N_eff = N / (1 + (N-1) rho_bar)."""
    N = W.shape[0]
    if N < 2: return float(N)
    rb = max(_rho_bar(W), -1.0 / max(N - 1, 1) + 1e-3)
    return float(N / (1.0 + (N - 1) * rb))


def _in_degree_variance(W_comm: np.ndarray) -> float:
    """Variance of in-degrees (column sums) of the comm adjacency."""
    if W_comm.size == 0: return 0.0
    in_deg = W_comm.sum(axis=0)
    return float(in_deg.var())


# -- Vote / answer features ---------------------------------------------
def _norm(s: str) -> str:
    return (s or "").strip().lower().strip(".\"' \t\n,;:!?")


def _plurality(agents_dict: dict) -> tuple[str, float, int]:
    """Returns (answer, vote_share, n_distinct) over agents_dict values."""
    answers = [_norm(r["answer"]) for r in agents_dict.values()]
    if not answers: return ("", 0.0, 0)
    cnt = Counter(answers)
    top, n_top = cnt.most_common(1)[0]
    return (top, n_top / len(answers), len(cnt))


# -- Per-panel feature vector (22-d) ------------------------------------
def panel_features(
    agents_dict_T: dict,
    agents_dict_0: dict | None,
    agents: list[str],
    W_T: np.ndarray,
    W_0: np.ndarray | None,
    w_comm: np.ndarray,
) -> np.ndarray:
    """Build the 22-d feature vector for one panel.

    Args:
      agents_dict_T: post-communication panel (this rollout). Each value
                     dict has at minimum 'answer' and optionally 'mean_logprob'.
      agents_dict_0: counterfactual-iid panel (same rollout, iid topology).
                     If None (e.g. topo == iid), G4 deltas are zero and G5
                     features are zero.
      agents: ordered agent ids that index W_T (and W_0).
      W_T:   post-communication correlation matrix (N x N).
      W_0:   counterfactual-iid correlation matrix (N x N), or None.
      w_comm: communication adjacency for the topology (N x N).
    """
    N = len(agents)

    # ---- G1 Vote (2) ----
    top_T, vote_share, n_distinct = _plurality(agents_dict_T)

    # ---- G2 Disagreement (3) ----
    # disco_uq_features returns 5: [vote_share, unique_ratio, length_ratio,
    # emb_cos, mean_logprob]. vote_share is duplicate of G1[0] and emb_cos
    # is always 0 (we don't pass an embedder). Drop both → 3 useful feats.
    g2_full = disco_uq_features(agents_dict_T, embedder=None)
    if not isinstance(g2_full, np.ndarray):
        g2_full = np.asarray(g2_full, dtype=np.float32)
    g2 = g2_full[[1, 2, 4]]  # unique_ratio, length_ratio, mean_logprob

    # ---- G3 Static post-comm graph (5) ----
    g3 = np.array([
        _rho_bar(W_T),
        _rho_max_laplacian(W_T),
        _freeman_centralization(W_T),
        _same_family_cluster_fraction(W_T, agents),
        _in_degree_variance(w_comm),
    ], dtype=np.float32)

    # ---- G4 Counterfactual ΔG (6) ----
    if W_0 is not None:
        d_rho_bar = _rho_bar(W_T) - _rho_bar(W_0)
        d_rho_max = _rho_max_laplacian(W_T) - _rho_max_laplacian(W_0)
        d_density = _density(W_T) - _density(W_0)
        d_mod = _spectral_modularity(W_T, agents) - _spectral_modularity(W_0, agents)
        neff_T = _n_eff(W_T)
        neff_0 = _n_eff(W_0)
    else:
        d_rho_bar = d_rho_max = d_density = d_mod = 0.0
        neff_T = neff_0 = float(N)
    g4 = np.array([d_rho_bar, d_rho_max, d_density, d_mod, neff_T, neff_0],
                   dtype=np.float32)

    # ---- G5 Regression + herding (4) ----
    if agents_dict_0 is not None and len(agents_dict_0) > 0:
        top_0, vote_share_0, _ = _plurality(agents_dict_0)
        # regressed share: agents whose iid answer = iid plurality but
        # topo answer != iid plurality
        regressed = 0
        eligible = 0
        for aid in agents:
            a_T = _norm(agents_dict_T.get(aid, {}).get("answer", ""))
            a_0 = _norm(agents_dict_0.get(aid, {}).get("answer", ""))
            if a_0 == top_0:
                eligible += 1
                if a_T != top_0:
                    regressed += 1
        regressed_share = regressed / max(eligible, 1)
        match = 1.0 if top_T == top_0 else 0.0
        # Δ agreement: change in plurality share
        d_agree = vote_share - vote_share_0
        # Δ confidence: change in mean logprob across agents
        lp_T = [r.get("mean_logprob") for r in agents_dict_T.values()
                if r.get("mean_logprob") is not None]
        lp_0 = [r.get("mean_logprob") for r in agents_dict_0.values()
                if r.get("mean_logprob") is not None]
        d_conf = (float(np.mean(lp_T)) if lp_T else 0.0) - \
                  (float(np.mean(lp_0)) if lp_0 else 0.0)
    else:
        regressed_share = 0.0
        match = 1.0
        d_agree = 0.0
        d_conf = 0.0
    g5 = np.array([regressed_share, match, d_agree, d_conf], dtype=np.float32)

    g1 = np.array([vote_share, float(n_distinct)], dtype=np.float32)
    feats = np.concatenate([g1, g2, g3, g4, g5]).astype(np.float32)
    # Group offsets after dropping the 2 redundant g2 features
    # (vote_share dup, always-zero emb_cos):
    #   G1: 0:2 (2)  G2: 2:5 (3)  G3: 5:10 (5)  G4: 10:16 (6)  G5: 16:20 (4)
    return feats


GROUP_SLICES = {
    "G1": slice(0, 2),
    "G2": slice(2, 5),
    "G3": slice(5, 10),
    "G4": slice(10, 16),
    "G5": slice(16, 20),
}
