from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score


# --------------------------------------------------------------------------
#  Discrimination (uncertainty distinguishes correct from incorrect)
# --------------------------------------------------------------------------


def auroc(confidence: np.ndarray, correctness: np.ndarray) -> float:
    """Area under ROC curve. Higher = confidence ranks correct higher than wrong."""
    correctness = np.asarray(correctness).astype(int)
    if len(np.unique(correctness)) < 2:
        return float("nan")  # all-correct or all-wrong → undefined
    return float(roc_auc_score(correctness, np.asarray(confidence)))


def selective_accuracy(
    confidence: np.ndarray,
    correctness: np.ndarray,
    coverage: float = 0.9,
) -> float:
    """Accuracy on the top `coverage` fraction of confidence-ranked predictions.

    coverage=0.9 → keep top 90% most confident, abstain on bottom 10%.
    """
    confidence = np.asarray(confidence)
    correctness = np.asarray(correctness).astype(int)
    n = len(confidence)
    n_keep = int(np.ceil(n * coverage))
    if n_keep == 0:
        return float("nan")
    keep_idx = np.argsort(-confidence)[:n_keep]   # highest-confidence first
    return float(correctness[keep_idx].mean())


def auarc(confidence: np.ndarray, correctness: np.ndarray) -> float:
    """Area under Accuracy-Rejection Curve (sweep coverage 0→1).

    Higher = model abstains on the wrong cases preferentially.
    """
    confidence = np.asarray(confidence)
    correctness = np.asarray(correctness).astype(int)
    n = len(confidence)
    sort_idx = np.argsort(-confidence)
    sorted_corr = correctness[sort_idx]
    cum_correct = np.cumsum(sorted_corr)
    coverage_grid = np.arange(1, n + 1)
    accuracy_at_cov = cum_correct / coverage_grid
    return float(np.trapz(accuracy_at_cov, dx=1.0 / n))


# --------------------------------------------------------------------------
#  Calibration (confidence value matches actual frequency of correctness)
# --------------------------------------------------------------------------


def ece(
    confidence: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error with equal-width bins.

    Lower = confidence-frequency match better. ECE in [0, 1].
    """
    confidence = np.asarray(confidence)
    correctness = np.asarray(correctness).astype(float)
    n = len(confidence)
    if n == 0:
        return float("nan")
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    total_err = 0.0
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        # last bin inclusive on right
        if b == n_bins - 1:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)
        m = mask.sum()
        if m == 0:
            continue
        avg_conf = confidence[mask].mean()
        avg_acc = correctness[mask].mean()
        total_err += (m / n) * abs(avg_conf - avg_acc)
    return float(total_err)


def mce(
    confidence: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Maximum Calibration Error: worst bin's |conf - acc|."""
    confidence = np.asarray(confidence)
    correctness = np.asarray(correctness).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    worst = 0.0
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        if b == n_bins - 1:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)
        if not mask.any():
            continue
        worst = max(worst, abs(confidence[mask].mean() - correctness[mask].mean()))
    return float(worst)


def brier(confidence: np.ndarray, correctness: np.ndarray) -> float:
    """Brier score = mean squared error between confidence and 0/1 correctness.

    Lower = better. Combines calibration + sharpness.
    """
    confidence = np.asarray(confidence, dtype=float)
    correctness = np.asarray(correctness, dtype=float)
    return float(np.mean((confidence - correctness) ** 2))


# --------------------------------------------------------------------------
#  Reliability diagram (Figure 5 / Appendix Table A1)
# --------------------------------------------------------------------------


def reliability_bins(
    confidence: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 10,
) -> dict[str, np.ndarray]:
    """Return per-bin (count, avg_conf, avg_acc) for plotting reliability diagram."""
    confidence = np.asarray(confidence)
    correctness = np.asarray(correctness).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    counts = np.zeros(n_bins, dtype=int)
    avg_conf = np.zeros(n_bins)
    avg_acc = np.zeros(n_bins)
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        if b == n_bins - 1:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)
        m = mask.sum()
        counts[b] = m
        if m > 0:
            avg_conf[b] = confidence[mask].mean()
            avg_acc[b] = correctness[mask].mean()
    return {
        "bin_edges": bin_edges,
        "counts": counts,
        "avg_conf": avg_conf,
        "avg_acc": avg_acc,
    }


# --------------------------------------------------------------------------
#  Oracle isotonic upper bound (paper baseline for "how close to oracle")
# --------------------------------------------------------------------------


def oracle_isotonic_calibrate(
    confidence: np.ndarray,
    correctness: np.ndarray,
) -> np.ndarray:
    """Fit isotonic regression on the SAME (confidence, correctness) data and
    return the calibrated confidence.

    NOTE: this fits + predicts on the same set, intentionally — it gives the
    *oracle* upper bound (best achievable ECE if you knew the test labels in
    advance). Use only as a reference baseline, not as a fair method.
    """
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(np.asarray(confidence), np.asarray(correctness).astype(float))
    return iso.predict(np.asarray(confidence))


# --------------------------------------------------------------------------
#  Theorem 2' — spectral lower bound on ECE
# --------------------------------------------------------------------------


def predicted_ece_spectral(
    rho_max_L: float,
    p: float,
    N: int | None = None,
) -> float:
    """Theorem 2' ECE lower bound prediction.

    ECE(g) ≥ (1/2) · √( p(1-p) · ρ_max(L_W) )    (asymptotic, N→∞)

    For finite N, the practical prediction divides by √N to reflect the
    1/√N concentration term (see Theorem 3' rate). Pass `N` to use the
    finite-N variant; leave None for the asymptotic form.

    Args:
        rho_max_L:  ρ_max(L_W), the largest eigenvalue of the graph Laplacian
                    of the pairwise failure-correlation matrix W.
        p:          Marginal correctness probability (≈ mean per-agent accuracy
                    estimated on train split).
        N:          Number of agents in the panel. If given, divide variance
                    bound by N for the finite-sample form.

    Returns:
        Predicted ECE lower bound ∈ [0, ∞).
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0,1], got {p}")
    if rho_max_L < 0:
        raise ValueError(f"rho_max_L must be ≥ 0, got {rho_max_L}")
    variance = p * (1.0 - p) * rho_max_L
    if N is not None:
        if N <= 0:
            raise ValueError(f"N must be > 0, got {N}")
        variance = variance / N
    return 0.5 * float(np.sqrt(variance))


def predicted_ece_scalar(
    rho_bar: float,
    p: float,
) -> float:
    """Scalar baseline prediction: ½ √(p(1-p) ρ̄).

    Comparison anchor for Fig Y sidebar — should fit ECE worse than
    `predicted_ece_spectral` (proves the spectral form is needed).
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0,1], got {p}")
    rho_bar = max(0.0, float(rho_bar))
    return 0.5 * float(np.sqrt(p * (1.0 - p) * rho_bar))


# --------------------------------------------------------------------------
#  Theorem 2'' — variance-aware finite-N ECE predictor
# --------------------------------------------------------------------------


def predicted_ece_variance_aware(
    W: np.ndarray,
    p: float,
    N: int | None = None,
    n_samples: int = 20000,
    seed: int = 0,
) -> dict[str, float]:
    """Finite-N expected ECE under joint Bernoulli with correlation matrix W.

    Models the binary "majority correct vs majority wrong" outcome of plurality
    vote on N agents with marginal accuracy p and pairwise error correlation W.
    Uses a Gaussian copula to simulate the joint Bernoulli distribution, then
    computes the expected ECE under the standard binned definition.

    This decomposes empirically into:
      - bias term: E[max(K, N-K)/N] − E[1[K > N/2]] (under-confidence on easy
        benchmarks where K is concentrated near N)
      - variance term: spread of vote share around its mean (over-confidence
        on high-correlation cells, the Theorem 2' lower-bound regime)

    Returns:
      dict with keys:
        - ece_finite_N: predicted expected ECE under this model
        - bias: predicted bias contribution (|E[S] − E[Y]|)
        - variance: predicted variance contribution (√Var(S))
        - mean_S: E[max(K, N-K)/N]
        - mean_Y: E[1[K > N/2]] (majority accuracy)
    """
    from scipy.stats import norm

    W = np.asarray(W, dtype=float)
    if N is None:
        N = W.shape[0]
    if W.shape != (N, N):
        raise ValueError(f"W shape mismatch: got {W.shape}, expected ({N},{N})")

    # Symmetrize and PSD-project the correlation matrix
    W_sym = (W + W.T) / 2.0
    eig_vals, eig_vecs = np.linalg.eigh(W_sym)
    eig_vals_clipped = np.clip(eig_vals, 1e-6, None)
    W_psd = eig_vecs @ np.diag(eig_vals_clipped) @ eig_vecs.T

    # Cholesky for sampling
    L = np.linalg.cholesky(W_psd + 1e-6 * np.eye(N))

    # Threshold so that P(Z_i > thr) = p
    threshold = norm.ppf(1 - p) if 0 < p < 1 else (np.inf if p == 0 else -np.inf)

    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n_samples, N)) @ L.T
    X = (Z > threshold).astype(int)
    K = X.sum(axis=1)
    S = np.maximum(K, N - K) / float(N)  # plurality vote share for majority answer
    Y = (K > N / 2).astype(int)          # 1 if majority answer is correct

    # ECE via 11-bin discrete binning (one per possible majority-vote-count value)
    ece = 0.0
    for k_bin in range(int(np.ceil(N / 2 + 1e-9)) + 1, N + 1):
        in_bin = (np.maximum(K, N - K) == k_bin)
        m = in_bin.sum()
        if m == 0:
            continue
        bin_mass = m / float(n_samples)
        bin_conf = k_bin / float(N)
        bin_acc = float(Y[in_bin].mean())
        ece += bin_mass * abs(bin_conf - bin_acc)

    return {
        "ece_finite_N": float(ece),
        "bias": float(abs(S.mean() - Y.mean())),
        "variance": float(np.std(S)),
        "mean_S": float(S.mean()),
        "mean_Y": float(Y.mean()),
    }


# --------------------------------------------------------------------------
#  Convenience: full metric suite
# --------------------------------------------------------------------------


def all_metrics(
    confidence: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 10,
    coverages: tuple[float, ...] = (0.5, 0.8, 0.9),
) -> dict[str, float]:
    """Compute the full headline metric suite for one (method, cell) result."""
    confidence = np.asarray(confidence)
    correctness = np.asarray(correctness).astype(int)
    out = {
        "auroc": auroc(confidence, correctness),
        "auarc": auarc(confidence, correctness),
        "ece": ece(confidence, correctness, n_bins=n_bins),
        "mce": mce(confidence, correctness, n_bins=n_bins),
        "brier": brier(confidence, correctness),
        "accuracy": float(correctness.mean()),
    }
    for c in coverages:
        out[f"sel_acc@{int(c*100)}"] = selective_accuracy(confidence, correctness, coverage=c)
    return out
