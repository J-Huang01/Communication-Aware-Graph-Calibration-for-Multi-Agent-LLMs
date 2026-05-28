from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from betacal import BetaCalibration
from calibration import PlattBinnerCalibrator

warnings.filterwarnings("ignore")
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE)); sys.path.insert(0, str(_HERE.parent))

from sklearn.decomposition import PCA
from cage_cal.per_query_W import PerQueryWEstimator
from data_utils import BENCHES, LOADERS, TOPOS, make_splits
from cage_gnn_hypergraph import (
    HyperData,  # noqa: F401  for pickle
    PCA_DIM, ROOT, ROLLOUTS, build_dataset, build_sbert_cache,
    per_rollout_summary, predict,
)
from cage_gnn_hyper_selfcal import train_one_selfcal
import sys as _sys
_sys.modules['__main__'].HyperData = HyperData


def fit_per_bench_beta_only(pva, yva, bva):
    out = {}
    for b in BENCHES:
        m = np.array(bva) == b
        if m.sum() < 50: continue
        pc = np.clip(pva[m], 1e-6, 1 - 1e-6)
        try:
            out[b] = BetaCalibration(parameters="abm").fit(pc.reshape(-1, 1), yva[m])
        except Exception:
            pass
    return out


def fit_per_bench_sb_only(pva, yva, bva):
    out = {}
    for b in BENCHES:
        m = np.array(bva) == b
        if m.sum() < 50: continue
        try:
            cal = PlattBinnerCalibrator(num_calibration=m.sum(), num_bins=10)
            cal.train_calibration(pva[m], yva[m].astype(int))
            out[b] = cal
        except Exception:
            pass
    return out


def apply_beta_only(p, b_, betas):
    out = p.copy()
    for b in BENCHES:
        m = np.array(b_) == b
        if m.sum() and b in betas:
            pc = np.clip(p[m], 1e-6, 1 - 1e-6)
            out[m] = betas[b].predict(pc.reshape(-1, 1)).ravel()
    return out


def apply_sb_only(p, b_, sbs):
    out = p.copy()
    for b in BENCHES:
        m = np.array(b_) == b
        if m.sum() and b in sbs:
            out[m] = np.asarray(sbs[b].calibrate(p[m]), dtype=np.float64)
    return out


def apply_betasb(p, b_, betas, sbs):
    out = p.copy()
    for b in BENCHES:
        m = np.array(b_) == b
        if m.sum() == 0: continue
        preds = []
        if b in betas:
            pc = np.clip(p[m], 1e-6, 1 - 1e-6)
            preds.append(betas[b].predict(pc.reshape(-1, 1)).ravel())
        if b in sbs:
            preds.append(np.asarray(sbs[b].calibrate(p[m]), dtype=np.float64))
        if preds:
            out[m] = np.mean(preds, axis=0)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_seeds", type=int, default=10)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--device", type=str, default="cuda:0")
    args = p.parse_args()
    device = torch.device(args.device)

    sbert = build_sbert_cache(ROOT)
    keys = list(sbert.keys()); emb_mat = np.stack([sbert[k] for k in keys])
    pca = PCA(n_components=PCA_DIM, random_state=0).fit(emb_mat)
    sbert_pca = {k: pca.transform(sbert[k].reshape(1, -1))[0].astype(np.float32) for k in keys}

    est = PerQueryWEstimator(root=ROOT, cache_dir=Path("cache/per_query_W"), k=20)
    bench_problems = {b: LOADERS[b](n=500, seed=0) for b in BENCHES}
    splits = {b: make_splits(bench_problems[b]) for b in BENCHES}
    tr = {b: splits[b][0] for b in BENCHES}
    te = {b: splits[b][2] for b in BENCHES}
    panels = build_dataset(ROOT, sbert_pca, est, tr, te)

    tr_ds = [p for p in panels if p["split"] == "train"]
    va_ds = [p for p in panels if p["split"] == "val"]
    te_ds = [p for p in panels if p["split"] == "test"]
    in_dim = tr_ds[0]["data"].x_T.shape[1]
    print(f"[cal-abl] train={len(tr_ds)} val={len(va_ds)} test={len(te_ds)}", file=sys.stderr)

    yva = np.array([p["label"] for p in va_ds])
    yte = np.array([p["label"] for p in te_ds])
    bva = [p["bench"] for p in va_ds]; bte = [p["bench"] for p in te_ds]
    tte = [p["topo"] for p in te_ds]; rte = [p["ro"] for p in te_ds]

    pva_seeds, pte_seeds = [], []
    for s in range(args.n_seeds):
        print(f"\n[cal-abl] === seed {s} ===", file=sys.stderr)
        # No MMCE (matches new headline lean variant)
        model = train_one_selfcal(tr_ds, va_ds, in_dim, device, seed=s,
                                    epochs=args.epochs,
                                    brier_weight=0.4, mmce_weight=0.0)
        pva_seeds.append(predict(model, va_ds, device))
        pte_seeds.append(predict(model, te_ds, device))
        del model; torch.cuda.empty_cache()

    pva_ens = np.mean(pva_seeds, axis=0)
    pte_ens = np.mean(pte_seeds, axis=0)
    pva_single = pva_seeds[0]
    pte_single = pte_seeds[0]

    out = {}
    out["seed0_raw"]     = per_rollout_summary(pte_single, yte, bte, tte, rte)
    out["ensemble_raw"]  = per_rollout_summary(pte_ens,    yte, bte, tte, rte)

    betas = fit_per_bench_beta_only(pva_ens, yva, bva)
    sbs   = fit_per_bench_sb_only(pva_ens, yva, bva)

    pte_beta_only = apply_beta_only(pte_ens, bte, betas)
    pte_sb_only   = apply_sb_only(pte_ens, bte, sbs)
    pte_full      = apply_betasb(pte_ens, bte, betas, sbs)

    out["ensemble_beta_only"] = per_rollout_summary(pte_beta_only, yte, bte, tte, rte)
    out["ensemble_sb_only"]   = per_rollout_summary(pte_sb_only,   yte, bte, tte, rte)
    out["ensemble_betasb"]    = per_rollout_summary(pte_full,      yte, bte, tte, rte)

    out_path = ROOT / f"cage_cal_results_n{args.n_seeds}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[cage-cal] wrote {out_path}\n", file=sys.stderr)

    print(f"\n{'recipe':<25s}  {'ECE':<14s}   {'AUROC':<14s}   {'AUARC':<14s}")
    print("-" * 80)
    for k in ["seed0_raw", "ensemble_raw", "ensemble_beta_only", "ensemble_sb_only", "ensemble_betasb"]:
        m = out[k].get("Mean", {})
        if not m: continue
        print(f"{k:<25s}  "
              f"{100*m['ece_mean']:5.2f}±{100*m['ece_std']:.2f}     "
              f"{100*m['auroc_mean']:5.2f}±{100*m['auroc_std']:.2f}    "
              f"{100*m['auarc_mean']:5.2f}±{100*m['auarc_std']:.2f}")


if __name__ == "__main__":
    sys.exit(main())
