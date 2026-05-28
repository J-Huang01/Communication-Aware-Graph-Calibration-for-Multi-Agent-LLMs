from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE)); sys.path.insert(0, str(_HERE.parent))

from sklearn.decomposition import PCA
from cage_cal.calibration import auarc, auroc, ece
from cage_cal.per_query_W import PerQueryWEstimator
from data_utils import BENCHES, LOADERS, TOPOS, make_splits
from cage_gnn_hypergraph import (
    HyperData,  # noqa: F401  needed for pickle
    HyperHybridGNN, PCA_DIM, ROOT, ROLLOUTS,
    apply_betasb, build_dataset, build_sbert_cache, collate,
    fit_per_bench_betasb, per_rollout_summary, predict,
)
# Make HyperData findable in __main__ for pickle compatibility
import sys as _sys
_sys.modules['__main__'].HyperData = HyperData


def mmce_loss(p, y, sigma=0.4, eps=1e-8):
    """RKHS-based ECE surrogate (Kumar 2018), CORRECT formulation.

    p: predicted probability that panel answer is correct, shape (B,)
    y: binary correctness label {0, 1}, shape (B,)
    """
    p = p.view(-1)
    y = y.float().view(-1)
    residual = y - p
    K = torch.exp(-((p[:, None] - p[None, :]) ** 2) / sigma)
    mask = 1.0 - torch.eye(p.size(0), device=p.device)
    return (residual[:, None] * residual[None, :] * K * mask).sum() / (mask.sum() + eps)


def brier_loss(p, y):
    """Proper scoring rule: squared error from truth."""
    return F.mse_loss(p, y.float())


def train_one_selfcal(
    tr_ds, va_ds, in_dim, device, *,
    hid=64, heads=4, dropout=0.3,
    lr=2e-3, weight_decay=3e-4,
    epochs=15, batch_size=128, seed=0,
    label_smooth=0.05,
    brier_weight=0.2,
    mmce_weight=0.1,
    mmce_sigma=0.4,
    selection_lambda=0.5,
):
    torch.manual_seed(seed); np.random.seed(seed)
    model = HyperHybridGNN(in_dim, hid=hid, heads=heads,
                            n_bench=len(BENCHES), dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_combined, best_state, patience, max_patience = -1e9, None, 0, 5
    best_auroc_recorded, best_ece_recorded = -1, -1
    n_tr = len(tr_ds)

    for ep in range(1, epochs + 1):
        model.train()
        np.random.shuffle(tr_ds)
        total_loss = total_bce = total_brier = total_mmce = 0.0
        for i in range(0, n_tr, batch_size):
            chunk = tr_ds[i:i + batch_size]
            batch, bench, label = collate(chunk)
            batch, bench, label = batch.to(device), bench.to(device), label.to(device)
            target = label * (1 - 2*label_smooth) + label_smooth
            logit = model(batch, bench)
            p = torch.sigmoid(logit)

            l_bce = F.binary_cross_entropy_with_logits(logit, target)
            l_brier = brier_loss(p, label) if brier_weight > 0 else torch.tensor(0.0, device=device)
            l_mmce = mmce_loss(p, label, sigma=mmce_sigma) if mmce_weight > 0 else torch.tensor(0.0, device=device)
            loss = l_bce + brier_weight * l_brier + mmce_weight * l_mmce

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * len(chunk)
            total_bce += l_bce.item() * len(chunk)
            total_brier += l_brier.item() * len(chunk)
            total_mmce += l_mmce.item() * len(chunk)
        sched.step()

        # Val: BOTH AUROC and ECE
        pva = predict(model, va_ds, device, batch_size=batch_size)
        yva = np.array([p["label"] for p in va_ds])
        bva = [p["bench"] for p in va_ds]
        auroc_l, ece_l = [], []
        for b in BENCHES:
            m = np.array(bva) == b
            yy = yva[m]
            if 0 < yy.sum() < len(yy):
                auroc_l.append(auroc(pva[m], yy))
                ece_l.append(ece(pva[m], yy.astype(int), n_bins=10))
        val_auroc = float(np.mean(auroc_l))
        val_ece = float(np.mean(ece_l))
        val_score = val_auroc - selection_lambda * val_ece  # combined

        if val_score > best_combined:
            best_combined = val_score
            best_auroc_recorded = val_auroc
            best_ece_recorded = val_ece
            patience = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        print(f"  [selfcal] seed{seed} ep{ep:02d} "
              f"loss={total_loss/n_tr:.4f}(bce={total_bce/n_tr:.4f} "
              f"brier={total_brier/n_tr:.4f} mmce={total_mmce/n_tr:.4f}) "
              f"val AUROC={val_auroc:.4f} ECE={val_ece:.4f} score={val_score:.4f} "
              f"best=({best_auroc_recorded:.4f},{best_ece_recorded:.4f})",
              file=sys.stderr)
        if patience >= max_patience:
            print(f"  [selfcal] seed{seed} early-stop at ep{ep}", file=sys.stderr)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_seeds", type=int, default=10)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--brier_weight", type=float, default=0.2)
    p.add_argument("--mmce_weight", type=float, default=0.1)
    p.add_argument("--mmce_sigma", type=float, default=0.4)
    p.add_argument("--selection_lambda", type=float, default=0.5)
    p.add_argument("--beta_weight", type=float, default=0.5,
                     help="Weight on Beta calibration in Beta+SB averaging "
                          "(0.0 = SB-only, 1.0 = Beta-only)")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--tag", type=str, default="selfcal",
                     help="suffix for the output JSON file")
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
    print(f"[selfcal] train={len(tr_ds)} val={len(va_ds)} test={len(te_ds)}", file=sys.stderr)
    print(f"[selfcal] brier_w={args.brier_weight} mmce_w={args.mmce_weight} "
          f"mmce_sigma={args.mmce_sigma} selection_lambda={args.selection_lambda}",
          file=sys.stderr)

    yva = np.array([p["label"] for p in va_ds])
    yte = np.array([p["label"] for p in te_ds])
    bva = [p["bench"] for p in va_ds]; bte = [p["bench"] for p in te_ds]
    tte = [p["topo"] for p in te_ds]; rte = [p["ro"] for p in te_ds]

    pva_seeds, pte_seeds = [], []
    for s in range(args.n_seeds):
        print(f"\n[selfcal] === seed {s} ===", file=sys.stderr)
        model = train_one_selfcal(
            tr_ds, va_ds, in_dim, device,
            seed=s, epochs=args.epochs,
            brier_weight=args.brier_weight,
            mmce_weight=args.mmce_weight,
            mmce_sigma=args.mmce_sigma,
            selection_lambda=args.selection_lambda,
        )
        pva_seeds.append(predict(model, va_ds, device))
        pte_seeds.append(predict(model, te_ds, device))
        del model; torch.cuda.empty_cache()

    pva = np.mean(pva_seeds, axis=0)
    pte = np.mean(pte_seeds, axis=0)
    betas, sbs = fit_per_bench_betasb(pva, yva, bva)
    pte_cal = apply_betasb(pte, bte, betas, sbs, beta_weight=args.beta_weight)

    out = {
        f"{args.tag}_raw":    per_rollout_summary(pte,     yte, bte, tte, rte),
        f"{args.tag}_BetaSB": per_rollout_summary(pte_cal, yte, bte, tte, rte),
    }
    out_path = ROOT / f"cage_gnn_hyper_{args.tag}_n{args.n_seeds}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[selfcal] wrote {out_path}\n", file=sys.stderr)

    print(f"\n(ref) DiscoUQ-LR raw             ECE= 6.85±0.49  AUROC=73.46±1.94  AUARC=72.55")
    print(f"(ref) hyper baseline (no selfcal) raw  ECE=10.38±0.24  AUROC=82.56±1.27  AUARC=76.08±0.38")
    print(f"(ref) hyper baseline +BetaSB           ECE= 6.75±0.37  AUROC=82.56±1.27  AUARC=76.08±0.38")
    for suffix in ["raw", "BetaSB"]:
        m = out[f"{args.tag}_{suffix}"].get("Mean", {})
        print(f"{args.tag}_{suffix:8s}   "
              f"ECE={100*m['ece_mean']:5.2f}±{100*m['ece_std']:.2f}  "
              f"AUROC={100*m['auroc_mean']:5.2f}±{100*m['auroc_std']:.2f}  "
              f"AUARC={100*m['auarc_mean']:5.2f}±{100*m['auarc_std']:.2f}")


if __name__ == "__main__":
    sys.exit(main())
