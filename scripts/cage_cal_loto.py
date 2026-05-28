from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import numpy as np
import torch
warnings.filterwarnings("ignore")
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE)); sys.path.insert(0, str(_HERE.parent))

from sklearn.decomposition import PCA
from cage_cal.per_query_W import PerQueryWEstimator
from data_utils import BENCHES, LOADERS, TOPOS, make_splits
from cage_gnn_hypergraph import (
    HyperData,  # noqa: F401  needed for pickle load
    PCA_DIM, ROOT, ROLLOUTS, apply_betasb, build_dataset,
    build_sbert_cache, fit_per_bench_betasb, per_rollout_summary,
    predict, train_one,
)
# Make HyperData findable in __main__ for pickle compatibility
import sys as _sys
_sys.modules['__main__'].HyperData = HyperData


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n_seeds", type=int, default=5)
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
    in_dim = panels[0]["data"].x_T.shape[1]

    loto_results = {}
    for held in TOPOS:
        print(f"\n=== LOTO held-out: {held} ===", file=sys.stderr, flush=True)
        tr_ds = [p for p in panels if p["topo"] != held and p["split"] == "train"]
        va_ds = [p for p in panels if p["topo"] != held and p["split"] == "val"]
        te_ds = [p for p in panels if p["topo"] == held and p["split"] == "test"]
        print(f"  {held}: train={len(tr_ds)} val={len(va_ds)} test={len(te_ds)}",
              file=sys.stderr)
        if not tr_ds or not te_ds:
            continue

        pva_seeds, pte_seeds = [], []
        for s in range(args.n_seeds):
            print(f"  [hyper-loto] {held} seed {s}", file=sys.stderr)
            model, _ = train_one(tr_ds, va_ds, in_dim, device,
                                  seed=s, epochs=args.epochs)
            pva_seeds.append(predict(model, va_ds, device))
            pte_seeds.append(predict(model, te_ds, device))
            del model; torch.cuda.empty_cache()

        pva = np.mean(pva_seeds, axis=0)
        pte = np.mean(pte_seeds, axis=0)
        yva = np.array([p["label"] for p in va_ds])
        yte = np.array([p["label"] for p in te_ds])
        bva = [p["bench"] for p in va_ds]; bte = [p["bench"] for p in te_ds]
        tte = [p["topo"] for p in te_ds]; rte = [p["ro"] for p in te_ds]

        betas, sbs = fit_per_bench_betasb(pva, yva, bva)
        pte_cal = apply_betasb(pte, bte, betas, sbs)
        loto_results[held] = per_rollout_summary(pte_cal, yte, bte, tte, rte)
        m = loto_results[held].get("Mean", {})
        if m:
            print(f"  {held} mean: ECE={100*m['ece_mean']:.2f}±{100*m['ece_std']:.2f}  "
                  f"AUROC={100*m['auroc_mean']:.2f}±{100*m['auroc_std']:.2f}  "
                  f"AUARC={100*m['auarc_mean']:.2f}±{100*m['auarc_std']:.2f}",
                  file=sys.stderr)

    out_path = ROOT / f"cage_gnn_hyper_loto_n{args.n_seeds}.json"
    out_path.write_text(json.dumps(loto_results, indent=2))
    print(f"\nwrote {out_path}", file=sys.stderr)

    # Per-bench aggregate
    bench_acc = {b: {"ece": [], "auroc": [], "auarc": []} for b in BENCHES + ["Mean"]}
    for held, summary in loto_results.items():
        for b, v in summary.items():
            if b not in bench_acc: continue
            bench_acc[b]["ece"].append(v["ece_mean"])
            bench_acc[b]["auroc"].append(v["auroc_mean"])
            bench_acc[b]["auarc"].append(v["auarc_mean"])
    print(f"\nPer-bench LOTO:")
    print(f"{'bench':<13} {'ECE':<14} {'AUROC':<14} {'AUARC':<14}")
    for b in BENCHES + ["Mean"]:
        e, a, aa = bench_acc[b]["ece"], bench_acc[b]["auroc"], bench_acc[b]["auarc"]
        if not e: continue
        em, es = 100*np.mean(e), 100*np.std(e, ddof=1)
        am, asd = 100*np.mean(a), 100*np.std(a, ddof=1)
        aam, aasd = 100*np.mean(aa), 100*np.std(aa, ddof=1)
        print(f"{b:<13} {em:5.2f}±{es:.2f}    {am:5.2f}±{asd:.2f}     {aam:5.2f}±{aasd:.2f}")


if __name__ == "__main__":
    sys.exit(main())
