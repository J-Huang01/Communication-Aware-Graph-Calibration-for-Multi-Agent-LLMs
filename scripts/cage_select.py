from __future__ import annotations

import argparse
import json
import pickle
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE)); sys.path.insert(0, str(_HERE.parent))

from sklearn.decomposition import PCA
from cage_cal.per_query_W import PerQueryWEstimator
from data_utils import BENCHES, LOADERS, TOPOS, make_splits, load_cell_panels
from cage_gnn_hypergraph import (
    HyperData,  # noqa: F401  for pickle
    PCA_DIM, ROOT, ROLLOUTS, apply_betasb, build_dataset, build_sbert_cache,
    fit_per_bench_betasb, predict,
)
from cage_gnn_hyper_selfcal import train_one_selfcal
import sys as _sys
_sys.modules['__main__'].HyperData = HyperData


def _norm(s):
    return (s or "").strip().lower().strip(".\"' \t\n,;:!?")


def plurality_share(panel_T_raw, common):
    """Return plurality answer and its vote share."""
    answers = [_norm(panel_T_raw[a]["answer"]) for a in common if a in panel_T_raw]
    if not answers: return "", 0.0
    cnt = Counter(answers)
    top, n_top = cnt.most_common(1)[0]
    return top, n_top / len(answers)


def panel_avg_logprob(panel_T_raw, common):
    lps = []
    for a in common:
        if a not in panel_T_raw: continue
        lp = panel_T_raw[a].get("mean_logprob")
        if lp is not None:
            lps.append(lp)
    return float(np.exp(np.mean(lps))) if lps else 0.0


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
    print(f"[cage-select] train={len(tr_ds)} val={len(va_ds)} test={len(te_ds)}",
          file=sys.stderr)

    # --- 1. Train + predict CAGE-Cal ---
    pva_seeds, pte_seeds = [], []
    for s in range(args.n_seeds):
        print(f"\n[cage-select] === seed {s} ===", file=sys.stderr)
        model = train_one_selfcal(
            tr_ds, va_ds, in_dim, device, seed=s, epochs=args.epochs,
            brier_weight=0.4, mmce_weight=0.0,
        )
        pva_seeds.append(predict(model, va_ds, device))
        pte_seeds.append(predict(model, te_ds, device))
        del model; torch.cuda.empty_cache()
    pva = np.mean(pva_seeds, axis=0)
    pte = np.mean(pte_seeds, axis=0)

    # Per-bench BetaSB
    yva = np.array([p["label"] for p in va_ds])
    bva = [p["bench"] for p in va_ds]
    bte = [p["bench"] for p in te_ds]
    betas, sbs = fit_per_bench_betasb(pva, yva, bva)
    pte_cage = apply_betasb(pte, bte, betas, sbs, beta_weight=0.0)

    # --- 2. Compute per-panel metadata (plurality answer, vote share, logprob) ---
    # We need to re-read raw panels to get per-agent answers
    panels_cell = {(topo, bench): load_cell_panels(ROOT, topo, bench)
                     for topo in TOPOS for bench in BENCHES}
    cell_W, cell_agents, cell_qid_idx = {}, {}, {}
    for topo in TOPOS:
        for bench in BENCHES:
            try:
                W, agents, qids_all = est.load(topo, bench)
                cell_W[(topo, bench)] = W
                cell_agents[(topo, bench)] = agents
                cell_qid_idx[(topo, bench)] = {q: i for i, q in enumerate(qids_all)}
            except FileNotFoundError:
                continue

    # For each test panel: extract plurality answer, vote_share, mean_logprob
    def make_records(ds, p_cal):
        out = []
        for i, p in enumerate(ds):
            qid, ro, topo, bench = p["qid"], p["ro"], p["topo"], p["bench"]
            agents_T = cell_agents.get((topo, bench), [])
            agents_0 = cell_agents.get(("iid", bench), [])
            common = [a for a in agents_T if a in agents_0]
            panel_T = panels_cell[(topo, bench)].get((qid, ro), {})
            pT = {a: panel_T[a] for a in common if a in panel_T}
            ans, share = plurality_share(pT, common)
            mean_lp = panel_avg_logprob(pT, common)
            out.append({
                "qid": qid, "ro": ro, "topo": topo, "bench": bench,
                "label": p["label"], "plurality_ans": ans,
                "plurality_share": share, "mean_logprob": mean_lp,
                "cage_conf": float(p_cal[i]),
            })
        return out

    pva_cage = apply_betasb(pva, bva, betas, sbs)
    val_records = make_records(va_ds, pva_cage)
    test_records = make_records(te_ds, pte_cage)

    # Group by (qid, ro, bench)
    groups = defaultdict(dict)  # (qid, ro, bench) -> {topo: record}
    for r in test_records:
        key = (r["qid"], r["ro"], r["bench"])
        groups[key][r["topo"]] = r
    # Keep only full-5-topology groups for fair comparison
    groups = {k: v for k, v in groups.items() if len(v) == 5}
    print(f"\n[cage-select] full 5-topo groups: {len(groups)}", file=sys.stderr)

    # --- 3. Selection strategies ---
    def strategy_fixed(topo):
        return [g[topo]["label"] for g in groups.values()]

    def strategy_majority_label():
        return [int(sum(t["label"] for t in g.values()) >= 3) for g in groups.values()]

    def strategy_pick_topo_by(score_fn):
        """Pick the topology with highest score_fn(panel); return its label."""
        out = []
        for g in groups.values():
            best_topo = max(g.keys(), key=lambda t: score_fn(g[t]))
            out.append(g[best_topo]["label"])
        return out

    def strategy_answer_select(reduce_fn):
        """For each distinct answer, aggregate confs across topos giving that
        answer, then pick answer with highest aggregated score.
        Label of final answer = label of the topology giving best score for that answer."""
        out = []
        for g in groups.values():
            # answer -> list of (topo, conf, label)
            ans_to_records = defaultdict(list)
            for topo, r in g.items():
                ans_to_records[r["plurality_ans"]].append(r)
            # Pick answer with best aggregated conf
            best_ans = max(ans_to_records.keys(),
                            key=lambda a: reduce_fn([r["cage_conf"] for r in ans_to_records[a]]))
            # "Correct" if at least one topology with this answer is correct
            # (since they all said the same answer, all labels should match)
            out.append(int(any(r["label"] for r in ans_to_records[best_ans])))
        return out

    def strategy_oracle():
        return [int(any(t["label"] for t in g.values())) for g in groups.values()]

    # Compute val accuracies per topology for "best fixed (val-tuned)"
    val_topo_acc = {t: np.mean([r["label"] for r in val_records if r["topo"] == t])
                    for t in TOPOS}
    best_fixed_topo = max(TOPOS, key=lambda t: val_topo_acc[t])
    print(f"[cage-select] val-tuned best fixed topology: {best_fixed_topo}",
          file=sys.stderr)

    results = {}
    for t in TOPOS:
        results[f"Fixed {t}"] = strategy_fixed(t)
    results[f"Best fixed (val-tuned: {best_fixed_topo})"] = strategy_fixed(best_fixed_topo)
    results["Majority over topologies"] = strategy_majority_label()
    results["Highest plurality share"] = strategy_pick_topo_by(lambda r: r["plurality_share"])
    results["Highest mean logprob"] = strategy_pick_topo_by(lambda r: r["mean_logprob"])
    results["CAGE-Select (max conf)"] = strategy_pick_topo_by(lambda r: r["cage_conf"])
    results["CAGE-AnswerSelect (max)"] = strategy_answer_select(max)
    results["CAGE-AnswerSelect (mean)"] = strategy_answer_select(lambda xs: sum(xs) / len(xs))
    results["Oracle topology (upper bound)"] = strategy_oracle()

    print(f"\n{'Strategy':<42s}  {'Acc':>8s}  {'Δ vs best fixed':>18s}")
    print("-" * 75)
    best_fixed_acc = np.mean(strategy_fixed(best_fixed_topo))
    out_json = {}
    for name, labels in results.items():
        acc = float(np.mean(labels))
        delta = 100 * (acc - best_fixed_acc)
        out_json[name] = acc
        print(f"{name:<42s}  {acc:7.4f}  {delta:+8.2f}pp")

    # Per-bench breakdown
    print(f"\n{'Strategy':<42s}", end="")
    for b in BENCHES:
        print(f"  {b:<12}", end="")
    print()
    print("-" * 110)
    for name, labels in results.items():
        print(f"{name:<42s}", end="")
        # Need to recompute per bench
        labels_by_bench = defaultdict(list)
        for (key, label) in zip(list(groups.keys()), labels):
            b = key[2]
            labels_by_bench[b].append(label)
        for b in BENCHES:
            ls = labels_by_bench[b]
            print(f"  {np.mean(ls):>8.4f}    ", end="")
        print()

    # Save summary
    out_path = ROOT / "cage_select_results.json"
    out_path.write_text(json.dumps({
        "strategies": out_json,
        "best_fixed_topology": best_fixed_topo,
        "n_groups": len(groups),
    }, indent=2))
    print(f"\nwrote {out_path}", file=sys.stderr)

    # Also dump raw per-panel records (for iterating selection rules)
    def group_records(records):
        g = defaultdict(dict)
        for r in records:
            key = f"{r['qid']}||{r['ro']}||{r['bench']}"
            g[key][r["topo"]] = {
                "cage_conf": r["cage_conf"],
                "plurality_ans": r["plurality_ans"],
                "plurality_share": r["plurality_share"],
                "mean_logprob": r["mean_logprob"],
                "label": r["label"],
            }
        return {k: v for k, v in g.items() if len(v) == 5}

    raw_path = ROOT / "cage_select_records.json"
    raw_path.write_text(json.dumps({
        "test_groups": group_records(test_records),
        "val_groups":  group_records(val_records),
    }, indent=2, default=float))
    print(f"wrote {raw_path}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
