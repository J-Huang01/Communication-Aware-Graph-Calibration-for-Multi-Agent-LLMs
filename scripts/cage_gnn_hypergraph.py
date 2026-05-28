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
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATv2Conv, HypergraphConv, global_max_pool, global_mean_pool

warnings.filterwarnings("ignore")

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE)); sys.path.insert(0, str(_HERE.parent))

from betacal import BetaCalibration
from calibration import PlattBinnerCalibrator
from cage_cal.cage_cal_features import comm_adjacency
from cage_cal.calibration import auarc, auroc, ece
from cage_cal.per_query_W import PerQueryWEstimator
from data_utils import (BENCHES, LOADERS, TOPOS, load_cell_panels,
                              make_splits, _panel_correctness)

ROOT = Path("results/panels")
SBERT_CACHE = Path("cache/sbert_answer_emb.pkl")
HYPER_DATA_CACHE = Path("cache/cage_gnn_hyper_panels.pkl")
PCA_DIM = 16
ROLLOUTS = [1, 2, 3]


def _norm(s):
    return (s or "").strip().lower().strip(".\"' \t\n,;:!?")


def _split_agent(agent_id):
    """agent_id format 'backbone::role'."""
    parts = agent_id.split("::")
    if len(parts) >= 2:
        return parts[0], parts[1]
    return agent_id, "default"


# ============================================================ SBERT cache ===
def build_sbert_cache(root):
    if SBERT_CACHE.exists():
        with SBERT_CACHE.open("rb") as f:
            return pickle.load(f)
    print("[hyper] building SBERT cache...", file=sys.stderr)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda:0")
    answers = set()
    for topo in TOPOS:
        for bench in BENCHES:
            panels = load_cell_panels(root, topo, bench)
            for panel in panels.values():
                for r in panel.values():
                    if r.get("answer"):
                        answers.add(r["answer"])
    all_answers = sorted(answers)
    embs = model.encode(all_answers, batch_size=512, show_progress_bar=False,
                          normalize_embeddings=True)
    out = {a: embs[i].astype(np.float32) for i, a in enumerate(all_answers)}
    SBERT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with SBERT_CACHE.open("wb") as f:
        pickle.dump(out, f)
    return out


# ============================================================ Hyperedges ===
def build_hyperedges(panel, agents, topology, include_comm=True):
    """Returns a list of hyperedges (each = list of node indices).

    Hyperedge types:
      - backbone groups (filter to size >= 2)
      - role groups (filter to size >= 2)
      - answer-cluster groups in this panel
      - topology-comm hyperedge (all agents) if non-iid AND include_comm
    """
    backbones = [_split_agent(a)[0] for a in agents]
    roles = [_split_agent(a)[1] for a in agents]

    edges = []
    bb_groups = defaultdict(list)
    for i, b in enumerate(backbones):
        bb_groups[b].append(i)
    for members in bb_groups.values():
        if len(members) >= 2:
            edges.append(members)

    role_groups = defaultdict(list)
    for i, r in enumerate(roles):
        role_groups[r].append(i)
    for members in role_groups.values():
        if len(members) >= 2:
            edges.append(members)

    ans_groups = defaultdict(list)
    for i, a in enumerate(agents):
        ans = _norm(panel[a].get("answer", ""))
        ans_groups[ans].append(i)
    for members in ans_groups.values():
        if len(members) >= 2:
            edges.append(members)

    if include_comm and topology != "iid":
        edges.append(list(range(len(agents))))

    return edges


def edges_to_index(edges_list):
    """Convert list of hyperedges to PyG hyperedge_index format.
    Shape: (2, E_total) where row 0 = node indices, row 1 = hyperedge id.
    """
    if not edges_list:
        return torch.zeros((2, 0), dtype=torch.long), 0
    node_idx, edge_idx = [], []
    for he_id, members in enumerate(edges_list):
        for n in members:
            node_idx.append(n)
            edge_idx.append(he_id)
    return torch.tensor([node_idx, edge_idx], dtype=torch.long), len(edges_list)


class HyperData(Data):
    """Custom Data class so PyG Batch offsets hyperedge_index correctly."""
    def __inc__(self, key, value, *args, **kwargs):
        if key in ("hyperedge_index_T", "hyperedge_index_0"):
            # row 0 offset by num_nodes, row 1 offset by num_hyperedges
            n_he = getattr(self, f"num_he_{key.split('_')[-1]}", 0)
            return torch.tensor([[self.num_nodes], [n_he]])
        return super().__inc__(key, value, *args, **kwargs)


# ============================================================ Panel -> Data
def panel_to_hyperdata(panel_T, panel_0, agents, W_T, W_0, w_comm, sbert_pca,
                         topology):
    """Returns a HyperData with both pairwise (two-tower) and hyperedge info."""
    N = len(agents)
    answers_T = [panel_T[a]["answer"] for a in agents]
    lp_T = np.array([panel_T[a].get("mean_logprob") or -5.0 for a in agents],
                      dtype=np.float32)
    lp_T = np.clip(lp_T, -10.0, 0.0)
    lp_T_norm = (lp_T + 5.0) / 5.0

    counts = Counter(answers_T).most_common()
    ans_to_rank = {a: r for r, (a, _) in enumerate(counts)}
    ranks_T = np.array([ans_to_rank[a] for a in answers_T], dtype=np.float32)
    ranks_T_norm = ranks_T / max(len(counts) - 1, 1)
    plurality_T = counts[0][0]
    plur_T = np.array([1.0 if a == plurality_T else 0.0 for a in answers_T],
                        dtype=np.float32)

    answers_0 = [panel_0[a]["answer"] for a in agents]
    lp_0 = np.array([panel_0[a].get("mean_logprob") or -5.0 for a in agents],
                      dtype=np.float32)
    lp_0 = np.clip(lp_0, -10.0, 0.0)
    lp_0_norm = (lp_0 + 5.0) / 5.0
    counts0 = Counter(answers_0).most_common()
    ans_to_rank0 = {a: r for r, (a, _) in enumerate(counts0)}
    ranks_0 = np.array([ans_to_rank0[a] for a in answers_0], dtype=np.float32)
    ranks_0_norm = ranks_0 / max(len(counts0) - 1, 1)
    plurality_0 = counts0[0][0]
    plur_0 = np.array([1.0 if a == plurality_0 else 0.0 for a in answers_0],
                        dtype=np.float32)

    W_abs = np.abs(W_T)
    row_mean = W_abs.mean(axis=1).astype(np.float32)
    row_var = W_abs.var(axis=1).astype(np.float32)
    row_max = W_abs.max(axis=1).astype(np.float32)
    W_abs_0 = np.abs(W_0)
    row_mean_0 = W_abs_0.mean(axis=1).astype(np.float32)
    row_var_0 = W_abs_0.var(axis=1).astype(np.float32)
    row_max_0 = W_abs_0.max(axis=1).astype(np.float32)

    panel_size_inv = np.full(N, 1.0 / max(N, 1), dtype=np.float32)
    sbert_T = np.stack([sbert_pca.get(a, np.zeros(PCA_DIM, dtype=np.float32))
                          for a in answers_T])
    sbert_0 = np.stack([sbert_pca.get(a, np.zeros(PCA_DIM, dtype=np.float32))
                          for a in answers_0])

    x_T = np.column_stack([lp_T_norm, ranks_T_norm, row_mean, row_var, row_max,
                             plur_T, panel_size_inv, sbert_T])
    x_0 = np.column_stack([lp_0_norm, ranks_0_norm, row_mean_0, row_var_0,
                             row_max_0, plur_0, panel_size_inv, sbert_0])

    # Pairwise edges for T tower (topology comm + W_T threshold)
    src_T, dst_T, attr_T = [], [], []
    for i in range(N):
        for j in range(N):
            if i == j: continue
            ic = float(w_comm[i, j] > 0)
            wt = float(W_T[i, j])
            if ic > 0 or abs(wt) > 0.05:
                src_T.append(i); dst_T.append(j); attr_T.append([ic, wt])
    if not src_T:
        for i in range(N):
            for j in range(N):
                if i == j: continue
                src_T.append(i); dst_T.append(j)
                attr_T.append([0.0, float(W_T[i, j])])

    # Pairwise edges for 0 tower (W_0 threshold only, is_comm = 0)
    src_0, dst_0, attr_0 = [], [], []
    for i in range(N):
        for j in range(N):
            if i == j: continue
            w0 = float(W_0[i, j])
            if abs(w0) > 0.05:
                src_0.append(i); dst_0.append(j); attr_0.append([0.0, w0])
    if not src_0:
        for i in range(N):
            for j in range(N):
                if i == j: continue
                src_0.append(i); dst_0.append(j)
                attr_0.append([0.0, float(W_0[i, j])])

    # Hyperedges
    he_T = build_hyperedges(panel_T, agents, topology, include_comm=True)
    he_0 = build_hyperedges(panel_0, agents, "iid", include_comm=False)
    he_index_T, num_he_T = edges_to_index(he_T)
    he_index_0, num_he_0 = edges_to_index(he_0)

    d = HyperData(
        x_T=torch.tensor(x_T, dtype=torch.float32),
        x_0=torch.tensor(x_0, dtype=torch.float32),
        edge_index_T=torch.tensor([src_T, dst_T], dtype=torch.long),
        edge_attr_T=torch.tensor(attr_T, dtype=torch.float32),
        edge_index_0=torch.tensor([src_0, dst_0], dtype=torch.long),
        edge_attr_0=torch.tensor(attr_0, dtype=torch.float32),
        hyperedge_index_T=he_index_T,
        hyperedge_index_0=he_index_0,
    )
    d.num_he_T = num_he_T
    d.num_he_0 = num_he_0
    # Use x_T as the canonical "x" so PyG knows num_nodes
    d.x = d.x_T
    return d


def build_dataset(root, sbert_pca, estimator, tr_qids, te_qids):
    if HYPER_DATA_CACHE.exists():
        with HYPER_DATA_CACHE.open("rb") as f:
            ds = pickle.load(f)
        print(f"[hyper] dataset cache hit: {len(ds)} panels", file=sys.stderr)
        return ds

    panels_cell = {}
    for topo in TOPOS:
        for bench in BENCHES:
            panels_cell[(topo, bench)] = load_cell_panels(root, topo, bench)
    cell_W, cell_agents, cell_qid_idx = {}, {}, {}
    for topo in TOPOS:
        for bench in BENCHES:
            try:
                W, agents, qids_all = estimator.load(topo, bench)
            except FileNotFoundError:
                continue
            cell_W[(topo, bench)] = W
            cell_agents[(topo, bench)] = agents
            cell_qid_idx[(topo, bench)] = {q: i for i, q in enumerate(qids_all)}

    out = []
    for topo in TOPOS:
        for bench in BENCHES:
            panels_T = panels_cell.get((topo, bench), {})
            panels_0 = panels_cell.get(("iid", bench), {})
            if not panels_T: continue
            agents_T = cell_agents.get((topo, bench))
            agents_0 = cell_agents.get(("iid", bench))
            if agents_T is None or agents_0 is None: continue
            common = [a for a in agents_T if a in agents_0]
            if len(common) < 2: continue
            idx_T = np.array([agents_T.index(a) for a in common])
            idx_0 = np.array([agents_0.index(a) for a in common])
            W_T_full = cell_W[(topo, bench)]
            W_0_full = cell_W[("iid", bench)]
            qid_idx_T = cell_qid_idx[(topo, bench)]
            qid_idx_0 = cell_qid_idx[("iid", bench)]
            w_comm = comm_adjacency(topo, common)
            train_set = tr_qids[bench]
            test_set = te_qids[bench]
            for (qid, ro), panel_T_raw in panels_T.items():
                pT = {a: panel_T_raw[a] for a in common if a in panel_T_raw}
                if len(pT) != len(common): continue
                p0_raw = panels_0.get((qid, ro), {})
                p0 = {a: p0_raw[a] for a in common if a in p0_raw}
                if len(p0) != len(common): continue
                if qid not in qid_idx_T: continue
                W_T = W_T_full[qid_idx_T[qid]][np.ix_(idx_T, idx_T)]
                if qid not in qid_idx_0: continue
                W_0 = W_0_full[qid_idx_0[qid]][np.ix_(idx_0, idx_0)]
                d = panel_to_hyperdata(pT, p0, common, W_T, W_0, w_comm,
                                        sbert_pca, topo)
                if qid in train_set: split = "train"
                elif qid in test_set: split = "test"
                else: split = "val"
                out.append({
                    "data": d, "bench": bench, "topo": topo,
                    "qid": qid, "ro": ro,
                    "label": int(_panel_correctness(pT)),
                    "split": split,
                })
    HYPER_DATA_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with HYPER_DATA_CACHE.open("wb") as f:
        pickle.dump(out, f)
    print(f"[hyper] built dataset: {len(out)} panels", file=sys.stderr)
    return out


# ============================================================ Model =========
class HyperHybridTower(nn.Module):
    """Two-layer hybrid: GATv2 pairwise + HypergraphConv hyperedges, concat.

    Layer 1: x(24-d) → GATv2(hid, heads) || HypergraphConv(hid*heads) → cat → 2*hid*heads
    Layer 2: cat → GATv2(hid, 1 head) || HypergraphConv(hid) → cat → 2*hid
    Readout: mean_pool + max_pool over each stream → 4*hid
    """
    def __init__(self, in_dim, hid=64, heads=4, dropout=0.3):
        super().__init__()
        # Pairwise stream
        self.gat1 = GATv2Conv(in_dim, hid, heads=heads, edge_dim=2,
                                dropout=dropout, concat=True)  # → hid*heads
        self.gat2 = GATv2Conv(hid*heads + hid*heads, hid, heads=1, edge_dim=2,
                                dropout=dropout, concat=False)  # → hid
        # Hyperedge stream
        self.hyper1 = HypergraphConv(in_dim, hid*heads)  # → hid*heads (match GAT1)
        self.hyper2 = HypergraphConv(hid*heads + hid*heads, hid)  # → hid
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, hyperedge_index, batch):
        # Layer 1
        h_g = self.gat1(x, edge_index, edge_attr)
        h_g = F.elu(h_g)
        if hyperedge_index.numel() > 0:
            h_h = self.hyper1(x, hyperedge_index)
        else:
            h_h = torch.zeros_like(h_g)
        h_h = F.elu(h_h)
        h = torch.cat([h_g, h_h], dim=-1)  # (N, 2*hid*heads)
        h = self.dropout(h)

        # Layer 2
        h_g2 = self.gat2(h, edge_index, edge_attr)
        h_g2 = F.elu(h_g2)
        if hyperedge_index.numel() > 0:
            h_h2 = self.hyper2(h, hyperedge_index)
        else:
            h_h2 = torch.zeros_like(h_g2)
        h_h2 = F.elu(h_h2)
        h_out = torch.cat([h_g2, h_h2], dim=-1)  # (N, 2*hid)

        # Readout: mean + max pool
        return torch.cat([
            global_mean_pool(h_out, batch),
            global_max_pool(h_out, batch),
        ], dim=-1)  # (B, 4*hid)


class HyperHybridGNN(nn.Module):
    def __init__(self, in_dim, hid=64, heads=4, n_bench=5, dropout=0.3):
        super().__init__()
        # Shared two-tower backbone
        self.tower = HyperHybridTower(in_dim, hid, heads, dropout)
        emb_dim = 4 * hid  # mean+max over (h_g + h_h) at layer 2
        head_in = emb_dim * 3 + n_bench
        self.head = nn.Sequential(
            nn.Linear(head_in, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, batch_data, bench_oh):
        # T tower
        h_T = self.tower(
            batch_data.x_T, batch_data.edge_index_T,
            batch_data.edge_attr_T, batch_data.hyperedge_index_T,
            batch_data.batch,
        )
        # 0 tower
        h_0 = self.tower(
            batch_data.x_0, batch_data.edge_index_0,
            batch_data.edge_attr_0, batch_data.hyperedge_index_0,
            batch_data.batch,
        )
        feat = torch.cat([h_T, h_0, h_T - h_0, bench_oh], dim=-1)
        return self.head(feat).squeeze(-1)


# ============================================================ Train / Eval ==
def collate(batch_panels):
    data_list = [p["data"] for p in batch_panels]
    batch = Batch.from_data_list(data_list)
    bench = torch.zeros(len(batch_panels), len(BENCHES), dtype=torch.float32)
    for i, p in enumerate(batch_panels):
        bench[i, BENCHES.index(p["bench"])] = 1.0
    label = torch.tensor([p["label"] for p in batch_panels], dtype=torch.float32)
    return batch, bench, label


def predict(model, ds, device, batch_size=128):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(ds), batch_size):
            chunk = ds[i:i + batch_size]
            batch, bench, _ = collate(chunk)
            batch, bench = batch.to(device), bench.to(device)
            logit = model(batch, bench)
            preds.append(torch.sigmoid(logit).cpu().numpy())
    return np.concatenate(preds)


def train_one(tr_ds, va_ds, in_dim, device, *, hid=64, heads=4,
                dropout=0.3, lr=2e-3, weight_decay=3e-4, epochs=15,
                batch_size=128, seed=0, label_smooth=0.05):
    torch.manual_seed(seed); np.random.seed(seed)
    model = HyperHybridGNN(in_dim, hid=hid, heads=heads,
                            n_bench=len(BENCHES), dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_score, best_state, patience, max_patience = -1e9, None, 0, 5
    n_tr = len(tr_ds)
    for ep in range(1, epochs + 1):
        model.train()
        np.random.shuffle(tr_ds)
        total_loss = 0.0
        for i in range(0, n_tr, batch_size):
            chunk = tr_ds[i:i + batch_size]
            batch, bench, label = collate(chunk)
            batch, bench, label = batch.to(device), bench.to(device), label.to(device)
            target = label * (1 - 2*label_smooth) + label_smooth
            logit = model(batch, bench)
            loss = F.binary_cross_entropy_with_logits(logit, target)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * len(chunk)
        sched.step()
        pva = predict(model, va_ds, device, batch_size)
        yva = np.array([p["label"] for p in va_ds])
        bva = [p["bench"] for p in va_ds]
        scores = []
        for b in BENCHES:
            m = np.array(bva) == b
            yy = yva[m]
            if 0 < yy.sum() < len(yy):
                scores.append(auroc(pva[m], yy))
        score = float(np.mean(scores))
        if score > best_score:
            best_score, patience = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        print(f"  [hyper] seed{seed} ep{ep:02d} loss={total_loss/n_tr:.4f} "
              f"val AUROC={score:.4f} best={best_score:.4f}", file=sys.stderr)
        if patience >= max_patience:
            print(f"  [hyper] seed{seed} early-stop at ep{ep}", file=sys.stderr)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_score


def fit_per_bench_betasb(pva, yva, bva):
    betas, sbs = {}, {}
    for b in BENCHES:
        m = np.array(bva) == b
        if m.sum() < 50: continue
        pc = np.clip(pva[m], 1e-6, 1 - 1e-6)
        try:
            betas[b] = BetaCalibration(parameters="abm").fit(pc.reshape(-1, 1), yva[m])
        except Exception: pass
        try:
            cal = PlattBinnerCalibrator(num_calibration=m.sum(), num_bins=10)
            cal.train_calibration(pva[m], yva[m].astype(int))
            sbs[b] = cal
        except Exception: pass
    return betas, sbs


def apply_betasb(p, b_, betas, sbs, beta_weight=0.5):
    """Weighted average of Beta-cal and Scaling-Binning predictions.
    beta_weight: weight on Beta (0.0 = SB-only, 1.0 = Beta-only, 0.5 = equal).
    """
    out = p.copy()
    w_beta = float(beta_weight)
    w_sb = 1.0 - w_beta
    for b in BENCHES:
        m = np.array(b_) == b
        if m.sum() == 0: continue
        beta_pred = None; sb_pred = None
        if b in betas:
            pc = np.clip(p[m], 1e-6, 1 - 1e-6)
            beta_pred = betas[b].predict(pc.reshape(-1, 1)).ravel()
        if b in sbs:
            sb_pred = np.asarray(sbs[b].calibrate(p[m]), dtype=np.float64)
        if beta_pred is not None and sb_pred is not None:
            out[m] = w_beta * beta_pred + w_sb * sb_pred
        elif beta_pred is not None:
            out[m] = beta_pred
        elif sb_pred is not None:
            out[m] = sb_pred
    return out


def per_rollout_summary(p, y, b_, t_, r_):
    b_, t_, r_ = map(np.array, (b_, t_, r_))
    per_b_per_ro = {}
    for bench in BENCHES:
        per_ro = {}
        for ro in ROLLOUTS:
            ece_l, au_l, aa_l, br_l = [], [], [], []
            for topo in TOPOS:
                m = (b_ == bench) & (t_ == topo) & (r_ == ro)
                if m.sum() < 20: continue
                pp, yy = p[m], y[m]
                if 0 < yy.sum() < len(yy):
                    ece_l.append(ece(pp, yy, n_bins=10))
                    au_l.append(auroc(pp, yy.astype(int)))
                    aa_l.append(auarc(pp, yy.astype(int)))
                    br_l.append(float(np.mean((pp - yy.astype(float)) ** 2)))
            if ece_l:
                per_ro[ro] = {"ece": float(np.mean(ece_l)),
                              "auroc": float(np.mean(au_l)),
                              "auarc": float(np.mean(aa_l)),
                              "brier": float(np.mean(br_l))}
        per_b_per_ro[bench] = per_ro

    summary = {}
    for bench, per_ro in per_b_per_ro.items():
        ros = sorted(per_ro.keys())
        if len(ros) >= 2:
            arr = np.array([[per_ro[ro]["ece"], per_ro[ro]["auroc"],
                              per_ro[ro]["auarc"], per_ro[ro]["brier"]]
                            for ro in ros])
            summary[bench] = {
                "ece_mean": float(arr[:, 0].mean()), "ece_std": float(arr[:, 0].std(ddof=1)),
                "auroc_mean": float(arr[:, 1].mean()), "auroc_std": float(arr[:, 1].std(ddof=1)),
                "auarc_mean": float(arr[:, 2].mean()), "auarc_std": float(arr[:, 2].std(ddof=1)),
                "brier_mean": float(arr[:, 3].mean()), "brier_std": float(arr[:, 3].std(ddof=1)),
            }
    per_ro_overall = {}
    for ro in ROLLOUTS:
        ece_b, au_b, aa_b, br_b = [], [], [], []
        for b in BENCHES:
            v = per_b_per_ro.get(b, {}).get(ro)
            if not v: continue
            ece_b.append(v["ece"]); au_b.append(v["auroc"])
            aa_b.append(v["auarc"]); br_b.append(v["brier"])
        if ece_b:
            per_ro_overall[ro] = (float(np.mean(ece_b)),
                                  float(np.mean(au_b)),
                                  float(np.mean(aa_b)),
                                  float(np.mean(br_b)))
    if len(per_ro_overall) >= 2:
        arr = np.array(list(per_ro_overall.values()))
        summary["Mean"] = {
            "ece_mean": float(arr[:, 0].mean()), "ece_std": float(arr[:, 0].std(ddof=1)),
            "auroc_mean": float(arr[:, 1].mean()), "auroc_std": float(arr[:, 1].std(ddof=1)),
            "auarc_mean": float(arr[:, 2].mean()), "auarc_std": float(arr[:, 2].std(ddof=1)),
            "brier_mean": float(arr[:, 3].mean()), "brier_std": float(arr[:, 3].std(ddof=1)),
        }
    return summary


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
    print(f"[hyper] train={len(tr_ds)} val={len(va_ds)} test={len(te_ds)} in_dim={in_dim}",
          file=sys.stderr)

    # Quick sanity check on hyperedge counts
    he_T_counts = [p["data"].num_he_T for p in tr_ds[:100]]
    he_0_counts = [p["data"].num_he_0 for p in tr_ds[:100]]
    print(f"[hyper] hyperedges per panel: T avg={np.mean(he_T_counts):.1f} "
          f"0 avg={np.mean(he_0_counts):.1f}", file=sys.stderr)

    yva = np.array([p["label"] for p in va_ds])
    yte = np.array([p["label"] for p in te_ds])
    bva = [p["bench"] for p in va_ds]; bte = [p["bench"] for p in te_ds]
    tte = [p["topo"] for p in te_ds]; rte = [p["ro"] for p in te_ds]

    pva_seeds, pte_seeds = [], []
    for s in range(args.n_seeds):
        print(f"\n[hyper] === seed {s} ===", file=sys.stderr)
        model, _ = train_one(tr_ds, va_ds, in_dim, device,
                              seed=s, epochs=args.epochs)
        pva_seeds.append(predict(model, va_ds, device))
        pte_seeds.append(predict(model, te_ds, device))
        del model; torch.cuda.empty_cache()

    pva = np.mean(pva_seeds, axis=0)
    pte = np.mean(pte_seeds, axis=0)
    betas, sbs = fit_per_bench_betasb(pva, yva, bva)
    pte_cal = apply_betasb(pte, bte, betas, sbs)

    out = {
        "hyper_raw":    per_rollout_summary(pte,     yte, bte, tte, rte),
        "hyper_BetaSB": per_rollout_summary(pte_cal, yte, bte, tte, rte),
    }
    out_path = ROOT / f"cage_gnn_hyper_n{args.n_seeds}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[hyper] wrote {out_path}\n", file=sys.stderr)

    print(f"\n(ref) two-tower BetaSB+smooth   ECE=6.78±0.33  AUROC=81.89±1.47  AUARC=75.79±0.24")
    for suffix in ["raw", "BetaSB"]:
        m = out[f"hyper_{suffix}"].get("Mean", {})
        print(f"hyper_{suffix:8s}   "
              f"ECE={100*m['ece_mean']:5.2f}±{100*m['ece_std']:.2f}  "
              f"AUROC={100*m['auroc_mean']:5.2f}±{100*m['auroc_std']:.2f}  "
              f"AUARC={100*m['auarc_mean']:5.2f}±{100*m['auarc_std']:.2f}")


if __name__ == "__main__":
    sys.exit(main())
