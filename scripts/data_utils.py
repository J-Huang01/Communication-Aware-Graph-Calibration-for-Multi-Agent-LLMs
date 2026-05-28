from __future__ import annotations

import collections
import json
from pathlib import Path

from cage_cal.benchmarks.bbh import load_bbh
from cage_cal.benchmarks.gsm8k import load_gsm8k
from cage_cal.benchmarks.mmlu_pro import load_mmlu_pro
from cage_cal.benchmarks.trivia_qa import load_trivia_qa
from cage_cal.benchmarks.truthful_qa import load_truthful_qa
from cage_cal.per_query_W import _strip_tree

LOADERS = {
    "trivia_qa": load_trivia_qa,
    "truthful_qa": load_truthful_qa,
    "mmlu_pro": load_mmlu_pro,
    "gsm8k": load_gsm8k,
    "bbh": load_bbh,
}
TOPOS = ["iid", "debate", "chain", "hub_spoke", "tree"]
BENCHES = ["trivia_qa", "truthful_qa", "mmlu_pro", "gsm8k", "bbh"]


def _norm(s):
    return (s or "").strip().lower().strip(".\"' \t\n,;:!?")


def make_splits(problems, frac_train=0.6, frac_val=0.2):
    """Deterministic 60/20/20 train/val/test split over problem qids."""
    n = len(problems)
    n_train = int(n * frac_train)
    n_val_end = int(n * (frac_train + frac_val))
    return (
        {p.qid for p in problems[:n_train]},
        {p.qid for p in problems[n_train:n_val_end]},
        {p.qid for p in problems[n_val_end:]},
    )


def load_cell_panels(root: Path, topo: str, bench: str) -> dict:
    """Load one (topology, benchmark) cell.

    Returns {(qid, rollout): {agent_id: {answer, mean_logprob, correct}}},
    reading the deduplicated rollouts and the grader labels written by the
    panel-generation stage.
    """
    lfn = root / topo / bench / "labels.jsonl"
    rfn = root / topo / bench / "rollouts.dedup.jsonl"
    if not lfn.exists() or not rfn.exists():
        return {}
    labels = {}
    for line in lfn.open():
        l = json.loads(line)
        labels[(l["qid"], l["rollout"], _strip_tree(l["agent_id"]))] = bool(l["correct"])
    rolls = [json.loads(line) for line in rfn.open()]
    out: dict = collections.defaultdict(dict)
    for r in rolls:
        aid = _strip_tree(r["agent_id"])
        key = (r["qid"], r["rollout"], aid)
        if key not in labels:
            continue
        out[(r["qid"], r["rollout"])][aid] = {
            "answer": _norm(r.get("answer", "")),
            "mean_logprob": r.get("mean_logprob"),
            "correct": labels[key],
        }
    return dict(out)


def _panel_correctness(panel: dict) -> bool:
    """Whether the panel's plurality-vote answer is correct."""
    answers = [r["answer"] for r in panel.values()]
    if not answers:
        return False
    from collections import Counter as C
    top, _ = C(answers).most_common(1)[0]
    for r in panel.values():
        if r["answer"] == top:
            return bool(r["correct"])
    return False
