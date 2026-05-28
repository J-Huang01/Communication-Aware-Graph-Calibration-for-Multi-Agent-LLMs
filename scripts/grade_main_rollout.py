from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cage_cal.benchmarks.bbh import load_bbh
from cage_cal.benchmarks.gsm8k import load_gsm8k
from cage_cal.benchmarks.mmlu_pro import load_mmlu_pro
from cage_cal.benchmarks.trivia_qa import load_trivia_qa
from cage_cal.benchmarks.truthful_qa import load_truthful_qa
from cage_cal.judge import judge_one
from cage_cal.scoring import score_one


LOADERS = {
    "trivia_qa":   load_trivia_qa,
    "truthful_qa": load_truthful_qa,
    "mmlu_pro":    load_mmlu_pro,
    "gsm8k":       load_gsm8k,
    "bbh":         load_bbh,
}
TOPOS = ["iid", "debate", "chain", "hub_spoke", "tree"]


def _norm_pred(s: str) -> str:
    return (s or "").strip().lower()


def _load_cache(path: Path) -> dict[tuple[str, str], bool]:
    cache: dict[tuple[str, str], bool] = {}
    if not path.exists():
        return cache
    for line in path.open():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        cache[(r["qid"], r["pred_norm"])] = bool(r["correct"])
    return cache


def _append_cache(fp, qid: str, pred_norm: str, correct: bool, method: str) -> None:
    fp.write(json.dumps({
        "qid": qid, "pred_norm": pred_norm, "correct": correct, "method": method,
    }) + "\n")
    fp.flush()


def _gold_index(bench: str) -> dict[str, tuple[str, str, dict]]:
    problems = LOADERS[bench](n=500, seed=0)
    return {p.qid: (p.question, p.answer, p.meta) for p in problems}


def grade_cell(
    topo: str,
    bench: str,
    root: Path,
    gold: dict[str, tuple[str, str, dict]],
    cache: dict[tuple[str, str], bool],
    cache_fp,
    workers: int,
    dry: bool,
) -> dict:
    cell_dir = root / topo / bench
    input_fn = cell_dir / "rollouts.dedup.jsonl"
    label_fn = cell_dir / "labels.jsonl"

    if not input_fn.exists():
        return {"status": "missing_input"}

    rows = [json.loads(l) for l in input_fn.open()]
    if label_fn.exists():
        existing = sum(1 for _ in label_fn.open())
        if existing == len(rows):
            return {"status": "skipped", "n": len(rows)}

    # 1. First pass: rule-based + cache lookup
    decisions: list[dict] = [None] * len(rows)  # type: ignore
    unique_judge_inputs: dict[tuple[str, str], tuple[str, str]] = {}  # key → (question, gold)
    judge_indices: dict[tuple[str, str], list[int]] = {}
    n_rule = n_cache = 0
    for i, r in enumerate(rows):
        qid = r["qid"]
        if qid not in gold:
            decisions[i] = {"correct": False, "method": "missing_gold"}
            continue
        q, g, meta = gold[qid]
        pred = r.get("answer", "") or ""
        pred_n = _norm_pred(pred)
        key = (qid, pred_n)
        if key in cache:
            decisions[i] = {"correct": cache[key], "method": "cache"}
            n_cache += 1
            continue
        rule = score_one(bench, pred, g, meta)
        if rule is not None:
            cache[key] = rule
            _append_cache(cache_fp, qid, pred_n, rule, "rule")
            decisions[i] = {"correct": rule, "method": "rule"}
            n_rule += 1
            continue
        # Need LLM judge
        unique_judge_inputs[key] = (q, g)
        judge_indices.setdefault(key, []).append(i)

    n_judge_unique = len(unique_judge_inputs)
    n_judge_rows = sum(len(v) for v in judge_indices.values())

    if dry:
        return {
            "status": "dry", "n": len(rows),
            "rule": n_rule, "cache": n_cache,
            "judge_unique": n_judge_unique, "judge_rows": n_judge_rows,
        }

    # 2. Second pass: parallel LLM judge calls on unique (qid, pred) keys only
    print(
        f"  [grade {topo}/{bench}] {len(rows)} rows: cache={n_cache} rule={n_rule} "
        f"judge_rows={n_judge_rows} judge_unique={n_judge_unique}"
    )
    if n_judge_unique > 0:
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            future_to_key = {
                ex.submit(judge_one, q, g, key[1]): key
                for key, (q, g) in unique_judge_inputs.items()
            }
            done = 0
            for fut in as_completed(future_to_key):
                key = future_to_key[fut]
                try:
                    correct = bool(fut.result())
                except Exception as e:
                    print(f"    judge error on {key}: {type(e).__name__}: {e}", file=sys.stderr)
                    correct = False
                cache[key] = correct
                _append_cache(cache_fp, key[0], key[1], correct, "judge")
                for idx in judge_indices[key]:
                    decisions[idx] = {"correct": correct, "method": "judge"}
                done += 1
                if done % 200 == 0:
                    rate = done / (time.time() - t0 + 1e-9)
                    print(f"    judge {done}/{n_judge_unique} ({rate:.1f}/s)")
        dt = time.time() - t0
        print(f"  [grade {topo}/{bench}] {n_judge_unique} judge calls in {dt:.1f}s")

    # 3. Write labels.jsonl
    tmp = label_fn.with_suffix(".jsonl.tmp")
    with tmp.open("w") as fout:
        for r, d in zip(rows, decisions):
            fout.write(json.dumps({
                "qid": r["qid"],
                "rollout": r["rollout"],
                "agent_id": r["agent_id"],
                "correct": d["correct"],
                "method": d["method"],
            }) + "\n")
    tmp.replace(label_fn)

    n_correct = sum(1 for d in decisions if d["correct"])
    return {
        "status": "ok", "n": len(rows),
        "rule": n_rule, "cache": n_cache,
        "judge_unique": n_judge_unique, "judge_rows": n_judge_rows,
        "accuracy": n_correct / len(rows),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("results/panels"))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--dry", action="store_true", help="don't call judge, just count")
    p.add_argument("--cells", nargs="*", default=None,
                   help="optional list of <topo>/<bench> to grade (default: all 25)")
    args = p.parse_args()

    # Force judge provider / model unless user already set them
    os.environ.setdefault("JUDGE_PROVIDER", "openrouter")
    os.environ.setdefault("JUDGE_MODEL", "openai/gpt-4o-mini")
    print(f"[grade] JUDGE_PROVIDER={os.environ['JUDGE_PROVIDER']} "
          f"JUDGE_MODEL={os.environ['JUDGE_MODEL']}")

    cache_path = args.root / ".judge_cache.jsonl"
    cache = _load_cache(cache_path)
    print(f"[grade] loaded {len(cache)} cache entries from {cache_path}")
    cache_fp = open(cache_path, "a")

    # Pre-load benchmarks (each only once)
    print("[grade] loading benchmarks...")
    gold_by_bench = {b: _gold_index(b) for b in LOADERS}

    targets = []
    if args.cells:
        for c in args.cells:
            topo, bench = c.split("/")
            targets.append((topo, bench))
    else:
        for t in TOPOS:
            for b in LOADERS:
                targets.append((t, b))

    summary = []
    for topo, bench in targets:
        res = grade_cell(
            topo, bench, args.root, gold_by_bench[bench],
            cache, cache_fp, args.workers, args.dry,
        )
        summary.append((topo, bench, res))
        if res.get("status") == "skipped":
            print(f"  [skip] {topo}/{bench} (labels.jsonl already complete)")

    cache_fp.close()
    print("\n=== summary ===")
    print(f"{'cell':<24} {'n':>5} {'cache':>5} {'rule':>5} {'judge':>5} {'acc%':>5}")
    tot_judge = 0
    for topo, bench, res in summary:
        if res.get("status") not in ("ok", "skipped", "dry"):
            print(f"{topo}/{bench:<14}  status={res.get('status')}")
            continue
        cell = f"{topo}/{bench}"
        acc = res.get("accuracy")
        acc_s = f"{100*acc:.1f}" if acc is not None else "--"
        print(f"{cell:<24} {res.get('n', '?'):>5} {res.get('cache', 0):>5} "
              f"{res.get('rule', 0):>5} {res.get('judge_unique', 0):>5} {acc_s:>5}")
        tot_judge += res.get("judge_unique", 0)
    print(f"\nTotal judge API calls this run: {tot_judge}")
    print(f"Estimated cost @ GPT-4o-mini ($0.15/M in + $0.60/M out, ~300+50 tok/call): "
          f"~${0.0001 * tot_judge:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
