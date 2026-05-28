from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Graceful exit on SIGTERM/SIGINT: finish the in-flight rollout, write its
# JSONL row, then exit. A re-run resumes from the cache + already-written rows.
_should_exit = False


def _on_sigterm(signum, frame):
    global _should_exit
    if not _should_exit:
        print(f"\n[rollout] signal {signum} received; will exit after the "
              f"current rollout completes and resume from cache on re-run.",
              flush=True)
    _should_exit = True


signal.signal(signal.SIGTERM, _on_sigterm)
signal.signal(signal.SIGINT, _on_sigterm)

from cage_cal.agent_population import (
    ALL_AGENTS,
    BACKBONES,
    HUB_AGENT,
    ROLES,
    SPOKE_AGENTS,
    TOPOLOGY_AGENTS,
    TREE_AGENTS,
    chain_order,
    export_manifest,
)
from cage_cal.benchmarks._common import Problem
from cage_cal.benchmarks.bbh import load_bbh
from cage_cal.benchmarks.gsm8k import load_gsm8k
from cage_cal.benchmarks.mmlu_pro import load_mmlu_pro
from cage_cal.benchmarks.trivia_qa import load_trivia_qa
from cage_cal.benchmarks.truthful_qa import load_truthful_qa
from cage_cal.cache import RolloutCache
from cage_cal.topology import run_topology

# Line-buffered stdout/stderr for live log streaming.
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


BENCHMARKS = {
    "trivia_qa":   load_trivia_qa,
    "truthful_qa": load_truthful_qa,
    "mmlu_pro":    load_mmlu_pro,
    "gsm8k":       load_gsm8k,
    "bbh":         load_bbh,
}

# Per-benchmark max_tokens cap. Answers are short for all of these; the cap
# stops LLMs from emitting long-tail CoT that we never read (graph node
# features only consume {answer, mean_logprob, length}). Critical for chain
# wall-clock since output token count dominates per-call latency.
BENCHMARK_MAX_TOKENS = {
    "trivia_qa":   64,    # short entity (e.g. "Lincoln")
    "truthful_qa": 128,   # 1-2 sentences
    "mmlu_pro":    128,   # single letter + brief reasoning
    "gsm8k":       256,   # CoT + numeric answer
    "bbh":         256,   # varied free-form
}

TOPOLOGIES_DEFAULT = ["iid", "debate", "chain", "hub_spoke", "tree"]
BENCHMARKS_DEFAULT = list(BENCHMARKS)


def agents_for(topology: str, qid: str, rollout_idx: int) -> list[tuple[str, str]]:
    """Return the per-rollout agent ordering required by the topology.

    Most topologies use the canonical TOPOLOGY_AGENTS list as-is. Chain is
    the exception: order matters, so we shuffle deterministically per
    (qid, rollout_idx) to average out the tail-agent confound across problems.
    """
    base = TOPOLOGY_AGENTS[topology]
    if topology == "chain":
        return chain_order(base, qid, rollout_idx)
    return base


def already_done(jsonl_path: Path, qid: str, ridx: int) -> bool:
    """Cheap resumability: if a (qid, rollout) row already exists in the
    cell's output file, skip. We only need to test for ANY agent row from
    that (qid, ridx) — `run_topology` is atomic per-cell because the
    RolloutCache is what makes the inner agent calls idempotent.
    """
    if not jsonl_path.exists():
        return False
    needle = f'"qid": "{qid}", "rollout": {ridx}'
    # Cheap line scan; cells are small (~25K rows). Could be replaced with
    # a sidecar set if this ever shows up in a profile.
    with jsonl_path.open() as f:
        for line in f:
            if needle in line:
                return True
    return False


def append_rollout(
    jsonl_path: Path,
    topology: str,
    benchmark: str,
    qid: str,
    ridx: int,
    rollout_resp: dict,
) -> None:
    with jsonl_path.open("a") as f:
        for aid, r in rollout_resp.items():
            row = {
                "topology":  topology,
                "benchmark": benchmark,
                "qid":       qid,
                "rollout":   ridx,
                "agent_id":  aid,
                "answer":    r.get("answer", ""),
                "text":      r.get("text", ""),
                "model":     r.get("model", ""),
                "backbone":  r.get("backbone", ""),
            }
            # mean_logprob is computed by vLLM for local backbones only
            # (frontier API agents return None). Persist when present so the
            # GNN node feature in graph_build_uq.py is non-zero.
            if r.get("logprobs"):
                lps = [tok.get("logprob") for tok in r["logprobs"]
                       if tok.get("logprob") is not None]
                if lps:
                    row["mean_logprob"] = sum(lps) / len(lps)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_cell(
    *,
    topology: str,
    benchmark: str,
    problems: list[Problem],
    n_rollouts: int,
    cache: RolloutCache,
    out_root: Path,
    return_logprobs: bool,
    temperature: float,
    log_every: int,
) -> tuple[int, int]:
    """Returns (n_done, n_skipped) for the cell."""
    cell_dir = out_root / topology / benchmark
    cell_dir.mkdir(parents=True, exist_ok=True)
    jsonl = cell_dir / "rollouts.jsonl"
    max_tokens = BENCHMARK_MAX_TOKENS.get(benchmark, 256)

    n_done = 0
    n_skip = 0
    n_total = len(problems) * n_rollouts
    t0 = time.time()
    print(f"[r0] cell start: topology={topology} benchmark={benchmark} "
          f"n_qid={len(problems)} n_rollouts={n_rollouts} max_tokens={max_tokens} "
          f"→ {n_total} rollouts")

    for q_i, prob in enumerate(problems):
        for ridx in range(n_rollouts):
            if _should_exit:
                print(f"[r0]   {topology}/{benchmark}: graceful exit at "
                      f"qid={prob.qid} ridx={ridx} (SIGTERM); "
                      f"done={n_done} skipped={n_skip}", flush=True)
                return n_done, n_skip
            if already_done(jsonl, prob.qid, ridx):
                n_skip += 1
                continue
            agents = agents_for(topology, prob.qid, ridx)
            try:
                resp = run_topology(
                    topology, agents, prob.question, prob.context,
                    rollout_idx=ridx, cache=cache,
                    return_logprobs=return_logprobs, temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception as e:
                print(f"[r0] FAIL topology={topology} bench={benchmark} "
                      f"qid={prob.qid} ridx={ridx}: {type(e).__name__}: {e}",
                      file=sys.stderr)
                continue
            append_rollout(jsonl, topology, benchmark, prob.qid, ridx, resp)
            n_done += 1
            done_idx = q_i * n_rollouts + ridx + 1
            if done_idx % log_every == 0:
                rate = (n_done) / max(time.time() - t0, 1e-6)
                eta = (n_total - n_skip - n_done) / max(rate, 1e-6)
                print(f"[r0]   {topology}/{benchmark}: {done_idx}/{n_total} "
                      f"(skipped {n_skip}, rate {rate:.1f}/s, eta {eta/60:.1f} min)")

    print(f"[r0] cell done:  topology={topology} benchmark={benchmark} "
          f"done={n_done} skipped={n_skip} elapsed={(time.time()-t0)/60:.1f} min")
    return n_done, n_skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topologies", nargs="+", default=TOPOLOGIES_DEFAULT,
                    choices=TOPOLOGIES_DEFAULT)
    ap.add_argument("--benchmarks", nargs="+", default=BENCHMARKS_DEFAULT,
                    choices=BENCHMARKS_DEFAULT)
    ap.add_argument("--n-per-bench", type=int, default=500,
                    help="problems per benchmark (60/20/20 split applied downstream)")
    ap.add_argument("--n-rollouts", type=int, default=3,
                    help="rollouts per problem; pilot v3 showed CI non-overlapping at N=3")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--return-logprobs", action="store_true", default=True,
                    help="request logprobs from local vLLM (needed for graph features)")
    ap.add_argument("--out-dir", type=Path, default=Path("results/panels"))
    ap.add_argument("--cache-dir", type=Path, default=Path("cache/panels"))
    ap.add_argument("--log-every", type=int, default=25,
                    help="how often (in rollouts) to print progress per cell")
    ap.add_argument("--seed", type=int, default=0,
                    help="benchmark sampling seed (kept consistent across all cells)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    # Early-exit if a previous run already wrote the completion sentinel, so
    # a re-run is a fast no-op instead of re-scanning every cell.
    sentinel_path = args.out_dir / ".completed"
    if sentinel_path.exists():
        print(f"[r0] sentinel already exists at {sentinel_path}.", flush=True)
        print(f"[rollout] already completed by a prior run; nothing to do.",
              flush=True)
        sys.exit(0)

    print(f"[r0] config: topologies={args.topologies} benchmarks={args.benchmarks} "
          f"n_per_bench={args.n_per_bench} n_rollouts={args.n_rollouts}")
    print(f"[r0] agent population: {len(ALL_AGENTS)} agents = "
          f"{len(BACKBONES)} backbones × {len(ROLES)} roles; "
          f"hub={HUB_AGENT}; tree subset={len(TREE_AGENTS)}")

    # 1. Load all benchmarks once (deterministic seed=0)
    bench_problems: dict[str, list[Problem]] = {}
    for b in args.benchmarks:
        loader = BENCHMARKS[b]
        # all loaders accept (n, seed); some also accept split (use default)
        try:
            probs = loader(n=args.n_per_bench, seed=args.seed)
        except TypeError:
            probs = loader(n=args.n_per_bench)
        print(f"[r0] loaded {b}: {len(probs)} problems")
        bench_problems[b] = probs

    # 2. Manifest snapshot — pin agent population + config for reproducibility
    manifest_path = args.out_dir / "manifest.json"
    pop_manifest = export_manifest(args.out_dir / "agent_population.json")
    json.dump({
        "config": vars(args) | {"out_dir": str(args.out_dir),
                                "cache_dir": str(args.cache_dir)},
        "agent_population": pop_manifest,
        "benchmark_sizes": {b: len(p) for b, p in bench_problems.items()},
    }, manifest_path.open("w"), indent=2, default=str)
    print(f"[r0] wrote {manifest_path}")

    # 3. Iterate cells
    cache = RolloutCache(args.cache_dir)
    grand_done = grand_skip = 0
    t_start = time.time()
    for topology in args.topologies:
        if _should_exit:
            break
        for benchmark in args.benchmarks:
            if _should_exit:
                break
            d, s = run_cell(
                topology=topology,
                benchmark=benchmark,
                problems=bench_problems[benchmark],
                n_rollouts=args.n_rollouts,
                cache=cache,
                out_root=args.out_dir,
                return_logprobs=args.return_logprobs,
                temperature=args.temperature,
                log_every=args.log_every,
            )
            grand_done += d
            grand_skip += s

    elapsed = (time.time() - t_start) / 3600

    # Compute the total work expected across this invocation's cell list.
    expected_total = sum(
        len(bench_problems[b]) * args.n_rollouts
        for _ in args.topologies for b in args.benchmarks
    )
    completed_total = grand_done + grand_skip

    if _should_exit:
        print(f"[r0] EXIT (SIGTERM) — done={grand_done} skipped={grand_skip} "
              f"of {expected_total}; elapsed={elapsed:.2f} h. "
              f"Resubmit to continue.", flush=True)
        sys.exit(0)

    if completed_total == expected_total:
        sentinel = args.out_dir / ".completed"
        sentinel.write_text(json.dumps({
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "topologies": args.topologies,
            "benchmarks": args.benchmarks,
            "n_per_bench": args.n_per_bench,
            "n_rollouts": args.n_rollouts,
            "grand_done": grand_done,
            "grand_skip": grand_skip,
            "elapsed_hours": elapsed,
        }, indent=2))
        print(f"[r0] ALL CELLS DONE — total done={grand_done} skipped={grand_skip} "
              f"elapsed={elapsed:.2f} h. Wrote sentinel: {sentinel}", flush=True)
    else:
        print(f"[r0] PARTIAL — done={grand_done} skipped={grand_skip} "
              f"of {expected_total}; elapsed={elapsed:.2f} h. "
              f"Resubmit to continue.", flush=True)


if __name__ == "__main__":
    main()
