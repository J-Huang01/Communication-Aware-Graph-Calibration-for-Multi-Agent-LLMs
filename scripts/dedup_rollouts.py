from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def dedup_cell(in_path: Path, out_path: Path) -> dict:
    seen: set[tuple] = set()
    kept = dropped = 0
    with in_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            r = json.loads(line)
            key = (r["qid"], r["rollout"], r["agent_id"])
            if key in seen:
                dropped += 1
                continue
            seen.add(key)
            fout.write(line)
            kept += 1
    return {"in": kept + dropped, "kept": kept, "dropped": dropped}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path, help="results dir, e.g. results/panels")
    p.add_argument(
        "--suffix",
        default="dedup.jsonl",
        help="output sibling filename suffix (default: dedup.jsonl)",
    )
    args = p.parse_args()

    if not args.root.exists():
        print(f"[dedup] {args.root} not found", file=sys.stderr)
        return 1

    print(f"{'cell':<28} {'in':>6} {'kept':>6} {'dropped':>7}")
    print("-" * 55)
    total_in = total_kept = total_dropped = 0
    cells = sorted(args.root.glob("*/*/rollouts.jsonl"))
    for in_path in cells:
        out_path = in_path.with_suffix("." + args.suffix)
        stats = dedup_cell(in_path, out_path)
        cell = f"{in_path.parent.parent.name}/{in_path.parent.name}"
        marker = "  DUP" if stats["dropped"] > 0 else ""
        print(
            f"{cell:<28} {stats['in']:>6} {stats['kept']:>6} {stats['dropped']:>7}{marker}"
        )
        total_in += stats["in"]
        total_kept += stats["kept"]
        total_dropped += stats["dropped"]

    print("-" * 55)
    print(
        f"{'TOTAL':<28} {total_in:>6} {total_kept:>6} {total_dropped:>7}  "
        f"({100*total_dropped/total_in:.2f}% removed)"
    )
    print(f"\n[dedup] wrote *.{args.suffix} alongside originals; originals untouched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
