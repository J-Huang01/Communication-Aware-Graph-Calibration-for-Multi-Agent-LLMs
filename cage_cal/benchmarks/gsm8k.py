from __future__ import annotations

import re

from datasets import load_dataset

from ._common import Problem


_FINAL_RE = re.compile(r"####\s*(-?\d[\d,]*(?:\.\d+)?)")


def _extract_gold(answer_field: str) -> str:
    m = _FINAL_RE.search(answer_field)
    if not m:
        return answer_field.strip()
    return m.group(1).replace(",", "")


def load_gsm8k(n: int = 500, split: str = "test", seed: int = 0) -> list[Problem]:
    ds = load_dataset("gsm8k", "main", split=split)
    ds = ds.shuffle(seed=seed)
    out: list[Problem] = []
    for i, ex in enumerate(ds.select(range(min(n, len(ds))))):
        gold = _extract_gold(str(ex["answer"]))
        out.append(
            Problem(
                qid=f"gsm8k_{i}",
                question=str(ex["question"]),
                answer=gold,
                context="",
                meta={"full_solution": str(ex["answer"]), "task": "gsm8k"},
            )
        )
    return out
