from __future__ import annotations

from datasets import load_dataset

from ._common import Problem


def load_trivia_qa(n: int = 500, split: str = "validation", seed: int = 0) -> list[Problem]:
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split=split, trust_remote_code=True)
    ds = ds.shuffle(seed=seed)
    out: list[Problem] = []
    for ex in ds.select(range(min(n, len(ds)))):
        ans_obj = ex["answer"]
        gold = str(ans_obj["value"])
        aliases = list(ans_obj.get("aliases", []))
        out.append(
            Problem(
                qid=str(ex["question_id"]),
                question=str(ex["question"]),
                answer=gold,
                context="",
                meta={"aliases": aliases, "task": "trivia_qa"},
            )
        )
    return out
