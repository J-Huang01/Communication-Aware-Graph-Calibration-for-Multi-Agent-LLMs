from __future__ import annotations

from datasets import load_dataset

from ._common import Problem


def load_truthful_qa(n: int = 500, split: str = "validation", seed: int = 0) -> list[Problem]:
    ds = load_dataset("truthful_qa", "generation", split=split, trust_remote_code=True)
    ds = ds.shuffle(seed=seed)
    out: list[Problem] = []
    for i, ex in enumerate(ds.select(range(min(n, len(ds))))):
        correct = list(ex.get("correct_answers", []))
        incorrect = list(ex.get("incorrect_answers", []))
        # Use the canonical 'best_answer' as primary gold; full list in meta
        gold = str(ex.get("best_answer") or (correct[0] if correct else ""))
        out.append(
            Problem(
                qid=f"tqa_{i}",
                question=str(ex["question"]),
                answer=gold,
                context="",
                meta={
                    "correct_answers": correct,
                    "incorrect_answers": incorrect,
                    "category": str(ex.get("category", "")),
                    "task": "truthful_qa",
                },
            )
        )
    return out
