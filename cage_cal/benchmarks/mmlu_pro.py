from __future__ import annotations

import random

from datasets import load_dataset

from ._common import Problem


def _format_question(q: str, options: list[str]) -> str:
    """Append A/B/C/... options to question text in stable format."""
    letters = "ABCDEFGHIJ"
    lines = [q.strip(), ""]
    for i, opt in enumerate(options):
        if i >= len(letters):
            break
        lines.append(f"{letters[i]}. {opt}")
    return "\n".join(lines)


def load_mmlu_pro(n: int = 500, seed: int = 0) -> list[Problem]:
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")

    # group by category
    by_cat: dict[str, list[dict]] = {}
    for ex in ds:
        by_cat.setdefault(str(ex["category"]), []).append(ex)

    n_cat = len(by_cat)
    per_cat = max(1, n // n_cat)
    rng = random.Random(seed)

    out: list[Problem] = []
    for cat in sorted(by_cat.keys()):
        items = list(by_cat[cat])
        rng.shuffle(items)
        for ex in items[:per_cat]:
            q_text = _format_question(str(ex["question"]), [str(o) for o in ex["options"]])
            out.append(
                Problem(
                    qid=f"mmlupro_{ex['question_id']}",
                    question=q_text,
                    answer=str(ex["answer"]),     # already a letter A-J
                    context="",
                    meta={
                        "category": cat,
                        "options": [str(o) for o in ex["options"]],
                        "answer_index": int(ex["answer_index"]),
                        "task": "mmlu_pro",
                    },
                )
            )
    # Trim or pad: we may overshoot/undershoot n by a few due to per_cat rounding
    rng.shuffle(out)
    return out[:n]
