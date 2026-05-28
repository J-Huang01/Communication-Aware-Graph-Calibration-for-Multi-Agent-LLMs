from __future__ import annotations

import random

from datasets import load_dataset

from ._common import Problem


# 23 official BBH subtasks (per Suzgun et al. 2022)
BBH_SUBTASKS: list[str] = [
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "dyck_languages",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    # only 23 in canonical BBH; we have 24 here, drop one
]
BBH_SUBTASKS = BBH_SUBTASKS[:23]


def load_bbh(n: int = 500, seed: int = 0) -> list[Problem]:
    rng = random.Random(seed)
    per_task = max(1, n // len(BBH_SUBTASKS))
    out: list[Problem] = []
    for task in BBH_SUBTASKS:
        try:
            ds = load_dataset("lukaemon/bbh", task, split="test")
        except Exception:
            # subtask 名字 mapping 偶尔有微调；跳过失败的
            continue
        items = list(ds)
        rng.shuffle(items)
        for i, ex in enumerate(items[:per_task]):
            out.append(
                Problem(
                    qid=f"bbh_{task}_{i}",
                    question=str(ex["input"]),
                    answer=str(ex["target"]).strip(),
                    context="",
                    meta={"subtask": task, "task": "bbh"},
                )
            )
    rng.shuffle(out)
    return out[:n]
