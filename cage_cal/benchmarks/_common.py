from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Problem:
    qid: str
    question: str
    answer: str            # gold answer (free-form short string, MCQA letter, numeric, code, etc.)
    context: str = ""      # optional context; "" for context-free benchmarks
    meta: dict = field(default_factory=dict)  # extra fields (e.g. multi-gold list, subtask id, options)


def deterministic_split(
    problems: list[Problem],
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    seed: int = 0,
) -> tuple[list[Problem], list[Problem], list[Problem]]:
    """Shuffle + 60/20/20 split. test_frac = 1 - train_frac - val_frac."""
    import random

    items = list(problems)
    random.Random(seed).shuffle(items)
    n = len(items)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = items[:n_train]
    val = items[n_train : n_train + n_val]
    test = items[n_train + n_val :]
    return train, val, test
