from __future__ import annotations

import re
from typing import Any


# --- normalization helpers --------------------------------------------------


def _norm(s: str) -> str:
    return (s or "").strip().lower().strip(".\"' \t\n,;:!?")


def _strip_articles(s: str) -> str:
    return re.sub(r"\b(a|an|the)\b", "", s).strip()


def _basic_match(pred: str, gold: str) -> bool:
    a = _strip_articles(_norm(pred))
    b = _strip_articles(_norm(gold))
    if not a or not b:
        return False
    return a == b or a in b or b in a


# --- mmlu_pro ---------------------------------------------------------------

_MMLU_LETTER_RE = re.compile(
    r"\b([A-J])\b", re.IGNORECASE
)


def score_mmlu_pro(pred: str, gold: str, meta: dict[str, Any]) -> bool | None:
    """Extract a single A-J letter from pred, compare to gold letter.

    Gold is always a single letter (A-J). Pure rule-based, no judge.
    """
    if not pred:
        return False
    p = pred.strip()
    # Common patterns: "Answer: C", "C", "C.", "(C)", "The answer is C"
    # Take the LAST single A-J letter in the string (most robust).
    matches = _MMLU_LETTER_RE.findall(p)
    if not matches:
        return False
    return matches[-1].upper() == gold.strip().upper()


# --- gsm8k ------------------------------------------------------------------

_GSM_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _norm_num(s: str) -> str:
    s = (s or "").strip().replace(",", "").replace("$", "").rstrip(".")
    if not s:
        return ""
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
        # Strip trailing zeros for stable comparison
        return f"{f:g}"
    except ValueError:
        return s


def score_gsm8k(pred: str, gold: str, meta: dict[str, Any]) -> bool | None:
    """Extract last numeric token, compare normalized form. Pure rule-based."""
    if not pred:
        return False
    nums = _GSM_NUM_RE.findall(pred)
    if not nums:
        return False
    pred_num = _norm_num(nums[-1])
    gold_num = _norm_num(gold)
    return pred_num == gold_num


# --- trivia_qa --------------------------------------------------------------


def score_trivia_qa(pred: str, gold: str, meta: dict[str, Any]) -> bool | None:
    """Match pred against gold + aliases (case-insensitive, articles-stripped).

    Returns None to escalate to LLM judge if no direct match found (paraphrase).
    """
    if not pred:
        return False
    candidates = [gold] + list(meta.get("aliases", []))
    pn = _strip_articles(_norm(pred))
    for cand in candidates:
        cn = _strip_articles(_norm(str(cand)))
        if not cn:
            continue
        if pn == cn or (len(cn) >= 3 and cn in pn):
            return True
    # No alias match — escalate
    return None


# --- truthful_qa ------------------------------------------------------------


def score_truthful_qa(pred: str, gold: str, meta: dict[str, Any]) -> bool | None:
    """truthful_qa gold answers are sentences; rule-based exact match almost
    never works. Try `correct_answers` list for trivial cases, otherwise
    escalate to LLM judge.

    Also flag as FALSE (cheaply) if pred matches one of `incorrect_answers`.
    """
    if not pred:
        return False
    pn = _norm(pred)
    # Quick exact-form match against any correct alternative
    correct = [gold] + list(meta.get("correct_answers", []))
    for cand in correct:
        cn = _norm(str(cand))
        if cn and cn == pn:
            return True
    # Quick exact-form match against incorrect → cheap False
    for cand in meta.get("incorrect_answers", []) or []:
        cn = _norm(str(cand))
        if cn and cn == pn:
            return False
    # Otherwise escalate
    return None


# --- bbh --------------------------------------------------------------------

_BBH_RULE_SUBTASKS = {
    # Subtasks whose target is yes/no/(A)/(B)/.../number/short token —
    # rule-based scoring is reliable.
    "boolean_expressions",
    "causal_judgement",          # "Yes"/"No"
    "date_understanding",        # MCQA (A)-(F)
    "disambiguation_qa",         # MCQA
    "formal_fallacies",          # "valid"/"invalid"
    "geometric_shapes",          # MCQA
    "hyperbaton",                # MCQA
    "logical_deduction_three_objects",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "movie_recommendation",      # MCQA
    "multistep_arithmetic_two",  # numeric
    "navigate",                  # Yes/No
    "object_counting",           # numeric
    "penguins_in_a_table",       # MCQA
    "reasoning_about_colored_objects",  # MCQA
    "ruin_names",                # MCQA
    "salient_translation_error_detection",  # MCQA
    "snarks",                    # MCQA
    "sports_understanding",      # Yes/No
    "temporal_sequences",        # MCQA
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
}

_BBH_PARENS_RE = re.compile(r"\(([A-Za-z]|\d+)\)")


def score_bbh(pred: str, gold: str, meta: dict[str, Any]) -> bool | None:
    """Per-subtask rule. Returns None to escalate for free-form subtasks
    (none of the canonical 23 are truly free-form; all should rule-resolve).
    """
    if not pred:
        return False
    subtask = meta.get("subtask", "")
    g = gold.strip()
    p = pred.strip()

    # Multiple-choice "(A)" format — most BBH subtasks
    if _BBH_PARENS_RE.match(g) or g.startswith("("):
        # Extract LAST parens-letter from pred
        pred_matches = _BBH_PARENS_RE.findall(p)
        if pred_matches:
            return pred_matches[-1].upper() == g.strip("()").upper()
        # Fallback: look for bare letter A-Z
        letter_matches = re.findall(r"\b([A-J])\b", p)
        if letter_matches:
            return letter_matches[-1].upper() == g.strip("()").upper()
        return False

    # Numeric (object_counting, multistep_arithmetic_two)
    if re.fullmatch(r"-?\d+(?:\.\d+)?", g):
        nums = _GSM_NUM_RE.findall(p)
        if not nums:
            return False
        return _norm_num(nums[-1]) == _norm_num(g)

    # Yes/No / valid/invalid / etc — case-insensitive exact match
    return _basic_match(p, g)


# --- registry ---------------------------------------------------------------

SCORERS = {
    "mmlu_pro":    score_mmlu_pro,
    "gsm8k":       score_gsm8k,
    "trivia_qa":   score_trivia_qa,
    "truthful_qa": score_truthful_qa,
    "bbh":         score_bbh,
}


def score_one(benchmark: str, pred: str, gold: str, meta: dict[str, Any]) -> bool | None:
    """Dispatch to the right scorer. Returns True/False if decided, else None."""
    scorer = SCORERS.get(benchmark)
    if scorer is None:
        raise KeyError(f"no scorer for benchmark {benchmark!r}")
    return scorer(pred, gold, meta)
