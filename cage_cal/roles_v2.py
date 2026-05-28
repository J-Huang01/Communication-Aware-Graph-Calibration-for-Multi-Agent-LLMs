from __future__ import annotations


_HEADER_FORMAT = (
    'IMPORTANT: Your response must end with exactly one line: '
    '"Answer: <your short answer>". '
    'Keep the Answer line concise (no qualifiers, no extra clauses). '
    'If you start running out of space, output "Answer: <X>" immediately, '
    'even before finishing your reasoning.\n\n'
)

_FOOTER_FORMAT = (
    ' Output the final answer at the end as exactly one line: '
    '"Answer: <your short answer>"'
)


ROLES_V2: dict[str, str] = {
    # zero-shot baseline (no reasoning shown).
    "direct": (
        _HEADER_FORMAT +
        "Skip reasoning. Output only the Answer line."
        + _FOOTER_FORMAT
    ),
    # Wei et al. 2022 NeurIPS — Chain-of-Thought.
    # Reasoning cap is generous enough to actually do CoT but tight enough
    # to prevent runaway generation that crowds out the Answer line.
    "cot": (
        _HEADER_FORMAT +
        "Think step by step in 2-4 short steps (≤60 words total of reasoning). "
        "Then output the Answer line."
        + _FOOTER_FORMAT
    ),
    # Wang et al. ACL 2023 — Plan-and-Solve.
    "plan_solve": (
        _HEADER_FORMAT +
        "State a 2-step plan, then carry it out. Reasoning ≤60 words total. "
        "Then output the Answer line."
        + _FOOTER_FORMAT
    ),
    # Zheng et al. ICLR 2024 — Step-Back Prompting.
    "step_back": (
        _HEADER_FORMAT +
        "State the relevant principle in one sentence (≤25 words), "
        "then use it to solve the problem. Then output the Answer line."
        + _FOOTER_FORMAT
    ),
    # Yasunaga et al. ICLR 2024 — Analogical Prompting.
    "analogical": (
        _HEADER_FORMAT +
        "Recall 1-2 analogous problems (≤25 words each, format: 'Q: ... -> ...'), "
        "then use what you learned to solve the problem. Then output the Answer line."
        + _FOOTER_FORMAT
    ),
}


def role_prompt_v2(role: str) -> str:
    if role not in ROLES_V2:
        raise KeyError(f"unknown role {role!r}; choose from {list(ROLES_V2)}")
    return ROLES_V2[role]
