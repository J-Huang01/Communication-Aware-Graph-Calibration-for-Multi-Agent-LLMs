from __future__ import annotations

# All roles return a single short answer at the end. We standardise the answer
# format so downstream parsing + judge are uniform across roles/benchmarks.

_FORMAT_INSTRUCTION = (
    'Output the final answer at the end as exactly one line: "Answer: <your short answer>"'
)


ROLES: dict[str, str] = {
    # zero-shot baseline (no reasoning shown)
    "direct": (
        "Answer the question directly. Do not show reasoning. " + _FORMAT_INSTRUCTION
    ),
    # Wei et al. 2022 NeurIPS — Chain-of-Thought
    "cot": (
        "Let's think step by step. Reason through the problem, then commit to a final answer. "
        + _FORMAT_INSTRUCTION
    ),
    # Wang et al. ACL 2023 — Plan-and-Solve
    "plan_solve": (
        "First, understand the problem and devise a brief plan in 2–4 steps. "
        "Then carry out the plan to solve the problem. "
        + _FORMAT_INSTRUCTION
    ),
    # Zheng et al. ICLR 2024 — Step-Back Prompting
    "step_back": (
        "Take a step back. State the high-level concept, principle, or category that this "
        "problem falls under. Then use that principle to solve the specific problem. "
        + _FORMAT_INSTRUCTION
    ),
    # Yasunaga et al. ICLR 2024 — Analogical Prompting
    "analogical": (
        "Recall 2–3 analogous problems you have seen before. Briefly describe each in one "
        "sentence. Then use what you learned from those analogies to solve this problem. "
        + _FORMAT_INSTRUCTION
    ),
}


def role_prompt(role: str) -> str:
    if role not in ROLES:
        raise KeyError(f"unknown role {role!r}; choose from {list(ROLES)}")
    return ROLES[role]


def agent_id(backbone: str, role: str) -> str:
    return f"{backbone}::{role}"
