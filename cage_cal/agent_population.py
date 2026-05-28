from __future__ import annotations

import hashlib
import itertools
import json
import os
import random
from pathlib import Path
from typing import Iterable

# Backbones, ordered for reproducibility.
BACKBONES: list[str] = [
    "qwen3-8b",
    "llama-3.1-8b",
    "gemma-3-12b",
    "phi-4",
]

# Atomic prompting roles, ordered for reproducibility.
ROLES: list[str] = [
    "direct",
    "cot",
    "plan_solve",
    "step_back",
    "analogical",
]

# Canonical population: 4 backbone × 5 role = 20 agents, deterministic order
ALL_AGENTS: list[tuple[str, str]] = list(itertools.product(BACKBONES, ROLES))
assert len(ALL_AGENTS) == 20, f"expected 20 agents, got {len(ALL_AGENTS)}"

# Tree subset: deterministic 15-of-20, binary tree shape 8+4+2+1
_TREE_SAMPLE_SEED = 0
TREE_AGENTS: list[tuple[str, str]] = sorted(
    random.Random(_TREE_SAMPLE_SEED).sample(ALL_AGENTS, 15),
    key=lambda x: (x[0], x[1]),
)

# Chain subset: N=5, one agent per role, backbones assigned by round-robin.
# Chain is sequential so its cost scales O(N); a 5-agent chain keeps balanced
# role coverage with cross-family backbone representation.
CHAIN_AGENTS: list[tuple[str, str]] = [
    (BACKBONES[i % len(BACKBONES)], role)
    for i, role in enumerate(ROLES)
]
assert len(CHAIN_AGENTS) == 5

_CHAIN_N10_SAMPLE_SEED = 0
CHAIN_AGENTS_N10: list[tuple[str, str]] = sorted(
    random.Random(_CHAIN_N10_SAMPLE_SEED).sample(ALL_AGENTS, 10),
    key=lambda x: (x[0], x[1]),
)

# Hub fixed at a middle-strength backbone with a neutral role; using the
# strongest model as hub would confound topology effect with hub quality.
HUB_AGENT: tuple[str, str] = ("llama-3.1-8b", "cot")
assert HUB_AGENT in ALL_AGENTS, f"HUB_AGENT {HUB_AGENT} not in ALL_AGENTS"
SPOKE_AGENTS: list[tuple[str, str]] = [a for a in ALL_AGENTS if a != HUB_AGENT]
assert len(SPOKE_AGENTS) == 19


# ---- Matched-N=10 design --------------------------------------------------
# To isolate the topology effect from panel size N, the matched design uses
# N=10 across topologies (chain stays at 5 for cost; tree at 7 for binary
# depth-2). iid/debate/hub_spoke share the SAME 10 role-stratified agents
# (HUB_AGENT forced in) so the comparison is over an identical population.
_MATCHED_N10_SEED = 1


def _role_stratified_sample(seed: int, per_role: int = 2) -> list[tuple[str, str]]:
    """For each role, sample `per_role` backbones. HUB_AGENT forced in for HUB's role.
    Total = len(ROLES) * per_role agents (e.g. 5 * 2 = 10).
    """
    rng = random.Random(seed)
    out: list[tuple[str, str]] = []
    for role in ROLES:
        candidates = [(bb, role) for bb in BACKBONES]
        if role == HUB_AGENT[1]:
            others = [a for a in candidates if a != HUB_AGENT]
            sampled = rng.sample(others, per_role - 1) + [HUB_AGENT]
        else:
            sampled = rng.sample(candidates, per_role)
        out.extend(sampled)
    return sorted(out, key=lambda x: (x[0], x[1]))


MATCHED_AGENTS_N10: list[tuple[str, str]] = _role_stratified_sample(
    _MATCHED_N10_SEED, per_role=2
)
assert HUB_AGENT in MATCHED_AGENTS_N10 and len(MATCHED_AGENTS_N10) == 10
# Hub_spoke uses the same 10 but with HUB last (run_hub_spoke convention).
HUB_SPOKE_AGENTS_N10: list[tuple[str, str]] = (
    [a for a in MATCHED_AGENTS_N10 if a != HUB_AGENT] + [HUB_AGENT]
)
assert len(HUB_SPOKE_AGENTS_N10) == 10 and HUB_SPOKE_AGENTS_N10[-1] == HUB_AGENT

# Tree at N=7 (binary depth 2: 4 leaves + 2 mid + 1 root), independent
# seed-0 sample; the binary tree structure is what matters, not identity.
_TREE_N7_SEED = 0
TREE_AGENTS_N7: list[tuple[str, str]] = sorted(
    random.Random(_TREE_N7_SEED).sample(ALL_AGENTS, 7),
    key=lambda x: (x[0], x[1]),
)


# Per-topology agent membership, selected at import time via env var
# MAS_UQ_AGENT_CONFIG:
#   "matched": N=10 across topologies (tree at N=7).
#   "natural": N=20 across topologies (tree at N=15).

_MATCHED_N10_TOPOLOGY_AGENTS: dict[str, list[tuple[str, str]]] = {
    "iid":       MATCHED_AGENTS_N10,                  # 10, parallel independent
    "debate":    MATCHED_AGENTS_N10,                  # 10, full mesh, 2 rounds
    "chain":     MATCHED_AGENTS_N10,                  # 10, sequential
    "hub_spoke": HUB_SPOKE_AGENTS_N10,                # 10 = 9 spokes + 1 hub
    "tree":      TREE_AGENTS_N7,                      # 7, binary tree (4+2+1)
}

TOPOLOGY_AGENTS_NATURAL_N: dict[str, list[tuple[str, str]]] = {
    "iid":       ALL_AGENTS,                          # 20
    "debate":    ALL_AGENTS,                          # 20
    "chain":     ALL_AGENTS,                          # 20
    "hub_spoke": SPOKE_AGENTS + [HUB_AGENT],          # 20 = 19 + 1
    "tree":      TREE_AGENTS,                         # 15 (8+4+2+1)
}

_config = os.environ.get("MAS_UQ_AGENT_CONFIG", "matched").lower()
if _config == "natural":
    TOPOLOGY_AGENTS = TOPOLOGY_AGENTS_NATURAL_N
elif _config == "matched":
    TOPOLOGY_AGENTS = _MATCHED_N10_TOPOLOGY_AGENTS
else:
    raise ValueError(
        f"MAS_UQ_AGENT_CONFIG={_config!r} not in {{'matched', 'natural'}}"
    )


def chain_order(
    agents: list[tuple[str, str]],
    qid: str,
    rollout_idx: int,
) -> list[tuple[str, str]]:
    """Deterministic but per-rollout shuffled order for chain topology.

    Avoids the confound where the tail agent (which sees all peer answers)
    becomes a fixed model, which would make the chain effect approximate a
    tail-agent effect.
    """
    seed_int = int(hashlib.sha256(f"{qid}::{rollout_idx}".encode()).hexdigest()[:8], 16)
    rng = random.Random(seed_int)
    return rng.sample(agents, len(agents))


# ---- Hub-identity variants (rotate the hub backbone) ----------------------

HUB_IDENTITY_VARIANTS: dict[str, tuple[str, str]] = {
    f"{bb}_cot_hub": (bb, "cot")
    for bb in BACKBONES
}


def hub_spoke_with_hub(hub: tuple[str, str]) -> list[tuple[str, str]]:
    """Build a hub-spoke agent list with an arbitrary hub."""
    if hub not in ALL_AGENTS:
        raise ValueError(f"hub {hub} not in ALL_AGENTS")
    spokes = [a for a in ALL_AGENTS if a != hub]
    return spokes + [hub]


# ---- Manifest export (reproducibility) ------------------------------------


def export_manifest(out_path: str | Path) -> dict:
    """Dump the canonical population + per-topology membership to JSON.

    Released alongside the paper so external researchers can verify they're
    replaying the exact same agent assignments.
    """
    manifest = {
        "backbones": BACKBONES,
        "roles": ROLES,
        "all_agents": [list(a) for a in ALL_AGENTS],
        "tree_agents_seed": _TREE_SAMPLE_SEED,
        "tree_agents": [list(a) for a in TREE_AGENTS],
        "chain_agents": [list(a) for a in CHAIN_AGENTS],
        "chain_agents_n10_robustness": [list(a) for a in CHAIN_AGENTS_N10],
        "chain_agents_n10_seed": _CHAIN_N10_SAMPLE_SEED,
        "hub_agent": list(HUB_AGENT),
        "spoke_agents": [list(a) for a in SPOKE_AGENTS],
        "topology_agents": {
            t: [list(a) for a in agents] for t, agents in TOPOLOGY_AGENTS.items()
        },
        "hub_identity_variants": {
            name: list(hub) for name, hub in HUB_IDENTITY_VARIANTS.items()
        },
        "chain_order_protocol": (
            "Per-rollout: rng = random.Random(int(sha256(f'{qid}::{rollout_idx}')[:8], 16)); "
            "rng.sample(CHAIN_AGENTS, 5)."
        ),
    }
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest


# ---- Convenience helpers --------------------------------------------------


def agent_id(backbone: str, role: str) -> str:
    return f"{backbone}::{role}"


def all_agent_ids(agents: Iterable[tuple[str, str]]) -> list[str]:
    return [agent_id(b, r) for b, r in agents]


if __name__ == "__main__":
    # Quick sanity printout
    print("=" * 60)
    print(f"BACKBONES ({len(BACKBONES)}): {BACKBONES}")
    print(f"ROLES     ({len(ROLES)}): {ROLES}")
    print(f"ALL_AGENTS: {len(ALL_AGENTS)} agents")
    print(f"TREE_AGENTS: {len(TREE_AGENTS)} agents (deterministic seed={_TREE_SAMPLE_SEED})")
    for a in TREE_AGENTS:
        print(f"  {a}")
    print(f"HUB_AGENT: {HUB_AGENT}")
    print(f"SPOKE_AGENTS: {len(SPOKE_AGENTS)} agents")
    print()
    print("TOPOLOGY_AGENTS:")
    for t, agents in TOPOLOGY_AGENTS.items():
        print(f"  {t:11s}: {len(agents)} agents")
    print()
    print("Chain order example for (qid='abc123', rollout_idx=0):")
    order = chain_order(ALL_AGENTS, "abc123", 0)
    for i, a in enumerate(order):
        print(f"  pos {i:2d}: {a}")
    print()
    print("HUB_IDENTITY_VARIANTS:")
    for name, hub in HUB_IDENTITY_VARIANTS.items():
        print(f"  {name}: hub = {hub}")
