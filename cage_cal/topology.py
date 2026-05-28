from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .cache import RolloutCache
from .inference.vllm_client import chat
from .roles import agent_id, role_prompt


# ----- answer extraction ----------------------------------------------------

_ANSWER_RE = re.compile(r"answer\s*[:：]\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def extract_answer(text: str) -> str:
    """Pick the last 'Answer: ...' line. Fall back to last non-empty line."""
    matches = _ANSWER_RE.findall(text or "")
    if matches:
        return matches[-1].strip().strip(".\"' ")
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    return lines[-1] if lines else ""


# ----- single-agent forward -------------------------------------------------


def _build_messages(role: str, query: str, context: str | None, peer_block: str | None) -> list[dict]:
    sys = role_prompt(role)
    user_parts = []
    if context:
        user_parts.append(f"Context:\n{context}")
    user_parts.append(f"Question:\n{query}")
    if peer_block:
        user_parts.append(
            "Other agents have already proposed the following answers. "
            "Consider them, but think for yourself; you may agree or disagree.\n"
            f"{peer_block}"
        )
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


def _agent_call(
    backbone: str,
    role: str,
    query: str,
    context: str | None,
    peer_block: str | None,
    *,
    rollout_idx: int,
    topology: str,
    cache: RolloutCache | None,
    return_logprobs: bool,
    temperature: float,
    max_tokens: int = 256,
) -> dict[str, Any]:
    key = {
        "backbone": backbone,
        "role": role,
        "topology": topology,
        "rollout_idx": rollout_idx,
        "query": query,
        "context": context,
        "peer_block": peer_block,
        "temperature": temperature,
        "return_logprobs": return_logprobs,
        "max_tokens": max_tokens,
    }
    if cache is not None:
        hit = cache.get(key)
        if hit is not None:
            return hit["value"]
    messages = _build_messages(role, query, context, peer_block)
    resp = chat(
        backbone,
        messages,
        return_logprobs=return_logprobs,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    resp["answer"] = extract_answer(resp["text"])
    resp["agent_id"] = agent_id(backbone, role)
    if cache is not None:
        cache.put(key, resp)
    return resp


# ----- topologies -----------------------------------------------------------


def run_iid(
    agents: list[tuple[str, str]],   # list of (backbone, role)
    query: str,
    context: str | None,
    *,
    rollout_idx: int,
    cache: RolloutCache | None = None,
    return_logprobs: bool = False,
    temperature: float = 0.7,
    max_tokens: int = 256,
    max_workers: int = 20,
) -> dict[str, dict[str, Any]]:
    """Independent calls, no communication. Reference for W_iid."""

    def _go(item: tuple[str, str]) -> tuple[str, dict[str, Any]]:
        bb, role = item
        out = _agent_call(
            bb, role, query, context, None,
            rollout_idx=rollout_idx, topology="iid",
            cache=cache, return_logprobs=return_logprobs, temperature=temperature, max_tokens=max_tokens,
        )
        return out["agent_id"], out

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = dict(ex.map(_go, agents))
    return results


def run_debate(
    agents: list[tuple[str, str]],
    query: str,
    context: str | None,
    *,
    rollout_idx: int,
    cache: RolloutCache | None = None,
    return_logprobs: bool = False,
    temperature: float = 0.7,
    max_tokens: int = 256,
    max_workers: int = 20,
) -> dict[str, dict[str, Any]]:
    """2-round debate: round 1 = independent, round 2 = see all peers' round-1 answers."""
    # Round 1: same as iid
    r1 = run_iid(
        agents, query, context,
        rollout_idx=rollout_idx, cache=cache,
        return_logprobs=False, temperature=temperature, max_tokens=max_tokens,
        max_workers=max_workers,
    )
    # Build peer block from round-1 answers
    peer_lines = [
        f"- Agent {aid.split('::')[1]}({aid.split('::')[0]}): {r['answer']}"
        for aid, r in r1.items()
    ]
    peer_block = "\n".join(peer_lines)

    def _go(item: tuple[str, str]) -> tuple[str, dict[str, Any]]:
        bb, role = item
        out = _agent_call(
            bb, role, query, context, peer_block,
            rollout_idx=rollout_idx, topology="debate-r2",
            cache=cache, return_logprobs=return_logprobs, temperature=temperature, max_tokens=max_tokens,
        )
        return out["agent_id"], out

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        r2 = dict(ex.map(_go, agents))
    return r2


def _peer_lines(results: dict[str, dict[str, Any]], label: str | None = None) -> str:
    """Format peer answers as a multi-line block injected into prompts.

    Tolerant to agent_id format: prefers the resp's own ``agent_id`` field
    (resolved as ``"<backbone>::<role>"`` or with extra suffix); falls back to
    the dict key if the resp doesn't carry one.
    """
    lines = []
    for key, r in results.items():
        aid = r.get("agent_id", key) if isinstance(r, dict) else key
        # try parse as backbone::role[::suffix]
        parts = aid.split("::")
        if len(parts) >= 2:
            bb, role = parts[0], parts[1]
            tag = f"{role}({bb})"
        else:
            tag = aid
        prefix = f"{label} " if label else ""
        lines.append(f"- {prefix}{tag}: {r['answer']}")
    return "\n".join(lines)


def run_chain(
    agents: list[tuple[str, str]],
    query: str,
    context: str | None,
    *,
    rollout_idx: int,
    cache: RolloutCache | None = None,
    return_logprobs: bool = False,
    temperature: float = 0.7,
    max_tokens: int = 256,
    max_workers: int = 20,
) -> dict[str, dict[str, Any]]:
    """Linear pipeline: a_1 (no peer) → a_2 (sees a_1) → ... → a_K (sees a_1..K-1).

    All agents return their own answer; final MAS answer = last agent's answer
    (downstream aggregator can also use intermediate answers).

    Order = the order of `agents` list (deterministic given caller's order).
    """
    results: dict[str, dict[str, Any]] = {}
    accumulated: dict[str, dict[str, Any]] = {}
    for bb, role in agents:
        peer = _peer_lines(accumulated) if accumulated else None
        out = _agent_call(
            bb, role, query, context, peer,
            rollout_idx=rollout_idx, topology="chain",
            cache=cache, return_logprobs=return_logprobs, temperature=temperature, max_tokens=max_tokens,
        )
        results[out["agent_id"]] = out
        accumulated[out["agent_id"]] = out
    return results


def run_hub_spoke(
    agents: list[tuple[str, str]],
    query: str,
    context: str | None,
    *,
    rollout_idx: int,
    cache: RolloutCache | None = None,
    return_logprobs: bool = False,
    temperature: float = 0.7,
    max_tokens: int = 256,
    max_workers: int = 20,
) -> dict[str, dict[str, Any]]:
    """Hub-and-spoke: K-1 workers answer independently in parallel, then 1 hub
    sees all worker answers and emits the final MAS answer.

    Convention: the LAST agent in `agents` is the hub; the rest are spokes.
    Caller controls hub identity by ordering. Returns ALL agent outputs (including hub).
    """
    if len(agents) < 2:
        raise ValueError("hub_spoke needs >=2 agents (>=1 spoke + 1 hub)")
    spokes = agents[:-1]
    hub_bb, hub_role = agents[-1]

    # Round 1: spokes parallel (independent)
    def _spoke(item: tuple[str, str]) -> tuple[str, dict[str, Any]]:
        bb, role = item
        out = _agent_call(
            bb, role, query, context, None,
            rollout_idx=rollout_idx, topology="hub-spoke-r1",
            cache=cache, return_logprobs=return_logprobs, temperature=temperature, max_tokens=max_tokens,
        )
        return out["agent_id"], out

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        spoke_results = dict(ex.map(_spoke, spokes))

    # Round 2: hub sees all spoke answers
    peer = _peer_lines(spoke_results, label="worker")
    hub_out = _agent_call(
        hub_bb, hub_role, query, context, peer,
        rollout_idx=rollout_idx, topology="hub-spoke-r2",
        cache=cache, return_logprobs=return_logprobs, temperature=temperature, max_tokens=max_tokens,
    )
    spoke_results[hub_out["agent_id"]] = hub_out
    return spoke_results


def run_tree(
    agents: list[tuple[str, str]],
    query: str,
    context: str | None,
    *,
    rollout_idx: int,
    cache: RolloutCache | None = None,
    return_logprobs: bool = False,
    temperature: float = 0.7,
    max_tokens: int = 256,
    max_workers: int = 20,
    branching: int = 2,
) -> dict[str, dict[str, Any]]:
    """Hierarchical aggregation tree.

    Layer 0 (leaves): N agents answer independently.
    Layer 1+:         each inner node sees `branching` children's answers and emits its own.
    Convention: ``agents`` lists ALL nodes in BFS order from leaves up.
    Tree shape with branching=2:  N must be of the form 2^L + 2^(L-1) + ... + 1 (e.g. 8+4+2+1=15).

    The MAS final answer = the root node's answer (last agent in BFS). All nodes' answers
    are returned for downstream graph construction.

    Each node is identified by ``"<backbone>::<role>::tree<i>"`` to keep agent_id unique
    across layers (otherwise two HGT layers using same (backbone, role) would collide).
    """
    if len(agents) < branching + 1:
        raise ValueError(f"tree needs >= {branching + 1} agents")

    # Decompose layered structure: greedily group bottom-up.
    # Simplest: binary tree, total N must satisfy:
    #   N = sum_{l=0..L} branching^l for some L≥1.
    # E.g. branching=2: 3 (1+2), 7 (1+2+4), 15 (1+2+4+8), 31 (...).
    # We accept any N where layered structure can fit and lay out leaves first.
    # For paper default K=15 (8 leaf, 4, 2, 1).

    layers: list[list[int]] = []  # each entry = list of agent indices in that layer
    n = len(agents)
    cursor = 0
    layer_size = 0
    # Determine layer 0 size (leaves) by reverse: sum of 1 + b + b^2 + ... + b^L = n
    L = 0
    s = 1
    while s + branching ** (L + 1) <= n:
        L += 1
        s += branching ** L
    if s != n:
        # If exact tree shape doesn't divide, fall back: leaves = remaining after building inner layers
        # But we keep clean: enforce paper-default shape.
        raise ValueError(
            f"tree of branching={branching} requires N in {{3, 7, 15, 31, ...}}, got {n}"
        )

    # Build layers bottom-up: leaves first.
    # Layer 0 = leaves = branching^L agents
    # Layer 1 = branching^(L-1) agents
    # ...
    # Layer L = 1 agent (root)
    sizes = [branching ** (L - l) for l in range(L + 1)]
    cursor = 0
    for sz in sizes:
        layers.append(list(range(cursor, cursor + sz)))
        cursor += sz

    results: dict[str, dict[str, Any]] = {}
    layer_outputs: list[dict[int, dict[str, Any]]] = []  # per-layer agent_idx -> resp

    # Layer 0: leaves run independently
    leaf_indices = layers[0]

    def _leaf(idx: int) -> tuple[int, dict[str, Any]]:
        bb, role = agents[idx]
        out = _agent_call(
            bb, role, query, context, None,
            rollout_idx=rollout_idx, topology=f"tree-leaf-{idx}",
            cache=cache, return_logprobs=return_logprobs, temperature=temperature, max_tokens=max_tokens,
        )
        # Make leaf's id unique (in case same (bb,role) reused upstream)
        out["agent_id"] = f"{out['agent_id']}::tree{idx}"
        return idx, out

    leaf_outputs: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, r in ex.map(_leaf, leaf_indices):
            leaf_outputs[i] = r
            results[r["agent_id"]] = r
    layer_outputs.append(leaf_outputs)

    # Layers 1..L: each inner node sees `branching` children sequentially below it
    for layer_idx in range(1, L + 1):
        cur_layer = layers[layer_idx]
        prev_outputs = layer_outputs[layer_idx - 1]
        prev_indices = layers[layer_idx - 1]

        # Each inner node i takes children prev_indices[branching*j : branching*(j+1)]
        # where j is its position in cur_layer
        cur_outputs: dict[int, dict[str, Any]] = {}

        def _inner(item: tuple[int, int]) -> tuple[int, dict[str, Any]]:
            j, idx = item
            children_global_idx = prev_indices[branching * j : branching * (j + 1)]
            children_resp = {f"child_{c}": prev_outputs[c] for c in children_global_idx}
            peer = _peer_lines(children_resp, label="child")
            bb, role = agents[idx]
            out = _agent_call(
                bb, role, query, context, peer,
                rollout_idx=rollout_idx, topology=f"tree-L{layer_idx}-{idx}",
                cache=cache, return_logprobs=return_logprobs, temperature=temperature, max_tokens=max_tokens,
            )
            out["agent_id"] = f"{out['agent_id']}::tree{idx}"
            return idx, out

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for j, idx in enumerate(cur_layer):
                i, r = _inner((j, idx))
                cur_outputs[i] = r
                results[r["agent_id"]] = r

        layer_outputs.append(cur_outputs)

    return results


TOPOLOGIES = {
    "iid":       run_iid,
    "debate":    run_debate,
    "chain":     run_chain,
    "hub_spoke": run_hub_spoke,
    "tree":      run_tree,
}


def run_topology(name: str, *args, **kwargs) -> dict[str, dict[str, Any]]:
    if name not in TOPOLOGIES:
        raise KeyError(f"unknown topology {name!r}; choose from {list(TOPOLOGIES)}")
    return TOPOLOGIES[name](*args, **kwargs)
