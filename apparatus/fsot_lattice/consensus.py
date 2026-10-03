"""Trit collapse, consensus aggregate, phase rotation.

Collapse threshold is the live C_eff·P_var. No softmax and no exponential.
"""

from __future__ import annotations

import math

from fsot_lattice.engine import COHERENCE_GATE, COLLAPSE


def collapse_code(value: float, threshold: float = COLLAPSE) -> int:
    """Map a continuous unit to {0 spin-down, 1 superposed, 2 spin-up}."""
    if value > threshold:
        return 2
    if value < -threshold:
        return 0
    return 1


def code_signed(code: int) -> int:
    return -1 if code == 0 else (0 if code == 1 else 1)


def trit_similarity(a: list[float], b: list[float], threshold: float = COLLAPSE) -> float:
    """Mean consensus. Match +1, opposite -1, either superposed 0."""
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    acc = 0
    for i in range(n):
        ta = collapse_code(a[i], threshold)
        tb = collapse_code(b[i], threshold)
        if ta == 1 or tb == 1:
            continue
        acc += 1 if ta == tb else -1
    return acc / n


def phase_rotate(h: list[float], position: int) -> list[float]:
    """Pair rotation. theta = 2 * position, matching the GPU consensus operator."""
    theta = 2.0 * position
    cs = math.cos(theta)
    sn = math.sin(theta)
    out = list(h)
    pairs = len(out) // 2
    for k in range(pairs):
        a = out[2 * k]
        b = out[2 * k + 1]
        out[2 * k] = cs * a - sn * b
        out[2 * k + 1] = sn * a + cs * b
    return out


def position_coherence(h: list[float], threshold: float = COLLAPSE) -> float:
    if not h:
        return 0.0
    hot = sum(1 for v in h if abs(v) > threshold)
    return hot / len(h)


def consensus_states(tokens: list[list[float]], threshold: float = COLLAPSE) -> list[list[float]]:
    """Causal consensus at every position.

    A key contributes only when its coherence clears COHERENCE_GATE and it
    is not in the future. Weights are trit similarities, divided by the
    active count. There is no softmax denominator.
    """
    seq = len(tokens)
    if seq == 0:
        return []
    dim = len(tokens[0])
    states: list[list[float]] = []
    for i in range(seq):
        num = [0.0] * dim
        active = 0
        for j in range(i + 1):
            if position_coherence(tokens[j], threshold) <= COHERENCE_GATE:
                continue
            w = trit_similarity(tokens[i], tokens[j], threshold)
            if w == 0.0:
                continue
            active += 1
            v = tokens[j]
            for d in range(dim):
                num[d] += w * v[d]
        if active == 0:
            states.append(list(tokens[i]))
        else:
            states.append([n / active for n in num])
    return states


def consensus_aggregate(tokens: list[list[float]], threshold: float = COLLAPSE) -> list[float]:
    """Last-position consensus state."""
    states = consensus_states(tokens, threshold)
    return states[-1] if states else []


def consensus_attend(
    query: list[float],
    keys: list[list[float]],
    threshold: float = COLLAPSE,
) -> tuple[list[float], int]:
    """Attend a query over value keys. The query itself is not a value.

    A key contributes when its coherence clears COHERENCE_GATE and its trit
    similarity with the query is not zero. The returned state is that active
    mean. An empty pass returns a zero state and an active count of zero.
    """
    dim = len(query)
    num = [0.0] * dim
    active = 0
    for key in keys:
        if len(key) != dim:
            raise ValueError("consensus key width does not match the query")
        if position_coherence(key, threshold) <= COHERENCE_GATE:
            continue
        weight = trit_similarity(query, key, threshold)
        if weight == 0.0:
            continue
        active += 1
        for unit in range(dim):
            num[unit] += weight * key[unit]
    if active == 0:
        return [0.0] * dim, 0
    return [value / active for value in num], active
