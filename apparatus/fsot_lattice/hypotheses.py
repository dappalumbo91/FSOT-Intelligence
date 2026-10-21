"""A new table proposes a shape the lattice already holds, then the fluid step tests it.

A fit is added beside the held shapes. A miss leaves those shapes where they were.
The open item is the shape that matched nothing. No question is written in advance.
"""

from __future__ import annotations

from dataclasses import dataclass

from fsot_lattice.layouts import (
    LayoutScore,
    _quantity,
    _shared,
    home_reading,
    settle_layout,
)


@dataclass(frozen=True)
class HeldShape:
    source: str
    kind: str


@dataclass(frozen=True)
class Reading:
    kind: str
    names: int
    homes: int
    gap: float
    span: float


@dataclass(frozen=True)
class Trial:
    source: str
    proposal: str
    fitted: bool
    reason: str
    joined: tuple[str, ...]
    score: LayoutScore | None = None


def _date(text: str) -> bool:
    parts = text.split("-")
    return len(parts) == 3 and len(parts[0]) == 4 and all(part.isdigit() for part in parts)


def _gap_span(homes: list[str]) -> tuple[float, float]:
    values = sorted({float(home) for home in homes})
    if len(values) < 2:
        return 0.0, 0.0
    gap = min(right - left for left, right in zip(values, values[1:]))
    return gap, values[-1] - values[0]


def read_table(settled: dict[str, str]) -> Reading:
    """The shape of a name-to-home table, before any fluid step."""
    homes = sorted(set(settled.values()))
    reading = home_reading(homes)
    gap, span = _gap_span(homes) if reading in {"quantity", "close"} else (0.0, 0.0)
    shared = _shared(settled) and len(homes) >= 2 and len(settled) >= 8
    if settled and all(_date(name) for name in settled):
        kind = "series"
    elif reading == "quantity" and shared:
        kind = "shared-quantity"
    elif reading == "quantity":
        kind = "unique-numbers"
    elif reading == "close" and shared:
        kind = "close-numbers"
    elif reading == "label" and shared:
        kind = "shared-label"
    else:
        kind = "unshared"
    return Reading(kind, len(settled), len(homes), gap, span)


def read_sequence(sample: str) -> Reading:
    """A FASTA header and a run of bases. The catalog shape has no column to hold it."""
    bases = "".join(line.strip() for line in sample.splitlines()[1:])
    alphabet = "".join(sorted(set(bases)))
    return Reading("sequence", 1, len(alphabet), 0.0, float(len(bases)))


def _matches(held: list[HeldShape], kind: str) -> tuple[str, ...]:
    return tuple(item.source for item in held if item.kind == kind)


def sign_magnitude(homes: list[str]) -> bool:
    """True when every numeric home has its negation among the homes."""
    values: list[float] = []
    for home in homes:
        value = _quantity(home)
        if value is None:
            return False
        values.append(value)
    have = set(values)
    return bool(have) and all(-value in have for value in have)


def magnitude_note(homes: list[str], held: list[HeldShape]) -> str:
    """A sign pair is one magnitude. A held label keeps the assignment."""
    if not sign_magnitude(homes):
        return "the homes are not sign pairs"
    labels = _matches(held, "shared-label")
    where = ", ".join(labels) if labels else "no held label"
    return f"the sign pair is one magnitude; the assignment stays on {where}"


def consider(
    source: str,
    settled: dict[str, str],
    held: list[HeldShape],
    *,
    chew: bool = True,
) -> Trial:
    """Propose the nearest held shape and test it. A miss does not edit `held`."""
    reading = read_table(settled)
    if reading.kind == "shared-quantity":
        return _test_known(source, settled, held, "shared-quantity", chew)
    if reading.kind == "shared-label":
        return _test_known(source, settled, held, "shared-label", chew)
    if reading.kind == "close-numbers":
        return _test_close(source, settled, held, reading, chew)
    joined = _matches(held, reading.kind)
    proposal = (
        f"same shape as {', '.join(joined)}"
        if joined
        else f"no held shape is a {reading.kind}"
    )
    detail = f"{reading.names} names, {reading.homes} homes"
    if reading.gap:
        detail = f"{detail}, gap {reading.gap:g}, span {reading.span:g}"
    return Trial(source, proposal, False, f"the proposal stays open, {detail}", ())


def consider_sequence(source: str, sample: str, held: list[HeldShape]) -> Trial:
    reading = read_sequence(sample)
    joined = _matches(held, "sequence")
    proposal = f"same shape as {', '.join(joined)}" if joined else "no held shape is a sequence"
    return Trial(
        source,
        proposal,
        False,
        f"the proposal stays open, alphabet {int(reading.homes)} over {int(reading.span)} bases",
        joined,
    )


def _test_known(
    source: str,
    settled: dict[str, str],
    held: list[HeldShape],
    kind: str,
    chew: bool,
) -> Trial:
    joined = _matches(held, kind)
    proposal = f"same shape as {', '.join(joined)}" if joined else f"first {kind}"
    if kind == "shared-quantity" and sign_magnitude(sorted(set(settled.values()))):
        note = magnitude_note(sorted(set(settled.values())), held)
        return Trial(source, note, False, "a sign pair stays one magnitude", joined)
    if not chew:
        return Trial(source, proposal, False, "held for a later chew", joined)
    score = settle_layout(settled)
    if score.chewable:
        held.append(HeldShape(source, kind))
        return Trial(source, proposal, True, score.reason, joined, score)
    return Trial(source, proposal, False, score.reason, joined, score)


def _test_close(
    source: str,
    settled: dict[str, str],
    held: list[HeldShape],
    reading: Reading,
    chew: bool,
) -> Trial:
    """The count shape misses. The label shape is the next proposal, beside it."""
    counts = _matches(held, "shared-quantity")
    labels = _matches(held, "shared-label")
    proposal = (
        f"count shape of {', '.join(counts)} misses at gap {reading.gap:g}; "
        f"label shape of {', '.join(labels)}"
    )
    if not chew:
        return Trial(source, proposal, False, "held for a later chew", ())
    score = settle_layout(settled)
    if score.chewable:
        held.append(HeldShape(source, "shared-label"))
        return Trial(source, proposal, True, score.reason, labels, score)
    return Trial(source, proposal, False, score.reason, (), score)
