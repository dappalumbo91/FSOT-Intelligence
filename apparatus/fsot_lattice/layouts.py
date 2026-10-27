"""A table layout is a name column and a home column.

The relation word is a prior. It has no gauge. The read is the consensus
remainder of the name gauge and the home gauge. A name with more than one
home is withheld. A layout is chewable when the homes compress the names
and the fluid step lands inside the floor.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from fsot_lattice.apply import DROP, GAUGE_FLOOR
from fsot_lattice.engine import fluid_lr
from fsot_lattice.lattice import GAUGE_GAIN, K, Lattice

RELATION = "is"


@dataclass(frozen=True)
class LayoutScore:
    name_column: str
    home_column: str
    names: int
    homes: int
    conflicted: tuple[str, ...]
    chewable: bool
    reason: str
    epochs: int
    train_acc: float
    hold_acc: float
    true_max: float
    false_min: float
    residual: float
    name_gauge: tuple[tuple[str, float], ...] = ()
    home_gauge: tuple[tuple[str, float], ...] = ()


def _clean(value: object) -> str:
    return " ".join(str(value).split())


def one_home(rows: list[dict[str, object]], name_column: str, home_column: str) -> tuple[dict[str, str], tuple[str, ...]]:
    """Names with a single home. A name with two homes is withheld."""
    found: dict[str, set[str]] = {}
    for row in rows:
        name = _clean(row.get(name_column, ""))
        home = _clean(row.get(home_column, ""))
        if not name or not home or home.lower() == "none":
            continue
        found.setdefault(name, set()).add(home)
    conflicted = tuple(sorted(name for name, homes in found.items() if len(homes) > 1))
    settled = {name: next(iter(homes)) for name, homes in found.items() if len(homes) == 1}
    return settled, conflicted


def _shared(settled: dict[str, str]) -> bool:
    """True when at least two names sit on one home."""
    counts: dict[str, int] = {}
    for home in settled.values():
        counts[home] = counts.get(home, 0) + 1
    return any(count >= 2 for count in counts.values())


def _quantity(text: str) -> float | None:
    """A home written as a number. A label such as chr17 stays a label."""
    body = text[1:] if text.startswith("-") else text
    if not body or body.count(".") > 1 or not body.replace(".", "", 1).isdigit():
        return None
    value = float(text)
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return value


def quantity_homes(homes: list[str]) -> dict[str, float] | None:
    """Place a numeric home at K times its value when neighbors sit at least K apart."""
    values: list[float] = []
    for home in homes:
        value = _quantity(home)
        if value is None:
            return None
        values.append(value)
    ordered = sorted(set(values))
    if len(ordered) >= 2 and min(right - left for left, right in zip(ordered, ordered[1:])) < 1:
        return None
    return {home: K * value for home, value in zip(homes, values)}


def home_reading(homes: list[str]) -> str:
    """quantity when neighbors sit at least one apart, close when they crowd, else label."""
    if quantity_homes(homes) is not None:
        return "quantity"
    if homes and all(_quantity(home) is not None for home in homes):
        return "close"
    return "label"


def settle_layout(
    settled: dict[str, str],
    epochs: int = 4000,
    observer=None,
    name_column: str = "",
    home_column: str = "",
) -> LayoutScore:
    """Pull each name onto its home with the fluid step. The home gauges stay put."""
    names = sorted(settled)
    homes = sorted(set(settled.values()))
    quantities = quantity_homes(homes)
    if quantities is None:
        home_gauge = {home: K * (index + 1) for index, home in enumerate(homes)}
        placed = "label"
    else:
        home_gauge = quantities
        placed = "quantity"
    name_gauge = {name: K for name in names}
    name_index = {name: index for index, name in enumerate(names)}
    home_index = {home: index for index, home in enumerate(homes)}
    pairs = [(name, home) for name in names for home in homes]
    hold = {
        pair for pair in pairs
        if (name_index[pair[0]] * 3 + home_index[pair[1]] * 4) % 5 == 0
    }
    train = [pair for pair in pairs if pair not in hold]
    lattice = Lattice.fresh()
    sign = lattice.routed_sign("-")
    if RELATION in lattice.acquired_value or RELATION in name_gauge:
        raise RuntimeError("the relation word took a gauge")

    def remainder(name: str, home: str) -> float:
        return lattice.consensus_quantity(name_gauge[name], home_gauge[home], sign)

    def score(split: list[tuple[str, str]]) -> tuple[float, float, float, int]:
        true_abs: list[float] = []
        false_abs: list[float] = []
        hits = 0
        for name, home in split:
            gap = remainder(name, home)
            want = settled[name] == home
            got = lattice.remainder_answer(gap) == "yes"
            gap = abs(gap)
            if want:
                true_abs.append(gap)
            else:
                false_abs.append(gap)
            if got == want:
                hits += 1
        return hits / len(split), max(true_abs), min(false_abs), len(split) - hits

    hits = 0.0
    used = 0
    residual = 1.0
    train_acc = 0.0
    hold_acc = 0.0
    if observer is not None:
        _, _, _, train_wrong = score(train)
        _, _, _, hold_wrong = score(list(hold))
        observer(0, train_wrong, hold_wrong, residual)
    for epoch in range(epochs):
        grad = {name: 0.0 for name in names}
        align = 0.0
        for name in names:
            err = home_gauge[settled[name]] - name_gauge[name]
            grad[name] += err
            align += err * err
        align_mse = align / len(names)
        step = fluid_lr(align_mse, hits) * GAUGE_GAIN
        for name in names:
            name_gauge[name] += step * grad[name] / len(names)
        residual = max(abs(name_gauge[name] - home_gauge[settled[name]]) for name in names)
        train_acc, _, _, train_wrong = score(train)
        hold_acc, _, _, hold_wrong = score(list(hold))
        hits = train_acc * len(train)
        used = epoch + 1
        if observer is not None:
            observer(used, train_wrong, hold_wrong, residual)
        if residual < GAUGE_FLOOR and train_acc == 1.0 and hold_acc == 1.0:
            break
    _, true_max, false_min, _ = score(pairs)
    landed = (
        train_acc == 1.0
        and hold_acc == 1.0
        and true_max < GAUGE_FLOOR
        and false_min > DROP
    )
    reason = f"landed on the {placed}" if landed else f"the step missed the floor on the {placed}"
    if placed == "quantity":
        readback = max(abs(name_gauge[name] / K - float(settled[name])) for name in names)
        reason = f"{reason}, readback {readback:.6e}"
    return LayoutScore(
        name_column=name_column,
        home_column=home_column,
        names=len(names),
        homes=len(homes),
        conflicted=(),
        chewable=landed,
        reason=reason,
        epochs=used,
        train_acc=train_acc,
        hold_acc=hold_acc,
        true_max=true_max,
        false_min=false_min,
        residual=residual,
        name_gauge=tuple(sorted(name_gauge.items())),
        home_gauge=tuple(sorted(home_gauge.items())),
    )


def save_layout(path: Path, score: LayoutScore, pin: str, source: str) -> None:
    """Write a landed layout. A miss is not kept."""
    if not score.chewable:
        raise RuntimeError("a missed layout is not kept")
    payload = {
        "pin": pin,
        "source": source,
        "name_column": score.name_column,
        "home_column": score.home_column,
        "names": score.names,
        "homes": score.homes,
        "epochs": score.epochs,
        "true_max": score.true_max,
        "false_min": score.false_min,
        "residual": score.residual,
        "name_gauge": {name: gauge for name, gauge in score.name_gauge},
        "home_gauge": {home: gauge for home, gauge in score.home_gauge},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_layout(path: Path) -> LayoutScore:
    """Read a kept layout back into gauges the asker can use."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    names = tuple(sorted((key, float(value)) for key, value in payload["name_gauge"].items()))
    homes = tuple(sorted((key, float(value)) for key, value in payload["home_gauge"].items()))
    return LayoutScore(
        name_column=payload["name_column"],
        home_column=payload["home_column"],
        names=len(names),
        homes=len(homes),
        conflicted=(),
        chewable=True,
        reason="kept",
        epochs=int(payload["epochs"]),
        train_acc=1.0,
        hold_acc=1.0,
        true_max=float(payload["true_max"]),
        false_min=float(payload["false_min"]),
        residual=float(payload["residual"]),
        name_gauge=names,
        home_gauge=homes,
    )


def organize_table(rows: list[dict[str, object]], columns: list[str] | None = None) -> list[LayoutScore]:
    """Try each ordered pair of columns. Keep the layouts the step can chew."""
    if columns is None:
        columns = list(rows[0].keys()) if rows else []
    found: list[LayoutScore] = []
    for name_column in columns:
        for home_column in columns:
            if name_column == home_column:
                continue
            settled, conflicted = one_home(rows, name_column, home_column)
            homes = len(set(settled.values()))
            if len(settled) < 8 or homes < 2 or not _shared(settled):
                found.append(
                    LayoutScore(
                        name_column=name_column,
                        home_column=home_column,
                        names=len(settled),
                        homes=homes,
                        conflicted=conflicted,
                        chewable=False,
                        reason="the column does not compress to a shared home",
                        epochs=0,
                        train_acc=0.0,
                        hold_acc=0.0,
                        true_max=0.0,
                        false_min=0.0,
                        residual=0.0,
                    )
                )
                continue
            score = settle_layout(settled)
            found.append(
                LayoutScore(
                    name_column=name_column,
                    home_column=home_column,
                    names=score.names,
                    homes=score.homes,
                    conflicted=conflicted,
                    chewable=score.chewable,
                    reason=score.reason,
                    epochs=score.epochs,
                    train_acc=score.train_acc,
                    hold_acc=score.hold_acc,
                    true_max=score.true_max,
                    false_min=score.false_min,
                    residual=score.residual,
                    name_gauge=score.name_gauge,
                    home_gauge=score.home_gauge,
                )
            )
    return found
