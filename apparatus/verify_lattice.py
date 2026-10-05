"""Verify the promoted lattice and the C attend against the Python attend.

The recorded package in results/lattice_run.json has to still clear the
promotion floors. A fresh lattice then re-runs the mechanism checks. The
compiled attend has to name the same cells as the Python attend.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

APPARATUS = Path(__file__).resolve().parent
PROJECT = APPARATUS.parent
if str(APPARATUS) not in sys.path:
    sys.path.insert(0, str(APPARATUS))

from fsot_lattice import fast_attend
from fsot_lattice.apply import check_generated
from fsot_lattice.engine import require_live_pin
from fsot_lattice.lattice import Lattice

FLOOR = 1e-4
PARITY_GAP = 1e-9
PACKAGE = APPARATUS / "results" / "lattice_run.json"
SURFACES = ("digit", "word")
BUCKETS = ("hold", "closed")


def _fail(issues: list[str]) -> int:
    print(f"lattice verification failed: {len(issues)}")
    for item in issues:
        print(f"  {item}")
    return 1


def _exact(report: dict, key: str, issues: list[str]) -> None:
    bucket = report.get(key)
    if not isinstance(bucket, dict):
        issues.append(f"missing {key}")
        return
    if bucket.get("acc") != 1.0 or bucket.get("misses"):
        issues.append(f"{key} acc {bucket.get('acc')} misses {len(bucket.get('misses') or [])}")
    gap = bucket.get("gap")
    if gap is None or float(gap) >= FLOOR:
        issues.append(f"{key} gap {gap}")
    margin = bucket.get("margin_min")
    drop = float(report.get("drop") or 0.0)
    if margin is None or float(margin) <= drop:
        issues.append(f"{key} margin {margin} drop {drop}")


def check_package(pin: str) -> list[str]:
    issues: list[str] = []
    if not PACKAGE.is_file():
        return [f"missing promoted package {PACKAGE}"]
    report = json.loads(PACKAGE.read_text(encoding="utf-8"))
    if report.get("promoted") is not True:
        issues.append("package is not promoted")
    if not str(report.get("pin", "")).startswith(pin[:12]):
        issues.append(f"package pin {str(report.get('pin', ''))[:12]} is not {pin[:12]}")
    if report.get("width") != 60 or report.get("learned_parameters") != 1940:
        issues.append(
            f"width {report.get('width')} learned {report.get('learned_parameters')}"
        )
    if report.get("digit_hold_after") != 1.0 or report.get("word_hold") != 1.0:
        issues.append("a holdout surface is not exact")
    for key in ("read_gap", "word_read_gap", "digit_gauge_residual"):
        value = report.get(key)
        if value is None or float(value) >= FLOOR:
            issues.append(f"{key} {value}")
    if report.get("digit_drift_after_words") != 0.0 or report.get("engram_reload_drift") != 0.0:
        issues.append("a frozen gauge moved")
    if float(report.get("pathway_confidence_after_words") or 0.0) < 0.5:
        issues.append("pathway confidence fell under 0.5")
    if report.get("kept_bonds") != 1920.0:
        issues.append(f"kept bonds {report.get('kept_bonds')}")
    for key in ("equality_ok", "composition_ok", "apply_ok", "retention_ok"):
        if report.get(key) is not True:
            issues.append(f"{key} is not true")
    applied = report.get("apply") or {}
    if applied.get("ok") is not True:
        issues.append("generated application is not ok")
    word_tail = (report.get("word_history_tail") or [{}])[-1]
    if float(word_tail.get("gauge_residual") or 1.0) >= FLOOR:
        issues.append(f"word gauge residual {word_tail.get('gauge_residual')}")
    if float(word_tail.get("zero_fold") or 1.0) >= FLOOR:
        issues.append(f"word zero fold {word_tail.get('zero_fold')}")
    for surface in SURFACES:
        for bucket in BUCKETS:
            _exact(applied, f"ten_thou_op_{surface}_{bucket}", issues)
    print(
        f"package promoted={report.get('promoted')} pin={str(report.get('pin'))[:12]} "
        f"width={report.get('width')} learned={report.get('learned_parameters')} "
        f"wall={report.get('wall_seconds')}"
    )
    return issues


def _consensus_grid(lattice: Lattice) -> list[tuple[int, float, float, int]]:
    gauges = [lattice.digit_value[str(digit)] for digit in range(10)]
    rows: list[tuple[int, float, float, int]] = []
    for left in gauges:
        for right in gauges:
            for sign in (1, -1):
                rows.append((lattice.consensus_quantity(left, right, sign), left, right, sign))
    return rows


def _place_grid(lattice: Lattice) -> list[int]:
    ten = lattice.ten_quantity("digit")
    hundred = lattice.hundred_quantity("digit")
    thousand = lattice.thousand_quantity("digit")
    ten_thousand = lattice.ten_thousand_quantity("digit")
    gauges = [lattice.digit_value[str(digit)] for digit in range(10)]
    named: list[int] = []
    for quantity in (ten, hundred, thousand, ten_thousand):
        named.append(lattice.read_place(quantity, "digit", ten, gauges, hundred, thousand, ten_thousand)[0])
    for left in gauges:
        for right in gauges[::3]:
            total = lattice.consensus_quantity(left, right, 1)
            named.append(
                lattice.read_place(total, "digit", ten, gauges, hundred, thousand, ten_thousand)[0]
            )
    return named


def check_attend(lattice: Lattice) -> list[str]:
    issues: list[str] = []
    fast_attend.ensure(lattice)
    if not fast_attend.enabled():
        return ["compiled attend did not load"]
    fast_attend.set_enabled(False)
    python_consensus = _consensus_grid(lattice)
    python_names = _place_grid(lattice)
    fast_attend.set_enabled(True)
    if not fast_attend.enabled():
        return ["compiled attend did not load after the Python pass"]
    c_consensus = _consensus_grid(lattice)
    c_names = _place_grid(lattice)
    if len(python_consensus) != len(c_consensus) or len(python_names) != len(c_names):
        return ["attend grids differed in length"]
    worst = 0.0
    for (py_value, _left, _right, _sign), (c_value, _, _, _) in zip(python_consensus, c_consensus):
        gap = abs(py_value - c_value)
        if gap > worst:
            worst = gap
    if worst >= PARITY_GAP:
        issues.append(f"consensus gap {worst}")
    name_misses = sum(py_name != c_name for py_name, c_name in zip(python_names, c_names))
    if name_misses:
        issues.append(f"place names differed on {name_misses} reads")
    print(
        f"attend rows={len(c_consensus)} names={len(c_names)} "
        f"worst_gap={worst:.3e} name_misses={name_misses}"
    )
    return issues


def main() -> int:
    pin = require_live_pin()
    print(f"engine pin {pin[:12]}")
    issues = check_package(pin)
    lattice = Lattice.fresh()
    issues.extend(check_attend(lattice))
    if issues:
        return _fail(issues)
    print("mechanism check")
    check_generated(lattice)
    print("lattice verification ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
