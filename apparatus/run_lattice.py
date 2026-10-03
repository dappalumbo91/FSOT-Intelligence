"""Run the closed arithmetic lattice and write results/lattice_run.json."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fsot_lattice.train import run


if __name__ == "__main__":
    report = run()
    raise SystemExit(0 if report["promoted"] else 1)
