# FSOT Intelligence

[![CI](https://github.com/dappalumbo91/FSOT-Intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/dappalumbo91/FSOT-Intelligence/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Own machine-learning apparatus under Fluid Spacetime Omni-Theory.
The law is S = K(T1 + T2 + T3), taken from the live engine in
[FSOT-2.1-Lean](https://github.com/dappalumbo91/FSOT-2.1-Lean) at pin **AEB2AD**
(`vendor/fsot_compute.py`). Width is 60. The promoted package learned 1940 numbers:
10 digit gauges, 10 number-word gauges, and 1920 bonds.

The scored read is a consensus attend on two orthogonal carriers. The same
gauges name closed arithmetic from a single digit through a ten-thousand name
used as an operand. The score writeup is [apparatus/README.md](apparatus/README.md).

## Run

```powershell
pip install -r requirements.txt
python apparatus\run_lattice.py
```

A full promotion re-scores the closed package. The run that promoted this
package finished in 2837.33 seconds. The compiled attend is
`apparatus/fsot_lattice/attend_core.c`. Build it beside the Python file and
the loader uses it. A missing library leaves the Python attend in place, and
that path names the same cells.

```powershell
clang -O2 -std=c11 -ffp-contract=off -fno-fast-math -fno-associative-math -shared -o apparatus\fsot_lattice\attend_core.dll apparatus\fsot_lattice\attend_core.c
```

On Linux, add `-fPIC` and write `apparatus/fsot_lattice/attend_core.so`.

## Promoted package

Pin `aeb2adad6e80`. Both surfaces exact. Digit holdout read gap 1.530e-6.
Learned count 1940. Width 60. Scores are in `apparatus/results/lattice_run.json`.
Surviving bonds are in `apparatus/data/engrams.json`.

## Authority

`vendor/fsot_compute.py` is the pinned engine. When a checkout of
FSOT-2.1-Lean is present at `fsot-2.1-lean/`, that copy is the file this
folder binds. The published clone ships the pin on its own.

## Related

- [FSOT-2.1-Lean](https://github.com/dappalumbo91/FSOT-2.1-Lean) — theory authority
- [FSOT-2.1-Cpp](https://github.com/dappalumbo91/FSOT-2.1-Cpp) — C++ port of the same engine
- [fsot-neuron-zig](https://github.com/dappalumbo91/fsot-neuron-zig) — fixed-point neural mind
- [FSOT-GPU](https://github.com/dappalumbo91/FSOT-GPU) — earlier consensus attend on a host model
