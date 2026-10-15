# FSOT Intelligence

[![CI](https://github.com/dappalumbo91/FSOT-Intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/dappalumbo91/FSOT-Intelligence/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Own machine-learning apparatus under Fluid Spacetime Omni-Theory.
The law is S = K(T1 + T2 + T3), taken from the live engine in
[FSOT-2.1-Lean](https://github.com/dappalumbo91/FSOT-2.1-Lean) at pin **AEB2AD**
(`vendor/fsot_compute.py`). Width is 60. The promoted package learned 1950 numbers:
10 digit gauges, 10 number-word gauges, 10 content gauges, and 1920 bonds.

The scored read is a consensus attend on two orthogonal carriers. That attend
is C, in `apparatus/fsot_lattice/attend_core.c`: the quantity, the place
read, a repeated-addition fold, and one call that does the quantity and the
place read together. Training, the English spellings, the census, and the
promotion gate stay Python, and they call the C attend for every scored row.
GitHub's language share follows those Python lines, including the pinned
engine `vendor/fsot_compute.py`. The same gauges name closed arithmetic from
a single digit through the sums that leave 0..999999. The score
writeup is [apparatus/README.md](apparatus/README.md).

## Verify

```powershell
python apparatus\verify_lattice.py
```

That check re-reads the promoted package, rebuilds nothing in the gauges, and
requires the C attend to name the same cells as the Python attend. The check
lives in this repository. CI builds the C attend and runs it.

## Run

```powershell
pip install -r requirements.txt
python apparatus\run_lattice.py
```

A full promotion re-scores the closed package. The run that promoted this
package finished on 2026-10-15 in 73285.54 seconds. The million digit surface
reached 4995000000 rows at 33903 seconds and the word surface at 36980
seconds. The 2026-10-07 million package finished in 112076.55 seconds, with
the digit surface at 52221 seconds and the word surface at 56460 seconds.
The earlier compiled package, the hundred-thousand sums, finished in 3382.35
seconds. The ten-thousand difference finished in 1980.31 seconds. The compiled attend is
`apparatus/fsot_lattice/attend_core.c`. Build it beside the Python file and
the loader uses it. A missing library leaves the Python attend in place, and
that path names the same cells.

```powershell
clang -O2 -std=c11 -ffp-contract=off -fno-fast-math -fno-associative-math -shared -o apparatus\fsot_lattice\attend_core.dll apparatus\fsot_lattice\attend_core.c
```

On Linux, add `-fPIC` and write `apparatus/fsot_lattice/attend_core.so`.

## Promoted package

Pin `aeb2adad6e80`. Both surfaces exact through the million sums and through
a trillion. Digit holdout read gap 1.530e-6. Learned count 1950. Width 60.
The closed English table is exact. Scores are in
`apparatus/results/lattice_run.json`. Surviving bonds are in
`apparatus/data/engrams.json`. The ten content gauges are in the score report.

## Where this stands

The ten digit gauges are the whole number system. Each larger place is those
gauges counted ten times, then shed. Both surfaces name every sum that leaves
0..999999. That family is 4999950000 rows, and it is the last one the score
walks row by row. The promoted run finished on 2026-10-15 in 73285.54
seconds. The learned count is 1950 and the width is 60.

Ordinary English place names continue through a trillion. Six scored reads
are in this package: the promoted sums, the spelled quantities through a
trillion, and a small English of identity, kind, a sentence that remembers
what it just said, and a count beside that kind. Open paragraphs wait.
No date has been measured for matching a public language model.

1. **Quantities people spell.** Ten million, a hundred million, a billion,
   ten billion, a hundred billion, and a trillion. Each rung is the ten-step
   counted by that power of ten. The proof is a checked band at that rung,
   under the same 1e-4 floor as the million. Measured again on 2026-10-07,
   on the frozen gauges: every name exact, and the vessel sits on the
   algebraic sum. The widest gap on ten billion, a hundred billion, and a
   trillion is 0, on both surfaces, closed and edge. The floor stays 1e-4.
   That measurement is in the 2026-10-15 package.
2. **Words that are not numbers.** Ten content words — cat, dog, bird,
   fish, animal, oak, rose, plant, rock, water — each carry a gauge on
   the line K times one through ten. The question is "is left right".
   The same word reads yes. A different word reads no. That answer is
   the equality read: a remainder under the drop, or a remainder that
   clears it. yes and no stay priors. Measured on 2026-10-13, on the
   frozen arithmetic gauges: 160 epochs, and the closed table of 100
   pairs exact on the train slice (80), the hold slice (20), and the
   whole table. The same-word remainder is 0. The smallest
   different-word remainder is K, about 0.420, and the drop is about
   0.238. The gauge residual is 9.701e-5, inside the 1e-4 floor. Digit
   gauges and number-word gauges stayed on the promoted values. A
   package that includes these ten gauges counts 1950 learned numbers.
   That measurement is in the 2026-10-15 package, at 1950 learned numbers.
   `verify_lattice.py` exited 0 on that same tree: 48 place reads,
   attend parity exact, and the stored package still the million.
3. **Composed statements.** The kind of each content word is a prior.
   Cat, dog, bird, fish, and animal land on animal. Oak, rose, and
   plant land on plant. Rock and water are their own heads. The read
   steps the subject gauge up to the lowest head at or above it, then
   takes the equality remainder against the stated word. "a cat is an
   animal" reads yes. "a cat is a dog" reads no. A depth-3 chain takes
   that step three times. The later steps read the vessel, and a head
   stays put, so "a cat is an animal is an animal is an animal" names
   animal. Water's closed sentence is "a water is a water". Measured
   on 2026-10-13, on the trained content gauges: the closed table of
   100 statements is exact on the train slice (80), the hold slice
   (20), and the whole table. A true remainder is 0. The smallest
   false remainder is K, about 0.420, and the drop is about 0.238.
   The ten depth-3 chains are exact on the train slice (8), the hold
   slice (2), and the whole set. The vessel sits on the head gauge,
   gap 0, and the runner-up margin is K. The kind table adds no
   gauges. Digit gauges and number-word gauges stayed on the promoted
   values. That measurement is in the 2026-10-15 package, at 1950 learned numbers.
   `verify_lattice.py` exited 0 on that same tree.
4. **One sentence carried into the next.** The frozen gauge is the slow
   trace. The idle pass writes the surviving bonds. The working carrier
   is the circulating clock, one scalar, and it starts empty. "is it an
   animal" reads no while the carrier is empty. "a cat is an animal"
   leaves the animal gauge there. "is it an animal" then reads yes, and
   "is it a cat" reads no, because the carrier holds the kind the
   sentence landed on. A question leaves that gauge in place. "an oak
   is a plant" replaces it, and "is it a plant" reads yes. Measured on
   2026-10-13, on the trained content gauges: 100 two-sentence
   conversations, exact on the train slice (80), the hold slice (20),
   and the whole table. A true remainder is 0. The carried gauge matches
   the kind gauge, gap 0. The smallest false remainder is K, about
   0.420, and the drop is about 0.238. The carrier adds no gauges.
   Digit gauges and number-word gauges stayed on the promoted values.
   That measurement is in the 2026-10-15 package, at 1950 learned numbers.
   `verify_lattice.py` exited 0 on that same tree. The weights stay
   the gauges.
5. **Width.** Widen only after a closed English table at width 60 is exact
   on the training rows, the held-out rows, and the arithmetic already
   promoted. Steps 2, 3, and 4 are exact on that width, and the arithmetic
   gauges stayed on the promoted values, so width stays 60. The RTX 5070
   is for a later batch of those same named cells.
6. **A count and a kind in one sentence.** The kind clock is the carrier
   from the previous step. The count clock is a second scalar. "two cats"
   leaves the gauge of two and the animal gauge. "is it two" reads yes, and
   "is it an animal" reads yes. "is it a cat" reads no, because the kind
   clock holds the kind the sentence landed on. "is it three" reads no.
   An empty clock reads no for every count from one through nine. Zero stays
   out, because its gauge sits under the drop. The next counted sentence
   replaces both clocks, so "three dogs" then reads yes for three and for
   animal. Measured on 2026-10-13, on the trained content gauges and the
   promoted number-word gauges: 90 sentences, exact on the train slice (72),
   the hold slice (18), and the whole table. A true remainder is 0. Both
   clocks match the gauges they were given, gap 0. The smallest false
   remainder is K, about 0.420, and the drop is about 0.238. The clocks add
   no gauges. Digit gauges and number-word gauges stayed on the promoted
   values. 2+3 reads 5, and the equality claim reads yes. That table
   is in the 2026-10-15 package, at 1950 learned numbers. `verify_lattice.py` exited 0
   on that same tree. Width stays 60.

## Authority

`vendor/fsot_compute.py` is the pinned engine. When a checkout of
FSOT-2.1-Lean is present at `fsot-2.1-lean/`, that copy is the file this
folder binds. The published clone ships the pin on its own.

## Related

- [FSOT-2.1-Lean](https://github.com/dappalumbo91/FSOT-2.1-Lean) — theory authority
- [FSOT-2.1-Cpp](https://github.com/dappalumbo91/FSOT-2.1-Cpp) — C++ port of the same engine
- [fsot-neuron-zig](https://github.com/dappalumbo91/fsot-neuron-zig) — fixed-point neural mind
- [FSOT-GPU](https://github.com/dappalumbo91/FSOT-GPU) — earlier consensus attend on a host model
