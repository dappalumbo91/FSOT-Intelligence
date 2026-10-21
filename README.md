# FSOT Intelligence

[![CI](https://github.com/dappalumbo91/FSOT-Intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/dappalumbo91/FSOT-Intelligence/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Own machine-learning apparatus under Fluid Spacetime Omni-Theory.
The law is S = K(T1 + T2 + T3), taken from the live engine in
[FSOT-2.1-Lean](https://github.com/dappalumbo91/FSOT-2.1-Lean) at pin **AEB2AD**
(`vendor/fsot_compute.py`). Width is 60. The promoted package learned 1970 numbers:
10 digit gauges, 10 number-word gauges, 10 content gauges, 10 kind gauges,
10 individual gauges, and 1920 bonds.

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
package finished on 2026-10-20 in 65590.94 seconds. The million digit surface
reached 4995000000 rows at 31682 seconds and the word surface at 31544
seconds. The 2026-10-15 package finished in 73285.54 seconds, with the digit
surface at 33903 seconds and the word surface at 36980 seconds. The 2026-10-07
million package finished in 112076.55 seconds, with the digit surface at
52221 seconds and the word surface at 56460 seconds.
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
a trillion. Digit holdout read gap 1.530e-6. Learned count 1970. Width 60.
The closed English table is exact, including the ten taught nouns, their
individual gauges, and a fact of three nouns. Scores are in
`apparatus/results/lattice_run.json`. Surviving bonds, the ten kind gauges,
and the ten individual gauges are in `apparatus/data/engrams.json`. The ten
content gauges are in the score report.

## Where this stands

The ten digit gauges are the whole number system. Each larger place is those
gauges counted ten times, then shed. Both surfaces name every sum that leaves
0..999999. That family is 4999950000 rows, and it is the last one the score
walks row by row. The promoted run finished on 2026-10-20 in 65590.94
seconds. The learned count is 1970 and the width is 60.

Ordinary English place names continue through a trillion. This package holds
the promoted sums, the spelled quantities through a trillion, and a closed
English of identity, kind, a sentence that remembers what it just said, and
a count beside that kind. A two-sentence paragraph keeps the first kind. A
counted sentence keeps its number across the next kind sentence. Ten nouns
outside that table settle onto the kind the sentence names, reload from idle
with drift 0, and each keeps an individual gauge beside its kind, so a wolf
is an animal and a wolf is not a cow. A later sentence moves that kind onto
another head and leaves the individual where it is. The resting curriculum
stays the first teaching. A counted sentence can name one of those nouns, so
two wolves leaves two and animal. Two of those nouns make a paragraph, so a
wolf then a pine keeps animal beside plant. Three of those nouns keep the
first kind after two later sentences, so a wolf, then a pine, then a clay
still holds animal. Those reads are in this package. Four of those
nouns make an open paragraph. A wolf, then a pine, then a clay, then a
mist still holds animal after the three-clock fact read has let it go.
That paragraph is measured locally. The resting fact stays the first
teaching, so a wolf is an animal. A second sense of the same name
belongs beside that fact, in the situation that names it. The
published package remains the 2026-10-20 report. No date has been
measured for matching a public language model.

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
7. **A paragraph keeps the first kind.** Two kind sentences. The second
   writes its kind on the latest clock and moves the first kind onto an
   earlier clock. "is it" reads the latest clock. After "a cat is an
   animal. an oak is a plant", that question reads yes for plant and no
   for animal. The paragraph read takes the closer of the two remainders.
   The animal question then reads yes, and rock reads no. A same-kind
   paragraph, "a cat is an animal. a dog is an animal", stays yes for
   animal. Measured on 2026-10-19, on the promoted
   content gauges: 100 paragraphs, exact on the train slice (80), the hold
   slice (20), and the whole table. A true remainder is 0. Both clocks
   match the kind gauges they were given, gap 0. The smallest false
   remainder is K, about 0.420, and the drop is about 0.238. The follow-up
   table stayed exact, because "is it" still reads the latest clock. The
   earlier clock adds no gauges. Digit gauges, number-word gauges, and the
   content gauges stayed on the promoted values. 2+3 reads 5, and the
   equality claim reads yes. `verify_lattice.py` exited 0 on this tree:
   learned 1950, 48 place reads, attend parity exact, worst consensus gap 0.
   That read is in the 2026-10-20 package, at 1970 learned numbers. Width stays 60.
8. **A count survives the next sentence.** "two cats. an oak is a plant"
   leaves two on the earlier count clock and animal on the earlier kind
   clock. Plant is the latest kind. The latest count is empty, so "is it
   two" on that clock reads no. The kept read takes the closer count
   remainder, so "is it two" reads yes and "is it three" reads no. The
   kept kind read still answers yes for animal and for plant, and no for
   rock. "is it a cat" reads no. Measured on 2026-10-19, on the promoted
   gauges: 900 paragraphs, exact on the train slice (720), the hold slice
   (180), and the whole table. A true remainder is 0. The three clocks
   match the gauges they were given, gap 0. The smallest false remainder
   is K, about 0.420, and the drop is about 0.238. The follow-up, the
   kind paragraph, and the counted sentence stayed on their previous
   answers. The clocks add no gauges. Digit gauges, number-word gauges,
   and the content gauges stayed on the promoted values. 2+3 reads 5,
   and the equality claim reads yes. `verify_lattice.py` exited 0 on
   this tree: learned 1950, 48 place reads, attend parity exact, worst
   consensus gap 0. That read is in the 2026-10-20 package, at 1970 learned
   numbers. Width stays 60.
9. **A noun outside the ten.** Wolf, clay, cow, hare, pine, moss, slate,
   boulder, mist, and pond each start at K. The same fluid step moves that
   gauge onto the live head the sentence names. Wolf, cow, and hare land
   on animal. Pine and moss land on plant. Clay, slate, and boulder land
   on rock. Mist and pond land on water. The ten content gauges stay on
   the promoted values. "a wolf is an animal" reads yes. "a wolf is a cat"
   reads no. "a pine is a plant" reads yes. A fresh gauge at K already
   kind-steps onto animal, so pine reads as animal until the step carries
   it to plant. The gauge stores the kind. Wolf, cow, and hare share one
   gauge, so "is wolf cow" reads yes. Measured on 2026-10-19, on the
   promoted gauges: 172 epochs. The closed table of 100 statements is
   exact on the train slice (80), the hold slice (20), and the whole
   table. The gauge residual is 9.584e-5, inside the 1e-4 floor. The
   widest true statement remainder is that same 9.584e-5, the gap still
   left on the water nouns, and it sits under the drop. The smallest
   false statement remainder is 0.420013, and the drop is about 0.238.
   The pair table of 100 is exact on the train slice (80), the hold slice
   (20), and the whole table. A same-kind remainder is 0. The smallest
   different-kind remainder is 0.420098. Wolf, then pine, leaves animal
   on the earlier clock and plant on the latest clock. "is it a plant"
   reads yes. "is it an animal" on the latest clock reads no, and the
   kept read answers yes for animal. The carry gap on that script is
   7.454e-5. The follow-up, the kind paragraph, the counted sentence, and
   the kept count stayed on their previous answers. Digit gauges,
   number-word gauges, and the content gauges stayed on the promoted
   values. 2+3 reads 5 and five, and the equality claim reads yes on both
   surfaces. `verify_lattice.py` exited 0 on this tree: learned 1950, 48
   place reads, attend parity exact, worst consensus gap 0. That read is in
   the 2026-10-20 package, at 1970 learned numbers. Width stays 60.
10. **The noun gauge survives idle.** After the ten nouns have settled, an
    idle snapshot writes those gauges and the reload reads them back. The
    four clocks stay in memory. Measured on 2026-10-19, on the promoted
    gauges: the drift is 0. The closed statement table stays exact on the
    train slice (80), the hold slice (20), and the whole table of 100.
    The widest true remainder stays 9.584e-5. The smallest false remainder
    stays 0.420013, and the drop is about 0.238. The pair table stays
    exact, a same-kind remainder stays 0, and the smallest different-kind
    remainder stays 0.420098. Wolf, then pine, still answers no, yes, no,
    no, yes, no, yes. The follow-up, the kind paragraph, the counted
    sentence, and the kept count stayed on their previous answers. Digit
    gauges, number-word gauges, and the content gauges stayed on the
    promoted values. 2+3 reads 5 and five, and the equality claim reads
    yes on both surfaces. `verify_lattice.py` exited 0 on this tree:
    learned 1950, 48 place reads, attend parity exact, worst consensus
    gap 0. That reload is in the 2026-10-20 package, at 1970 learned
    numbers. Width stays 60.
11. **The individual beside the kind.** Each new noun keeps the kind gauge
    and gains a second gauge on the same line the content words use, K
    times its place. Wolf and cow still share the animal gauge, so "is
    wolf cow" reads yes and "a wolf is an animal" reads yes. The second
    gauge is the individual. "a wolf is a wolf" reads yes. "a wolf is a
    cow" reads no. "a wolf is a hare" reads no. Measured on 2026-10-19,
    on the promoted gauges: 160 epochs. The closed table of 100 is exact
    on the train slice (80), the hold slice (20), and the whole table.
    A same-noun remainder is 0. The smallest different-noun remainder is
    K, about 0.420, and the drop is about 0.238. The gauge residual is
    9.701e-5, inside the 1e-4 floor. The places land on the same values
    as cat through water, because that is the same line, and the kind
    question still answers no for "a wolf is a cat". The kind table stayed
    exact. Idle reloads both gauges with drift 0. Digit gauges,
    number-word gauges, the content gauges, and the kind gauges stayed
    on their values. 2+3 reads 5 and five, and the equality claim reads
    yes on both surfaces. `verify_lattice.py` exited 0 on this tree:
    learned 1950, 48 place reads, attend parity exact, worst consensus
    gap 0. That read is in the 2026-10-20 package, at 1970 learned numbers.
    Width stays 60.
12. **A later sentence moves the kind.** The individual gauge stays. One
    sentence moves wolf onto plant and leaves cow on animal. "a wolf is a
    plant" reads yes. "a wolf is an animal" reads no. "is wolf cow" reads
    no. "a wolf is a wolf" stays yes, and "a wolf is a cow" stays no. The
    closed table then moves every noun onto a new head with that same
    alignment step. Measured on 2026-10-19, on the promoted gauges: the
    single sentence took 10 epochs and left wolf 5.332e-5 from plant. The
    closed table took 160 epochs. Train 80/80, hold 20/20, and the whole
    table of 100 are exact. The widest true remainder is 9.966e-5. The
    smallest false remainder is 0.420009, and the drop is about 0.238. The
    widest gap between two nouns on the same new head is 9.965e-5, inside
    the 1e-4 floor. Wolf and cow differ by about 0.840. Clay and mist meet
    on animal. Pine stays with moss on rock, and cow stays with hare on
    water. The individual table stayed exact, a same-noun remainder is 0,
    and that residual stays 9.701e-5. Idle reloads both gauges with drift
    0. Digit gauges, number-word gauges, and the content gauges stayed on
    the promoted values. Putting the first teaching back leaves those
    original tables exact, and wolf then pine still answers no, yes, no,
    no, yes, no, yes. The follow-up, the kind paragraph, the counted
    sentence, and the kept count stayed on their previous answers. 2+3
    reads 5 and five, and the equality claim reads yes on both surfaces.
    `verify_lattice.py` exited 0 on this tree: learned 1950, 48 place
    reads, attend parity exact, worst consensus gap 0. That read is in
    the 2026-10-20 package, at 1970 learned numbers. This sentence adds
    no gauges. Width stays 60.
13. **The new noun inside a count.** "two wolves" writes the gauge of two
    on the count clock and the animal gauge on the kind clock. The noun
    gauge and the individual stay put. "is it two" reads yes. "is it an
    animal" reads yes. "is it a cat" reads no. The next counted noun
    replaces both clocks. One pine, then moss, keeps plant. Nine ponds,
    then a wolf, leaves water and then animal. Zero stays out. The prior
    plural makes two wolves and two mosses, and it still makes two cats
    and two fish. Measured on 2026-10-19, on the promoted gauges: 90
    sentences, exact on the train slice (72), the hold slice (18), and
    the whole table. The widest true remainder is 9.584e-5. The carry
    gap is that same 9.584e-5. The smallest false remainder is 0.420013,
    and the drop is about 0.238. The kind gauges and the individual
    gauges stayed on their values. Digit gauges, number-word gauges, and
    the content gauges stayed on the promoted values. One cat, one rock,
    and five water stayed on their previous lists. 2+3 reads 5 and five,
    and the equality claim reads yes on both surfaces.
    `verify_lattice.py` exited 0 on this tree: learned 1950, 48 place
    reads, attend parity exact, worst consensus gap 0. That read is in
    the 2026-10-20 package, at 1970 learned numbers. This sentence adds
    no gauges. Width stays 60.
14. **The new noun inside a paragraph.** "a wolf is an animal. a pine is
    a plant" writes the animal gauge, then keeps that kind and writes
    plant. The paragraph read answers yes for animal and for plant, and
    no for rock. "is it" reads the latest clock, so plant is the latest
    answer. "a cow is an animal. a hare is an animal" stays on animal.
    "a mist is a water. a pond is a water" stays on water. The noun
    gauge and the individual stay put. Measured on 2026-10-19, on the
    promoted gauges: 100 paragraphs, exact on the train slice (80), the
    hold slice (20), and the whole table. The widest true remainder is
    9.584e-5. The carry gap is that same 9.584e-5. The smallest false
    remainder is 0.420013, and the drop is about 0.238. Wolf then pine
    answers no, yes, no, yes, yes, no, and its carry is 7.454e-5. Cow
    then hare, and mist then pond, answer no, yes, yes, yes, yes, no.
    The kind gauges and the individual gauges stayed on their values.
    Digit gauges, number-word gauges, and the content gauges stayed on
    the promoted values. The follow-up, the kind paragraph, one cat,
    the kept count, wolf then pine on the latest clock, two wolves, one
    pine, and nine ponds stayed on their previous lists. 2+3 reads 5
    and five, and the equality claim reads yes on both surfaces.
    `verify_lattice.py` exited 0 on this tree in 107.49 seconds:
    learned 1950, 48 place reads, attend parity exact, worst consensus
    gap 0. That read is in the 2026-10-20 package, at 1970 learned
    numbers. This paragraph adds no gauges. Width stays 60.
15. **A fact survives two replacements.** "a wolf is an animal. a pine
    is a plant. a clay is a rock" writes animal, then plant, then rock.
    Animal stays on the clock two sentences back. Plant stays on the
    clock one sentence back. Rock is the latest kind. "is it" reads
    the latest clock, so the latest answer is rock. The fact read
    answers yes for animal, plant, and rock, and no for water. "a cow
    is an animal. a hare is an animal. a wolf is an animal" stays on
    animal. "a mist is a water. a pond is a water. a boulder is a rock"
    keeps water after rock arrives. The noun gauge and the individual
    stay put. Measured on 2026-10-19, on the promoted gauges: 1000
    facts, exact on the train slice (800), the hold slice (200), and
    the whole table. The widest true remainder is 9.584e-5. The carry
    gap is that same 9.584e-5. The smallest false remainder is
    0.420013, and the drop is about 0.238. Wolf then pine then clay
    answers no, yes, no, yes, yes, no, yes, yes, yes, no, and its
    carry is 8.519e-5. Cow then hare then wolf answers no, yes, yes,
    yes, yes, yes, yes, yes, yes, no. Mist then pond then boulder
    answers no, yes, yes, yes, yes, no, yes, yes, yes, no. The kind
    gauges and the individual gauges stayed on their values. Digit
    gauges, number-word gauges, and the content gauges stayed on the
    promoted values. The follow-up, the kind paragraph, one cat, the
    kept count, wolf then pine, two wolves, one pine, and nine ponds
    stayed on their previous lists. 2+3 reads 5 and five, and the
    equality claim reads yes on both surfaces. `verify_lattice.py`
    exited 0 on this tree in 109.34 seconds: learned 1950, 48 place
    reads, attend parity exact, worst consensus gap 0. That read is in
    the 2026-10-20 package, at 1970 learned numbers. This fact adds no
    gauges. Width stays 60.
16. **An open paragraph keeps the opening kind.** Four kind sentences.
    The fourth writes its kind on the latest clock and moves the opening
    kind onto one more clock. "is it" reads the latest clock. The fact
    read sees the latest three clocks, so after "a wolf is an animal. a
    pine is a plant. a clay is a rock. a mist is a water" that read
    answers no for animal. The open read answers yes for animal, plant,
    rock, and water. "is it" answers yes for water and no for animal.
    Wolf then pine then clay then pine leaves water unnamed, so water
    reads no. Wolf then cow then hare then wolf stays on animal, and
    plant reads no. Three sentences leave the opening clock empty, so
    wolf then pine then clay still answers no, yes, no, yes, yes, no,
    yes, yes, yes, no. Measured on 2026-10-20, on the promoted gauges:
    10000 paragraphs, exact on the train slice (8000), the hold slice
    (2000), and the whole table. The widest true remainder is 9.584e-5.
    The carry gap is that same 9.584e-5. The smallest false remainder
    is 0.420013, and the drop is about 0.238. Wolf then pine then clay
    then mist answers no, yes, no, yes, yes, yes, yes, yes, yes, and
    its carry is 9.584e-5. Wolf then pine then clay then pine answers
    no, yes, no, yes, yes, yes, yes, yes, yes, no, and its carry is
    8.519e-5. Wolf then cow then hare then wolf answers no, yes, yes,
    yes, yes, yes, yes, yes, yes, no, and its carry is 4.259e-5. The
    train, hold, and closed passes together took 16.10 seconds. The
    kind gauges and the individual gauges stayed on their values.
    Digit gauges, number-word gauges, and the content gauges stayed on
    the promoted values. The follow-up, the kind paragraph, one cat,
    the kept count, wolf then pine, two wolves, one pine, nine ponds,
    and wolf then pine then clay stayed on their previous lists. This
    paragraph adds no gauges. The published package remains the
    2026-10-20 report. Width stays 60.

## Authority

`vendor/fsot_compute.py` is the pinned engine. When a checkout of
FSOT-2.1-Lean is present at `fsot-2.1-lean/`, that copy is the file this
folder binds. The published clone ships the pin on its own.

## Related

- [FSOT-2.1-Lean](https://github.com/dappalumbo91/FSOT-2.1-Lean) — theory authority
- [FSOT-2.1-Cpp](https://github.com/dappalumbo91/FSOT-2.1-Cpp) — C++ port of the same engine
- [fsot-neuron-zig](https://github.com/dappalumbo91/fsot-neuron-zig) — fixed-point neural mind
- [FSOT-GPU](https://github.com/dappalumbo91/FSOT-GPU) — earlier consensus attend on a host model
