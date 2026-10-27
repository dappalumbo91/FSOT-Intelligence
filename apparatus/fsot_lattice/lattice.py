"""60-unit lattice with three clocks.

Quantity lives on a learned gauge per symbol. The scored read is the
consensus state on two orthogonal carriers: operand A on the plus carrier,
operand B on that same carrier for plus and on the minus carrier for minus.
Equality subtracts the stated digit from that sum. A superposed remainder
is yes. A remainder that clears collapse is no. A two-step chain feeds that
same vessel back in as the next operand. Order, product, and a third step
are generated from that vessel. A name past nine sheds the consensus of
nine-plus-one until the remainder is a digit gauge. That same ten-step
folds back up when the name re-enters as an operand. Two rebuilt names
share one consensus pass, and the same spelling reads their sum or difference.
A sum that leaves 0..99 sheds the ten-step counted ten times, and that
hundred place names the result. A hundred name re-enters as an operand:
the hundred-step folds back up, the lower place joins it, and one
consensus pass adds or subtracts a name in 0..99. The result stays
inside 0..999. A sum that leaves 0..999 sheds the hundred-step counted
ten times, and that thousand place names the result. A thousand name
re-enters as an operand: the thousand-step folds back up, one name in
0..999 joins it, and one consensus pass adds or subtracts that name.
The result stays inside 0..9999. Two such thousand names share one
consensus pass, and the same spelling reads their sum or difference
while it stays inside 0..9999. A sum that leaves 0..9999 sheds the
thousand-step counted ten times, and that ten-thousand place names
the result. A ten-thousand name re-enters as an operand: that step
folds back up, one name in 0..999 joins it, and one consensus pass
adds or subtracts a name in 0..9999. The result stays inside 0..10999.
Two ten-thousand names share one consensus pass. The spelling reads their
difference while it stays inside 0..999. Their sum leaves the table.
A sum that leaves 0..99999 sheds the ten-thousand step counted ten
times, and that hundred-thousand place names the result.
A sum that leaves 0..999999 sheds the hundred-thousand step counted
ten times, and that million place names the result. Ten million through
a trillion are that same count. Each is the previous rung counted ten
times. The million is the last place the score walks row by row.
Trit masks identify symbols and gate the operator. Bonds are the
long-term pathways written at idle.

Digit gauges freeze at consolidation. Word training moves only word gauges.
"""

from __future__ import annotations

import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path

from fsot_lattice.consensus import position_coherence, trit_similarity
from fsot_lattice.engine import (
    COHERENCE_GATE,
    COLLAPSE,
    K,
    PHI,
    POOF,
    PSI_CON,
    SUCTION,
    THETA_S,
    fluid_lr,
)
from fsot_lattice.tasks import (
    DIGITS,
    OP_WORD,
    ACQUIRED_NOUNS,
    CONTENT_WORDS,
    KIND_HEADS,
    KIND_STEPS,
    OPPOSED_KIND,
    WORD_OF,
    AcquiredCount,
    AcquiredFact,
    AcquiredOpen,
    AcquiredParagraph,
    Counted,
    FollowUp,
    AcquiredRevision,
    acquired_kind,
    all_acquired,
    all_acquired_pairs,
    split_revisions,
    content_kind,
    split_acquired,
    split_acquired_pairs,
    split_acquired_selves,
    Chain,
    Claim,
    HundredOp,
    HundredSum,
    Lexeme,
    HundredThousandSum,
    MillionSum,
    TenThousandOp,
    TenThousandPair,
    TenThousandSum,
    ThousandOp,
    ThousandPair,
    ThousandSum,
    PlacePair,
    PlaceStep,
    Product,
    Span,
    Triple,
    decimal_name,
    number_name,
)

# Batch mean gradient is O(1/n). PHI/POOF brings the raw fluid unit onto that mean.
GAUGE_GAIN = PHI / POOF
# A K-scale gauge on this amplitude clears collapse. Not a fitted coefficient.
AMP = PHI / K

WIDTH = 60
DIGIT_SYMBOLS = DIGITS + ["+", "-", "="]
WORD_SYMBOLS = [WORD_OF[i] for i in range(10)] + ["plus", "minus", "equals", "what", "is"]
NUMERIC_WORDS = {WORD_OF[i]: str(i) for i in range(10)}
QUANTITY_SYMBOLS = set(DIGITS) | {WORD_OF[i] for i in range(10)} | set(CONTENT_WORDS)


def _mask(index: int) -> list[float]:
    """Fixed sign pattern. Superposed band is the poof width, not a fit."""
    out: list[float] = []
    for unit in range(WIDTH):
        x = math.sin(THETA_S * (unit + 1) * (index + 1) + PSI_CON * index)
        if x > POOF:
            out.append(1.0)
        elif x < -POOF:
            out.append(-1.0)
        else:
            out.append(0.0)
    return out


def _quantity_carriers() -> tuple[list[float], list[float], list[float]]:
    """Plus, minus, and an identity axis for symbols that are not quantities.

    Plus is uniform. Minus alternates, so the two axes are orthogonal at even
    width and their trit similarity is zero. Operators sit on ++-- and are
    invisible to both quantity queries.
    """
    if WIDTH % 4 != 0:
        raise RuntimeError("quantity carriers need a width divisible by 4")
    plus = [AMP] * WIDTH
    minus = [AMP if unit % 2 == 0 else -AMP for unit in range(WIDTH)]
    pattern = (AMP, AMP, -AMP, -AMP)
    ortho = [pattern[unit % 4] for unit in range(WIDTH)]
    return plus, minus, ortho


@dataclass
class Clocks:
    working: list[float]
    trace: list[float]
    working_norm: float
    trace_norm: float
    consensus_sim: float


@dataclass
class Lattice:
    digit_value: dict[str, float]
    word_value: dict[str, float]
    masks: dict[str, list[float]]
    carrier_plus: list[float]
    carrier_minus: list[float]
    carrier_ortho: list[float]
    bonds: list[list[float]]
    content_value: dict[str, float]
    # Nouns taught by a sentence. Each starts at K and moves onto a kind head.
    # The head gauges stay put. These gauges are not masks.
    acquired_value: dict[str, float]
    # The individual beside the kind. Same offset line as the content words,
    # in its own dict. Wolf and cow share a kind gauge and keep two of these.
    acquired_self: dict[str, float]
    frozen_digits: dict[str, float] | None = None
    width: int = WIDTH
    # The circulating clocks. Zero is empty. A named sentence writes its kind
    # gauge on `working`, moves the previous kind to `earlier`, moves the
    # kind before that to `earliest`, and moves the kind before that to
    # `opening`. A counted sentence also writes the number-word gauge on
    # `counted` and shifts the count clocks the same way. A question reads
    # the clocks and leaves them. "is it" reads the latest clock. A paragraph
    # read also sees `earlier`. A fact read also sees `earliest`. An open
    # paragraph also sees `opening`. A kept count also sees `earlier_count`.
    # The latest count read stays on `counted`.
    working: float = 0.0
    counted: float = 0.0
    earlier: float = 0.0
    earlier_count: float = 0.0
    earliest: float = 0.0
    earliest_count: float = 0.0
    opening: float = 0.0
    opening_count: float = 0.0

    @classmethod
    def fresh(cls) -> "Lattice":
        symbols = DIGIT_SYMBOLS + WORD_SYMBOLS + list(CONTENT_WORDS)
        masks = {sym: _mask(i) for i, sym in enumerate(symbols)}
        # Ordered prior K*(count+1). Plus and minus constraints pull the offset into gauge.
        digit_value = {str(d): K * (d + 1) for d in range(10)}
        # Words start flat at the rest unit. They have to move onto the frozen digits.
        word_value = {WORD_OF[d]: K for d in range(10)}
        word_value["plus"] = K
        word_value["minus"] = K
        word_value["equals"] = K
        word_value["what"] = K
        word_value["is"] = K
        # Content words start flat at the rest unit, then move onto the offset line.
        content_value = {word: K for word in CONTENT_WORDS}
        acquired_value = {word: K for word in ACQUIRED_NOUNS}
        acquired_self = {word: K for word in ACQUIRED_NOUNS}
        bonds = [[0.0] * WIDTH for _ in range(WIDTH)]
        plus, minus, ortho = _quantity_carriers()
        return cls(
            digit_value=digit_value,
            word_value=word_value,
            masks=masks,
            carrier_plus=plus,
            carrier_minus=minus,
            carrier_ortho=ortho,
            bonds=bonds,
            content_value=content_value,
            acquired_value=acquired_value,
            acquired_self=acquired_self,
            working=0.0,
            counted=0.0,
            earlier=0.0,
            earlier_count=0.0,
            earliest=0.0,
            earliest_count=0.0,
            opening=0.0,
            opening_count=0.0,
        )

    def symbol_vector(self, symbol: str, surface: str) -> list[float]:
        if symbol in self.content_value:
            value = self.content_value[symbol]
        elif surface == "digit":
            value = self.digit_value.get(symbol, K)
        else:
            value = self.word_value.get(symbol, K)
        axis = self.carrier_plus if symbol in QUANTITY_SYMBOLS else self.carrier_ortho
        return [value * unit for unit in axis]

    def _on_axis(self, value: float, axis: list[float]) -> list[float]:
        return [value * unit for unit in axis]

    def consensus_quantity(self, left: float, right: float, sign: int) -> float:
        """Signed sum carried by the consensus state.

        The query is the carrier, never one of the value keys. Projection
        times the active count undoes the mean, so the read is v[a] ± v[b]
        when both operands clear collapse. A sub-threshold operand drops out.
        The compiled attend is the same loops. A miss falls back to Python.
        """
        from fsot_lattice import fast_attend

        fast_attend.ensure(self)
        if fast_attend.enabled():
            got = fast_attend.consensus_quantity(left, right, sign)
            if got is not None:
                return got
        return self._consensus_quantity_py(left, right, sign)

    def _weighted_scale(self, scale: float, axis: list[float]) -> float:
        """Trit weight times this scale.

        The key is the scale written on one carrier, so the projection is
        the weight times the scale. Summing the carrier components first
        drops the low digits once a place reaches the billions.
        """
        key = self._on_axis(scale, axis)
        if position_coherence(key) <= COHERENCE_GATE:
            return 0.0
        weight = trit_similarity(axis, key)
        if weight == 0.0:
            return 0.0
        return weight * scale

    def _consensus_quantity_py(self, left: float, right: float, sign: int) -> float:
        plus_q = self._weighted_scale(left, self.carrier_plus)
        if sign > 0:
            plus_q += self._weighted_scale(right, self.carrier_plus)
        minus_q = 0.0
        if sign < 0:
            minus_q = self._weighted_scale(right, self.carrier_minus)
        return plus_q - minus_q

    def routed_sign(self, op_symbol: str) -> int:
        """Trit-route an operator mask onto plus or minus."""
        plus_sim = trit_similarity(self.masks[op_symbol], self.masks["+"])
        minus_sim = trit_similarity(self.masks[op_symbol], self.masks["-"])
        if plus_sim == minus_sim:
            raise RuntimeError(f"{op_symbol} is trit-tied between plus and minus")
        return 1 if plus_sim > minus_sim else -1

    def phrase_clocks(self, symbols: list[str], surface: str) -> Clocks:
        """Trace flush across one phrase. Working here is the current token.

        The scored quantity is consensus_quantity. This path is the one-token
        flush check, where there is no operand pair to read.
        """
        trace = [0.0] * self.width
        working = [0.0] * self.width
        for sym in symbols:
            vec = self.symbol_vector(sym, surface)
            working = vec
            trace = [
                (1.0 - SUCTION) * trace[i] + SUCTION * vec[i] for i in range(self.width)
            ]
        return Clocks(
            working=working,
            trace=trace,
            working_norm=_norm(working),
            trace_norm=_norm(trace),
            consensus_sim=0.0,
        )

    def read_clocks(self, row: Triple, surface: str) -> Clocks:
        """Working state is the consensus vessel. Trace averages the logged phrase.

        The answer token is in the trace and is the trit reference. It is not
        a value key. Keys are the two operand gauges.
        """
        if surface == "digit":
            symbols = [str(row.a), row.op, str(row.b), "=", str(row.c)]
            answer = str(row.c)
            left = self.digit_value[str(row.a)]
            right = self.digit_value[str(row.b)]
        else:
            symbols = [
                "what",
                "is",
                WORD_OF[row.a],
                OP_WORD[row.op],
                WORD_OF[row.b],
                "equals",
                WORD_OF[row.c],
            ]
            answer = WORD_OF[row.c]
            left = self.word_value[WORD_OF[row.a]]
            right = self.word_value[WORD_OF[row.b]]
        trace = [0.0] * self.width
        for sym in symbols:
            vec = self.symbol_vector(sym, surface)
            trace = [
                (1.0 - SUCTION) * trace[i] + SUCTION * vec[i] for i in range(self.width)
            ]
        quantity = self.consensus_quantity(left, right, self.routed_sign(row.op))
        working = [quantity * unit for unit in self.carrier_plus]
        sim = trit_similarity(working, self.symbol_vector(answer, surface))
        return Clocks(
            working=working,
            trace=trace,
            working_norm=_norm(working),
            trace_norm=_norm(trace),
            consensus_sim=sim,
        )

    def predict_digit(self, row: Triple) -> str:
        sign = self.routed_sign(row.op)
        pred = self.consensus_quantity(
            self.digit_value[str(row.a)],
            self.digit_value[str(row.b)],
            sign,
        )
        return _nearest(pred, self.digit_value, DIGITS)

    def predict_word(self, row: Triple) -> str:
        sign = self.routed_sign(row.op)
        pred = self.consensus_quantity(
            self.word_value[WORD_OF[row.a]],
            self.word_value[WORD_OF[row.b]],
            sign,
        )
        return _nearest(pred, self.word_value, [WORD_OF[i] for i in range(10)])

    def equality_remainder(self, claim: Claim, surface: str) -> float:
        """Consensus sum minus the stated digit. The claim is a value key."""
        sign = self.routed_sign(claim.op)
        if surface == "digit":
            left = self.digit_value[str(claim.a)]
            right = self.digit_value[str(claim.b)]
            stated = self.digit_value[str(claim.claimed)]
        else:
            left = self.word_value[WORD_OF[claim.a]]
            right = self.word_value[WORD_OF[claim.b]]
            stated = self.word_value[WORD_OF[claim.claimed]]
        total = self.consensus_quantity(left, right, sign)
        return self.consensus_quantity(total, stated, -1)

    def remainder_answer(self, remainder: float) -> str:
        """Yes when the remainder vessel stays under the coherence gate."""
        vessel = [remainder * unit for unit in self.carrier_plus]
        if position_coherence(vessel) <= COHERENCE_GATE:
            return "yes"
        return "no"

    def predict_equal(self, claim: Claim, surface: str) -> str:
        return self.remainder_answer(self.equality_remainder(claim, surface))

    def content_remainder(self, left: str, right: str) -> float:
        """Difference of two content gauges. The same word cancels."""
        return self.consensus_quantity(
            self.content_value[left],
            self.content_value[right],
            self.routed_sign("-"),
        )

    def predict_content(self, left: str, right: str) -> str:
        return self.remainder_answer(self.content_remainder(left, right))

    def kind_head_value(self, vessel: float) -> float:
        """Lowest kind-head gauge at or above this vessel.

        A head the vessel already sits on, inside the drop, wins.
        The head list is a prior. Which head wins is the gauge line.
        """
        drop = COLLAPSE / AMP
        heads = [
            (self.content_value[name], place)
            for place, name in enumerate(KIND_HEADS)
        ]
        on = [item for item in heads if abs(item[0] - vessel) <= drop]
        if on:
            gauge, _place = min(on, key=lambda item: (abs(item[0] - vessel), item[0], item[1]))
            return gauge
        above = [item for item in heads if item[0] > vessel]
        if not above:
            raise RuntimeError("a content gauge sat above every kind")
        gauge, _place = min(above, key=lambda item: (item[0], item[1]))
        return gauge

    def kind_step(self, vessel: float) -> float:
        """One kind step. The gap is the head gauge minus this vessel."""
        gap = self.kind_head_value(vessel) - vessel
        return self.consensus_quantity(vessel, gap, self.routed_sign("+"))

    def statement_remainder(self, subject: str, stated: str) -> float:
        """Kind step of the subject, minus the stated gauge."""
        vessel = self.kind_step(self.content_value[subject])
        return self.consensus_quantity(
            vessel,
            self.content_value[stated],
            self.routed_sign("-"),
        )

    def predict_statement(self, subject: str, stated: str) -> str:
        return self.remainder_answer(self.statement_remainder(subject, stated))

    def kind_chain_quantity(self, subject: str) -> float:
        """Three kind steps. Step two and step three read the running vessel."""
        vessel = self.content_value[subject]
        for _ in range(KIND_STEPS):
            vessel = self.kind_step(vessel)
        return vessel

    def predict_kind_chain(self, subject: str) -> str:
        values = [self.content_value[name] for name in CONTENT_WORDS]
        best, _margin, _dist = _nearest_detail(self.kind_chain_quantity(subject), values)
        return CONTENT_WORDS[best]

    def clear_working(self) -> None:
        """An empty carrier. The next question has no noun and no count."""
        self.working = 0.0
        self.counted = 0.0
        self.earlier = 0.0
        self.earlier_count = 0.0
        self.earliest = 0.0
        self.earliest_count = 0.0
        self.opening = 0.0
        self.opening_count = 0.0

    def _keep_previous(self) -> None:
        """The sentence about to be written becomes the latest. The older three stay."""
        self.opening = self.earliest
        self.opening_count = self.earliest_count
        self.earliest = self.earlier
        self.earliest_count = self.earlier_count
        self.earlier = self.working
        self.earlier_count = self.counted

    def deposit_kind(self, subject: str) -> float:
        """Leave this subject's kind gauge. A kind sentence names no count."""
        self._keep_previous()
        self.working = self.kind_step(self.content_value[subject])
        self.counted = 0.0
        return self.working

    def deposit_count(self, count: str, subject: str) -> tuple[float, float]:
        """Leave the number-word gauge and this subject's kind gauge."""
        self._keep_previous()
        self.counted = self.word_value[count]
        self.working = self.kind_step(self.content_value[subject])
        return self.counted, self.working

    def deposit_acquired_count(self, count: str, noun: str) -> tuple[float, float]:
        """Leave the number-word gauge and the kind this taught noun steps to.

        The noun gauge and the individual gauge stay put. The clocks hold
        the number and the kind.
        """
        self._keep_previous()
        self.counted = self.word_value[count]
        self.working = self.kind_step(self.acquired_value[noun])
        return self.counted, self.working

    def it_remainder(self, stated: str) -> float:
        """Kind clock minus the stated gauge. An empty clock drops out."""
        return self.consensus_quantity(
            self.working,
            self.content_value[stated],
            self.routed_sign("-"),
        )

    def predict_it(self, stated: str) -> str:
        return self.remainder_answer(self.it_remainder(stated))

    def it_count_remainder(self, count: str) -> float:
        """Count clock minus the number-word gauge. An empty clock drops out."""
        return self.consensus_quantity(
            self.counted,
            self.word_value[count],
            self.routed_sign("-"),
        )

    def predict_count(self, count: str) -> str:
        return self.remainder_answer(self.it_count_remainder(count))

    def it_kept_remainder(self, stated: str) -> float:
        """The closer of the latest kind and the kind kept from the sentence before."""
        latest = self.it_remainder(stated)
        previous = self.consensus_quantity(
            self.earlier,
            self.content_value[stated],
            self.routed_sign("-"),
        )
        if abs(previous) < abs(latest):
            return previous
        return latest

    def it_surviving_remainder(self, stated: str) -> float:
        """The closer of the latest kind and the two kinds kept before it."""
        best = self.it_kept_remainder(stated)
        older = self.consensus_quantity(
            self.earliest,
            self.content_value[stated],
            self.routed_sign("-"),
        )
        if abs(older) < abs(best):
            return older
        return best

    def it_open_remainder(self, stated: str) -> float:
        """The closer of the fact read and the kind kept from the opening sentence."""
        best = self.it_surviving_remainder(stated)
        older = self.consensus_quantity(
            self.opening,
            self.content_value[stated],
            self.routed_sign("-"),
        )
        if abs(older) < abs(best):
            return older
        return best

    def predict_kept(self, stated: str) -> str:
        return self.remainder_answer(self.it_kept_remainder(stated))

    def it_kept_count_remainder(self, count: str) -> float:
        """The closer of the latest count and the count kept from the sentence before."""
        latest = self.it_count_remainder(count)
        previous = self.consensus_quantity(
            self.earlier_count,
            self.word_value[count],
            self.routed_sign("-"),
        )
        if abs(previous) < abs(latest):
            return previous
        return latest

    def predict_kept_count(self, count: str) -> str:
        return self.remainder_answer(self.it_kept_count_remainder(count))

    def acquired_remainder(self, noun: str, stated: str) -> float:
        """One kind step of a taught noun, minus the stated content gauge."""
        vessel = self.kind_step(self.acquired_value[noun])
        return self.consensus_quantity(
            vessel,
            self.content_value[stated],
            self.routed_sign("-"),
        )

    def predict_acquired(self, noun: str, stated: str) -> str:
        return self.remainder_answer(self.acquired_remainder(noun, stated))

    def acquired_pair_remainder(self, left: str, right: str) -> float:
        """Difference of two taught gauges. Same kind agrees."""
        return self.consensus_quantity(
            self.acquired_value[left],
            self.acquired_value[right],
            self.routed_sign("-"),
        )

    def predict_acquired_pair(self, left: str, right: str) -> str:
        return self.remainder_answer(self.acquired_pair_remainder(left, right))

    def acquired_self_remainder(self, left: str, right: str) -> float:
        """Difference of two individual gauges. The same noun cancels."""
        return self.consensus_quantity(
            self.acquired_self[left],
            self.acquired_self[right],
            self.routed_sign("-"),
        )

    def predict_acquired_self(self, left: str, right: str) -> str:
        return self.remainder_answer(self.acquired_self_remainder(left, right))

    def deposit_acquired(self, noun: str) -> float:
        """Leave the kind this taught noun steps to. The noun gauge stays put."""
        self._keep_previous()
        self.working = self.kind_step(self.acquired_value[noun])
        self.counted = 0.0
        return self.working

    def order_answer(self, remainder: float) -> str:
        """Same when the remainder collapses. The sign names less or more."""
        if self.remainder_answer(remainder) == "yes":
            return "same"
        if remainder > 0:
            return "more"
        return "less"

    def predict_order(self, claim: Claim, surface: str) -> str:
        return self.order_answer(self.equality_remainder(claim, surface))

    def _gauge(self, digit: int, surface: str) -> float:
        if surface == "digit":
            return self.digit_value[str(digit)]
        return self.word_value[WORD_OF[digit]]

    def fold_steps(self, left: float, steps: list[tuple[int, float]]) -> float:
        """Each step is (sign, right-hand gauge) applied to the running vessel.

        The compiled attend folds the chain in one call. A miss keeps this loop.
        """
        from fsot_lattice import fast_attend

        fast_attend.ensure(self)
        if fast_attend.enabled() and steps:
            got = fast_attend.fold_steps(left, steps)
            if got is not None:
                return got
        acc = left
        for sign, right in steps:
            acc = self.consensus_quantity(acc, right, sign)
        return acc

    def product_steps(self, row: Product, surface: str) -> tuple[float, list[tuple[int, float]]]:
        addend = self._gauge(row.addend, surface)
        return 0.0, [(1, addend) for _ in range(row.count)]

    def product_quantity(self, row: Product, surface: str) -> float:
        left, steps = self.product_steps(row, surface)
        return self.fold_steps(left, steps)

    def algebraic_product(self, row: Product, surface: str) -> float:
        return row.count * self._gauge(row.addend, surface)

    def predict_product(self, row: Product, surface: str) -> str:
        """Name the product through the place reader, including a result past nine."""
        named = self.read_place(self.product_quantity(row, surface), surface)[0]
        if surface == "digit":
            return decimal_name(named)
        return number_name(named)

    def span_steps(self, span: Span, surface: str) -> tuple[float, list[tuple[int, float]]]:
        steps = [
            (self.routed_sign(span.op1), self._gauge(span.b, surface)),
            (self.routed_sign(span.op2), self._gauge(span.c, surface)),
            (self.routed_sign(span.op3), self._gauge(span.d, surface)),
        ]
        return self._gauge(span.a, surface), steps

    def span_quantity(self, span: Span, surface: str) -> float:
        left, steps = self.span_steps(span, surface)
        return self.fold_steps(left, steps)

    def algebraic_span(self, span: Span, surface: str) -> float:
        left, steps = self.span_steps(span, surface)
        total = left
        for sign, right in steps:
            total += sign * right
        return total

    def predict_span(self, span: Span, surface: str) -> str:
        pred = self.span_quantity(span, surface)
        if surface == "digit":
            return _nearest(pred, self.digit_value, DIGITS)
        return _nearest(pred, self.word_value, [WORD_OF[i] for i in range(10)])

    def gauge_line(self, surface: str) -> list[float]:
        return [self._gauge(digit, surface) for digit in range(len(DIGITS))]

    def ten_quantity(self, surface: str) -> float:
        """The place past nine: consensus of the nine gauge and the one gauge."""
        return self.consensus_quantity(
            self._gauge(9, surface),
            self._gauge(1, surface),
            self.routed_sign("+"),
        )

    def hundred_quantity(self, surface: str) -> float:
        """The place past ninety-nine: the ten-step counted ten times."""
        ten = self.ten_quantity(surface)
        steps = [(self.routed_sign("+"), ten) for _ in range(len(DIGITS))]
        return self.fold_steps(0.0, steps)

    def thousand_quantity(self, surface: str) -> float:
        """The place past nine hundred ninety-nine: the hundred-step counted ten times."""
        hundred = self.hundred_quantity(surface)
        steps = [(self.routed_sign("+"), hundred) for _ in range(len(DIGITS))]
        return self.fold_steps(0.0, steps)

    def ten_thousand_quantity(self, surface: str) -> float:
        """The place past 9999: the thousand-step counted ten times."""
        thousand = self.thousand_quantity(surface)
        steps = [(self.routed_sign("+"), thousand) for _ in range(len(DIGITS))]
        return self.fold_steps(0.0, steps)

    def hundred_thousand_quantity(self, surface: str) -> float:
        """The place past 99999: the ten-thousand step counted ten times."""
        ten_thousand = self.ten_thousand_quantity(surface)
        steps = [(self.routed_sign("+"), ten_thousand) for _ in range(len(DIGITS))]
        return self.fold_steps(0.0, steps)

    def million_quantity(self, surface: str) -> float:
        """The place past 999999: the hundred-thousand step counted ten times."""
        hundred_thousand = self.hundred_thousand_quantity(surface)
        steps = [(self.routed_sign("+"), hundred_thousand) for _ in range(len(DIGITS))]
        return self.fold_steps(0.0, steps)

    def place_steps(self, surface: str) -> list[float]:
        """Index is the power of ten. Zero is unused. Twelve is a trillion.

        Each rung is the ten-step counted 10^(exp-1) times. One product is
        that count. The shed still removes one step at a time.
        """
        ten = self.ten_quantity(surface)
        steps = [0.0]
        for exp in range(1, 13):
            steps.append(float(10 ** (exp - 1)) * ten)
        return steps

    def _rungs_above(self, million: float) -> tuple[float, float, float, float, float, float]:
        """Ten million through a trillion, each the previous rung counted ten times."""
        plus = self.routed_sign("+")
        width = len(DIGITS)
        step = million
        rungs: list[float] = []
        for _ in range(6):
            step = self.fold_steps(0.0, [(plus, step) for _ in range(width)])
            rungs.append(step)
        return (rungs[0], rungs[1], rungs[2], rungs[3], rungs[4], rungs[5])

    def _shed(self, quantity: float, step: float) -> tuple[int, float]:
        """Count steps that leave a remainder above the negative drop.

        The cap is the largest digit this lexicon can name in that place.
        """
        drop = COLLAPSE / AMP
        rem = quantity
        count = 0
        if step == 0.0:
            return 0, quantity
        while count < len(DIGITS) - 1:
            nxt = self.consensus_quantity(rem, step, self.routed_sign("-"))
            if nxt < -drop:
                break
            rem = nxt
            count += 1
        return count, rem

    def shed_place(self, quantity: float, ten: float) -> tuple[int, float]:
        """Count how many ten-steps leave a remainder above the negative drop."""
        return self._shed(quantity, ten)

    def shed_hundred(self, quantity: float, hundred: float) -> tuple[int, float]:
        """Count how many hundred-steps leave a remainder above the negative drop."""
        return self._shed(quantity, hundred)

    def shed_thousand(self, quantity: float, thousand: float) -> tuple[int, float]:
        """Count how many thousand-steps leave a remainder above the negative drop."""
        return self._shed(quantity, thousand)

    def shed_ten_thousand(self, quantity: float, ten_thousand: float) -> tuple[int, float]:
        """Count how many ten-thousand-steps leave a remainder above the negative drop."""
        return self._shed(quantity, ten_thousand)

    def shed_hundred_thousand(self, quantity: float, hundred_thousand: float) -> tuple[int, float]:
        """Count how many hundred-thousand-steps leave a remainder above the negative drop."""
        return self._shed(quantity, hundred_thousand)

    def shed_million(self, quantity: float, million: float) -> tuple[int, float]:
        """Count how many million-steps leave a remainder above the negative drop."""
        return self._shed(quantity, million)

    def read_place(
        self,
        quantity: float,
        surface: str,
        ten: float | None = None,
        gauges: list[float] | None = None,
        hundred: float | None = None,
        thousand: float | None = None,
        ten_thousand: float | None = None,
        hundred_thousand: float | None = None,
        million: float | None = None,
        above: tuple[float, float, float, float, float, float] | None = None,
    ) -> tuple[int, float, float, float]:
        """Integer, units remainder, runner-up margin, distance to the units gauge.

        A caller can pass a gauge line. The default line is the surface being read.
        A passed ten without a hundred uses that ten counted ten times, so a
        synthetic line does not shed against the lattice's own hundred.
        A passed hundred without a thousand uses that hundred counted ten times,
        so a synthetic line does not shed against the lattice's own thousand.
        A passed thousand without a ten-thousand uses that thousand counted ten
        times, so a synthetic line does not shed against the lattice's own
        ten-thousand. A passed ten-thousand without a hundred-thousand uses
        that ten-thousand counted ten times. A passed hundred-thousand
        without a million uses that hundred-thousand counted ten times.
        A passed lower step without the six rungs above the million uses the
        passed step counted ten times, six times. A bare read folds those rungs.
        The trillion shed runs first.
        """
        passed_ten = ten is not None
        passed_hundred = hundred is not None
        passed_thousand = thousand is not None
        passed_ten_thousand = ten_thousand is not None
        passed_hundred_thousand = hundred_thousand is not None
        passed_million = million is not None
        if ten is None:
            ten = self.ten_quantity(surface)
        if gauges is None:
            gauges = self.gauge_line(surface)
        if hundred is None:
            hundred = ten * len(gauges) if passed_ten else self.hundred_quantity(surface)
        if thousand is None:
            if passed_hundred or passed_ten:
                thousand = hundred * len(gauges)
            else:
                thousand = self.thousand_quantity(surface)
        if ten_thousand is None:
            if passed_thousand or passed_hundred or passed_ten:
                ten_thousand = thousand * len(gauges)
            else:
                ten_thousand = self.ten_thousand_quantity(surface)
        if hundred_thousand is None:
            if passed_ten_thousand or passed_thousand or passed_hundred or passed_ten:
                hundred_thousand = ten_thousand * len(gauges)
            else:
                hundred_thousand = self.hundred_thousand_quantity(surface)
        if million is None:
            if (
                passed_hundred_thousand
                or passed_ten_thousand
                or passed_thousand
                or passed_hundred
                or passed_ten
            ):
                million = hundred_thousand * len(gauges)
            else:
                million = self.million_quantity(surface)
        if above is None:
            if (
                passed_million
                or passed_hundred_thousand
                or passed_ten_thousand
                or passed_thousand
                or passed_hundred
                or passed_ten
            ):
                step = million
                span = len(gauges)
                built: list[float] = []
                for _ in range(6):
                    step = step * span
                    built.append(step)
                above = (built[0], built[1], built[2], built[3], built[4], built[5])
            else:
                above = self._rungs_above(million)
        elif len(above) != 6:
            raise RuntimeError("the place read wants six rungs above the million")
        from fsot_lattice import fast_attend

        fast_attend.ensure(self)
        if fast_attend.enabled():
            got = fast_attend.read_place(
                quantity,
                gauges,
                ten,
                hundred,
                thousand,
                ten_thousand,
                hundred_thousand,
                million,
                above,
            )
            if got is not None:
                return got
        return self._read_place_py(
            quantity, gauges, ten, hundred, thousand, ten_thousand, hundred_thousand, million, above
        )

    def _read_place_py(
        self,
        quantity: float,
        gauges: list[float],
        ten: float,
        hundred: float,
        thousand: float,
        ten_thousand: float,
        hundred_thousand: float,
        million: float,
        above: tuple[float, float, float, float, float, float],
    ) -> tuple[int, float, float, float]:
        """The same sheds as the compiled read, through the Python attend."""
        ten_million, hundred_million, billion, ten_billion, hundred_billion, trillion = above
        trillions, quantity = self._shed(quantity, trillion)
        hundred_billions, quantity = self._shed(quantity, hundred_billion)
        ten_billions, quantity = self._shed(quantity, ten_billion)
        billions, quantity = self._shed(quantity, billion)
        hundred_millions, quantity = self._shed(quantity, hundred_million)
        ten_millions, quantity = self._shed(quantity, ten_million)
        millions, quantity = self.shed_million(quantity, million)
        hundred_thousands, quantity = self.shed_hundred_thousand(quantity, hundred_thousand)
        ten_thousands, quantity = self.shed_ten_thousand(quantity, ten_thousand)
        thousands, quantity = self.shed_thousand(quantity, thousand)
        hundreds, quantity = self.shed_hundred(quantity, hundred)
        tens, rem = self.shed_place(quantity, ten)
        units, margin, dist = _nearest_detail(rem, gauges)
        span = len(gauges)
        named = 0
        for count in (
            trillions,
            hundred_billions,
            ten_billions,
            billions,
            hundred_millions,
            ten_millions,
            millions,
            hundred_thousands,
            ten_thousands,
            thousands,
            hundreds,
            tens,
            units,
        ):
            named = named * span + count
        return named, rem, margin, dist

    def lexeme_quantity(self, row: Lexeme, surface: str) -> float:
        if row.kind == "sum":
            return self.consensus_quantity(
                self._gauge(row.left, surface),
                self._gauge(row.right, surface),
                self.routed_sign("+"),
            )
        wide = Product(row.left, row.right, row.result)
        return self.product_quantity(wide, surface)

    def algebraic_lexeme(self, row: Lexeme, surface: str) -> float:
        if row.kind == "sum":
            return self._gauge(row.left, surface) + self._gauge(row.right, surface)
        return row.left * self._gauge(row.right, surface)

    def predict_lexeme(self, row: Lexeme, surface: str) -> str:
        named = self.read_place(self.lexeme_quantity(row, surface), surface)[0]
        if surface == "digit":
            return decimal_name(named)
        return number_name(named)

    def compose_place(self, place: int, ten: float, gauges: list[float]) -> float:
        """Fold the ten-step `tens` times, then add the units gauge when it is present."""
        tens, units = divmod(place, len(gauges))
        acc = self.fold_steps(0.0, [(self.routed_sign("+"), ten) for _ in range(tens)])
        if units:
            acc = self.consensus_quantity(acc, gauges[units], self.routed_sign("+"))
        return acc

    def place_quantity(self, place: int, surface: str) -> float:
        return self.compose_place(place, self.ten_quantity(surface), self.gauge_line(surface))

    def algebraic_place(self, place: int, surface: str) -> float:
        tens, units = divmod(place, len(DIGITS))
        total = tens * (self._gauge(9, surface) + self._gauge(1, surface))
        if units:
            total += self._gauge(units, surface)
        return total

    def place_step_quantity(self, row: PlaceStep, surface: str) -> float:
        """The rebuilt name, then the digit. A zero digit enters and drops."""
        base = self.place_quantity(row.place, surface)
        return self.consensus_quantity(
            base,
            self._gauge(row.digit, surface),
            self.routed_sign(row.op),
        )

    def algebraic_place_step(self, row: PlaceStep, surface: str) -> float:
        sign = 1.0 if row.op == "+" else -1.0
        return self.algebraic_place(row.place, surface) + sign * self._gauge(row.digit, surface)

    def predict_place_step(self, row: PlaceStep, surface: str) -> str:
        named = self.read_place(self.place_step_quantity(row, surface), surface)[0]
        if named < 0 or named >= len(DIGITS) * len(DIGITS):
            return "?"
        if surface == "digit":
            return decimal_name(named)
        return number_name(named)

    def place_pair_quantity(self, row: PlacePair, surface: str) -> float:
        """Both names are rebuilt, then one consensus pass adds or subtracts them."""
        left = self.place_quantity(row.left, surface)
        right = self.place_quantity(row.right, surface)
        return self.consensus_quantity(left, right, self.routed_sign(row.op))

    def algebraic_place_pair(self, row: PlacePair, surface: str) -> float:
        sign = 1.0 if row.op == "+" else -1.0
        return self.algebraic_place(row.left, surface) + sign * self.algebraic_place(row.right, surface)

    def predict_place_pair(self, row: PlacePair, surface: str) -> str:
        named = self.read_place(self.place_pair_quantity(row, surface), surface)[0]
        if named < 0 or named >= len(DIGITS) * len(DIGITS):
            return "?"
        if surface == "digit":
            return decimal_name(named)
        return number_name(named)

    def hundred_sum_quantity(self, row: HundredSum, surface: str) -> float:
        """Both operands are rebuilt. One consensus pass adds them."""
        left = self.place_quantity(row.left, surface)
        if row.kind == "step":
            right = self._gauge(row.right, surface)
        else:
            right = self.place_quantity(row.right, surface)
        return self.consensus_quantity(left, right, self.routed_sign("+"))

    def algebraic_hundred_sum(self, row: HundredSum, surface: str) -> float:
        left = self.algebraic_place(row.left, surface)
        if row.kind == "step":
            right = self._gauge(row.right, surface)
        else:
            right = self.algebraic_place(row.right, surface)
        return left + right

    def predict_hundred(self, row: HundredSum, surface: str) -> str:
        named = self.read_place(self.hundred_sum_quantity(row, surface), surface)[0]
        if named < 0 or named >= len(DIGITS) ** 3:
            return "?"
        if surface == "digit":
            return decimal_name(named)
        return number_name(named)

    def compose_hundred(self, number: int, hundred: float, ten: float, gauges: list[float]) -> float:
        """Fold the hundred-step `hundreds` times, then add the lower place when it is present."""
        span = len(gauges)
        hundreds, rest = divmod(number, span * span)
        acc = self.fold_steps(0.0, [(self.routed_sign("+"), hundred) for _ in range(hundreds)])
        if rest:
            acc = self.consensus_quantity(
                acc,
                self.compose_place(rest, ten, gauges),
                self.routed_sign("+"),
            )
        return acc

    def algebraic_hundred(self, number: int, surface: str) -> float:
        span = len(DIGITS)
        hundreds, rest = divmod(number, span * span)
        total = hundreds * span * (self._gauge(9, surface) + self._gauge(1, surface))
        if rest:
            total += self.algebraic_place(rest, surface)
        return total

    def hundred_op_quantity(self, row: HundredOp, surface: str) -> float:
        """The rebuilt hundred, then a digit or a rebuilt place. One consensus pass."""
        hundred = self.hundred_quantity(surface)
        ten = self.ten_quantity(surface)
        gauges = self.gauge_line(surface)
        left = self.compose_hundred(row.left, hundred, ten, gauges)
        if row.right < len(DIGITS):
            right = self._gauge(row.right, surface)
        else:
            right = self.compose_place(row.right, ten, gauges)
        return self.consensus_quantity(left, right, self.routed_sign(row.op))

    def algebraic_hundred_op(self, row: HundredOp, surface: str) -> float:
        sign = 1.0 if row.op == "+" else -1.0
        left = self.algebraic_hundred(row.left, surface)
        if row.right < len(DIGITS):
            right = self._gauge(row.right, surface)
        else:
            right = self.algebraic_place(row.right, surface)
        return left + sign * right

    def predict_hundred_op(self, row: HundredOp, surface: str) -> str:
        named = self.read_place(self.hundred_op_quantity(row, surface), surface)[0]
        if named < 0 or named >= len(DIGITS) ** 3:
            return "?"
        if surface == "digit":
            return decimal_name(named)
        return number_name(named)

    def thousand_sum_quantity(self, row: ThousandSum, surface: str) -> float:
        """The rebuilt hundred name, then a digit or a rebuilt place. One consensus pass."""
        hundred = self.hundred_quantity(surface)
        ten = self.ten_quantity(surface)
        gauges = self.gauge_line(surface)
        left = self.compose_hundred(row.left, hundred, ten, gauges)
        if row.kind == "step":
            right = self._gauge(row.right, surface)
        else:
            right = self.compose_place(row.right, ten, gauges)
        return self.consensus_quantity(left, right, self.routed_sign("+"))

    def algebraic_thousand_sum(self, row: ThousandSum, surface: str) -> float:
        left = self.algebraic_hundred(row.left, surface)
        if row.kind == "step":
            right = self._gauge(row.right, surface)
        else:
            right = self.algebraic_place(row.right, surface)
        return left + right

    def predict_thousand(self, row: ThousandSum, surface: str) -> str:
        named = self.read_place(self.thousand_sum_quantity(row, surface), surface)[0]
        if named < 0 or named >= len(DIGITS) ** 4:
            return "?"
        if surface == "digit":
            return decimal_name(named)
        return number_name(named)

    def compose_thousand(
        self,
        number: int,
        thousand: float,
        hundred: float,
        ten: float,
        gauges: list[float],
    ) -> float:
        """Fold the thousand-step `thousands` times, then add the lower name when it is present."""
        block = len(gauges) ** 3
        thousands, rest = divmod(number, block)
        acc = self.fold_steps(0.0, [(self.routed_sign("+"), thousand) for _ in range(thousands)])
        if rest:
            acc = self.consensus_quantity(
                acc,
                self.compose_hundred(rest, hundred, ten, gauges),
                self.routed_sign("+"),
            )
        return acc

    def algebraic_thousand(self, number: int, surface: str) -> float:
        span = len(DIGITS)
        thousands, rest = divmod(number, span ** 3)
        total = thousands * span * span * (self._gauge(9, surface) + self._gauge(1, surface))
        if rest:
            total += self.algebraic_hundred(rest, surface)
        return total

    def thousand_op_quantity(self, row: ThousandOp, surface: str) -> float:
        """The rebuilt thousand, then a name in 0..999. One consensus pass."""
        thousand = self.thousand_quantity(surface)
        hundred = self.hundred_quantity(surface)
        ten = self.ten_quantity(surface)
        gauges = self.gauge_line(surface)
        left = self.compose_thousand(row.left, thousand, hundred, ten, gauges)
        radix = len(DIGITS)
        if row.right < radix:
            right = self._gauge(row.right, surface)
        elif row.right < radix * radix:
            right = self.compose_place(row.right, ten, gauges)
        else:
            right = self.compose_hundred(row.right, hundred, ten, gauges)
        return self.consensus_quantity(left, right, self.routed_sign(row.op))

    def algebraic_thousand_op(self, row: ThousandOp, surface: str) -> float:
        sign = 1.0 if row.op == "+" else -1.0
        left = self.algebraic_thousand(row.left, surface)
        radix = len(DIGITS)
        if row.right < radix:
            right = self._gauge(row.right, surface)
        elif row.right < radix * radix:
            right = self.algebraic_place(row.right, surface)
        else:
            right = self.algebraic_hundred(row.right, surface)
        return left + sign * right

    def predict_thousand_op(self, row: ThousandOp, surface: str) -> str:
        named = self.read_place(self.thousand_op_quantity(row, surface), surface)[0]
        if named < 0 or named >= len(DIGITS) ** 4:
            return "?"
        if surface == "digit":
            return decimal_name(named)
        return number_name(named)

    def thousand_pair_quantity(self, row: ThousandPair, surface: str) -> float:
        """Both thousand names are rebuilt, then one consensus pass adds or subtracts them."""
        thousand = self.thousand_quantity(surface)
        hundred = self.hundred_quantity(surface)
        ten = self.ten_quantity(surface)
        gauges = self.gauge_line(surface)
        left = self.compose_thousand(row.left, thousand, hundred, ten, gauges)
        right = self.compose_thousand(row.right, thousand, hundred, ten, gauges)
        return self.consensus_quantity(left, right, self.routed_sign(row.op))

    def algebraic_thousand_pair(self, row: ThousandPair, surface: str) -> float:
        sign = 1.0 if row.op == "+" else -1.0
        return self.algebraic_thousand(row.left, surface) + sign * self.algebraic_thousand(
            row.right, surface
        )

    def predict_thousand_pair(self, row: ThousandPair, surface: str) -> str:
        named = self.read_place(self.thousand_pair_quantity(row, surface), surface)[0]
        if named < 0 or named >= len(DIGITS) ** 4:
            return "?"
        if surface == "digit":
            return decimal_name(named)
        return number_name(named)

    def ten_thousand_sum_quantity(self, row: TenThousandSum, surface: str) -> float:
        """The rebuilt thousand name, then a name in 0..999. One consensus pass."""
        thousand = self.thousand_quantity(surface)
        hundred = self.hundred_quantity(surface)
        ten = self.ten_quantity(surface)
        gauges = self.gauge_line(surface)
        left = self.compose_thousand(row.left, thousand, hundred, ten, gauges)
        radix = len(DIGITS)
        if row.right < radix:
            right = self._gauge(row.right, surface)
        elif row.right < radix * radix:
            right = self.compose_place(row.right, ten, gauges)
        else:
            right = self.compose_hundred(row.right, hundred, ten, gauges)
        return self.consensus_quantity(left, right, self.routed_sign("+"))

    def algebraic_ten_thousand_sum(self, row: TenThousandSum, surface: str) -> float:
        left = self.algebraic_thousand(row.left, surface)
        radix = len(DIGITS)
        if row.right < radix:
            right = self._gauge(row.right, surface)
        elif row.right < radix * radix:
            right = self.algebraic_place(row.right, surface)
        else:
            right = self.algebraic_hundred(row.right, surface)
        return left + right

    def predict_ten_thousand(self, row: TenThousandSum, surface: str) -> str:
        named = self.read_place(self.ten_thousand_sum_quantity(row, surface), surface)[0]
        cap = len(DIGITS) ** 4
        if named < cap or named >= cap + len(DIGITS) ** 3:
            return "?"
        if surface == "digit":
            return decimal_name(named)
        return number_name(named)

    def compose_below_ten_thousand(
        self,
        number: int,
        thousand: float,
        hundred: float,
        ten: float,
        gauges: list[float],
    ) -> float:
        """A name in 0..9999. A thousand name uses the thousand fold. A lower name does not."""
        span = len(gauges)
        if number >= span ** 3:
            return self.compose_thousand(number, thousand, hundred, ten, gauges)
        if number >= span * span:
            return self.compose_hundred(number, hundred, ten, gauges)
        if number >= span:
            return self.compose_place(number, ten, gauges)
        return gauges[number]

    def algebraic_below_ten_thousand(self, number: int, surface: str) -> float:
        span = len(DIGITS)
        if number >= span ** 3:
            return self.algebraic_thousand(number, surface)
        if number >= span * span:
            return self.algebraic_hundred(number, surface)
        if number >= span:
            return self.algebraic_place(number, surface)
        return self._gauge(number, surface)

    def compose_ten_thousand(
        self,
        number: int,
        ten_thousand: float,
        thousand: float,
        hundred: float,
        ten: float,
        gauges: list[float],
    ) -> float:
        """Fold the ten-thousand step `count` times, then add the lower name when it is present."""
        block = len(gauges) ** 4
        count, rest = divmod(number, block)
        acc = self.fold_steps(0.0, [(self.routed_sign("+"), ten_thousand) for _ in range(count)])
        if rest:
            acc = self.consensus_quantity(
                acc,
                self.compose_below_ten_thousand(rest, thousand, hundred, ten, gauges),
                self.routed_sign("+"),
            )
        return acc

    def algebraic_ten_thousand_name(self, number: int, surface: str) -> float:
        span = len(DIGITS)
        count, rest = divmod(number, span ** 4)
        total = count * (span ** 3) * (self._gauge(9, surface) + self._gauge(1, surface))
        if rest:
            total += self.algebraic_below_ten_thousand(rest, surface)
        return total

    def ten_thousand_op_quantity(self, row: TenThousandOp, surface: str) -> float:
        """The rebuilt ten-thousand name, then a name in 0..9999. One consensus pass."""
        ten_thousand = self.ten_thousand_quantity(surface)
        thousand = self.thousand_quantity(surface)
        hundred = self.hundred_quantity(surface)
        ten = self.ten_quantity(surface)
        gauges = self.gauge_line(surface)
        left = self.compose_ten_thousand(row.left, ten_thousand, thousand, hundred, ten, gauges)
        right = self.compose_below_ten_thousand(row.right, thousand, hundred, ten, gauges)
        return self.consensus_quantity(left, right, self.routed_sign(row.op))

    def algebraic_ten_thousand_op(self, row: TenThousandOp, surface: str) -> float:
        sign = 1.0 if row.op == "+" else -1.0
        left = self.algebraic_ten_thousand_name(row.left, surface)
        right = self.algebraic_below_ten_thousand(row.right, surface)
        return left + sign * right

    def predict_ten_thousand_op(self, row: TenThousandOp, surface: str) -> str:
        named = self.read_place(self.ten_thousand_op_quantity(row, surface), surface)[0]
        if named < 0 or named >= len(DIGITS) ** 4 + len(DIGITS) ** 3:
            return "?"
        if surface == "digit":
            return decimal_name(named)
        return number_name(named)

    def ten_thousand_pair_quantity(self, row: TenThousandPair, surface: str) -> float:
        """Both ten-thousand names are rebuilt, then one consensus pass subtracts them."""
        ten_thousand = self.ten_thousand_quantity(surface)
        thousand = self.thousand_quantity(surface)
        hundred = self.hundred_quantity(surface)
        ten = self.ten_quantity(surface)
        gauges = self.gauge_line(surface)
        left = self.compose_ten_thousand(row.left, ten_thousand, thousand, hundred, ten, gauges)
        right = self.compose_ten_thousand(row.right, ten_thousand, thousand, hundred, ten, gauges)
        return self.consensus_quantity(left, right, self.routed_sign("-"))

    def algebraic_ten_thousand_pair(self, row: TenThousandPair, surface: str) -> float:
        return self.algebraic_ten_thousand_name(row.left, surface) - self.algebraic_ten_thousand_name(
            row.right, surface
        )

    def predict_ten_thousand_pair(self, row: TenThousandPair, surface: str) -> str:
        named = self.read_place(self.ten_thousand_pair_quantity(row, surface), surface)[0]
        if named < 0 or named >= len(DIGITS) ** 3:
            return "?"
        if surface == "digit":
            return decimal_name(named)
        return number_name(named)

    def hundred_thousand_sum_quantity(self, row: HundredThousandSum, surface: str) -> float:
        """A name in 90001..99999, then a name in 1..9999. One consensus pass."""
        ten_thousand = self.ten_thousand_quantity(surface)
        thousand = self.thousand_quantity(surface)
        hundred = self.hundred_quantity(surface)
        ten = self.ten_quantity(surface)
        gauges = self.gauge_line(surface)
        left = self.compose_ten_thousand(row.left, ten_thousand, thousand, hundred, ten, gauges)
        right = self.compose_below_ten_thousand(row.right, thousand, hundred, ten, gauges)
        return self.consensus_quantity(left, right, self.routed_sign("+"))

    def algebraic_hundred_thousand_sum(self, row: HundredThousandSum, surface: str) -> float:
        left = self.algebraic_ten_thousand_name(row.left, surface)
        right = self.algebraic_below_ten_thousand(row.right, surface)
        return left + right

    def predict_hundred_thousand(self, row: HundredThousandSum, surface: str) -> str:
        named = self.read_place(self.hundred_thousand_sum_quantity(row, surface), surface)[0]
        start = len(DIGITS) ** 5
        if named < start or named >= start + len(DIGITS) ** 4:
            return "?"
        if surface == "digit":
            return decimal_name(named)
        return number_name(named)

    def compose_below_million(
        self,
        number: int,
        ten_thousand: float,
        thousand: float,
        hundred: float,
        ten: float,
        gauges: list[float],
    ) -> float:
        """A name in 0..99999. A ten-thousand name uses that fold. A lower name does not."""
        if number >= len(gauges) ** 4:
            return self.compose_ten_thousand(number, ten_thousand, thousand, hundred, ten, gauges)
        return self.compose_below_ten_thousand(number, thousand, hundred, ten, gauges)

    def algebraic_below_million(self, number: int, surface: str) -> float:
        if number >= len(DIGITS) ** 4:
            return self.algebraic_ten_thousand_name(number, surface)
        return self.algebraic_below_ten_thousand(number, surface)

    def compose_hundred_thousand(
        self,
        number: int,
        hundred_thousand: float,
        ten_thousand: float,
        thousand: float,
        hundred: float,
        ten: float,
        gauges: list[float],
    ) -> float:
        """Fold the hundred-thousand step `count` times, then add the lower name when it is present."""
        block = len(gauges) ** 5
        count, rest = divmod(number, block)
        acc = self.fold_steps(0.0, [(self.routed_sign("+"), hundred_thousand) for _ in range(count)])
        if rest:
            acc = self.consensus_quantity(
                acc,
                self.compose_below_million(rest, ten_thousand, thousand, hundred, ten, gauges),
                self.routed_sign("+"),
            )
        return acc

    def algebraic_hundred_thousand_name(self, number: int, surface: str) -> float:
        span = len(DIGITS)
        count, rest = divmod(number, span ** 5)
        total = count * (span ** 4) * (self._gauge(9, surface) + self._gauge(1, surface))
        if rest:
            total += self.algebraic_below_million(rest, surface)
        return total

    def million_sum_quantity(self, row: MillionSum, surface: str) -> float:
        """A name in 900001..999999, then a name in 1..99999. One consensus pass."""
        hundred_thousand = self.hundred_thousand_quantity(surface)
        ten_thousand = self.ten_thousand_quantity(surface)
        thousand = self.thousand_quantity(surface)
        hundred = self.hundred_quantity(surface)
        ten = self.ten_quantity(surface)
        gauges = self.gauge_line(surface)
        left = self.compose_hundred_thousand(
            row.left, hundred_thousand, ten_thousand, thousand, hundred, ten, gauges
        )
        right = self.compose_below_million(row.right, ten_thousand, thousand, hundred, ten, gauges)
        return self.consensus_quantity(left, right, self.routed_sign("+"))

    def algebraic_million_sum(self, row: MillionSum, surface: str) -> float:
        return self.algebraic_hundred_thousand_name(row.left, surface) + self.algebraic_below_million(
            row.right, surface
        )

    def predict_million(self, row: MillionSum, surface: str) -> str:
        named = self.read_place(self.million_sum_quantity(row, surface), surface)[0]
        start = len(DIGITS) ** 6
        if named < start or named >= start + len(DIGITS) ** 5:
            return "?"
        if surface == "digit":
            return decimal_name(named)
        return number_name(named)

    def compose_integer(self, number: int, steps: list[float], gauges: list[float]) -> float:
        """Each place is that step counted by its digit, then one consensus add.

        The product is the same count the algebraic sum writes. Units add
        that gauge when the digit is present.
        """
        if number < 0:
            raise RuntimeError("a place name was below zero")
        if len(steps) < 2:
            raise RuntimeError("a place name needs the ten-step")
        plus = self.routed_sign("+")
        ten = steps[1]
        acc = 0.0
        rest = number
        for exp in range(len(steps) - 1, 0, -1):
            place = 10 ** exp
            digit = rest // place
            if digit:
                rest -= digit * place
                counted = (digit * 10 ** (exp - 1)) * ten
                acc = self.consensus_quantity(acc, counted, plus)
        if rest:
            acc = self.consensus_quantity(acc, gauges[rest], plus)
        return acc

    def algebraic_integer(self, number: int, surface: str) -> float:
        """Same places as compose_integer, as the ten-step times the digit."""
        ten = self._gauge(9, surface) + self._gauge(1, surface)
        total = 0.0
        rest = number
        for exp in range(12, 0, -1):
            place = 10 ** exp
            digit = rest // place
            if digit:
                rest -= digit * place
                total += digit * (10 ** (exp - 1)) * ten
        if rest:
            total += self._gauge(rest, surface)
        return total

    def predict_power_sum(self, left: int, right: int, surface: str) -> str:
        """One consensus of two place names, read through the trillion shed."""
        steps = self.place_steps(surface)
        gauges = self.gauge_line(surface)
        left_q = self.compose_integer(left, steps, gauges)
        right_q = self.compose_integer(right, steps, gauges)
        quantity = self.consensus_quantity(left_q, right_q, self.routed_sign("+"))
        above = (steps[7], steps[8], steps[9], steps[10], steps[11], steps[12])
        named = self.read_place(
            quantity,
            surface,
            steps[1],
            gauges,
            steps[2],
            steps[3],
            steps[4],
            steps[5],
            steps[6],
            above,
        )[0]
        if surface == "digit":
            return decimal_name(named)
        return number_name(named)

    def _step_values(self, chain: Chain, surface: str) -> tuple[float, float, float, int, int]:
        sign1 = self.routed_sign(chain.op1)
        sign2 = self.routed_sign(chain.op2)
        if surface == "digit":
            left = self.digit_value[str(chain.a)]
            mid = self.digit_value[str(chain.b)]
            right = self.digit_value[str(chain.c)]
        else:
            left = self.word_value[WORD_OF[chain.a]]
            mid = self.word_value[WORD_OF[chain.b]]
            right = self.word_value[WORD_OF[chain.c]]
        return left, mid, right, sign1, sign2

    def compose_quantity(self, chain: Chain, surface: str) -> float:
        """Second consensus pass reads the first vessel as its left operand."""
        left, mid, right, sign1, sign2 = self._step_values(chain, surface)
        vessel = self.consensus_quantity(left, mid, sign1)
        return self.consensus_quantity(vessel, right, sign2)

    def algebraic_chain(self, chain: Chain, surface: str) -> float:
        left, mid, right, sign1, sign2 = self._step_values(chain, surface)
        return (left + sign1 * mid) + sign2 * right

    def predict_chain(self, chain: Chain, surface: str) -> str:
        pred = self.compose_quantity(chain, surface)
        if surface == "digit":
            return _nearest(pred, self.digit_value, DIGITS)
        return _nearest(pred, self.word_value, [WORD_OF[i] for i in range(10)])

    def hebbian(self, symbols: list[str], lr: float) -> None:
        acc = [0.0] * self.width
        for sym in symbols:
            mask = self.masks[sym]
            for i, bit in enumerate(mask):
                acc[i] += bit
        fired = [0.0] * self.width
        for i, total in enumerate(acc):
            if total > 0:
                fired[i] = 1.0
            elif total < 0:
                fired[i] = -1.0
        hot = [i for i, bit in enumerate(fired) if bit != 0.0]
        for i in hot:
            for j in hot:
                if i == j:
                    continue
                self.bonds[i][j] += lr * fired[i] * fired[j]

    def consolidate(self, rows: list[Triple], lr: float) -> dict[str, float]:
        """Replay digit phrases, decay bonds by poof, drop the weak tail, freeze gauges."""
        for row in rows:
            symbols = [str(row.a), row.op, str(row.b), "=", str(row.c)]
            self.hebbian(symbols, lr)
        peak = 0.0
        for row_b in self.bonds:
            for weight in row_b:
                peak = max(peak, abs(weight))
        cut = peak * POOF
        kept = 0
        for i in range(self.width):
            for j in range(self.width):
                self.bonds[i][j] *= 1.0 - POOF
                if abs(self.bonds[i][j]) <= cut * (1.0 - POOF):
                    self.bonds[i][j] = 0.0
                elif self.bonds[i][j] != 0.0:
                    kept += 1
        self.frozen_digits = dict(self.digit_value)
        return {"peak_before_decay": peak, "cut": cut * (1.0 - POOF), "kept_bonds": float(kept)}

    def pathway_confidence(self, symbols: list[str]) -> float:
        """Fraction of firing units that still have a surviving bond to another firing unit."""
        acc = [0.0] * self.width
        for sym in symbols:
            for i, bit in enumerate(self.masks[sym]):
                acc[i] += bit
        hot = [i for i, total in enumerate(acc) if total != 0.0]
        if len(hot) < 2:
            return 0.0
        supported = 0
        for i in hot:
            if any(self.bonds[i][j] != 0.0 for j in hot if j != i):
                supported += 1
        return supported / len(hot)

    def learned_parameter_count(self) -> int:
        """Digit gauges, number-word gauges, and surviving bonds. Masks and carriers are priors."""
        bonds = sum(1 for row in self.bonds for weight in row if weight != 0.0)
        return (
            len(self.digit_value)
            + len(WORD_OF)
            + len(self.content_value)
            + len(self.acquired_value)
            + len(self.acquired_self)
            + bonds
        )

    def snapshot_digits(self) -> dict[str, float]:
        return dict(self.digit_value)


def _norm(vec: list[float]) -> float:
    return math.sqrt(sum(v * v for v in vec))


def _operand_values(lattice: Lattice, row: Triple, surface: str) -> tuple[float, float]:
    if surface == "digit":
        return lattice.digit_value[str(row.a)], lattice.digit_value[str(row.b)]
    return lattice.word_value[WORD_OF[row.a]], lattice.word_value[WORD_OF[row.b]]


def read_gap(lattice: Lattice, rows: list[Triple], surface: str) -> float:
    """Mean |consensus quantity − algebraic sum| on these rows.

    This is how far the consensus state sits from v[a] ± v[b]. The answer
    token is not part of the read.
    """
    if not rows:
        return 0.0
    err = 0.0
    for row in rows:
        sign = lattice.routed_sign(row.op)
        left, right = _operand_values(lattice, row, surface)
        got = lattice.consensus_quantity(left, right, sign)
        err += abs(got - (left + sign * right))
    return err / len(rows)


def _nearest_detail(pred: float, values: list[float]) -> tuple[int, float, float]:
    """Index of the nearest gauge, margin to the runner-up, and that distance."""
    order = sorted(range(len(values)), key=lambda index: (abs(pred - values[index]), index))
    best = order[0]
    dist = abs(pred - values[best])
    runner = abs(pred - values[order[1]]) if len(order) > 1 else dist
    return best, runner - dist, dist


def _nearest(pred: float, table: dict[str, float], keys: list[str]) -> str:
    best = keys[0]
    best_dist = abs(pred - table[best])
    for key in keys[1:]:
        dist = abs(pred - table[key])
        if dist < best_dist - 1e-15:
            best = key
            best_dist = dist
        elif abs(dist - best_dist) <= 1e-15 and key < best:
            best = key
            best_dist = dist
    return best


def score_digits(lattice: Lattice, rows: list[Triple]) -> tuple[float, int, list[str]]:
    if not rows:
        return 0.0, 0, []
    misses: list[str] = []
    correct = 0
    for row in rows:
        got = lattice.predict_digit(row)
        if got == row.digit_answer():
            correct += 1
        elif len(misses) < 8:
            misses.append(f"{row.digit_prompt()}{row.digit_answer()} -> {got}")
    return correct / len(rows), correct, misses


def score_equality(
    lattice: Lattice, claims: list[Claim], surface: str
) -> tuple[float, int, list[str], dict[str, float]]:
    """Yes/no from the remainder vessel. Margins are absolute remainders."""
    if not claims:
        return 0.0, 0, [], {}
    misses: list[str] = []
    correct = 0
    true_abs: list[float] = []
    false_abs: list[float] = []
    true_hit = 0
    false_hit = 0
    zero_hit = 0
    zero_n = 0
    zero_plus_hit = 0
    zero_plus_n = 0
    for claim in claims:
        got = lattice.predict_equal(claim, surface)
        remainder = abs(lattice.equality_remainder(claim, surface))
        if claim.holds:
            true_abs.append(remainder)
            if got == "yes":
                true_hit += 1
        else:
            false_abs.append(remainder)
            if got == "no":
                false_hit += 1
        if claim.holds and claim.claimed == 0:
            zero_n += 1
            if got == "yes":
                zero_hit += 1
        if claim.holds and claim.a == 0 and claim.op == "+":
            zero_plus_n += 1
            if got == "yes":
                zero_plus_hit += 1
        if got == claim.answer():
            correct += 1
        elif len(misses) < 8:
            shown = claim.digit_prompt() if surface == "digit" else claim.word_prompt()
            misses.append(f"{shown} -> {got} (want {claim.answer()})")
    stats = {
        "true_max_abs": max(true_abs) if true_abs else 0.0,
        "false_min_abs": min(false_abs) if false_abs else 0.0,
        "true_n": float(len(true_abs)),
        "false_n": float(len(false_abs)),
        "true_acc": true_hit / len(true_abs) if true_abs else 0.0,
        "false_acc": false_hit / len(false_abs) if false_abs else 0.0,
        "zero_true_n": float(zero_n),
        "zero_true_acc": zero_hit / zero_n if zero_n else 0.0,
        "zero_plus_true_n": float(zero_plus_n),
        "zero_plus_true_acc": zero_plus_hit / zero_plus_n if zero_plus_n else 0.0,
    }
    return correct / len(claims), correct, misses, stats


def chain_gap(lattice: Lattice, chains: list[Chain], surface: str) -> float:
    """Mean |composed vessel − algebraic two-step sum|."""
    if not chains:
        return 0.0
    err = 0.0
    for chain in chains:
        err += abs(lattice.compose_quantity(chain, surface) - lattice.algebraic_chain(chain, surface))
    return err / len(chains)


def score_chains(
    lattice: Lattice, chains: list[Chain], surface: str
) -> tuple[float, int, list[str], int, int]:
    if not chains:
        return 0.0, 0, [], 0, 0
    misses: list[str] = []
    correct = 0
    zero_hit = 0
    zero_n = 0
    for chain in chains:
        got = lattice.predict_chain(chain, surface)
        want = chain.digit_answer() if surface == "digit" else chain.word_answer()
        if chain.intermediate_is_zero():
            zero_n += 1
            if got == want:
                zero_hit += 1
        if got == want:
            correct += 1
        elif len(misses) < 8:
            if surface == "digit":
                misses.append(f"{chain.digit_prompt()}{want} -> {got}")
            else:
                misses.append(f"{chain.word_prompt()} -> {got} (want {want})")
    return correct / len(chains), correct, misses, zero_hit, zero_n


def score_words(lattice: Lattice, rows: list[Triple]) -> tuple[float, int, list[str]]:
    if not rows:
        return 0.0, 0, []
    misses: list[str] = []
    correct = 0
    for row in rows:
        got = lattice.predict_word(row)
        if got == row.word_answer():
            correct += 1
        elif len(misses) < 8:
            misses.append(f"{row.word_prompt()} -> {got} (want {row.word_answer()})")
    return correct / len(rows), correct, misses


def mean_confidence(lattice: Lattice, rows: list[Triple]) -> float:
    if not rows:
        return 0.0
    total = 0.0
    for row in rows:
        symbols = [str(row.a), row.op, str(row.b), "=", str(row.c)]
        total += lattice.pathway_confidence(symbols)
    return total / len(rows)


def gauge_residual(values: dict[str, float]) -> float:
    """Mean absolute miss of v[a]±v[b] against v[a±b] on the full closed set.

    This is a measurement of the learned gauge, not a second training loss.
    """
    err = 0.0
    n = 0
    for a in range(10):
        for b in range(10):
            if a + b <= 9:
                pred = values[str(a)] + values[str(b)]
                err += abs(pred - values[str(a + b)])
                n += 1
            if a - b >= 0:
                pred = values[str(a)] - values[str(b)]
                err += abs(pred - values[str(a - b)])
                n += 1
    return err / n if n else 0.0


def write_engrams(path: Path, lattice: Lattice, pin: str, meta: dict) -> None:
    """Idle snapshot. Noun gauges are stored. The clocks stay in memory."""
    bonds = []
    for i, row in enumerate(lattice.bonds):
        for j, weight in enumerate(row):
            if weight != 0.0:
                bonds.append([i, j, weight])
    payload = {
        "pin": pin,
        "width": lattice.width,
        "collapse": COLLAPSE,
        "coherence_gate": COHERENCE_GATE,
        "frozen_digit_value": lattice.frozen_digits,
        "word_value": lattice.word_value,
        "acquired_value": dict(lattice.acquired_value),
        "acquired_self": dict(lattice.acquired_self),
        "bonds": bonds,
        "meta": meta,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def reload_frozen_digits(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frozen = payload["frozen_digit_value"]
    return {key: float(value) for key, value in frozen.items()}


def reload_acquired(path: Path) -> dict[str, float]:
    """Kind gauges from an idle snapshot. The clocks are not in that file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    stored = payload["acquired_value"]
    return {key: float(value) for key, value in stored.items()}


def reload_acquired_self(path: Path) -> dict[str, float]:
    """Individual gauges from an idle snapshot."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    stored = payload["acquired_self"]
    return {key: float(value) for key, value in stored.items()}


def acquired_idle_drift(lattice: Lattice) -> float:
    """Write the noun gauges to a scratch snapshot and read them back.

    Memory is cleared before the read, so a drift of 0 is the file. The
    clocks stay as they were. The lattice keeps the reloaded gauges.
    """
    snapshot = dict(lattice.acquired_value)
    snapshot_self = dict(lattice.acquired_self)
    clocks = (
        lattice.working,
        lattice.counted,
        lattice.earlier,
        lattice.earlier_count,
        lattice.earliest,
        lattice.earliest_count,
        lattice.opening,
        lattice.opening_count,
    )
    handle = tempfile.NamedTemporaryFile(
        prefix="fsot-acquired-idle-", suffix=".json", delete=False
    )
    path = Path(handle.name)
    handle.close()
    try:
        write_engrams(path, lattice, "acquired-idle", {"phase": "acquired_idle"})
        payload = json.loads(path.read_text(encoding="utf-8"))
        for key in (
            "working",
            "counted",
            "earlier",
            "earlier_count",
            "earliest",
            "earliest_count",
            "opening",
            "opening_count",
        ):
            if key in payload:
                raise RuntimeError("idle stored a clock")
        if set(payload.get("acquired_value") or {}) != set(ACQUIRED_NOUNS):
            raise RuntimeError("idle dropped a noun")
        if set(payload.get("acquired_self") or {}) != set(ACQUIRED_NOUNS):
            raise RuntimeError("idle dropped an individual")
        for name in ACQUIRED_NOUNS:
            lattice.acquired_value[name] = 0.0
            lattice.acquired_self[name] = 0.0
        loaded = reload_acquired(path)
        loaded_self = reload_acquired_self(path)
        lattice.acquired_value = {name: loaded[name] for name in ACQUIRED_NOUNS}
        lattice.acquired_self = {name: loaded_self[name] for name in ACQUIRED_NOUNS}
    finally:
        path.unlink(missing_ok=True)
    if (
        lattice.working,
        lattice.counted,
        lattice.earlier,
        lattice.earlier_count,
        lattice.earliest,
        lattice.earliest_count,
        lattice.opening,
        lattice.opening_count,
    ) != clocks:
        raise RuntimeError("idle reload moved a clock")
    kind_drift = max(abs(snapshot[name] - lattice.acquired_value[name]) for name in ACQUIRED_NOUNS)
    self_drift = max(
        abs(snapshot_self[name] - lattice.acquired_self[name]) for name in ACQUIRED_NOUNS
    )
    return max(kind_drift, self_drift)


def apply_digit_epoch(lattice: Lattice, rows: list[Triple], hits: float) -> tuple[float, float]:
    """One batch step on digit gauges. Returns mse and the fluid step size."""
    grad = {sym: 0.0 for sym in DIGITS}
    loss = 0.0
    for row in rows:
        sign = float(lattice.routed_sign(row.op))
        pred = lattice.digit_value[str(row.a)] + sign * lattice.digit_value[str(row.b)]
        err = lattice.digit_value[str(row.c)] - pred
        loss += err * err
        grad[str(row.a)] += err
        grad[str(row.b)] += sign * err
        grad[str(row.c)] -= err
    n = max(len(rows), 1)
    mse = loss / n
    lr = fluid_lr(mse, hits) * GAUGE_GAIN
    for sym in DIGITS:
        lattice.digit_value[sym] += lr * grad[sym] / n
    return mse, lr


def apply_word_epoch(lattice: Lattice, rows: list[Triple], hits: float) -> tuple[float, float]:
    """Bind number-words onto frozen digits, then one compose step on the words.

    Alignment runs alone until it is tight. Compose before that fight pulls
    every word toward the same value and the names never separate.
    """
    if lattice.frozen_digits is None:
        raise RuntimeError("word phase requires consolidated digit gauges")
    names = [WORD_OF[i] for i in range(10)]
    align = 0.0
    grad = {name: 0.0 for name in names}
    for name in names:
        err = lattice.frozen_digits[NUMERIC_WORDS[name]] - lattice.word_value[name]
        grad[name] += err
        align += err * err
    align_mse = align / len(names)
    lr = fluid_lr(align_mse, hits) * GAUGE_GAIN
    for name in names:
        lattice.word_value[name] += lr * grad[name] / len(names)
    if align_mse > 1e-4:
        return align_mse, lr

    loss = 0.0
    grad = {name: 0.0 for name in names}
    for row in rows:
        sign = float(lattice.routed_sign(row.op))
        wa, wb, wc = WORD_OF[row.a], WORD_OF[row.b], WORD_OF[row.c]
        pred = lattice.word_value[wa] + sign * lattice.word_value[wb]
        err = lattice.word_value[wc] - pred
        loss += err * err
        grad[wa] += err
        grad[wb] += sign * err
        grad[wc] -= err
    n = max(len(rows), 1)
    mse = loss / n
    lr = fluid_lr(max(mse, align_mse), hits) * GAUGE_GAIN
    for name in names:
        lattice.word_value[name] += lr * grad[name] / n
    return mse + align_mse, lr


def content_offset(index: int) -> float:
    """The same K*(count+1) line the digit gauges start on."""
    return K * (index + 1)


def content_residual(lattice: Lattice) -> float:
    """Widest miss of a content gauge against its offset."""
    worst = 0.0
    for index, name in enumerate(CONTENT_WORDS):
        worst = max(worst, abs(lattice.content_value[name] - content_offset(index)))
    return worst


def apply_content_epoch(lattice: Lattice, rows: list, hits: float) -> tuple[float, float]:
    """Pull content gauges onto the offset line, then fit the train questions.

    Alignment runs alone until it is tight. The pair term is the train
    slice only. Digit gauges and number-word gauges are not in the step.
    """
    names = list(CONTENT_WORDS)
    index = {name: place for place, name in enumerate(names)}
    grad = {name: 0.0 for name in names}
    align = 0.0
    for name in names:
        err = content_offset(index[name]) - lattice.content_value[name]
        grad[name] += err
        align += err * err
    align_mse = align / len(names)
    lr = fluid_lr(align_mse, hits) * GAUGE_GAIN
    for name in names:
        lattice.content_value[name] += lr * grad[name] / len(names)
    if align_mse > 1e-4:
        return align_mse, lr

    loss = 0.0
    grad = {name: 0.0 for name in names}
    for row in rows:
        target = K * (index[row.left] - index[row.right])
        pred = lattice.content_value[row.left] - lattice.content_value[row.right]
        err = target - pred
        grad[row.left] += err
        grad[row.right] -= err
        loss += err * err
    n = max(len(rows), 1)
    mse = loss / n
    lr = fluid_lr(max(mse, align_mse), hits) * GAUGE_GAIN
    for name in names:
        lattice.content_value[name] += lr * grad[name] / n
    return mse + align_mse, lr


def acquired_residual(lattice: Lattice) -> float:
    """Widest miss of a taught noun against the live head it was given."""
    worst = 0.0
    for name in ACQUIRED_NOUNS:
        target = lattice.content_value[acquired_kind(name)]
        worst = max(worst, abs(lattice.acquired_value[name] - target))
    return worst


def apply_acquired_epoch(lattice: Lattice, hits: float) -> tuple[float, float]:
    """Pull each new noun onto the live gauge of its kind.

    Same alignment step as the content offset. The head is the target and
    is not updated. Digit gauges, number-word gauges, and the ten content
    gauges stay out of the step.
    """
    names = list(ACQUIRED_NOUNS)
    grad = {name: 0.0 for name in names}
    align = 0.0
    for name in names:
        err = lattice.content_value[acquired_kind(name)] - lattice.acquired_value[name]
        grad[name] += err
        align += err * err
    align_mse = align / len(names)
    lr = fluid_lr(align_mse, hits) * GAUGE_GAIN
    for name in names:
        lattice.acquired_value[name] += lr * grad[name] / len(names)
    return align_mse, lr


def revision_residual(lattice: Lattice, targets: dict[str, str]) -> float:
    """Widest miss of a revised noun against the head the later sentence named."""
    worst = 0.0
    for name, head in targets.items():
        worst = max(worst, abs(lattice.acquired_value[name] - lattice.content_value[head]))
    return worst


def revision_spread(lattice: Lattice, targets: dict[str, str]) -> float:
    """Widest gap between two nouns a later sentence put on one head."""
    groups: dict[str, list[float]] = {}
    for name, head in targets.items():
        groups.setdefault(head, []).append(lattice.acquired_value[name])
    worst = 0.0
    for gauges in groups.values():
        if len(gauges) > 1:
            worst = max(worst, max(gauges) - min(gauges))
    return worst


def apply_revision_epoch(lattice: Lattice, targets: dict[str, str], hits: float) -> tuple[float, float]:
    """Pull the named nouns onto a new head. The individual gauges stay out."""
    names = list(targets)
    grad = {name: 0.0 for name in names}
    align = 0.0
    for name, head in targets.items():
        err = lattice.content_value[head] - lattice.acquired_value[name]
        grad[name] += err
        align += err * err
    align_mse = align / len(names)
    lr = fluid_lr(align_mse, hits) * GAUGE_GAIN
    for name in names:
        lattice.acquired_value[name] += lr * grad[name] / len(names)
    return align_mse, lr


def settle_revision(lattice: Lattice, targets: dict[str, str], epochs: int = 500) -> list[dict]:
    """Move the named kind gauges onto the later sentence. Individuals stay put.

    Nouns that meet on one head keep the same step until their spread is
    inside the floor. A gauge that arrives from below and a gauge that
    arrives from above can each sit inside the residual and still disagree.
    """
    if set(targets) == set(ACQUIRED_NOUNS):
        train, hold = split_revisions()
    else:
        rows = [
            AcquiredRevision(noun, stated, targets[noun])
            for noun in targets
            for stated in CONTENT_WORDS
        ]
        train, hold = rows, rows
    history: list[dict] = []
    hits = 0.0
    for epoch in range(epochs):
        mse, lr = apply_revision_epoch(lattice, targets, hits)
        train_acc, train_n, _, _ = score_acquired(lattice, train)
        hold_acc, hold_n, _, _ = score_acquired(lattice, hold)
        residual = revision_residual(lattice, targets)
        spread = revision_spread(lattice, targets)
        hits = train_acc * len(train)
        history.append(
            {
                "epoch": epoch,
                "mse": mse,
                "lr": lr,
                "train_acc": train_acc,
                "hold_acc": hold_acc,
                "train_correct": train_n,
                "hold_correct": hold_n,
                "gauge_residual": residual,
            }
        )
        if residual < 1e-4 and spread < 1e-4 and train_acc >= 1.0 and hold_acc >= 1.0:
            break
    return history


def score_acquired(lattice: Lattice, rows: list) -> tuple[float, int, list[str], dict[str, float]]:
    """Yes when the stated content word is the head the noun was taught."""
    if not rows:
        return 0.0, 0, [], {}
    misses: list[str] = []
    correct = 0
    true_abs: list[float] = []
    false_abs: list[float] = []
    for row in rows:
        remainder = lattice.acquired_remainder(row.noun, row.stated)
        got = lattice.remainder_answer(remainder)
        if row.holds():
            true_abs.append(abs(remainder))
        else:
            false_abs.append(abs(remainder))
        if got == row.answer():
            correct += 1
        elif len(misses) < 8:
            misses.append(f"{row.prompt()} -> {got} (want {row.answer()})")
    stats = {
        "true_max_abs": max(true_abs) if true_abs else 0.0,
        "false_min_abs": min(false_abs) if false_abs else 0.0,
    }
    return correct / len(rows), correct, misses, stats


def score_acquired_pairs(lattice: Lattice, rows: list) -> tuple[float, int, list[str], dict[str, float]]:
    """Same taught kind agrees. A different kind clears the drop."""
    if not rows:
        return 0.0, 0, [], {}
    misses: list[str] = []
    correct = 0
    true_abs: list[float] = []
    false_abs: list[float] = []
    for row in rows:
        remainder = lattice.acquired_pair_remainder(row.left, row.right)
        got = lattice.remainder_answer(remainder)
        if row.holds():
            true_abs.append(abs(remainder))
        else:
            false_abs.append(abs(remainder))
        if got == row.answer():
            correct += 1
        elif len(misses) < 8:
            misses.append(f"{row.prompt()} -> {got} (want {row.answer()})")
    stats = {
        "true_max_abs": max(true_abs) if true_abs else 0.0,
        "false_min_abs": min(false_abs) if false_abs else 0.0,
    }
    return correct / len(rows), correct, misses, stats


def settle_acquired(lattice: Lattice, epochs: int = 500) -> list[dict]:
    """Move the new nouns onto their heads. Stops once the table is exact."""
    train, hold = split_acquired()
    history: list[dict] = []
    hits = 0.0
    for epoch in range(epochs):
        mse, lr = apply_acquired_epoch(lattice, hits)
        train_acc, train_n, _, _ = score_acquired(lattice, train)
        hold_acc, hold_n, _, _ = score_acquired(lattice, hold)
        residual = acquired_residual(lattice)
        hits = train_acc * len(train)
        history.append(
            {
                "epoch": epoch,
                "mse": mse,
                "lr": lr,
                "train_acc": train_acc,
                "hold_acc": hold_acc,
                "train_correct": train_n,
                "hold_correct": hold_n,
                "gauge_residual": residual,
            }
        )
        if residual < 1e-4 and train_acc >= 1.0 and hold_acc >= 1.0:
            break
    return history


def acquired_self_residual(lattice: Lattice) -> float:
    """Widest miss of an individual gauge against its offset."""
    worst = 0.0
    for place, name in enumerate(ACQUIRED_NOUNS):
        worst = max(worst, abs(lattice.acquired_self[name] - content_offset(place)))
    return worst


def apply_acquired_self_epoch(lattice: Lattice, hits: float) -> tuple[float, float]:
    """Pull each individual onto K times its place.

    Same alignment step as the content offset, then the train-split
    difference. The kind gauges stay out of the step.
    """
    names = list(ACQUIRED_NOUNS)
    index = {name: place for place, name in enumerate(names)}
    grad = {name: 0.0 for name in names}
    align = 0.0
    for name in names:
        err = content_offset(index[name]) - lattice.acquired_self[name]
        grad[name] += err
        align += err * err
    align_mse = align / len(names)
    lr = fluid_lr(align_mse, hits) * GAUGE_GAIN
    for name in names:
        lattice.acquired_self[name] += lr * grad[name] / len(names)
    if align_mse > 1e-4:
        return align_mse, lr

    rows, _hold = split_acquired_selves()
    loss = 0.0
    grad = {name: 0.0 for name in names}
    for row in rows:
        target = K * (index[row.left] - index[row.right])
        pred = lattice.acquired_self[row.left] - lattice.acquired_self[row.right]
        err = target - pred
        grad[row.left] += err
        grad[row.right] -= err
        loss += err * err
    n = max(len(rows), 1)
    mse = loss / n
    lr = fluid_lr(max(mse, align_mse), hits) * GAUGE_GAIN
    for name in names:
        lattice.acquired_self[name] += lr * grad[name] / n
    return mse + align_mse, lr


def score_acquired_selves(lattice: Lattice, rows: list) -> tuple[float, int, list[str], dict[str, float]]:
    """Yes when the two names are the same individual."""
    if not rows:
        return 0.0, 0, [], {}
    misses: list[str] = []
    correct = 0
    true_abs: list[float] = []
    false_abs: list[float] = []
    for row in rows:
        remainder = lattice.acquired_self_remainder(row.left, row.right)
        got = lattice.remainder_answer(remainder)
        if row.holds():
            true_abs.append(abs(remainder))
        else:
            false_abs.append(abs(remainder))
        if got == row.answer():
            correct += 1
        elif len(misses) < 8:
            misses.append(f"{row.prompt()} -> {got} (want {row.answer()})")
    stats = {
        "true_max_abs": max(true_abs) if true_abs else 0.0,
        "false_min_abs": min(false_abs) if false_abs else 0.0,
    }
    return correct / len(rows), correct, misses, stats


def settle_acquired_self(lattice: Lattice, epochs: int = 500) -> list[dict]:
    """Move the individuals apart on the offset line. Stops once the table is exact."""
    train, hold = split_acquired_selves()
    history: list[dict] = []
    hits = 0.0
    for epoch in range(epochs):
        mse, lr = apply_acquired_self_epoch(lattice, hits)
        train_acc, train_n, _, _ = score_acquired_selves(lattice, train)
        hold_acc, hold_n, _, _ = score_acquired_selves(lattice, hold)
        residual = acquired_self_residual(lattice)
        hits = train_acc * len(train)
        history.append(
            {
                "epoch": epoch,
                "mse": mse,
                "lr": lr,
                "train_acc": train_acc,
                "hold_acc": hold_acc,
                "train_correct": train_n,
                "hold_correct": hold_n,
                "gauge_residual": residual,
            }
        )
        if residual < 1e-4 and train_acc >= 1.0 and hold_acc >= 1.0:
            break
    return history


def play_acquired(lattice: Lattice) -> dict[str, float | list[str] | bool]:
    """Wolf then pine, on the clocks. The carrier holds the kind, not the noun.

    A question does not move the clocks. The caller keeps whatever they write.
    """
    answers: list[str] = []
    true_abs: list[float] = []
    false_abs: list[float] = []
    misses: list[str] = []

    def clocks() -> tuple[float, float, float, float, float, float, float, float]:
        return (
            lattice.working,
            lattice.counted,
            lattice.earlier,
            lattice.earlier_count,
            lattice.earliest,
            lattice.earliest_count,
            lattice.opening,
            lattice.opening_count,
        )

    def ask(stated: str, holds: bool, label: str, kept_read: bool) -> None:
        kept = clocks()
        remainder = (
            lattice.it_kept_remainder(stated)
            if kept_read
            else lattice.it_remainder(stated)
        )
        got = lattice.remainder_answer(remainder)
        want = "yes" if holds else "no"
        answers.append(got)
        if holds:
            true_abs.append(abs(remainder))
        else:
            false_abs.append(abs(remainder))
        if got != want and len(misses) < 8:
            misses.append(f"{label} -> {got} (want {want})")
        if clocks() != kept and len(misses) < 8:
            misses.append(f"{label} moved the carrier")

    lattice.clear_working()
    ask("animal", False, "empty is it an animal", False)
    lattice.deposit_acquired("wolf")
    ask("animal", True, "is it an animal", False)
    ask("cat", False, "is it a cat", False)
    ask("plant", False, "is it a plant", False)
    lattice.deposit_acquired("pine")
    ask("plant", True, "is it a plant", False)
    ask("animal", False, "latest is it an animal", False)
    ask("animal", True, "kept is it an animal", True)
    return {
        "ok": not misses,
        "answers": answers,
        "misses": misses,
        "true_max_abs": max(true_abs) if true_abs else 0.0,
        "false_min_abs": min(false_abs) if false_abs else 0.0,
        "carry_max": max(
            abs(lattice.working - lattice.content_value["plant"]),
            abs(lattice.earlier - lattice.content_value["animal"]),
        ),
    }


def score_content(lattice: Lattice, rows: list) -> tuple[float, int, list[str], dict[str, float]]:
    """Yes/no from the difference of two content gauges."""
    if not rows:
        return 0.0, 0, [], {}
    misses: list[str] = []
    correct = 0
    true_abs: list[float] = []
    false_abs: list[float] = []
    for row in rows:
        got = lattice.predict_content(row.left, row.right)
        remainder = abs(lattice.content_remainder(row.left, row.right))
        if row.holds():
            true_abs.append(remainder)
        else:
            false_abs.append(remainder)
        if got == row.answer():
            correct += 1
        elif len(misses) < 8:
            misses.append(f"{row.prompt()} -> {got} (want {row.answer()})")
    stats = {
        "true_max_abs": max(true_abs) if true_abs else 0.0,
        "false_min_abs": min(false_abs) if false_abs else 0.0,
        "true_n": float(len(true_abs)),
        "false_n": float(len(false_abs)),
    }
    return correct / len(rows), correct, misses, stats


def score_statements(lattice: Lattice, rows: list) -> tuple[float, int, list[str], dict[str, float]]:
    """Yes/no from one kind step against the stated word."""
    if not rows:
        return 0.0, 0, [], {}
    misses: list[str] = []
    correct = 0
    true_abs: list[float] = []
    false_abs: list[float] = []
    for row in rows:
        got = lattice.predict_statement(row.subject, row.stated)
        remainder = abs(lattice.statement_remainder(row.subject, row.stated))
        if row.holds():
            true_abs.append(remainder)
        else:
            false_abs.append(remainder)
        if got == row.answer():
            correct += 1
        elif len(misses) < 8:
            misses.append(f"{row.prompt()} -> {got} (want {row.answer()})")
    stats = {
        "true_max_abs": max(true_abs) if true_abs else 0.0,
        "false_min_abs": min(false_abs) if false_abs else 0.0,
        "true_n": float(len(true_abs)),
        "false_n": float(len(false_abs)),
    }
    return correct / len(rows), correct, misses, stats


def score_kind_chains(lattice: Lattice, rows: list) -> tuple[float, int, list[str], dict[str, float]]:
    """The depth-3 kind fold names the head, and the vessel sits on that gauge."""
    if not rows:
        return 0.0, 0, [], {}
    misses: list[str] = []
    correct = 0
    margins: list[float] = []
    gaps: list[float] = []
    values = [lattice.content_value[name] for name in CONTENT_WORDS]
    for row in rows:
        quantity = lattice.kind_chain_quantity(row.subject)
        best, margin, _dist = _nearest_detail(quantity, values)
        got = CONTENT_WORDS[best]
        gaps.append(abs(quantity - lattice.content_value[row.answer()]))
        margins.append(margin)
        if got == row.answer():
            correct += 1
        elif len(misses) < 8:
            misses.append(f"{row.prompt()} -> {got} (want {row.answer()})")
    stats = {
        "margin_min": min(margins) if margins else 0.0,
        "gap_max": max(gaps) if gaps else 0.0,
    }
    return correct / len(rows), correct, misses, stats


def play_follow(lattice: Lattice, first: str, second: str) -> dict[str, float | list[str] | bool]:
    """Empty carrier, then two named sentences and the questions about what they left.

    A question does not replace the carrier. The second sentence does.
    The caller keeps whatever clock this writes.
    """
    first_kind = content_kind(first)
    second_kind = content_kind(second)
    same = first_kind == second_kind
    answers: list[str] = []
    true_abs: list[float] = []
    false_abs: list[float] = []
    misses: list[str] = []

    def ask(stated: str, holds: bool, label: str) -> None:
        kept = (
            lattice.working,
            lattice.counted,
            lattice.earlier,
            lattice.earlier_count,
            lattice.earliest,
            lattice.earliest_count,
            lattice.opening,
            lattice.opening_count,
        )
        remainder = lattice.it_remainder(stated)
        got = lattice.remainder_answer(remainder)
        want = "yes" if holds else "no"
        answers.append(got)
        if holds:
            true_abs.append(abs(remainder))
        else:
            false_abs.append(abs(remainder))
        if got != want and len(misses) < 8:
            misses.append(f"{label} -> {got} (want {want})")
        if (
            lattice.working,
            lattice.counted,
            lattice.earlier,
            lattice.earlier_count,
            lattice.earliest,
            lattice.earliest_count,
            lattice.opening,
            lattice.opening_count,
        ) != kept and len(misses) < 8:
            misses.append(f"{label} moved the carrier")

    lattice.clear_working()
    ask(first_kind, False, f"empty is it {first_kind}")
    lattice.deposit_kind(first)
    carry = abs(lattice.working - lattice.content_value[first_kind])
    ask(first_kind, True, f"is it {first_kind}")
    ask(first, first == first_kind, f"is it {first}")
    ask(second_kind, same, f"is it {second_kind}")
    lattice.deposit_kind(second)
    carry = max(carry, abs(lattice.working - lattice.content_value[second_kind]))
    ask(second_kind, True, f"is it {second_kind}")
    ask(second, second == second_kind, f"is it {second}")
    ask(first_kind, same, f"again is it {first_kind}")
    return {
        "ok": not misses,
        "answers": answers,
        "misses": misses,
        "true_max_abs": max(true_abs) if true_abs else 0.0,
        "false_min_abs": min(false_abs) if false_abs else 0.0,
        "carry_max": carry,
    }


def score_follows(lattice: Lattice, rows: list) -> tuple[float, int, list[str], dict[str, float]]:
    """Each row is its own conversation. The clock returns to where it started."""
    if not rows:
        return 0.0, 0, [], {}
    saved = lattice.working
    saved_count = lattice.counted
    saved_earlier = lattice.earlier
    saved_earlier_count = lattice.earlier_count
    saved_earliest = lattice.earliest
    saved_earliest_count = lattice.earliest_count
    saved_opening = lattice.opening
    saved_opening_count = lattice.opening_count
    misses: list[str] = []
    correct = 0
    true_abs: list[float] = []
    false_abs: list[float] = []
    carry: list[float] = []
    try:
        for row in rows:
            played = play_follow(lattice, row.first, row.second)
            true_abs.append(float(played["true_max_abs"]))
            false_abs.append(float(played["false_min_abs"]))
            carry.append(float(played["carry_max"]))
            if played["ok"]:
                correct += 1
            elif len(misses) < 8:
                detail = played["misses"][0] if played["misses"] else "miss"
                misses.append(f"{row.opening()} / {row.closing()} {detail}")
    finally:
        lattice.working = saved
        lattice.counted = saved_count
        lattice.earlier = saved_earlier
        lattice.earlier_count = saved_earlier_count
        lattice.earliest = saved_earliest
        lattice.earliest_count = saved_earliest_count
        lattice.opening = saved_opening
        lattice.opening_count = saved_opening_count
    stats = {
        "true_max_abs": max(true_abs) if true_abs else 0.0,
        "false_min_abs": min(false_abs) if false_abs else 0.0,
        "carry_max": max(carry) if carry else 0.0,
    }
    return correct / len(rows), correct, misses, stats


def play_count(lattice: Lattice, count: str, subject: str) -> dict[str, float | list[str] | bool]:
    """Empty clocks, one counted sentence, then the next count on the next noun.

    A question does not replace either clock. The next counted sentence does.
    The caller keeps whatever clocks this writes.
    """
    row = Counted(count, subject)
    kind = content_kind(subject)
    other_count = row.next_count()
    other_subject = row.next_subject()
    other_kind = content_kind(other_subject)
    opposed = OPPOSED_KIND[kind]
    answers: list[str] = []
    true_abs: list[float] = []
    false_abs: list[float] = []
    misses: list[str] = []

    def ask(which: str, stated: str, holds: bool, label: str) -> None:
        kept = (
            lattice.working,
            lattice.counted,
            lattice.earlier,
            lattice.earlier_count,
            lattice.earliest,
            lattice.earliest_count,
            lattice.opening,
            lattice.opening_count,
        )
        if which == "count":
            remainder = lattice.it_count_remainder(stated)
        else:
            remainder = lattice.it_remainder(stated)
        got = lattice.remainder_answer(remainder)
        want = "yes" if holds else "no"
        answers.append(got)
        if holds:
            true_abs.append(abs(remainder))
        else:
            false_abs.append(abs(remainder))
        if got != want and len(misses) < 8:
            misses.append(f"{label} -> {got} (want {want})")
        if (
            lattice.working,
            lattice.counted,
            lattice.earlier,
            lattice.earlier_count,
            lattice.earliest,
            lattice.earliest_count,
            lattice.opening,
            lattice.opening_count,
        ) != kept and len(misses) < 8:
            misses.append(f"{label} moved the carrier")

    lattice.clear_working()
    ask("count", count, False, f"empty is it {count}")
    ask("kind", kind, False, f"empty is it {kind}")
    lattice.deposit_count(count, subject)
    carry = max(
        abs(lattice.counted - lattice.word_value[count]),
        abs(lattice.working - lattice.content_value[kind]),
    )
    ask("count", count, True, f"is it {count}")
    ask("kind", kind, True, f"is it {kind}")
    ask("kind", subject, subject == kind, f"is it {subject}")
    ask("count", other_count, False, f"is it {other_count}")
    ask("kind", opposed, False, f"is it {opposed}")
    lattice.deposit_count(other_count, other_subject)
    carry = max(
        carry,
        abs(lattice.counted - lattice.word_value[other_count]),
        abs(lattice.working - lattice.content_value[other_kind]),
    )
    ask("count", other_count, True, f"is it {other_count}")
    ask("kind", other_kind, True, f"is it {other_kind}")
    ask("count", count, False, f"again is it {count}")
    ask("kind", kind, kind == other_kind, f"again is it {kind}")
    return {
        "ok": not misses,
        "answers": answers,
        "misses": misses,
        "true_max_abs": max(true_abs) if true_abs else 0.0,
        "false_min_abs": min(false_abs) if false_abs else 0.0,
        "carry_max": carry,
    }


def score_counts(lattice: Lattice, rows: list) -> tuple[float, int, list[str], dict[str, float]]:
    """Each row is its own sentence. Both clocks return to where they started."""
    if not rows:
        return 0.0, 0, [], {}
    saved = lattice.working
    saved_count = lattice.counted
    saved_earlier = lattice.earlier
    saved_earlier_count = lattice.earlier_count
    saved_earliest = lattice.earliest
    saved_earliest_count = lattice.earliest_count
    saved_opening = lattice.opening
    saved_opening_count = lattice.opening_count
    misses: list[str] = []
    correct = 0
    true_abs: list[float] = []
    false_abs: list[float] = []
    carry: list[float] = []
    try:
        for row in rows:
            played = play_count(lattice, row.count, row.subject)
            true_abs.append(float(played["true_max_abs"]))
            false_abs.append(float(played["false_min_abs"]))
            carry.append(float(played["carry_max"]))
            if played["ok"]:
                correct += 1
            elif len(misses) < 8:
                detail = played["misses"][0] if played["misses"] else "miss"
                misses.append(f"{row.opening()} {detail}")
    finally:
        lattice.working = saved
        lattice.counted = saved_count
        lattice.earlier = saved_earlier
        lattice.earlier_count = saved_earlier_count
        lattice.earliest = saved_earliest
        lattice.earliest_count = saved_earliest_count
        lattice.opening = saved_opening
        lattice.opening_count = saved_opening_count
    stats = {
        "true_max_abs": max(true_abs) if true_abs else 0.0,
        "false_min_abs": min(false_abs) if false_abs else 0.0,
        "carry_max": max(carry) if carry else 0.0,
    }
    return correct / len(rows), correct, misses, stats


def _count_beside(kind: str) -> str:
    """A content word of this kind that is not the head, or cat when the head stands alone."""
    for word in CONTENT_WORDS:
        if word != kind and content_kind(word) == kind:
            return word
    return "cat"


def play_acquired_count(lattice: Lattice, count: str, noun: str) -> dict[str, float | list[str] | bool]:
    """Empty clocks, one counted taught noun, then the next count on the next noun.

    The count clock holds the number word. The kind clock holds the kind
    that noun steps to. A question does not replace either clock.
    """
    row = AcquiredCount(count, noun)
    kind = acquired_kind(noun)
    other_count = row.next_count()
    other_noun = row.next_noun()
    other_kind = acquired_kind(other_noun)
    opposed = OPPOSED_KIND[kind]
    member = _count_beside(kind)
    answers: list[str] = []
    true_abs: list[float] = []
    false_abs: list[float] = []
    misses: list[str] = []

    def ask(which: str, stated: str, holds: bool, label: str) -> None:
        kept = (
            lattice.working,
            lattice.counted,
            lattice.earlier,
            lattice.earlier_count,
            lattice.earliest,
            lattice.earliest_count,
            lattice.opening,
            lattice.opening_count,
        )
        if which == "count":
            remainder = lattice.it_count_remainder(stated)
        else:
            remainder = lattice.it_remainder(stated)
        got = lattice.remainder_answer(remainder)
        want = "yes" if holds else "no"
        answers.append(got)
        if holds:
            true_abs.append(abs(remainder))
        else:
            false_abs.append(abs(remainder))
        if got != want and len(misses) < 8:
            misses.append(f"{label} -> {got} (want {want})")
        if (
            lattice.working,
            lattice.counted,
            lattice.earlier,
            lattice.earlier_count,
            lattice.earliest,
            lattice.earliest_count,
            lattice.opening,
            lattice.opening_count,
        ) != kept and len(misses) < 8:
            misses.append(f"{label} moved the carrier")

    lattice.clear_working()
    ask("count", count, False, f"empty is it {count}")
    ask("kind", kind, False, f"empty is it {kind}")
    lattice.deposit_acquired_count(count, noun)
    carry = max(
        abs(lattice.counted - lattice.word_value[count]),
        abs(lattice.working - lattice.content_value[kind]),
    )
    ask("count", count, True, f"is it {count}")
    ask("kind", kind, True, f"is it {kind}")
    ask("kind", member, False, f"is it {member}")
    ask("count", other_count, False, f"is it {other_count}")
    ask("kind", opposed, False, f"is it {opposed}")
    lattice.deposit_acquired_count(other_count, other_noun)
    carry = max(
        carry,
        abs(lattice.counted - lattice.word_value[other_count]),
        abs(lattice.working - lattice.content_value[other_kind]),
    )
    ask("count", other_count, True, f"is it {other_count}")
    ask("kind", other_kind, True, f"is it {other_kind}")
    ask("count", count, False, f"again is it {count}")
    ask("kind", kind, kind == other_kind, f"again is it {kind}")
    return {
        "ok": not misses,
        "answers": answers,
        "misses": misses,
        "true_max_abs": max(true_abs) if true_abs else 0.0,
        "false_min_abs": min(false_abs) if false_abs else 0.0,
        "carry_max": carry,
    }


def score_acquired_counts(lattice: Lattice, rows: list) -> tuple[float, int, list[str], dict[str, float]]:
    """Each taught noun is its own counted sentence. Both clocks return."""
    if not rows:
        return 0.0, 0, [], {}
    saved = lattice.working
    saved_count = lattice.counted
    saved_earlier = lattice.earlier
    saved_earlier_count = lattice.earlier_count
    saved_earliest = lattice.earliest
    saved_earliest_count = lattice.earliest_count
    saved_opening = lattice.opening
    saved_opening_count = lattice.opening_count
    misses: list[str] = []
    correct = 0
    true_abs: list[float] = []
    false_abs: list[float] = []
    carry: list[float] = []
    try:
        for row in rows:
            played = play_acquired_count(lattice, row.count, row.noun)
            true_abs.append(float(played["true_max_abs"]))
            false_abs.append(float(played["false_min_abs"]))
            carry.append(float(played["carry_max"]))
            if played["ok"]:
                correct += 1
            elif len(misses) < 8:
                detail = played["misses"][0] if played["misses"] else "miss"
                misses.append(f"{row.opening()} {detail}")
    finally:
        lattice.working = saved
        lattice.counted = saved_count
        lattice.earlier = saved_earlier
        lattice.earlier_count = saved_earlier_count
        lattice.earliest = saved_earliest
        lattice.earliest_count = saved_earliest_count
        lattice.opening = saved_opening
        lattice.opening_count = saved_opening_count
    stats = {
        "true_max_abs": max(true_abs) if true_abs else 0.0,
        "false_min_abs": min(false_abs) if false_abs else 0.0,
        "carry_max": max(carry) if carry else 0.0,
    }
    return correct / len(rows), correct, misses, stats


def play_paragraph(lattice: Lattice, first: str, second: str) -> dict[str, float | list[str] | bool]:
    """Two kind sentences. The second keeps the first kind instead of dropping it.

    "is it" still reads only the latest clock. This read sees both.
    A question does not move the clocks. The caller keeps whatever they write.
    """
    row = FollowUp(first, second)
    kind = content_kind(first)
    other = content_kind(second)
    outside = row.other_kind()
    answers: list[str] = []
    true_abs: list[float] = []
    false_abs: list[float] = []
    misses: list[str] = []

    def clocks() -> tuple[float, float, float, float, float, float, float, float]:
        return (
            lattice.working,
            lattice.counted,
            lattice.earlier,
            lattice.earlier_count,
            lattice.earliest,
            lattice.earliest_count,
            lattice.opening,
            lattice.opening_count,
        )

    def ask(stated: str, holds: bool, label: str) -> None:
        kept = clocks()
        remainder = lattice.it_kept_remainder(stated)
        got = lattice.remainder_answer(remainder)
        want = "yes" if holds else "no"
        answers.append(got)
        if holds:
            true_abs.append(abs(remainder))
        else:
            false_abs.append(abs(remainder))
        if got != want and len(misses) < 8:
            misses.append(f"{label} -> {got} (want {want})")
        if clocks() != kept and len(misses) < 8:
            misses.append(f"{label} moved the carrier")

    lattice.clear_working()
    ask(kind, False, f"empty is it {kind}")
    lattice.deposit_kind(first)
    ask(kind, True, f"is it {kind}")
    ask(other, kind == other, f"early is it {other}")
    lattice.deposit_kind(second)
    carry = max(
        abs(lattice.working - lattice.content_value[other]),
        abs(lattice.earlier - lattice.content_value[kind]),
    )
    ask(other, True, f"is it {other}")
    ask(kind, True, f"kept is it {kind}")
    ask(outside, False, f"is it {outside}")
    return {
        "ok": not misses,
        "answers": answers,
        "misses": misses,
        "true_max_abs": max(true_abs) if true_abs else 0.0,
        "false_min_abs": min(false_abs) if false_abs else 0.0,
        "carry_max": carry,
    }


def score_paragraphs(lattice: Lattice, rows: list) -> tuple[float, int, list[str], dict[str, float]]:
    """Each pair is its own paragraph. The clocks return to where they started."""
    if not rows:
        return 0.0, 0, [], {}
    saved = lattice.working
    saved_count = lattice.counted
    saved_earlier = lattice.earlier
    saved_earlier_count = lattice.earlier_count
    saved_earliest = lattice.earliest
    saved_earliest_count = lattice.earliest_count
    saved_opening = lattice.opening
    saved_opening_count = lattice.opening_count
    misses: list[str] = []
    correct = 0
    true_abs: list[float] = []
    false_abs: list[float] = []
    carry: list[float] = []
    try:
        for row in rows:
            played = play_paragraph(lattice, row.first, row.second)
            true_abs.append(float(played["true_max_abs"]))
            false_abs.append(float(played["false_min_abs"]))
            carry.append(float(played["carry_max"]))
            if played["ok"]:
                correct += 1
            elif len(misses) < 8:
                detail = played["misses"][0] if played["misses"] else "miss"
                misses.append(f"{row.text()} {detail}")
    finally:
        lattice.working = saved
        lattice.counted = saved_count
        lattice.earlier = saved_earlier
        lattice.earlier_count = saved_earlier_count
        lattice.earliest = saved_earliest
        lattice.earliest_count = saved_earliest_count
        lattice.opening = saved_opening
        lattice.opening_count = saved_opening_count
    stats = {
        "true_max_abs": max(true_abs) if true_abs else 0.0,
        "false_min_abs": min(false_abs) if false_abs else 0.0,
        "carry_max": max(carry) if carry else 0.0,
    }
    return correct / len(rows), correct, misses, stats


def play_acquired_paragraph(lattice: Lattice, first: str, second: str) -> dict[str, float | list[str] | bool]:
    """Two taught nouns. The second sentence keeps the kind the first left.

    "is it" still reads only the latest clock. This read sees both.
    A question does not move the clocks. The noun gauges stay put.
    """
    row = AcquiredParagraph(first, second)
    kind = acquired_kind(first)
    other = acquired_kind(second)
    outside = row.other_kind()
    answers: list[str] = []
    true_abs: list[float] = []
    false_abs: list[float] = []
    misses: list[str] = []

    def clocks() -> tuple[float, float, float, float, float, float, float, float]:
        return (
            lattice.working,
            lattice.counted,
            lattice.earlier,
            lattice.earlier_count,
            lattice.earliest,
            lattice.earliest_count,
            lattice.opening,
            lattice.opening_count,
        )

    def ask(stated: str, holds: bool, label: str) -> None:
        kept = clocks()
        remainder = lattice.it_kept_remainder(stated)
        got = lattice.remainder_answer(remainder)
        want = "yes" if holds else "no"
        answers.append(got)
        if holds:
            true_abs.append(abs(remainder))
        else:
            false_abs.append(abs(remainder))
        if got != want and len(misses) < 8:
            misses.append(f"{label} -> {got} (want {want})")
        if clocks() != kept and len(misses) < 8:
            misses.append(f"{label} moved the carrier")

    lattice.clear_working()
    ask(kind, False, f"empty is it {kind}")
    lattice.deposit_acquired(first)
    ask(kind, True, f"is it {kind}")
    ask(other, kind == other, f"early is it {other}")
    lattice.deposit_acquired(second)
    carry = max(
        abs(lattice.working - lattice.content_value[other]),
        abs(lattice.earlier - lattice.content_value[kind]),
    )
    ask(other, True, f"is it {other}")
    ask(kind, True, f"kept is it {kind}")
    ask(outside, False, f"is it {outside}")
    return {
        "ok": not misses,
        "answers": answers,
        "misses": misses,
        "true_max_abs": max(true_abs) if true_abs else 0.0,
        "false_min_abs": min(false_abs) if false_abs else 0.0,
        "carry_max": carry,
    }


def score_acquired_paragraphs(lattice: Lattice, rows: list) -> tuple[float, int, list[str], dict[str, float]]:
    """Each pair of taught nouns is its own paragraph. The clocks return."""
    if not rows:
        return 0.0, 0, [], {}
    saved = lattice.working
    saved_count = lattice.counted
    saved_earlier = lattice.earlier
    saved_earlier_count = lattice.earlier_count
    saved_earliest = lattice.earliest
    saved_earliest_count = lattice.earliest_count
    saved_opening = lattice.opening
    saved_opening_count = lattice.opening_count
    misses: list[str] = []
    correct = 0
    true_abs: list[float] = []
    false_abs: list[float] = []
    carry: list[float] = []
    try:
        for row in rows:
            played = play_acquired_paragraph(lattice, row.first, row.second)
            true_abs.append(float(played["true_max_abs"]))
            false_abs.append(float(played["false_min_abs"]))
            carry.append(float(played["carry_max"]))
            if played["ok"]:
                correct += 1
            elif len(misses) < 8:
                detail = played["misses"][0] if played["misses"] else "miss"
                misses.append(f"{row.text()} {detail}")
    finally:
        lattice.working = saved
        lattice.counted = saved_count
        lattice.earlier = saved_earlier
        lattice.earlier_count = saved_earlier_count
        lattice.earliest = saved_earliest
        lattice.earliest_count = saved_earliest_count
        lattice.opening = saved_opening
        lattice.opening_count = saved_opening_count
    stats = {
        "true_max_abs": max(true_abs) if true_abs else 0.0,
        "false_min_abs": min(false_abs) if false_abs else 0.0,
        "carry_max": max(carry) if carry else 0.0,
    }
    return correct / len(rows), correct, misses, stats


def play_acquired_fact(lattice: Lattice, first: str, second: str, third: str) -> dict[str, float | list[str] | bool]:
    """Three taught nouns. The first kind survives both later sentences.

    "is it" still reads only the latest clock. This read sees all three.
    A question does not move the clocks. The noun gauges stay put.
    """
    row = AcquiredFact(first, second, third)
    kind = acquired_kind(first)
    middle = acquired_kind(second)
    last = acquired_kind(third)
    outside = row.outside_kind()
    answers: list[str] = []
    true_abs: list[float] = []
    false_abs: list[float] = []
    misses: list[str] = []

    def clocks() -> tuple[float, float, float, float, float, float, float, float]:
        return (
            lattice.working,
            lattice.counted,
            lattice.earlier,
            lattice.earlier_count,
            lattice.earliest,
            lattice.earliest_count,
            lattice.opening,
            lattice.opening_count,
        )

    def ask(stated: str, holds: bool, label: str) -> None:
        kept = clocks()
        remainder = lattice.it_surviving_remainder(stated)
        got = lattice.remainder_answer(remainder)
        want = "yes" if holds else "no"
        answers.append(got)
        if holds:
            true_abs.append(abs(remainder))
        else:
            false_abs.append(abs(remainder))
        if got != want and len(misses) < 8:
            misses.append(f"{label} -> {got} (want {want})")
        if clocks() != kept and len(misses) < 8:
            misses.append(f"{label} moved the carrier")

    lattice.clear_working()
    ask(kind, False, f"empty is it {kind}")
    lattice.deposit_acquired(first)
    ask(kind, True, f"is it {kind}")
    ask(middle, kind == middle, f"early is it {middle}")
    lattice.deposit_acquired(second)
    ask(middle, True, f"is it {middle}")
    ask(kind, True, f"kept is it {kind}")
    ask(last, last == kind or last == middle, f"early is it {last}")
    lattice.deposit_acquired(third)
    carry = max(
        abs(lattice.working - lattice.content_value[last]),
        abs(lattice.earlier - lattice.content_value[middle]),
        abs(lattice.earliest - lattice.content_value[kind]),
    )
    ask(last, True, f"is it {last}")
    ask(middle, True, f"kept is it {middle}")
    ask(kind, True, f"surviving is it {kind}")
    ask(outside, False, f"is it {outside}")
    return {
        "ok": not misses,
        "answers": answers,
        "misses": misses,
        "true_max_abs": max(true_abs) if true_abs else 0.0,
        "false_min_abs": min(false_abs) if false_abs else 0.0,
        "carry_max": carry,
    }


def score_acquired_facts(lattice: Lattice, rows: list) -> tuple[float, int, list[str], dict[str, float]]:
    """Each triple of taught nouns is its own fact. The clocks return."""
    if not rows:
        return 0.0, 0, [], {}
    saved = lattice.working
    saved_count = lattice.counted
    saved_earlier = lattice.earlier
    saved_earlier_count = lattice.earlier_count
    saved_earliest = lattice.earliest
    saved_earliest_count = lattice.earliest_count
    saved_opening = lattice.opening
    saved_opening_count = lattice.opening_count
    misses: list[str] = []
    correct = 0
    true_abs: list[float] = []
    false_abs: list[float] = []
    carry: list[float] = []
    try:
        for row in rows:
            played = play_acquired_fact(lattice, row.first, row.second, row.third)
            true_abs.append(float(played["true_max_abs"]))
            false_abs.append(float(played["false_min_abs"]))
            carry.append(float(played["carry_max"]))
            if played["ok"]:
                correct += 1
            elif len(misses) < 8:
                detail = played["misses"][0] if played["misses"] else "miss"
                misses.append(f"{row.text()} {detail}")
    finally:
        lattice.working = saved
        lattice.counted = saved_count
        lattice.earlier = saved_earlier
        lattice.earlier_count = saved_earlier_count
        lattice.earliest = saved_earliest
        lattice.earliest_count = saved_earliest_count
        lattice.opening = saved_opening
        lattice.opening_count = saved_opening_count
    stats = {
        "true_max_abs": max(true_abs) if true_abs else 0.0,
        "false_min_abs": min(false_abs) if false_abs else 0.0,
        "carry_max": max(carry) if carry else 0.0,
    }
    return correct / len(rows), correct, misses, stats


def play_acquired_open(
    lattice: Lattice, first: str, second: str, third: str, fourth: str
) -> dict[str, float | list[str] | bool]:
    """Four taught nouns. The opening kind survives the three sentences after it.

    "is it" still reads only the latest clock. The fact read still sees
    three clocks, so a fourth sentence drops the opening kind from that
    read when no later sentence repeats it. This read sees the opening
    clock. A question does not move the clocks. The noun gauges stay put.
    """
    row = AcquiredOpen(first, second, third, fourth)
    kind = acquired_kind(first)
    middle = acquired_kind(second)
    third_kind = acquired_kind(third)
    last = acquired_kind(fourth)
    outside = row.outside_kind()
    answers: list[str] = []
    true_abs: list[float] = []
    false_abs: list[float] = []
    misses: list[str] = []

    def clocks() -> tuple[float, float, float, float, float, float, float, float]:
        return (
            lattice.working,
            lattice.counted,
            lattice.earlier,
            lattice.earlier_count,
            lattice.earliest,
            lattice.earliest_count,
            lattice.opening,
            lattice.opening_count,
        )

    def ask(stated: str, holds: bool, label: str) -> None:
        kept = clocks()
        remainder = lattice.it_open_remainder(stated)
        got = lattice.remainder_answer(remainder)
        want = "yes" if holds else "no"
        answers.append(got)
        if holds:
            true_abs.append(abs(remainder))
        else:
            false_abs.append(abs(remainder))
        if got != want and len(misses) < 8:
            misses.append(f"{label} -> {got} (want {want})")
        if clocks() != kept and len(misses) < 8:
            misses.append(f"{label} moved the carrier")

    lattice.clear_working()
    ask(kind, False, f"empty is it {kind}")
    lattice.deposit_acquired(first)
    ask(kind, True, f"is it {kind}")
    ask(middle, kind == middle, f"early is it {middle}")
    lattice.deposit_acquired(second)
    ask(middle, True, f"is it {middle}")
    ask(kind, True, f"kept is it {kind}")
    lattice.deposit_acquired(third)
    ask(third_kind, True, f"is it {third_kind}")
    ask(kind, True, f"surviving is it {kind}")
    lattice.deposit_acquired(fourth)
    if lattice.opening == 0.0:
        misses.append("four sentences left the opening clock empty")
    carry = max(
        abs(lattice.working - lattice.content_value[last]),
        abs(lattice.earlier - lattice.content_value[third_kind]),
        abs(lattice.earliest - lattice.content_value[middle]),
        abs(lattice.opening - lattice.content_value[kind]),
    )
    ask(last, True, f"is it {last}")
    ask(kind, True, f"open is it {kind}")
    if outside is not None:
        ask(outside, False, f"is it {outside}")
    if row.first_alone():
        kept = clocks()
        if lattice.remainder_answer(lattice.it_surviving_remainder(kind)) != "no":
            misses.append("fact read kept the opening kind")
        if lattice.predict_it(kind) != "no" or lattice.predict_it(last) != "yes":
            misses.append("is it left the latest clock")
        if clocks() != kept:
            misses.append("the fact question moved the carrier")
    return {
        "ok": not misses,
        "answers": answers,
        "misses": misses,
        "true_max_abs": max(true_abs) if true_abs else 0.0,
        "false_min_abs": min(false_abs) if false_abs else 0.0,
        "carry_max": carry,
    }


def score_acquired_opens(lattice: Lattice, rows: list) -> tuple[float, int, list[str], dict[str, float]]:
    """Each four taught nouns is its own paragraph. The clocks return."""
    if not rows:
        return 0.0, 0, [], {}
    saved = lattice.working
    saved_count = lattice.counted
    saved_earlier = lattice.earlier
    saved_earlier_count = lattice.earlier_count
    saved_earliest = lattice.earliest
    saved_earliest_count = lattice.earliest_count
    saved_opening = lattice.opening
    saved_opening_count = lattice.opening_count
    misses: list[str] = []
    correct = 0
    true_abs: list[float] = []
    false_abs: list[float] = []
    carry: list[float] = []
    try:
        for row in rows:
            played = play_acquired_open(
                lattice, row.first, row.second, row.third, row.fourth
            )
            true_abs.append(float(played["true_max_abs"]))
            false_abs.append(float(played["false_min_abs"]))
            carry.append(float(played["carry_max"]))
            if played["ok"]:
                correct += 1
            elif len(misses) < 8:
                detail = played["misses"][0] if played["misses"] else "miss"
                misses.append(f"{row.text()} {detail}")
    finally:
        lattice.working = saved
        lattice.counted = saved_count
        lattice.earlier = saved_earlier
        lattice.earlier_count = saved_earlier_count
        lattice.earliest = saved_earliest
        lattice.earliest_count = saved_earliest_count
        lattice.opening = saved_opening
        lattice.opening_count = saved_opening_count
    stats = {
        "true_max_abs": max(true_abs) if true_abs else 0.0,
        "false_min_abs": min(false_abs) if false_abs else 0.0,
        "carry_max": max(carry) if carry else 0.0,
    }
    return correct / len(rows), correct, misses, stats


def play_kept_count(lattice: Lattice, count: str, subject: str, second: str) -> dict[str, float | list[str] | bool]:
    """A counted sentence, then a kind sentence. The count stays on the earlier clock.

    "is it two" still reads only the latest count. This read sees both count
    clocks, and the kind read sees both kind clocks. A question does not
    move the clocks. The caller keeps whatever they write.
    """
    counted = Counted(count, subject)
    kind = content_kind(subject)
    other = content_kind(second)
    outside = next(head for head in KIND_HEADS if head != kind and head != other)
    other_count = counted.next_count()
    answers: list[str] = []
    true_abs: list[float] = []
    false_abs: list[float] = []
    misses: list[str] = []

    def clocks() -> tuple[float, float, float, float, float, float, float, float]:
        return (
            lattice.working,
            lattice.counted,
            lattice.earlier,
            lattice.earlier_count,
            lattice.earliest,
            lattice.earliest_count,
            lattice.opening,
            lattice.opening_count,
        )

    def ask(which: str, stated: str, holds: bool, label: str, kept_read: bool) -> None:
        kept = clocks()
        if which == "count":
            remainder = (
                lattice.it_kept_count_remainder(stated)
                if kept_read
                else lattice.it_count_remainder(stated)
            )
        else:
            remainder = (
                lattice.it_kept_remainder(stated)
                if kept_read
                else lattice.it_remainder(stated)
            )
        got = lattice.remainder_answer(remainder)
        want = "yes" if holds else "no"
        answers.append(got)
        if holds:
            true_abs.append(abs(remainder))
        else:
            false_abs.append(abs(remainder))
        if got != want and len(misses) < 8:
            misses.append(f"{label} -> {got} (want {want})")
        if clocks() != kept and len(misses) < 8:
            misses.append(f"{label} moved the carrier")

    lattice.clear_working()
    ask("count", count, False, f"empty is it {count}", False)
    ask("kind", kind, False, f"empty is it {kind}", False)
    lattice.deposit_count(count, subject)
    ask("count", count, True, f"is it {count}", False)
    ask("kind", kind, True, f"is it {kind}", False)
    ask("kind", subject, subject == kind, f"is it {subject}", False)
    lattice.deposit_kind(second)
    carry = max(
        abs(lattice.working - lattice.content_value[other]),
        abs(lattice.earlier - lattice.content_value[kind]),
        abs(lattice.earlier_count - lattice.word_value[count]),
    )
    ask("count", count, False, f"latest is it {count}", False)
    ask("count", count, True, f"kept is it {count}", True)
    ask("count", other_count, False, f"kept is it {other_count}", True)
    ask("kind", kind, True, f"kept is it {kind}", True)
    ask("kind", other, True, f"kept is it {other}", True)
    ask("kind", outside, False, f"kept is it {outside}", True)
    return {
        "ok": not misses,
        "answers": answers,
        "misses": misses,
        "true_max_abs": max(true_abs) if true_abs else 0.0,
        "false_min_abs": min(false_abs) if false_abs else 0.0,
        "carry_max": carry,
    }


def score_kept_counts(lattice: Lattice, rows: list) -> tuple[float, int, list[str], dict[str, float]]:
    """Each row is its own paragraph. The clocks return to where they started."""
    if not rows:
        return 0.0, 0, [], {}
    saved = lattice.working
    saved_count = lattice.counted
    saved_earlier = lattice.earlier
    saved_earlier_count = lattice.earlier_count
    saved_earliest = lattice.earliest
    saved_earliest_count = lattice.earliest_count
    saved_opening = lattice.opening
    saved_opening_count = lattice.opening_count
    misses: list[str] = []
    correct = 0
    true_abs: list[float] = []
    false_abs: list[float] = []
    carry: list[float] = []
    try:
        for row in rows:
            played = play_kept_count(lattice, row.count, row.subject, row.second)
            true_abs.append(float(played["true_max_abs"]))
            false_abs.append(float(played["false_min_abs"]))
            carry.append(float(played["carry_max"]))
            if played["ok"]:
                correct += 1
            elif len(misses) < 8:
                detail = played["misses"][0] if played["misses"] else "miss"
                misses.append(f"{row.text()} {detail}")
    finally:
        lattice.working = saved
        lattice.counted = saved_count
        lattice.earlier = saved_earlier
        lattice.earlier_count = saved_earlier_count
        lattice.earliest = saved_earliest
        lattice.earliest_count = saved_earliest_count
        lattice.opening = saved_opening
        lattice.opening_count = saved_opening_count
    stats = {
        "true_max_abs": max(true_abs) if true_abs else 0.0,
        "false_min_abs": min(false_abs) if false_abs else 0.0,
        "carry_max": max(carry) if carry else 0.0,
    }
    return correct / len(rows), correct, misses, stats


def digit_drift(current: dict[str, float], frozen: dict[str, float]) -> float:
    return max(abs(current[key] - frozen[key]) for key in frozen)


def clock_means(lattice: Lattice, rows: list[Triple], surface: str) -> dict[str, float]:
    work = 0.0
    trace = 0.0
    sim = 0.0
    for row in rows:
        clocks = lattice.read_clocks(row, surface)
        work += clocks.working_norm
        trace += clocks.trace_norm
        sim += clocks.consensus_sim
    n = max(len(rows), 1)
    return {"working_norm": work / n, "trace_norm": trace / n, "consensus_sim": sim / n}
