"""Generated application pass.

A new closed family is a generator plus the same consensus vessel.
Order, single-digit products, a third step, names past nine,
two-digit names re-entering as operands, two places in one
expression, sums that leave 0..99, a hundred name used as an
operand, sums that leave 0..999, a thousand name used as an
operand, two thousand names in one expression, sums that leave
0..9999, a ten-thousand name used as an operand whose result
stays inside 0..10999, two ten-thousand names whose difference
stays inside 0..999, and sums that leave 0..99999 are produced here
and scored together. They add no gauges.
"""

from __future__ import annotations

import time

from fsot_lattice.engine import COLLAPSE, K
from fsot_lattice.lattice import AMP, Lattice
from fsot_lattice.tasks import (
    OP_WORD,
    RADIX,
    HUNDRED_THOUSAND,
    HUNDRED_THOUSAND_CAP,
    TEN_THOUSAND_CAP,
    THOUSAND_CAP,
    THOUSAND_SPAN,
    WORD_OF,
    Claim,
    HundredOp,
    HundredSum,
    HundredThousandSum,
    Lexeme,
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
    all_hundred_ops,
    all_hundred_sums,
    all_lexemes,
    all_thousand_sums,
    census_hundred_thousand_sums,
    census_ten_thousand_ops,
    census_ten_thousand_pairs,
    census_ten_thousand_sums,
    census_thousand_ops,
    census_thousand_pairs,
    all_place_pairs,
    all_place_steps,
    all_products,
    all_spans,
    decimal_name,
    number_name,
    order_digit,
    order_label,
    order_word,
    split_claims,
    split_hundred_ops,
    split_hundred_sums,
    split_lexemes,
    split_thousand_sums,
    hundred_thousand_held,
    hundred_thousand_kind,
    ten_thousand_held,
    ten_thousand_kind,
    ten_thousand_op_held,
    ten_thousand_pair_held,
    thousand_op_held,
    thousand_pair_held,
    split_place_pairs,
    split_place_steps,
    split_products,
    split_spans,
)

DROP = COLLAPSE / AMP
# Same floor as the additive gauge. A generated fold has to land inside it.
GAUGE_FLOOR = 1e-4


def _partition(rows, train, hold, name: str) -> None:
    universe = set(rows)
    train_set = set(train)
    hold_set = set(hold)
    if train_set | hold_set != universe or train_set & hold_set:
        raise RuntimeError(f"{name} split is not a partition")


def check_generated(lattice: Lattice) -> None:
    """Mechanism checks that hold before the gauges have settled."""
    if lattice.order_answer(0.0) != "same":
        raise RuntimeError("a zero remainder did not read as same")
    if lattice.order_answer(K) != "more" or lattice.order_answer(-K) != "less":
        raise RuntimeError("the order sign did not separate more from less")
    once = Product(1, 3, 3)
    if abs(lattice.product_quantity(once, "digit") - lattice.digit_value["3"]) > 1e-9:
        raise RuntimeError("one times a gauge did not return that gauge")
    none = Product(0, 5, 0)
    if lattice.product_quantity(none, "digit") != 0.0:
        raise RuntimeError("zero times a gauge did not stay empty")
    twice = Product(2, 3, 6)
    if abs(lattice.product_quantity(twice, "digit") - lattice.algebraic_product(twice, "digit")) > 1e-9:
        raise RuntimeError("repeated addition left the algebraic product")
    prod_train, prod_hold = split_products()
    _partition(all_products(), prod_train, prod_hold, "product")
    zeros = [row for row in all_products() if row.uses_zero()]
    if not any(row in set(prod_train) for row in zeros) or not any(row in set(prod_hold) for row in zeros):
        raise RuntimeError("product split hid every zero factor on one side")
    span_train, span_hold = split_spans()
    _partition(all_spans(), span_train, span_hold, "span")
    cancelled = [row for row in all_spans() if row.first_is_zero()]
    if not any(row in set(span_train) for row in cancelled) or not any(row in set(span_hold) for row in cancelled):
        raise RuntimeError("third-step split hid every zero start on one side")
    back = Span(4, "-", 4, "+", 3, "-", 3, 0)
    if abs(lattice.span_quantity(back, "digit")) > 1e-9:
        raise RuntimeError("a cancel-add-cancel span did not return to zero")
    if lattice.predict_span(back, "digit") != "0":
        raise RuntimeError("nearest gauge missed the third-step zero")
    _check_lexicon(lattice)
    _check_reentry(lattice)
    _check_pairs(lattice)
    _check_hundreds(lattice)
    _check_ops(lattice)
    _check_thousands(lattice)
    _check_thou_ops(lattice)
    _check_thou_pairs(lattice)
    _check_ten_thousands(lattice)
    _check_ten_thou_ops(lattice)
    _check_ten_thou_pairs(lattice)
    _check_hundred_thousands(lattice)


def _check_lexicon(lattice: Lattice) -> None:
    """Place-value names on a pure line, and the two init sums that already equal ten."""
    if number_name(10) != "ten" or number_name(18) != "eighteen":
        raise RuntimeError("teen spellings left the place table")
    if number_name(20) != "twenty" or number_name(40) != "forty":
        raise RuntimeError("exact tens kept a units word")
    if number_name(42) != "forty-two" or number_name(81) != "eighty-one":
        raise RuntimeError("a compound place lost its units word")
    if decimal_name(42) != "42":
        raise RuntimeError("the decimal spelling left the integer")
    step = K
    gauges = [index * step for index in range(RADIX)]
    ten = gauges[9] + gauges[1]
    for named in range(RADIX, 9 * 9 + 1):
        got, _rem, margin, _dist = lattice.read_place(named * step, "digit", ten, gauges)
        if got != named:
            raise RuntimeError(f"place read of {named} returned {got}")
        if margin <= DROP:
            raise RuntimeError(f"place read of {named} sat inside the drop")
    nine_plus = Lexeme("sum", 9, 1, 10)
    two_fives = Lexeme("product", 2, 5, 10)
    if lattice.predict_lexeme(nine_plus, "digit") != "10":
        raise RuntimeError("nine plus one did not read as 10")
    if lattice.predict_lexeme(two_fives, "digit") != "10":
        raise RuntimeError("two times five did not read as 10")
    rows = all_lexemes()
    train, hold = split_lexemes()
    _partition(rows, train, hold, "lexeme")
    closed_sums = sum(1 for left in range(RADIX) for right in range(RADIX) if left + right < RADIX)
    sums = [row for row in rows if row.kind == "sum"]
    products = [row for row in rows if row.kind == "product"]
    if len(sums) != RADIX * RADIX - closed_sums:
        raise RuntimeError("overflow sums are not the pairs past nine")
    if not products or any(row.result < RADIX for row in rows):
        raise RuntimeError("the lexicon below ten belongs to the digit gauges")
    if any(row.result > 9 * 9 for row in products):
        raise RuntimeError("a product left the digit square")

    def _both(pred, label: str) -> None:
        if not any(pred(row) for row in train) or not any(pred(row) for row in hold):
            raise RuntimeError(label)

    _both(lambda row: row.kind == "sum", "lexeme split hid every overflow sum on one side")
    _both(lambda row: row.kind == "product", "lexeme split hid every wide product on one side")
    _both(lambda row: row.exact_ten(), "lexeme split hid every exact ten on one side")
    _both(lambda row: row.result < 2 * RADIX, "lexeme split hid every teen on one side")
    _both(lambda row: row.result >= 2 * RADIX, "lexeme split hid every higher place on one side")


def _check_reentry(lattice: Lattice) -> None:
    """The ten-step folds back into the name, on a pure line and on the fresh digit gauges."""
    step = K
    gauges = [index * step for index in range(RADIX)]
    ten = gauges[9] + gauges[1]
    cases = (
        (24, "+", 3, 27),
        (24, "-", 7, 17),
        (10, "-", 9, 1),
        (20, "-", 9, 11),
        (24, "+", 0, 24),
        (15, "+", 5, 20),
        (90, "+", 9, 99),
        (99, "-", 0, 99),
    )
    for place, op, digit, result in cases:
        quantity = lattice.compose_place(place, ten, gauges)
        sign = 1 if op == "+" else -1
        quantity = lattice.consensus_quantity(quantity, gauges[digit], sign)
        got = lattice.read_place(quantity, "digit", ten, gauges)[0]
        if got != result:
            raise RuntimeError(f"rebuilt {place}{op}{digit} read {got}")
    for place in range(RADIX, RADIX * RADIX):
        named, remainder, margin, _dist = lattice.read_place(
            lattice.place_quantity(place, "digit"),
            "digit",
        )
        if named != place:
            raise RuntimeError(f"rebuilt place {place} read {named}")
        if margin <= DROP:
            raise RuntimeError(f"rebuilt place {place} sat inside the drop")
        if place % RADIX == 0 and abs(remainder) >= DROP:
            raise RuntimeError(f"exact ten {place} left a remainder")
    rows = all_place_steps()
    train, hold = split_place_steps()
    _partition(rows, train, hold, "place")
    plus = sum(
        1
        for place in range(RADIX, RADIX * RADIX)
        for digit in range(RADIX)
        if place + digit < RADIX * RADIX
    )
    minus = (RADIX * RADIX - RADIX) * RADIX
    if len(rows) != plus + minus:
        raise RuntimeError("place steps left the two-digit table")

    def _both(pred, label: str) -> None:
        if not any(pred(row) for row in train) or not any(pred(row) for row in hold):
            raise RuntimeError(label)

    _both(lambda row: row.op == "+", "place split hid every addition on one side")
    _both(lambda row: row.op == "-", "place split hid every subtraction on one side")
    _both(lambda row: row.uses_zero(), "place split hid every zero digit on one side")
    _both(lambda row: row.borrows(), "place split hid every borrow on one side")
    _both(lambda row: row.carries(), "place split hid every carry on one side")
    _both(lambda row: row.under_ten(), "place split hid every return under ten on one side")
    _both(lambda row: row.exact_ten(), "place split hid every exact ten result on one side")
    _both(lambda row: row.place < 2 * RADIX, "place split hid every teen operand on one side")
    _both(lambda row: row.place >= 90, "place split hid every ninety on one side")


def _check_pairs(lattice: Lattice) -> None:
    """Two rebuilt names share one pass. Mixed units are checked on a pure line.

    The fresh digit gauges carry an extra K on every units place, so a mixed
    units sum is not the result gauge yet. Rows whose extras cancel are.
    """
    step = K
    gauges = [index * step for index in range(RADIX)]
    ten = gauges[9] + gauges[1]
    cases = (
        (24, "+", 17, 41),
        (24, "-", 17, 7),
        (24, "-", 24, 0),
        (20, "+", 10, 30),
        (50, "+", 49, 99),
        (15, "+", 15, 30),
        (30, "-", 11, 19),
        (99, "-", 90, 9),
        (10, "+", 10, 20),
        (18, "-", 11, 7),
    )
    for left, op, right, result in cases:
        quantity = lattice.compose_place(left, ten, gauges)
        other = lattice.compose_place(right, ten, gauges)
        sign = 1 if op == "+" else -1
        quantity = lattice.consensus_quantity(quantity, other, sign)
        got, remainder, margin, _dist = lattice.read_place(quantity, "digit", ten, gauges)
        if got != result:
            raise RuntimeError(f"rebuilt {left}{op}{right} read {got}")
        if margin <= DROP:
            raise RuntimeError(f"rebuilt {left}{op}{right} sat inside the drop")
        if result % RADIX == 0 and abs(remainder) >= DROP:
            raise RuntimeError(f"exact ten {left}{op}{right} left a remainder")
    fresh = (
        PlacePair(24, "-", 24, 0),
        PlacePair(20, "+", 10, 30),
        PlacePair(10, "-", 10, 0),
    )
    for row in fresh:
        got = lattice.predict_place_pair(row, "digit")
        if got != row.digit_answer():
            raise RuntimeError(f"fresh {row.digit_prompt()}{row.digit_answer()} read {got}")
    rows = all_place_pairs()
    train, hold = split_place_pairs()
    _partition(rows, train, hold, "pair")
    limit = RADIX * RADIX
    plus_span = limit - 2 * RADIX
    minus_span = limit - RADIX
    plus_n = plus_span * (plus_span + 1) // 2
    minus_n = minus_span * (minus_span + 1) // 2
    if sum(1 for row in rows if row.op == "+") != plus_n:
        raise RuntimeError("place-pair additions left the two-digit table")
    if sum(1 for row in rows if row.op == "-") != minus_n:
        raise RuntimeError("place-pair subtractions left the two-digit table")
    if len(rows) != plus_n + minus_n:
        raise RuntimeError("place pairs left the two-digit table")

    def _both(pred, label: str) -> None:
        if not any(pred(row) for row in train) or not any(pred(row) for row in hold):
            raise RuntimeError(label)

    _both(lambda row: row.op == "+", "pair split hid every addition on one side")
    _both(lambda row: row.op == "-", "pair split hid every subtraction on one side")
    _both(lambda row: row.is_zero(), "pair split hid every zero result on one side")
    _both(lambda row: row.under_ten(), "pair split hid every return under ten on one side")
    _both(lambda row: row.exact_ten(), "pair split hid every exact ten result on one side")
    _both(lambda row: row.carries(), "pair split hid every carry on one side")
    _both(lambda row: row.borrows(), "pair split hid every borrow on one side")
    _both(lambda row: row.left < 2 * RADIX, "pair split hid every teen left operand on one side")
    _both(lambda row: row.left >= 90, "pair split hid every ninety on the left on one side")
    _both(lambda row: row.right >= 90, "pair split hid every ninety on the right on one side")


def _check_hundreds(lattice: Lattice) -> None:
    """The hundred place is the ten-step counted ten times.

    Mixed units are checked on a pure line. The fresh digit gauges are exact
    when the extra K on each units place cancels into that hundred.
    """
    if number_name(100) != "one hundred" or number_name(101) != "one hundred one":
        raise RuntimeError("the hundred spelling dropped the hundreds word")
    if number_name(110) != "one hundred ten" or number_name(122) != "one hundred twenty-two":
        raise RuntimeError("a hundred compound lost its lower places")
    if number_name(198) != "one hundred ninety-eight":
        raise RuntimeError("the top of this hundred lost its tens word")
    if number_name(42) != "forty-two":
        raise RuntimeError("the hundred spelling rewrote a two-digit name")
    if decimal_name(100) != "100":
        raise RuntimeError("the decimal hundred left the integer")
    step = K
    gauges = [index * step for index in range(RADIX)]
    ten = gauges[9] + gauges[1]
    hundred = ten * RADIX
    cases = (
        (50, 50, 100, "pair"),
        (99, 1, 100, "step"),
        (24, 76, 100, "pair"),
        (60, 41, 101, "pair"),
        (99, 9, 108, "step"),
        (99, 99, 198, "pair"),
        (15, 95, 110, "pair"),
        (42, 80, 122, "pair"),
        (25, 75, 100, "pair"),
        (33, 67, 100, "pair"),
        (50, 49, 99, "pair"),
        (89, 10, 99, "pair"),
        (99, 0, 99, "step"),
        (91, 8, 99, "step"),
    )

    def _read(left: int, right: int, kind: str) -> tuple[int, float, float]:
        quantity = lattice.compose_place(left, ten, gauges)
        other = gauges[right] if kind == "step" else lattice.compose_place(right, ten, gauges)
        quantity = lattice.consensus_quantity(quantity, other, 1)
        named, remainder, margin, _dist = lattice.read_place(quantity, "digit", ten, gauges, hundred)
        return named, remainder, margin

    for left, right, result, kind in cases:
        named, remainder, margin = _read(left, right, kind)
        if named != result:
            raise RuntimeError(f"rebuilt {left}+{right} read {named}")
        if margin <= DROP:
            raise RuntimeError(f"rebuilt {left}+{right} sat inside the drop")
        if result % RADIX == 0 and abs(remainder) >= DROP:
            raise RuntimeError(f"exact ten {left}+{right} left a remainder")
    for n in range(RADIX * RADIX, 2 * RADIX * RADIX - 1):
        hundreds, rest = divmod(n, RADIX * RADIX)
        tens, units = divmod(rest, RADIX)
        quantity = 0.0
        for _ in range(hundreds):
            quantity = lattice.consensus_quantity(quantity, hundred, 1)
        for _ in range(tens):
            quantity = lattice.consensus_quantity(quantity, ten, 1)
        if units:
            quantity = lattice.consensus_quantity(quantity, gauges[units], 1)
        named, remainder, margin, _dist = lattice.read_place(quantity, "digit", ten, gauges, hundred)
        if named != n:
            raise RuntimeError(f"rebuilt hundred {n} read {named}")
        if margin <= DROP:
            raise RuntimeError(f"rebuilt hundred {n} sat inside the drop")
        if n % RADIX == 0 and abs(remainder) >= DROP:
            raise RuntimeError(f"exact ten {n} left a remainder")
    live_ten = lattice.ten_quantity("digit")
    live_hundred = lattice.hundred_quantity("digit")
    if abs(live_hundred - RADIX * live_ten) > 1e-9:
        raise RuntimeError("the hundred-step left ten counted ten times")
    if lattice.read_place(live_hundred, "digit")[0] != RADIX * RADIX:
        raise RuntimeError("the hundred-step did not read as 100")
    fresh = (
        HundredSum("pair", 50, 50, 100),
        HundredSum("pair", 10, 90, 100),
        HundredSum("pair", 24, 76, 100),
        HundredSum("step", 99, 1, 100),
        HundredSum("step", 91, 9, 100),
    )
    for row in fresh:
        got = lattice.predict_hundred(row, "digit")
        if got != row.digit_answer():
            raise RuntimeError(f"fresh {row.digit_prompt()}{row.digit_answer()} read {got}")
    if lattice.predict_place_pair(PlacePair(50, "+", 49, 99), "digit") != "99":
        raise RuntimeError("fifty plus forty-nine shed into the hundred")
    if lattice.predict_place_step(PlaceStep(91, "+", 8, 99), "digit") != "99":
        raise RuntimeError("ninety-one plus eight shed into the hundred")
    rows = all_hundred_sums()
    train, hold = split_hundred_sums()
    _partition(rows, train, hold, "hundred")
    limit = RADIX * RADIX
    pair_n = sum(
        1
        for left in range(RADIX, limit)
        for right in range(RADIX, limit)
        if left + right >= limit
    )
    step_n = (RADIX - 1) * RADIX // 2
    if sum(1 for row in rows if row.kind == "pair") != pair_n:
        raise RuntimeError("hundred pairs left the sums past ninety-nine")
    if sum(1 for row in rows if row.kind == "step") != step_n:
        raise RuntimeError("hundred steps left the sums past ninety-nine")
    if len(rows) != pair_n + step_n:
        raise RuntimeError("hundred sums left 100..999")
    exact = sum(1 for row in rows if row.exact_hundred())
    if exact != (limit - 2 * RADIX + 1) + (RADIX - 1):
        raise RuntimeError("the exact hundreds are not the ways to write 100")

    def _both(pred, label: str) -> None:
        if not any(pred(row) for row in train) or not any(pred(row) for row in hold):
            raise RuntimeError(label)

    _both(lambda row: row.kind == "pair", "hundred split hid every pair on one side")
    _both(lambda row: row.kind == "step", "hundred split hid every digit step on one side")
    _both(lambda row: row.exact_hundred(), "hundred split hid every exact hundred on one side")
    _both(lambda row: row.exact_ten(), "hundred split hid every exact ten on one side")
    _both(lambda row: row.carries(), "hundred split hid every carry on one side")
    _both(lambda row: not row.carries(), "hundred split hid every sum without a carry on one side")
    _both(lambda row: row.low(), "hundred split hid the low band on one side")
    _both(lambda row: row.high(), "hundred split hid the high band on one side")
    _both(lambda row: row.left < 2 * RADIX, "hundred split hid every teen left operand on one side")
    _both(lambda row: row.left >= 90, "hundred split hid every ninety on the left on one side")
    _both(lambda row: row.right >= 90, "hundred split hid every ninety on the right on one side")


def _op_counts(rows: list[HundredOp]) -> dict[str, int]:
    span = RADIX * RADIX
    counts = {
        "plus": 0,
        "minus": 0,
        "zero": 0,
        "borrow": 0,
        "carry": 0,
        "under": 0,
        "ten": 0,
        "hundred": 0,
        "cross": 0,
        "ones": 0,
        "wide": 0,
        "left100": 0,
        "left900": 0,
        "right90": 0,
        "top": 0,
    }
    for row in rows:
        if row.op == "+":
            counts["plus"] += 1
        else:
            counts["minus"] += 1
        if row.uses_zero():
            counts["zero"] += 1
        if row.borrows():
            counts["borrow"] += 1
        if row.carries():
            counts["carry"] += 1
        if row.under_hundred():
            counts["under"] += 1
        if row.exact_ten():
            counts["ten"] += 1
        if row.exact_hundred():
            counts["hundred"] += 1
        if row.crosses():
            counts["cross"] += 1
        if row.ones():
            counts["ones"] += 1
        if row.wide():
            counts["wide"] += 1
        if row.left < 2 * span:
            counts["left100"] += 1
        if row.left >= 9 * span:
            counts["left900"] += 1
        if row.right >= 9 * RADIX:
            counts["right90"] += 1
        if row.result >= 9 * span:
            counts["top"] += 1
    return counts


def _check_ops(lattice: Lattice) -> None:
    """A hundred name re-enters as an operand. The result stays inside 0..999.

    Mixed units are checked on a pure line. The fresh digit gauges are exact
    when the extra K on each units place cancels.
    """
    if number_name(124) != "one hundred twenty-four" or number_name(200) != "two hundred":
        raise RuntimeError("the hundred operand spelling dropped a hundreds word")
    if number_name(297) != "two hundred ninety-seven" or number_name(990) != "nine hundred ninety":
        raise RuntimeError("a hundred operand compound lost its lower places")
    if number_name(76) != "seventy-six":
        raise RuntimeError("the hundred operand spelling rewrote a two-digit name")
    step = K
    gauges = [index * step for index in range(RADIX)]
    ten = gauges[9] + gauges[1]
    hundred = ten * RADIX
    span = RADIX * RADIX
    thousand = span * RADIX
    syn_big: dict[int, float] = {}

    def _syn_big(number: int) -> float:
        if number not in syn_big:
            syn_big[number] = lattice.compose_hundred(number, hundred, ten, gauges)
        return syn_big[number]

    syn_place = {place: lattice.compose_place(place, ten, gauges) for place in range(RADIX, span)}
    for number in (100, 124, 200, 297, 550, 990, 999):
        if abs(_syn_big(number) - number * step) > 1e-9:
            raise RuntimeError(f"synthetic {number} left the integer line")
    cases = (
        (100, "+", 24, 124),
        (100, "-", 24, 76),
        (100, "+", 3, 103),
        (100, "-", 99, 1),
        (100, "+", 0, 100),
        (100, "-", 1, 99),
        (150, "+", 50, 200),
        (198, "+", 99, 297),
        (250, "-", 50, 200),
        (124, "+", 3, 127),
        (101, "+", 9, 110),
        (190, "+", 10, 200),
        (999, "-", 9, 990),
        (999, "-", 0, 999),
        (900, "+", 99, 999),
        (500, "+", 50, 550),
        (200, "-", 99, 101),
    )

    def _syn_read(left: int, op: str, right: int) -> tuple[int, float, float]:
        other = gauges[right] if right < RADIX else syn_place[right]
        sign = 1 if op == "+" else -1
        quantity = lattice.consensus_quantity(_syn_big(left), other, sign)
        named, remainder, margin, _dist = lattice.read_place(quantity, "digit", ten, gauges, hundred)
        return named, remainder, margin

    for left, op, right, result in cases:
        named, remainder, margin = _syn_read(left, op, right)
        if named != result:
            raise RuntimeError(f"rebuilt {left}{op}{right} read {named}")
        if margin <= DROP:
            raise RuntimeError(f"rebuilt {left}{op}{right} sat inside the drop")
        if result % RADIX == 0 and abs(remainder) >= DROP:
            raise RuntimeError(f"exact ten {left}{op}{right} left a remainder")
    stay = lattice.consensus_quantity(syn_place[50], syn_place[49], 1)
    if lattice.read_place(stay, "digit", ten, gauges, hundred)[0] != 99:
        raise RuntimeError("fifty plus forty-nine shed into the hundred")
    stay_step = lattice.consensus_quantity(syn_place[91], gauges[8], 1)
    if lattice.read_place(stay_step, "digit", ten, gauges, hundred)[0] != 99:
        raise RuntimeError("ninety-one plus eight shed into the hundred")
    live_ten = lattice.ten_quantity("digit")
    live_gauges = lattice.gauge_line("digit")
    live_hundred = lattice.hundred_quantity("digit")
    live_plus = lattice.routed_sign("+")
    lowers = {rest: lattice.compose_place(rest, live_ten, live_gauges) for rest in range(1, span)}
    built: dict[int, float] = {}
    acc = 0.0
    for hundreds in range(1, RADIX):
        acc = lattice.consensus_quantity(acc, live_hundred, live_plus)
        base = hundreds * span
        built[base] = acc
        for rest, place_q in lowers.items():
            built[base + rest] = lattice.consensus_quantity(acc, place_q, live_plus)
    for number in (100, 101, 124, 200, 250, 999):
        direct = lattice.compose_hundred(number, live_hundred, live_ten, live_gauges)
        if abs(direct - built[number]) > 1e-9:
            raise RuntimeError(f"compose_hundred left the cached fold at {number}")
        if abs(lattice.algebraic_hundred(number, "digit") - built[number]) > 1e-9:
            raise RuntimeError(f"fresh hundred {number} left its algebraic value")
    for number in range(span, thousand):
        named, remainder, margin, _dist = lattice.read_place(
            built[number], "digit", live_ten, live_gauges, live_hundred
        )
        if named != number:
            raise RuntimeError(f"rebuilt hundred operand {number} read {named}")
        if margin <= DROP:
            raise RuntimeError(f"rebuilt hundred operand {number} sat inside the drop")
        if number % RADIX == 0 and abs(remainder) >= DROP:
            raise RuntimeError(f"exact ten {number} left a remainder")
    fresh = (
        HundredOp(100, "+", 24, 124),
        HundredOp(100, "-", 24, 76),
        HundredOp(100, "+", 0, 100),
        HundredOp(100, "-", 99, 1),
        HundredOp(100, "+", 99, 199),
        HundredOp(150, "+", 50, 200),
        HundredOp(124, "-", 24, 100),
        HundredOp(200, "-", 50, 150),
    )
    for row in fresh:
        if abs(lattice.hundred_op_quantity(row, "digit") - lattice.algebraic_hundred_op(row, "digit")) > 1e-9:
            raise RuntimeError(f"fresh {row.digit_prompt()}{row.digit_answer()} left the algebraic sum")
        got = lattice.predict_hundred_op(row, "digit")
        if got != row.digit_answer():
            raise RuntimeError(f"fresh {row.digit_prompt()}{row.digit_answer()} read {got}")
    rows = all_hundred_ops()
    train, hold = split_hundred_ops()
    _partition(rows, train, hold, "hundred operand")
    counts = _op_counts(rows)
    expected = {
        "plus": 85050,
        "minus": 90000,
        "zero": 1800,
        "borrow": 40500,
        "carry": 38025,
        "under": 4950,
        "ten": 17460,
        "hundred": 1701,
        "cross": 84150,
        "ones": 17955,
        "wide": 157095,
        "left100": 20000,
        "left900": 15050,
        "right90": 17055,
        "top": 15050,
    }
    for key, value in expected.items():
        if counts[key] != value:
            raise RuntimeError(f"hundred operand {key} count is {counts[key]}")
    if len(rows) != expected["plus"] + expected["minus"]:
        raise RuntimeError("hundred operands left 0..999")
    if len(train) != 140040 or len(hold) != 35010:
        raise RuntimeError(f"hundred operand split is {len(train)}/{len(hold)} of {len(rows)}")
    for row in rows:
        if not (span <= row.left < thousand and 0 <= row.right < span and 0 <= row.result < thousand):
            raise RuntimeError("hundred operand left the operand table")
        if row.op == "+" and row.result != row.left + row.right:
            raise RuntimeError("hundred operand addition left the sum")
        if row.op == "-" and row.result != row.left - row.right:
            raise RuntimeError("hundred operand subtraction left the difference")
        if row.op not in ("+", "-"):
            raise RuntimeError("hundred operand used an operator outside plus and minus")

    def _both(pred, label: str) -> None:
        if not any(pred(row) for row in train) or not any(pred(row) for row in hold):
            raise RuntimeError(label)

    _both(lambda row: row.op == "+", "hundred operand split hid every addition on one side")
    _both(lambda row: row.op == "-", "hundred operand split hid every subtraction on one side")
    _both(lambda row: row.uses_zero(), "hundred operand split hid every zero on one side")
    _both(lambda row: row.borrows(), "hundred operand split hid every borrow on one side")
    _both(lambda row: row.carries(), "hundred operand split hid every carry on one side")
    _both(lambda row: row.under_hundred(), "hundred operand split hid every result under one hundred on one side")
    _both(lambda row: row.exact_ten(), "hundred operand split hid every exact ten on one side")
    _both(lambda row: row.exact_hundred(), "hundred operand split hid every exact hundred on one side")
    _both(lambda row: row.crosses(), "hundred operand split hid every hundred crossing on one side")
    _both(lambda row: row.ones(), "hundred operand split hid every digit on the right on one side")
    _both(lambda row: row.wide(), "hundred operand split hid every place on the right on one side")
    _both(lambda row: row.left < 2 * span, "hundred operand split hid every left name under two hundred on one side")
    _both(lambda row: row.left >= 9 * span, "hundred operand split hid every nine hundred on the left on one side")
    _both(lambda row: row.right >= 9 * RADIX, "hundred operand split hid every ninety on the right on one side")
    _both(lambda row: row.result >= 9 * span, "hundred operand split hid every result from nine hundred up on one side")


def _check_thousands(lattice: Lattice) -> None:
    """The thousand place is the hundred-step counted ten times.

    Mixed units are checked on a pure line. The fresh digit gauges are exact
    when the extra K on each units place cancels into that thousand.
    """
    spells = (
        (1000, "one thousand"),
        (1001, "one thousand one"),
        (1008, "one thousand eight"),
        (1010, "one thousand ten"),
        (1025, "one thousand twenty-five"),
        (1098, "one thousand ninety-eight"),
        (1100, "one thousand one hundred"),
        (2000, "two thousand"),
        (9999, "nine thousand nine hundred ninety-nine"),
        (100, "one hundred"),
        (999, "nine hundred ninety-nine"),
        (42, "forty-two"),
    )
    for number, spelling in spells:
        got = number_name(number)
        if got != spelling:
            raise RuntimeError(f"{number} spelled {got}")
    if number_name(10000) != "ten thousand":
        raise RuntimeError("ten thousand left the spelling table")
    if number_name(11000) != "eleven thousand" or number_name(110000) != "?":
        raise RuntimeError("the spelling table left eleven thousand or passed one hundred ten thousand")
    if decimal_name(1000) != "1000":
        raise RuntimeError("the decimal thousand left the integer")
    step = K
    gauges = [index * step for index in range(RADIX)]
    ten = gauges[9] + gauges[1]
    hundred = ten * RADIX
    thousand = hundred * RADIX
    span = RADIX * RADIX
    syn_place = {place: lattice.compose_place(place, ten, gauges) for place in range(RADIX, span)}
    syn_big: dict[int, float] = {}

    def _syn_big(number: int) -> float:
        if number not in syn_big:
            syn_big[number] = lattice.compose_hundred(number, hundred, ten, gauges)
        return syn_big[number]

    def _syn_read(left: int, right: int) -> tuple[int, float, float]:
        other = gauges[right] if right < RADIX else syn_place[right]
        quantity = lattice.consensus_quantity(_syn_big(left), other, 1)
        named, remainder, margin, _dist = lattice.read_place(
            quantity, "digit", ten, gauges, hundred, thousand
        )
        return named, remainder, margin

    cases = (
        (999, 1, 1000),
        (901, 99, 1000),
        (950, 50, 1000),
        (990, 10, 1000),
        (991, 9, 1000),
        (999, 9, 1008),
        (999, 99, 1098),
        (960, 41, 1001),
        (975, 50, 1025),
        (990, 20, 1010),
        (999, 2, 1001),
    )
    for left, right, result in cases:
        named, remainder, margin = _syn_read(left, right)
        if named != result:
            raise RuntimeError(f"rebuilt {left}+{right} read {named}")
        if margin <= DROP:
            raise RuntimeError(f"rebuilt {left}+{right} sat inside the drop")
        if result % RADIX == 0 and abs(remainder) >= DROP:
            raise RuntimeError(f"exact ten {left}+{right} left a remainder")
    stay = (
        (999, 0, 999),
        (950, 49, 999),
        (991, 8, 999),
        (901, 98, 999),
    )
    for left, right, result in stay:
        named, _remainder, margin = _syn_read(left, right)
        if named != result:
            raise RuntimeError(f"rebuilt {left}+{right} shed into the thousand")
        if margin <= DROP:
            raise RuntimeError(f"rebuilt {left}+{right} sat inside the drop")

    def _syn_name(number: int) -> float:
        rest = number - span * RADIX
        acc = lattice.consensus_quantity(0.0, thousand, 1)
        if rest:
            place = gauges[rest] if rest < RADIX else lattice.compose_place(rest, ten, gauges)
            acc = lattice.consensus_quantity(acc, place, 1)
        return acc

    for number in (1000, 1001, 1008, 1010, 1025, 1098):
        if abs(_syn_name(number) - number * step) > 1e-9:
            raise RuntimeError(f"synthetic {number} left the integer line")
        named, remainder, margin, _dist = lattice.read_place(
            _syn_name(number), "digit", ten, gauges, hundred, thousand
        )
        if named != number:
            raise RuntimeError(f"rebuilt thousand {number} read {named}")
        if margin <= DROP:
            raise RuntimeError(f"rebuilt thousand {number} sat inside the drop")
        if number % RADIX == 0 and abs(remainder) >= DROP:
            raise RuntimeError(f"exact ten {number} left a remainder")
    live_ten = lattice.ten_quantity("digit")
    live_gauges = lattice.gauge_line("digit")
    live_hundred = lattice.hundred_quantity("digit")
    live_thousand = lattice.thousand_quantity("digit")
    live_plus = lattice.routed_sign("+")
    if abs(live_thousand - RADIX * live_hundred) > 1e-9:
        raise RuntimeError("the thousand-step left the hundred counted ten times")
    alg_thousand = RADIX * RADIX * (lattice._gauge(9, "digit") + lattice._gauge(1, "digit"))
    if abs(live_thousand - alg_thousand) > 1e-9:
        raise RuntimeError("the fresh thousand left its algebraic value")
    if lattice.read_place(live_thousand, "digit")[0] != span * RADIX:
        raise RuntimeError("the thousand-step did not read as 1000")
    lowers = {rest: lattice.compose_place(rest, live_ten, live_gauges) for rest in range(1, span - 1)}
    base = lattice.consensus_quantity(0.0, live_thousand, live_plus)
    if abs(base - live_thousand) > 1e-9:
        raise RuntimeError("one thousand left the thousand-step")
    built = {span * RADIX: base}
    for rest, place_q in lowers.items():
        built[span * RADIX + rest] = lattice.consensus_quantity(base, place_q, live_plus)
    for number in (1000, 1001, 1010, 1025, 1098):
        rest = number - span * RADIX
        alg = alg_thousand if rest == 0 else alg_thousand + lattice.algebraic_place(rest, "digit")
        if abs(built[number] - alg) > 1e-9:
            raise RuntimeError(f"fresh thousand {number} left its algebraic value")
    for number in range(span * RADIX, span * RADIX + span - 1):
        named, remainder, margin, _dist = lattice.read_place(
            built[number], "digit", live_ten, live_gauges, live_hundred, live_thousand
        )
        if named != number:
            raise RuntimeError(f"rebuilt thousand {number} read {named}")
        if margin <= DROP:
            raise RuntimeError(f"rebuilt thousand {number} sat inside the drop")
        if number % RADIX == 0 and abs(remainder) >= DROP:
            raise RuntimeError(f"exact ten {number} left a remainder")
    fresh = (
        ThousandSum("step", 999, 1, 1000),
        ThousandSum("wide", 901, 99, 1000),
        ThousandSum("wide", 950, 50, 1000),
        ThousandSum("wide", 990, 10, 1000),
        ThousandSum("step", 991, 9, 1000),
    )
    for row in fresh:
        if abs(lattice.thousand_sum_quantity(row, "digit") - lattice.algebraic_thousand_sum(row, "digit")) > 1e-9:
            raise RuntimeError(f"fresh {row.digit_prompt()}{row.digit_answer()} left the algebraic sum")
        got = lattice.predict_thousand(row, "digit")
        if got != row.digit_answer():
            raise RuntimeError(f"fresh {row.digit_prompt()}{row.digit_answer()} read {got}")
    for row in (
        HundredOp(999, "+", 0, 999),
        HundredOp(950, "+", 49, 999),
        HundredOp(991, "+", 8, 999),
        HundredOp(901, "+", 98, 999),
    ):
        got = lattice.predict_hundred_op(row, "digit")
        if got != row.digit_answer():
            raise RuntimeError(f"fresh {row.digit_prompt()}{row.digit_answer()} read {got}")
    rows = all_thousand_sums()
    train, hold = split_thousand_sums()
    _partition(rows, train, hold, "thousand")
    universe = set(rows)
    demos = fresh + (
        ThousandSum("step", 999, 9, 1008),
        ThousandSum("wide", 960, 41, 1001),
        ThousandSum("wide", 975, 50, 1025),
        ThousandSum("wide", 990, 20, 1010),
        ThousandSum("wide", 999, 99, 1098),
    )
    for row in demos:
        if row not in universe:
            raise RuntimeError(f"{row.digit_prompt()}{row.digit_answer()} is outside the thousand family")
    thousand = span * RADIX
    cap = thousand * RADIX
    if len(rows) != span * (span - 1) // 2:
        raise RuntimeError("thousand sums left 1000..1098")
    step_n = (RADIX - 1) * RADIX // 2
    if sum(1 for row in rows if row.kind == "step") != step_n:
        raise RuntimeError("thousand steps left the digit additions past 999")
    if sum(1 for row in rows if row.kind == "wide") != len(rows) - step_n:
        raise RuntimeError("thousand wide rows left the place additions past 999")
    if sum(1 for row in rows if row.exact_thousand()) != span - 1:
        raise RuntimeError("the exact thousands are not the ways to write 1000")
    if sum(1 for row in rows if row.exact_ten()) != sum(span - 1 - RADIX * band for band in range(RADIX)):
        raise RuntimeError("thousand exact tens left 1000, 1010, ... 1090")
    low_n = (span - RADIX) * RADIX + step_n
    if sum(1 for row in rows if row.low()) != low_n:
        raise RuntimeError("the thousand low band left 1000..1009")
    if sum(1 for row in rows if row.high()) != step_n:
        raise RuntimeError("the thousand high band left 1090..1098")
    if sum(1 for row in rows if row.carries()) != 2475:
        raise RuntimeError("thousand carries left the units sums past nine")
    edge_n = sum(range(9 * RADIX, span))
    if sum(1 for row in rows if row.right >= 9 * RADIX) != edge_n:
        raise RuntimeError("thousand rows with a ninety on the right left that band")
    if sum(1 for row in rows if row.left >= thousand - RADIX) != edge_n:
        raise RuntimeError("thousand rows with a nine hundred ninety on the left left that band")
    if sum(1 for row in rows if row.left == thousand - 1) != span - 1:
        raise RuntimeError("nine hundred ninety-nine lost an addition into the thousand")
    if any(row.right == 0 for row in rows):
        raise RuntimeError("a zero on the right left 0..999")
    if min(row.result for row in rows) != thousand or max(row.result for row in rows) != thousand + span - 2:
        raise RuntimeError("thousand results left 1000..1098")
    for row in rows:
        if not (span <= row.left < thousand and 0 <= row.right < span and thousand <= row.result < cap):
            raise RuntimeError("thousand sum left the operand table")
        if row.result != row.left + row.right:
            raise RuntimeError("thousand addition left the sum")
        if row.kind == "step" and row.right >= RADIX:
            raise RuntimeError("a thousand step carried a place on the right")
        if row.kind == "wide" and row.right < RADIX:
            raise RuntimeError("a thousand wide row carried a digit on the right")
        if row.kind not in ("step", "wide"):
            raise RuntimeError("thousand sum used a kind outside step and wide")
    if len(hold) * 4 != len(train) or len(train) + len(hold) != len(rows):
        raise RuntimeError(f"thousand split is {len(train)}/{len(hold)} of {len(rows)}")

    def _both(pred, label: str) -> None:
        if not any(pred(row) for row in train) or not any(pred(row) for row in hold):
            raise RuntimeError(label)

    _both(lambda row: row.kind == "step", "thousand split hid every digit step on one side")
    _both(lambda row: row.kind == "wide", "thousand split hid every place on the right on one side")
    _both(lambda row: row.exact_thousand(), "thousand split hid every exact thousand on one side")
    _both(lambda row: row.exact_ten(), "thousand split hid every exact ten on one side")
    _both(lambda row: row.carries(), "thousand split hid every carry on one side")
    _both(lambda row: not row.carries(), "thousand split hid every sum without a carry on one side")
    _both(lambda row: row.low(), "thousand split hid the low band on one side")
    _both(lambda row: row.high(), "thousand split hid the high band on one side")
    _both(lambda row: row.right >= 9 * RADIX, "thousand split hid every ninety on the right on one side")
    _both(lambda row: row.left >= thousand - RADIX, "thousand split hid every nine hundred ninety on the left on one side")
    _both(lambda row: row.left == thousand - 1, "thousand split hid every nine hundred ninety-nine on the left on one side")


def _check_thou_ops(lattice: Lattice) -> None:
    """A thousand name re-enters as an operand. The result stays inside 0..9999.

    Mixed units are checked on a pure line. The fresh digit gauges are exact
    when the extra K on each units place cancels into that thousand.
    """
    spells = (
        (1024, "one thousand twenty-four"),
        (976, "nine hundred seventy-six"),
        (1003, "one thousand three"),
        (1041, "one thousand forty-one"),
        (1550, "one thousand five hundred fifty"),
        (1999, "one thousand nine hundred ninety-nine"),
        (2000, "two thousand"),
        (2097, "two thousand ninety-seven"),
        (5500, "five thousand five hundred"),
        (9990, "nine thousand nine hundred ninety"),
        (9998, "nine thousand nine hundred ninety-eight"),
        (9999, "nine thousand nine hundred ninety-nine"),
        (1010, "one thousand ten"),
        (1001, "one thousand one"),
        (1, "one"),
    )
    for number, spelling in spells:
        got = number_name(number)
        if got != spelling:
            raise RuntimeError(f"{number} spelled {got}")
    if decimal_name(1024) != "1024":
        raise RuntimeError("the decimal thousand operand left the integer")
    counts = census_thousand_ops()
    width = THOUSAND_CAP - THOUSAND_SPAN
    plus_formula = sum(width - right for right in range(THOUSAND_SPAN))
    if counts["plus"] != plus_formula or counts["minus"] != width * THOUSAND_SPAN:
        raise RuntimeError("thousand operands left the plus and minus counts")
    if counts["n"] != counts["plus"] + counts["minus"]:
        raise RuntimeError("thousand operands left 0..9999")
    if counts["zero"] != 2 * width:
        raise RuntimeError("thousand operands with a zero on the right left that band")
    if counts["under"] != (THOUSAND_SPAN - 1) * THOUSAND_SPAN // 2:
        raise RuntimeError("results under one thousand left that band")
    ones_plus = sum(width - right for right in range(RADIX))
    if counts["ones"] != ones_plus + width * RADIX:
        raise RuntimeError("thousand operands with a digit on the right left that band")
    if counts["ones"] + counts["place"] + counts["block"] != counts["n"]:
        raise RuntimeError("ones, place, and block left the thousand operand family")
    if counts["thousand"] != 8001 + 9 * THOUSAND_SPAN:
        raise RuntimeError("exact thousands left the ways to write 1000 through 9000")
    if counts["left1000"] != 2 * THOUSAND_SPAN * THOUSAND_SPAN:
        raise RuntimeError("left names from one thousand through nineteen hundred ninety-nine left that band")
    if counts["hold"] * 4 != counts["train"] or counts["train"] + counts["hold"] != counts["n"]:
        raise RuntimeError(f"thousand operand split is {counts['train']}/{counts['hold']} of {counts['n']}")
    if counts["borrow_ten"] != 0:
        raise RuntimeError("a borrow landed on an exact ten")
    if counts["out"] != 0 or counts["zero_result"] != 0:
        raise RuntimeError("a thousand operand left 0..9999")
    if counts["min_result"] != 1 or counts["max_result"] != THOUSAND_CAP - 1:
        raise RuntimeError(f"thousand operand results span {counts['min_result']}..{counts['max_result']}")
    expected = {
        "n": 17500500,
        "train": 14000400,
        "hold": 3500100,
        "plus": 8500500,
        "minus": 9000000,
        "zero": 18000,
        "zero_train": 14400,
        "zero_hold": 3600,
        "borrow": 4050000,
        "borrow_train": 3240000,
        "borrow_hold": 810000,
        "carry": 3822750,
        "carry_train": 3058200,
        "carry_hold": 764550,
        "nocarry": 4677750,
        "noborrow": 4950000,
        "under": 499500,
        "under_train": 399600,
        "under_hold": 99900,
        "ten": 1749600,
        "ten_train": 1399600,
        "ten_hold": 350000,
        "hundred": 174510,
        "hundred_train": 139600,
        "hundred_hold": 34910,
        "thousand": 17001,
        "thousand_train": 13600,
        "thousand_hold": 3401,
        "cross": 8491500,
        "ones": 179955,
        "ones_train": 143964,
        "ones_hold": 35991,
        "place": 1615095,
        "place_train": 1292076,
        "place_hold": 323019,
        "block": 15705450,
        "block_train": 12564360,
        "block_hold": 3141090,
        "left1000": 2000000,
        "left9000": 1500500,
        "right900": 1705050,
        "top": 1500500,
    }
    for key, value in expected.items():
        if counts[key] != value:
            raise RuntimeError(f"thousand operand {key} count is {counts[key]}")
    both_keys = (
        "plus",
        "minus",
        "zero",
        "borrow",
        "carry",
        "nocarry",
        "noborrow",
        "under",
        "ten",
        "hundred",
        "thousand",
        "cross",
        "ones",
        "place",
        "block",
        "left1000",
        "left9000",
        "right900",
        "top",
    )
    for key in both_keys:
        if counts[key + "_train"] == 0 or counts[key + "_hold"] == 0:
            raise RuntimeError(f"thousand operand split hid every {key} on one side")
    step = K
    gauges = [index * step for index in range(RADIX)]
    ten = gauges[9] + gauges[1]
    hundred = ten * RADIX
    thousand = hundred * RADIX
    place = RADIX * RADIX

    def _syn_right(right: int) -> float:
        if right < RADIX:
            return gauges[right]
        if right < place:
            return lattice.compose_place(right, ten, gauges)
        return lattice.compose_hundred(right, hundred, ten, gauges)

    def _syn_left(left: int) -> float:
        if left < THOUSAND_SPAN:
            return lattice.compose_hundred(left, hundred, ten, gauges)
        return lattice.compose_thousand(left, thousand, hundred, ten, gauges)

    def _syn_read(left: int, op: str, right: int) -> tuple[int, float, float]:
        sign = 1 if op == "+" else -1
        quantity = lattice.consensus_quantity(_syn_left(left), _syn_right(right), sign)
        named, remainder, margin, _dist = lattice.read_place(
            quantity, "digit", ten, gauges, hundred, thousand
        )
        return named, remainder, margin

    cases = (
        (1000, "+", 24, 1024),
        (1000, "-", 24, 976),
        (1000, "+", 3, 1003),
        (1000, "-", 999, 1),
        (1000, "+", 0, 1000),
        (1000, "-", 1, 999),
        (1500, "+", 500, 2000),
        (1998, "+", 99, 2097),
        (2500, "-", 500, 2000),
        (1024, "+", 3, 1027),
        (1001, "+", 9, 1010),
        (1990, "+", 10, 2000),
        (9999, "-", 9, 9990),
        (9999, "-", 0, 9999),
        (9000, "+", 999, 9999),
        (5000, "+", 500, 5500),
        (2000, "-", 999, 1001),
        (1024, "+", 17, 1041),
        (1500, "+", 50, 1550),
        (9999, "+", 0, 9999),
    )
    for number in (1000, 1024, 1998, 2000, 2500, 5000, 5500, 9990, 9999):
        if abs(_syn_left(number) - number * step) > 1e-9:
            raise RuntimeError(f"synthetic {number} left the integer line")
    for left, op, right, result in cases:
        named, remainder, margin = _syn_read(left, op, right)
        if named != result:
            raise RuntimeError(f"rebuilt {left}{op}{right} read {named}")
        if margin <= DROP:
            raise RuntimeError(f"rebuilt {left}{op}{right} sat inside the drop")
        if result % RADIX == 0 and abs(remainder) >= DROP:
            raise RuntimeError(f"exact ten {left}{op}{right} left a remainder")
    stay = (
        (999, "+", 0, 999),
        (950, "+", 49, 999),
        (500, "+", 499, 999),
        (100, "+", 24, 124),
    )
    for left, op, right, result in stay:
        named, _remainder, margin = _syn_read(left, op, right)
        if named != result:
            raise RuntimeError(f"rebuilt {left}{op}{right} shed into the thousand")
        if margin <= DROP:
            raise RuntimeError(f"rebuilt {left}{op}{right} sat inside the drop")
    cache = _thou_op_cache(lattice, "digit")
    live_thousand = cache["thousand"]
    live_hundred = cache["hundred"]
    live_ten = cache["ten"]
    live_gauges = cache["gauges"]
    for number in (1000, 1001, 1024, 2000, 2500, 5000, 5500, 9990, 9999):
        direct = lattice.compose_thousand(number, live_thousand, live_hundred, live_ten, live_gauges)
        cached = cache["built"][number - THOUSAND_SPAN]
        if abs(direct - cached) > 1e-9:
            raise RuntimeError(f"compose_thousand left the cached fold at {number}")
        if abs(lattice.algebraic_thousand(number, "digit") - cached) > 1e-9:
            raise RuntimeError(f"fresh thousand {number} left its algebraic value")
    for rest in (10, 24, 50, 99):
        folded = lattice.compose_hundred(rest, live_hundred, live_ten, live_gauges)
        placed = lattice.compose_place(rest, live_ten, live_gauges)
        if abs(folded - placed) > 1e-9:
            raise RuntimeError(f"compose_hundred left the place at {rest}")
        if abs(lattice.algebraic_hundred(rest, "digit") - lattice.algebraic_place(rest, "digit")) > 1e-9:
            raise RuntimeError(f"algebraic hundred left the place at {rest}")
    print("thousand operand names", flush=True)
    for number in range(THOUSAND_SPAN, THOUSAND_CAP):
        quantity = cache["built"][number - THOUSAND_SPAN]
        if abs(quantity - lattice.algebraic_thousand(number, "digit")) > 1e-9:
            raise RuntimeError(f"fresh thousand {number} left its algebraic value")
        named, remainder, margin, _dist = lattice.read_place(
            quantity,
            "digit",
            live_ten,
            live_gauges,
            live_hundred,
            live_thousand,
        )
        if named != number:
            raise RuntimeError(f"rebuilt thousand operand {number} read {named}")
        if margin <= DROP:
            raise RuntimeError(f"rebuilt thousand operand {number} sat inside the drop")
        if number % RADIX == 0 and abs(remainder) >= DROP:
            raise RuntimeError(f"exact ten {number} left a remainder")
    print("thousand operand names done", flush=True)
    fresh = (
        ThousandOp(1000, "+", 24, 1024),
        ThousandOp(1000, "-", 24, 976),
        ThousandOp(1000, "+", 0, 1000),
        ThousandOp(1000, "-", 999, 1),
        ThousandOp(1000, "+", 999, 1999),
        ThousandOp(1500, "+", 500, 2000),
        ThousandOp(1024, "-", 24, 1000),
        ThousandOp(2000, "-", 500, 1500),
        ThousandOp(9999, "-", 9, 9990),
        ThousandOp(9000, "+", 999, 9999),
        ThousandOp(2000, "-", 999, 1001),
        ThousandOp(1500, "+", 50, 1550),
        ThousandOp(9999, "+", 0, 9999),
        ThousandOp(1000, "-", 1, 999),
    )
    for row in fresh:
        if abs(lattice.thousand_op_quantity(row, "digit") - lattice.algebraic_thousand_op(row, "digit")) > 1e-9:
            raise RuntimeError(f"fresh {row.digit_prompt()}{row.digit_answer()} left the algebraic sum")
        got = lattice.predict_thousand_op(row, "digit")
        if got != row.digit_answer():
            raise RuntimeError(f"fresh {row.digit_prompt()}{row.digit_answer()} read {got}")
    demos = fresh + (
        ThousandOp(1000, "+", 3, 1003),
        ThousandOp(1024, "+", 17, 1041),
        ThousandOp(1998, "+", 99, 2097),
        ThousandOp(1024, "+", 3, 1027),
        ThousandOp(1001, "+", 9, 1010),
        ThousandOp(1990, "+", 10, 2000),
        ThousandOp(9999, "-", 0, 9999),
        ThousandOp(5000, "+", 500, 5500),
    )
    for row in demos:
        if not (THOUSAND_SPAN <= row.left < THOUSAND_CAP and 0 <= row.right < THOUSAND_SPAN):
            raise RuntimeError(f"{row.digit_prompt()}{row.digit_answer()} is outside the thousand operand family")
        if row.op == "+" and (row.result != row.left + row.right or row.result >= THOUSAND_CAP):
            raise RuntimeError(f"{row.digit_prompt()}{row.digit_answer()} left the sum")
        if row.op == "-" and row.result != row.left - row.right:
            raise RuntimeError(f"{row.digit_prompt()}{row.digit_answer()} left the difference")
        if row.result < 0 or row.result >= THOUSAND_CAP:
            raise RuntimeError(f"{row.digit_prompt()}{row.digit_answer()} left 0..9999")
        if row.op not in ("+", "-"):
            raise RuntimeError("thousand operand used an operator outside plus and minus")


def _check_thou_pairs(lattice: Lattice) -> None:
    """Two rebuilt thousand names share one pass. The result stays inside 0..9999.

    Mixed units are checked on a pure line. The fresh digit gauges are exact
    when the extra K on each units place cancels into the thousand, or when
    the two sides are the same quantity and the difference is zero.
    """
    if number_name(10000) != "ten thousand":
        raise RuntimeError("ten thousand left the spelling table")
    if number_name(11000) != "eleven thousand" or number_name(110000) != "?":
        raise RuntimeError("the spelling table left eleven thousand or passed one hundred ten thousand")
    if number_name(0) != "zero":
        raise RuntimeError("a cancelled thousand pair lost the word zero")
    spells = (
        (100, "one hundred"),
        (500, "five hundred"),
        (2000, "two thousand"),
        (2124, "two thousand one hundred twenty-four"),
        (2130, "two thousand one hundred thirty"),
        (3000, "three thousand"),
        (4000, "four thousand"),
        (6000, "six thousand"),
        (8999, "eight thousand nine hundred ninety-nine"),
        (9000, "nine thousand"),
    )
    for number, spelling in spells:
        got = number_name(number)
        if got != spelling:
            raise RuntimeError(f"{number} spelled {got}")
    step = K
    gauges = [index * step for index in range(RADIX)]
    ten = gauges[9] + gauges[1]
    hundred = ten * RADIX
    thousand = hundred * RADIX

    def _syn_name(number: int) -> float:
        return lattice.compose_thousand(number, thousand, hundred, ten, gauges)

    def _syn_read(left: int, op: str, right: int) -> tuple[int, float, float]:
        sign = 1 if op == "+" else -1
        quantity = lattice.consensus_quantity(_syn_name(left), _syn_name(right), sign)
        named, remainder, margin, _dist = lattice.read_place(
            quantity, "digit", ten, gauges, hundred, thousand
        )
        return named, remainder, margin

    for number in (1000, 1024, 1100, 1103, 1104, 1500, 1999, 2000, 2048, 4499, 4999, 5000, 8999, 9000, 9998, 9999):
        if abs(_syn_name(number) - number * step) > 1e-9:
            raise RuntimeError(f"synthetic {number} left the integer line")
    cases = (
        (1000, "+", 1000, 2000),
        (1000, "-", 1000, 0),
        (2000, "-", 1000, 1000),
        (4000, "+", 5000, 9000),
        (9000, "-", 1000, 8000),
        (1500, "+", 2500, 4000),
        (2500, "-", 1500, 1000),
        (1100, "-", 1000, 100),
        (2000, "-", 1500, 500),
        (5500, "-", 1500, 4000),
        (9999, "-", 1000, 8999),
        (9999, "-", 9999, 0),
        (1999, "+", 1001, 3000),
        (1024, "+", 1100, 2124),
        (1026, "+", 1104, 2130),
        (5000, "-", 5000, 0),
        (8000, "+", 1000, 9000),
        (3000, "+", 3000, 6000),
        (1024, "+", 1103, 2127),
        (1024, "+", 1024, 2048),
        (1234, "-", 1234, 0),
        (2345, "-", 1234, 1111),
        (1000, "+", 8999, 9999),
        (5000, "+", 4999, 9999),
        (9999, "-", 9998, 1),
        (4500, "+", 4499, 8999),
    )
    for left, op, right, result in cases:
        named, remainder, margin = _syn_read(left, op, right)
        if named != result:
            raise RuntimeError(f"rebuilt {left}{op}{right} read {named}")
        if margin <= DROP:
            raise RuntimeError(f"rebuilt {left}{op}{right} sat inside the drop")
        if result % RADIX == 0 and abs(remainder) >= DROP:
            raise RuntimeError(f"exact ten {left}{op}{right} left a remainder")
    fresh = (
        ThousandPair(1000, "+", 1000, 2000),
        ThousandPair(1000, "-", 1000, 0),
        ThousandPair(2000, "-", 1000, 1000),
        ThousandPair(4000, "+", 5000, 9000),
        ThousandPair(9000, "-", 1000, 8000),
        ThousandPair(1500, "+", 2500, 4000),
        ThousandPair(2500, "-", 1500, 1000),
        ThousandPair(1100, "-", 1000, 100),
        ThousandPair(2000, "-", 1500, 500),
        ThousandPair(5500, "-", 1500, 4000),
        ThousandPair(9999, "-", 1000, 8999),
        ThousandPair(9999, "-", 9999, 0),
        ThousandPair(1999, "+", 1001, 3000),
        ThousandPair(1024, "+", 1100, 2124),
        ThousandPair(1026, "+", 1104, 2130),
        ThousandPair(5000, "-", 5000, 0),
        ThousandPair(8000, "+", 1000, 9000),
        ThousandPair(3000, "+", 3000, 6000),
        ThousandPair(1234, "-", 1234, 0),
    )
    for row in fresh:
        if abs(lattice.thousand_pair_quantity(row, "digit") - lattice.algebraic_thousand_pair(row, "digit")) > 1e-9:
            raise RuntimeError(f"fresh {row.digit_prompt()}{row.digit_answer()} left the algebraic sum")
        got = lattice.predict_thousand_pair(row, "digit")
        if got != row.digit_answer():
            raise RuntimeError(f"fresh {row.digit_prompt()}{row.digit_answer()} read {got}")
    counts = census_thousand_pairs()
    width = THOUSAND_CAP - THOUSAND_SPAN
    plus_span = width - THOUSAND_SPAN
    plus_n = plus_span * (plus_span + 1) // 2
    minus_n = width * (width + 1) // 2
    if counts["plus"] != plus_n or counts["minus"] != minus_n:
        raise RuntimeError("thousand pairs left the plus and minus counts")
    if counts["n"] != plus_n + minus_n:
        raise RuntimeError("thousand pairs left 0..9999")
    if counts["zero"] != width:
        raise RuntimeError("cancelled thousand pairs left the pairs of equal names")
    under_n = THOUSAND_SPAN * width - (THOUSAND_SPAN - 1) * THOUSAND_SPAN // 2
    if counts["under"] != under_n:
        raise RuntimeError("results under one thousand left that band")
    plus_exact = sum(THOUSAND_SPAN * k - (2 * THOUSAND_SPAN - 1) for k in range(2, RADIX))
    minus_exact = sum(width - THOUSAND_SPAN * k for k in range(0, RADIX - 1))
    if counts["thousand"] != plus_exact + minus_exact:
        raise RuntimeError("exact thousands left the ways to write 0 through 9000")
    band = THOUSAND_SPAN
    low_cut = width - 2 * band
    left1000_plus = plus_n - low_cut * (low_cut + 1) // 2
    left1000_minus = band * (band + 1) // 2
    if counts["left1000"] != left1000_plus + left1000_minus:
        raise RuntimeError("left names from one thousand through nineteen hundred ninety-nine left that band")
    if counts["left9000"] != minus_n - plus_n:
        raise RuntimeError("left names from nine thousand up left that band")
    if counts["right9000"] != band * (band + 1) // 2:
        raise RuntimeError("right names from nine thousand up left that band")
    if counts["top"] != left1000_plus:
        raise RuntimeError("results from nine thousand up left that band")
    if counts["hold"] * 4 != counts["train"] or counts["train"] + counts["hold"] != counts["n"]:
        raise RuntimeError(f"thousand pair split is {counts['train']}/{counts['hold']} of {counts['n']}")
    if counts["borrow_ten"] != 0:
        raise RuntimeError("a borrow landed on an exact ten")
    if counts["out"] != 0:
        raise RuntimeError("a thousand pair left 0..9999")
    if counts["cross"] != counts["n"]:
        raise RuntimeError("a thousand pair kept the same thousand digit")
    if counts["min_result"] != 0 or counts["max_result"] != THOUSAND_CAP - 1:
        raise RuntimeError(f"thousand pair results span {counts['min_result']}..{counts['max_result']}")
    if counts["carry"] + counts["nocarry"] != counts["plus"]:
        raise RuntimeError("carries left the additions")
    if counts["borrow"] + counts["noborrow"] != counts["minus"]:
        raise RuntimeError("borrows left the subtractions")
    expected = {
        "n": 72508500,
        "train": 58006800,
        "hold": 14501700,
        "plus": 32004000,
        "minus": 40504500,
        "zero": 9000,
        "zero_train": 7200,
        "zero_hold": 1800,
        "borrow": 18204750,
        "borrow_train": 14563800,
        "borrow_hold": 3640950,
        "carry": 14382000,
        "carry_train": 11505600,
        "carry_hold": 2876400,
        "nocarry": 17622000,
        "noborrow": 22299750,
        "under": 8500500,
        "under_train": 6800400,
        "under_hold": 1700100,
        "ten": 7251300,
        "ten_train": 5800400,
        "ten_hold": 1450900,
        "hundred": 725580,
        "hundred_train": 580400,
        "hundred_hold": 145180,
        "thousand": 73008,
        "thousand_train": 58400,
        "thousand_hold": 14608,
        "cross": 72508500,
        "left1000": 8001000,
        "left1000_train": 6400800,
        "left1000_hold": 1600200,
        "left9000": 8500500,
        "left9000_train": 6800400,
        "left9000_hold": 1700100,
        "right9000": 500500,
        "right9000_train": 400400,
        "right9000_hold": 100100,
        "top": 7500500,
        "top_train": 6000400,
        "top_hold": 1500100,
    }
    for key, value in expected.items():
        if counts[key] != value:
            raise RuntimeError(f"thousand pair {key} count is {counts[key]}")
    both_keys = (
        "plus",
        "minus",
        "zero",
        "borrow",
        "carry",
        "nocarry",
        "noborrow",
        "under",
        "ten",
        "hundred",
        "thousand",
        "left1000",
        "left9000",
        "right9000",
        "top",
    )
    for key in both_keys:
        if counts[key + "_train"] == 0 or counts[key + "_hold"] == 0:
            raise RuntimeError(f"thousand pair split hid every {key} on one side")
    demos = fresh + (
        ThousandPair(1024, "+", 1103, 2127),
        ThousandPair(1024, "+", 1024, 2048),
        ThousandPair(2345, "-", 1234, 1111),
        ThousandPair(1000, "+", 8999, 9999),
        ThousandPair(5000, "+", 4999, 9999),
        ThousandPair(9999, "-", 9998, 1),
    )
    for row in demos:
        if not (
            THOUSAND_SPAN <= row.left < THOUSAND_CAP and THOUSAND_SPAN <= row.right < THOUSAND_CAP
        ):
            raise RuntimeError(f"{row.digit_prompt()}{row.digit_answer()} is outside the thousand pair family")
        if row.op == "+" and (row.result != row.left + row.right or row.result >= THOUSAND_CAP):
            raise RuntimeError(f"{row.digit_prompt()}{row.digit_answer()} left the sum")
        if row.op == "-" and (row.left < row.right or row.result != row.left - row.right):
            raise RuntimeError(f"{row.digit_prompt()}{row.digit_answer()} left the difference")
        if row.result < 0 or row.result >= THOUSAND_CAP:
            raise RuntimeError(f"{row.digit_prompt()}{row.digit_answer()} left 0..9999")
        if row.op not in ("+", "-"):
            raise RuntimeError("thousand pair used an operator outside plus and minus")
        if not row.crosses():
            raise RuntimeError(f"{row.digit_prompt()}{row.digit_answer()} kept the same thousand digit")


def _check_ten_thousands(lattice: Lattice) -> None:
    """The ten-thousand place is the thousand-step counted ten times.

    Mixed units are checked on a pure line. The fresh digit gauges are exact
    when the extra K on each units place cancels into that ten-thousand, or
    when a units digit is absent on both sides. A name still inside 0..9999
    sheds zero ten-thousands.
    """
    spells = (
        (10000, "ten thousand"),
        (10001, "ten thousand one"),
        (10008, "ten thousand eight"),
        (10010, "ten thousand ten"),
        (10050, "ten thousand fifty"),
        (10098, "ten thousand ninety-eight"),
        (10100, "ten thousand one hundred"),
        (10998, "ten thousand nine hundred ninety-eight"),
        (10999, "ten thousand nine hundred ninety-nine"),
        (9999, "nine thousand nine hundred ninety-nine"),
        (2000, "two thousand"),
    )
    for number, spelling in spells:
        got = number_name(number)
        if got != spelling:
            raise RuntimeError(f"{number} spelled {got}")
    if number_name(11000) != "eleven thousand" or number_name(12000) != "twelve thousand":
        raise RuntimeError("eleven thousand or twelve thousand left the spelling table")
    if number_name(110000) != "?":
        raise RuntimeError("a name past one hundred nine thousand nine hundred ninety-nine entered the table")
    if decimal_name(10000) != "10000":
        raise RuntimeError("the decimal ten thousand left the integer")
    counts = census_ten_thousand_sums()
    if counts["n"] != 999 * 1000 // 2:
        raise RuntimeError("ten-thousand sums left the ways to write 10000..10998")
    if counts["ones"] != 45 or counts["place"] != 4905 or counts["block"] != 494550:
        raise RuntimeError("ten-thousand right-hand bands left digit, place, and block")
    if counts["ones"] + counts["place"] + counts["block"] != counts["n"]:
        raise RuntimeError("ten-thousand bands do not cover the family")
    if counts["thousand"] != 999:
        raise RuntimeError("the exact ten-thousands are not the ways to write 10000")
    if counts["hundred"] != 5490:
        raise RuntimeError("ten-thousand exact hundreds left 10000, 10100, ... 10900")
    if counts["ten"] != 50400:
        raise RuntimeError("ten-thousand exact tens left that band")
    if counts["low"] != 9945:
        raise RuntimeError("the ten-thousand low band left 10000..10009")
    if counts["high"] != 4950:
        raise RuntimeError("the ten-thousand high band left 10900..10998")
    if counts["carry"] + counts["nocarry"] != counts["n"]:
        raise RuntimeError("ten-thousand carries left the units sums")
    if counts["min_result"] != THOUSAND_CAP or counts["max_result"] != 10998:
        raise RuntimeError("ten-thousand results left 10000..10998")
    if counts["train"] + counts["hold"] != counts["n"] or counts["hold"] * 4 != counts["train"]:
        raise RuntimeError(
            f"ten-thousand split is {counts['train']}/{counts['hold']} of {counts['n']}"
        )
    for key in (
        "ones",
        "place",
        "block",
        "ten",
        "hundred",
        "thousand",
        "carry",
        "nocarry",
        "low",
        "high",
    ):
        if counts[key + "_train"] == 0 or counts[key + "_hold"] == 0:
            raise RuntimeError(f"ten-thousand split hid every {key} on one side")
    step = K
    gauges = [index * step for index in range(RADIX)]
    ten = gauges[9] + gauges[1]
    hundred = ten * RADIX
    thousand = hundred * RADIX
    ten_thousand = thousand * RADIX

    def _syn_name(number: int) -> float:
        return lattice.compose_thousand(number, thousand, hundred, ten, gauges)

    def _syn_right(right: int) -> float:
        if right < RADIX:
            return gauges[right]
        if right < RADIX * RADIX:
            return lattice.compose_place(right, ten, gauges)
        return lattice.compose_hundred(right, hundred, ten, gauges)

    def _syn_read(left: int, right: int) -> tuple[int, float, float]:
        quantity = lattice.consensus_quantity(_syn_name(left), _syn_right(right), 1)
        named, remainder, margin, _dist = lattice.read_place(
            quantity, "digit", ten, gauges, hundred, thousand, ten_thousand
        )
        return named, remainder, margin

    for number in (1000, 9000, 9001, 9500, 9800, 9900, 9901, 9950, 9998, 9999):
        if abs(_syn_name(number) - number * step) > 1e-9:
            raise RuntimeError(f"synthetic {number} left the integer line")
    if abs(ten_thousand - RADIX * thousand) > 1e-9:
        raise RuntimeError("the synthetic ten-thousand left the thousand counted ten times")
    cases = (
        (9999, 1, 10000),
        (9999, 9, 10008),
        (9901, 99, 10000),
        (9500, 500, 10000),
        (9001, 999, 10000),
        (9999, 99, 10098),
        (9999, 999, 10998),
        (9900, 200, 10100),
        (9950, 50, 10000),
        (9800, 250, 10050),
    )
    for left, right, result in cases:
        named, remainder, margin = _syn_read(left, right)
        if named != result:
            raise RuntimeError(f"rebuilt {left}+{right} read {named}")
        if margin <= DROP:
            raise RuntimeError(f"rebuilt {left}+{right} sat inside the drop")
        if result % RADIX == 0 and abs(remainder) >= DROP:
            raise RuntimeError(f"exact ten {left}+{right} left a remainder")
    stay = (
        (9000, 999, 9999),
        (9500, 499, 9999),
        (9998, 1, 9999),
        (9001, 998, 9999),
    )
    for left, right, result in stay:
        named, _remainder, margin = _syn_read(left, right)
        if named != result:
            raise RuntimeError(f"rebuilt {left}+{right} shed into the ten-thousand")
        if margin <= DROP:
            raise RuntimeError(f"rebuilt {left}+{right} sat inside the drop")
    named, _remainder, margin, _dist = lattice.read_place(
        _syn_name(9999), "digit", ten, gauges, hundred, thousand, ten_thousand
    )
    if named != 9999 or margin <= DROP:
        raise RuntimeError(f"synthetic 9999 read {named}")
    live_ten = lattice.ten_quantity("digit")
    live_gauges = lattice.gauge_line("digit")
    live_hundred = lattice.hundred_quantity("digit")
    live_thousand = lattice.thousand_quantity("digit")
    live_ten_thousand = lattice.ten_thousand_quantity("digit")
    if abs(live_thousand - RADIX * live_hundred) > 1e-9:
        raise RuntimeError("the thousand-step left the hundred counted ten times")
    if abs(live_ten_thousand - RADIX * live_thousand) > 1e-9:
        raise RuntimeError("the fresh ten-thousand left the thousand counted ten times")
    if lattice.read_place(live_ten_thousand, "digit")[0] != THOUSAND_CAP:
        raise RuntimeError("the ten-thousand step did not read as 10000")
    fresh_nine = lattice.compose_thousand(9999, live_thousand, live_hundred, live_ten, live_gauges)
    if lattice.read_place(fresh_nine, "digit")[0] != 9999:
        raise RuntimeError("9999 shed a ten-thousand on the fresh line")
    if lattice.read_place(fresh_nine, "digit", live_ten, live_gauges, live_hundred, live_thousand)[0] != 9999:
        raise RuntimeError("9999 shed a counted ten-thousand on the fresh line")
    fresh = (
        TenThousandSum("digit", 9999, 1, 10000),
        TenThousandSum("place", 9901, 99, 10000),
        TenThousandSum("block", 9500, 500, 10000),
        TenThousandSum("block", 9001, 999, 10000),
        TenThousandSum("block", 9900, 200, 10100),
        TenThousandSum("place", 9950, 50, 10000),
        TenThousandSum("block", 9800, 250, 10050),
    )
    for row in fresh:
        if abs(
            lattice.ten_thousand_sum_quantity(row, "digit") - lattice.algebraic_ten_thousand_sum(row, "digit")
        ) > 1e-9:
            raise RuntimeError(f"fresh {row.digit_prompt()}{row.digit_answer()} left the algebraic sum")
        got = lattice.predict_ten_thousand(row, "digit")
        if got != row.digit_answer():
            raise RuntimeError(f"fresh {row.digit_prompt()}{row.digit_answer()} read {got}")
    demos = fresh + (
        TenThousandSum("digit", 9999, 9, 10008),
        TenThousandSum("place", 9999, 99, 10098),
        TenThousandSum("block", 9999, 999, 10998),
    )
    for row in demos:
        kind = ten_thousand_kind(row.right)
        if row.kind != ("digit", "place", "block")[kind]:
            raise RuntimeError(f"{row.digit_prompt()}{row.digit_answer()} left its right-hand band")
        if not (THOUSAND_SPAN <= row.left < THOUSAND_CAP and 0 <= row.right < THOUSAND_SPAN):
            raise RuntimeError(f"{row.digit_prompt()}{row.digit_answer()} is outside the ten-thousand family")
        if row.result != row.left + row.right or not (THOUSAND_CAP <= row.result < TEN_THOUSAND_CAP):
            raise RuntimeError(f"{row.digit_prompt()}{row.digit_answer()} left 10000..10998")
        if row.right == 0:
            raise RuntimeError("a zero on the right left 0..9999")


def _check_ten_thou_ops(lattice: Lattice) -> None:
    """A ten-thousand name re-enters as an operand. The result stays inside 0..10999.

    Mixed units are checked on a pure line. The fresh digit gauges are exact
    when the extra K on each units place cancels, or when a units digit is
    absent on both sides. A sum that would reach 11000 stays outside.
    """
    spells = (
        (10000, "ten thousand"),
        (10001, "ten thousand one"),
        (10024, "ten thousand twenty-four"),
        (10050, "ten thousand fifty"),
        (10100, "ten thousand one hundred"),
        (10990, "ten thousand nine hundred ninety"),
        (10999, "ten thousand nine hundred ninety-nine"),
        (9999, "nine thousand nine hundred ninety-nine"),
        (1, "one"),
    )
    for number, spelling in spells:
        got = number_name(number)
        if got != spelling:
            raise RuntimeError(f"{number} spelled {got}")
    if number_name(11000) != "eleven thousand" or number_name(110000) != "?":
        raise RuntimeError("the spelling table left eleven thousand or passed one hundred ten thousand")
    if decimal_name(10000) != "10000":
        raise RuntimeError("the decimal ten-thousand operand left the integer")
    counts = census_ten_thousand_ops()
    width = TEN_THOUSAND_CAP - THOUSAND_CAP
    place = RADIX * RADIX
    if counts["plus"] != width * (width + 1) // 2 or counts["minus"] != width * THOUSAND_CAP:
        raise RuntimeError("ten-thousand operands left the plus and minus counts")
    if counts["n"] != counts["plus"] + counts["minus"]:
        raise RuntimeError("ten-thousand operands left 0..10999")
    if counts["zero"] != 2 * width:
        raise RuntimeError("ten-thousand operands with a zero on the right left that band")
    ones_plus = sum(width - right for right in range(RADIX))
    if counts["ones"] != ones_plus + width * RADIX:
        raise RuntimeError("ten-thousand operands with a digit on the right left that band")
    place_plus = sum(width - right for right in range(RADIX, place))
    if counts["place"] != place_plus + width * (place - RADIX):
        raise RuntimeError("ten-thousand operands with a place on the right left that band")
    block_plus = sum(width - right for right in range(place, THOUSAND_SPAN))
    if counts["block"] != block_plus + width * (THOUSAND_SPAN - place):
        raise RuntimeError("ten-thousand operands with a hundred on the right left that band")
    if counts["right1000"] != width * (THOUSAND_CAP - THOUSAND_SPAN):
        raise RuntimeError("ten-thousand operands with a thousand name on the right left that band")
    if counts["ones"] + counts["place"] + counts["block"] + counts["right1000"] != counts["n"]:
        raise RuntimeError("ones, place, block, and thousand-right left the ten-thousand operand family")
    if counts["mark"] != 1 + width:
        raise RuntimeError("the rows that land on ten thousand left that count")
    under = sum(THOUSAND_CAP - 1 - rest for rest in range(width))
    if counts["under"] != under or counts["cross"] != counts["under"]:
        raise RuntimeError("results under ten thousand left that band")
    borrow = sum(range(RADIX)) * (width // RADIX) * (THOUSAND_CAP // RADIX)
    if counts["borrow"] != borrow or counts["borrow"] + counts["noborrow"] != counts["minus"]:
        raise RuntimeError("ten-thousand borrows left the units pairs")
    if counts["carry"] + counts["nocarry"] != counts["plus"]:
        raise RuntimeError("ten-thousand carries left the units sums")
    low_plus = sum(width - digit for digit in range(RADIX))
    if counts["low"] != low_plus + RADIX * THOUSAND_CAP:
        raise RuntimeError("left names from ten thousand through ten thousand nine left that band")
    high_n = place
    if counts["high"] != high_n * (high_n + 1) // 2 + high_n * THOUSAND_CAP:
        raise RuntimeError("left names from ten thousand nine hundred left that band")
    if counts["right9000"] != width * (THOUSAND_CAP - 9 * THOUSAND_SPAN):
        raise RuntimeError("right names from nine thousand left that band")
    if counts["hold"] * 4 != counts["train"] or counts["train"] + counts["hold"] != counts["n"]:
        raise RuntimeError(
            f"ten-thousand operand split is {counts['train']}/{counts['hold']} of {counts['n']}"
        )
    if counts["borrow_ten"] != 0:
        raise RuntimeError("a borrow landed on an exact ten")
    if counts["out"] != 0 or counts["zero_result"] != 0:
        raise RuntimeError("a ten-thousand operand left 0..10999")
    if counts["min_result"] != 1 or counts["max_result"] != TEN_THOUSAND_CAP - 1:
        raise RuntimeError(
            f"ten-thousand operand results span {counts['min_result']}..{counts['max_result']}"
        )
    expected = {
        "n": 10500500,
        "train": 8400400,
        "hold": 2100100,
        "plus": 500500,
        "minus": 10000000,
        "zero": 2000,
        "zero_train": 1600,
        "zero_hold": 400,
        "borrow": 4500000,
        "borrow_train": 3600000,
        "borrow_hold": 900000,
        "carry": 222750,
        "carry_train": 178200,
        "carry_hold": 44550,
        "nocarry": 277750,
        "noborrow": 5500000,
        "under": 9499500,
        "under_train": 7599600,
        "under_hold": 1899900,
        "ten": 1049600,
        "ten_train": 839600,
        "ten_hold": 210000,
        "hundred": 104510,
        "hundred_train": 83600,
        "hundred_hold": 20910,
        "thousand": 10001,
        "thousand_train": 8000,
        "thousand_hold": 2001,
        "mark": 1001,
        "cross": 9499500,
        "ones": 19955,
        "ones_train": 15964,
        "ones_hold": 3991,
        "place": 175095,
        "place_train": 140076,
        "place_hold": 35019,
        "block": 1305450,
        "block_train": 1044360,
        "block_hold": 261090,
        "right1000": 9000000,
        "right1000_train": 7200000,
        "right1000_hold": 1800000,
        "low": 109955,
        "high": 1005050,
        "right9000": 1000000,
        "top": 100100,
        "top_train": 80080,
        "top_hold": 20020,
    }
    for key, value in expected.items():
        if counts[key] != value:
            raise RuntimeError(f"ten-thousand operand {key} count is {counts[key]}")
    both_keys = (
        "plus",
        "minus",
        "zero",
        "borrow",
        "carry",
        "nocarry",
        "noborrow",
        "under",
        "ten",
        "hundred",
        "thousand",
        "mark",
        "cross",
        "ones",
        "place",
        "block",
        "right1000",
        "low",
        "high",
        "right9000",
        "top",
    )
    for key in both_keys:
        if counts[key + "_train"] == 0 or counts[key + "_hold"] == 0:
            raise RuntimeError(f"ten-thousand operand split hid every {key} on one side")
    step = K
    gauges = [index * step for index in range(RADIX)]
    ten = gauges[9] + gauges[1]
    hundred = ten * RADIX
    thousand = hundred * RADIX
    folded = 0.0
    for _ in range(RADIX):
        folded = lattice.consensus_quantity(folded, thousand, 1)
    if abs(folded - RADIX * thousand) > 1e-9:
        raise RuntimeError("the synthetic ten-thousand left the thousand counted ten times")

    def _syn_read(left: int, op: str, right: int) -> tuple[int, float, float]:
        sign = 1 if op == "+" else -1
        left_q = lattice.compose_ten_thousand(left, folded, thousand, hundred, ten, gauges)
        right_q = lattice.compose_below_ten_thousand(right, thousand, hundred, ten, gauges)
        quantity = lattice.consensus_quantity(left_q, right_q, sign)
        named, remainder, margin, _dist = lattice.read_place(
            quantity, "digit", ten, gauges, hundred, thousand, folded
        )
        return named, remainder, margin

    for number in (10000, 10001, 10024, 10050, 10100, 10250, 10500, 10990, 10999):
        built = lattice.compose_ten_thousand(number, folded, thousand, hundred, ten, gauges)
        if abs(built - number * step) > 1e-9:
            raise RuntimeError(f"synthetic {number} left the integer line")
    cases = (
        (10000, "+", 0, 10000),
        (10000, "-", 0, 10000),
        (10000, "+", 1, 10001),
        (10000, "+", 24, 10024),
        (10000, "+", 50, 10050),
        (10000, "+", 999, 10999),
        (10000, "-", 1, 9999),
        (10000, "-", 9999, 1),
        (10001, "+", 1, 10002),
        (10001, "+", 9, 10010),
        (10008, "+", 1, 10009),
        (10024, "-", 24, 10000),
        (10050, "+", 50, 10100),
        (10250, "-", 250, 10000),
        (10500, "+", 499, 10999),
        (10999, "+", 0, 10999),
        (10999, "-", 0, 10999),
        (10999, "-", 9, 10990),
        (10999, "-", 999, 10000),
    )
    for left, op, right, result in cases:
        named, remainder, margin = _syn_read(left, op, right)
        if named != result:
            raise RuntimeError(f"rebuilt {left}{op}{right} read {named}")
        if margin <= DROP:
            raise RuntimeError(f"rebuilt {left}{op}{right} sat inside the drop")
        if result % RADIX == 0 and abs(remainder) >= DROP:
            raise RuntimeError(f"exact ten {left}{op}{right} left a remainder")
    named, _remainder, margin, _dist = lattice.read_place(
        lattice.compose_below_ten_thousand(9999, thousand, hundred, ten, gauges),
        "digit",
        ten,
        gauges,
        hundred,
        thousand,
        folded,
    )
    if named != 9999 or margin <= DROP:
        raise RuntimeError(f"synthetic 9999 read {named}")
    cache = _ten_thou_op_cache(lattice, "digit")
    live_ten = cache["ten"]
    live_gauges = cache["gauges"]
    live_hundred = cache["hundred"]
    live_thousand = cache["thousand"]
    live_ten_thousand = cache["ten_thousand"]
    if abs(live_ten_thousand - lattice.ten_thousand_quantity("digit")) > 1e-9:
        raise RuntimeError("the cached ten-thousand left the folded step")
    print("ten thousand operand names", flush=True)
    for number in range(THOUSAND_CAP, TEN_THOUSAND_CAP):
        rest = number - THOUSAND_CAP
        direct = lattice.compose_ten_thousand(
            number, live_ten_thousand, live_thousand, live_hundred, live_ten, live_gauges
        )
        if abs(direct - cache["left_q"][rest]) > 1e-9:
            raise RuntimeError(f"compose_ten_thousand left the cached fold at {number}")
        if abs(lattice.algebraic_ten_thousand_name(number, "digit") - cache["left_alg"][rest]) > 1e-9:
            raise RuntimeError(f"fresh ten-thousand {number} left its algebraic value")
        named, remainder, margin, _dist = lattice.read_place(
            cache["left_q"][rest],
            "digit",
            live_ten,
            live_gauges,
            live_hundred,
            live_thousand,
            live_ten_thousand,
        )
        if named != number:
            raise RuntimeError(f"rebuilt ten-thousand operand {number} read {named}")
        if margin <= DROP:
            raise RuntimeError(f"rebuilt ten-thousand operand {number} sat inside the drop")
        if number % RADIX == 0 and abs(remainder) >= DROP:
            raise RuntimeError(f"exact ten {number} left a remainder")
    print("ten thousand operand names done", flush=True)
    for right in (0, 1, 24, 50, 99, 100, 250, 999, 1000, 1024, 5000, 9999):
        direct = lattice.compose_below_ten_thousand(
            right, live_thousand, live_hundred, live_ten, live_gauges
        )
        if abs(direct - cache["right_q"][right]) > 1e-9:
            raise RuntimeError(f"the cached right left compose at {right}")
        if abs(lattice.algebraic_below_ten_thousand(right, "digit") - cache["right_alg"][right]) > 1e-9:
            raise RuntimeError(f"fresh right {right} left its algebraic value")
    fresh = (
        TenThousandOp(10000, "+", 0, 10000),
        TenThousandOp(10000, "+", 1, 10001),
        TenThousandOp(10000, "+", 24, 10024),
        TenThousandOp(10000, "+", 50, 10050),
        TenThousandOp(10000, "+", 999, 10999),
        TenThousandOp(10000, "-", 1, 9999),
        TenThousandOp(10000, "-", 9999, 1),
        TenThousandOp(10001, "+", 9, 10010),
        TenThousandOp(10024, "-", 24, 10000),
        TenThousandOp(10050, "+", 50, 10100),
        TenThousandOp(10250, "-", 250, 10000),
        TenThousandOp(10500, "+", 499, 10999),
        TenThousandOp(10999, "-", 9, 10990),
        TenThousandOp(10999, "-", 999, 10000),
    )
    for row in fresh:
        if abs(
            lattice.ten_thousand_op_quantity(row, "digit") - lattice.algebraic_ten_thousand_op(row, "digit")
        ) > 1e-9:
            raise RuntimeError(f"fresh {row.digit_prompt()}{row.digit_answer()} left the algebraic sum")
        got = lattice.predict_ten_thousand_op(row, "digit")
        if got != row.digit_answer():
            raise RuntimeError(f"fresh {row.digit_prompt()}{row.digit_answer()} read {got}")
    demos = fresh + (
        TenThousandOp(10000, "-", 0, 10000),
        TenThousandOp(10001, "+", 1, 10002),
        TenThousandOp(10008, "+", 1, 10009),
        TenThousandOp(10999, "+", 0, 10999),
        TenThousandOp(10999, "-", 0, 10999),
    )
    for row in demos:
        if not (THOUSAND_CAP <= row.left < TEN_THOUSAND_CAP and 0 <= row.right < THOUSAND_CAP):
            raise RuntimeError(f"{row.digit_prompt()}{row.digit_answer()} is outside the ten-thousand operand family")
        if row.op == "+" and (row.result != row.left + row.right or row.result >= TEN_THOUSAND_CAP):
            raise RuntimeError(f"{row.digit_prompt()}{row.digit_answer()} left the sum")
        if row.op == "-" and row.result != row.left - row.right:
            raise RuntimeError(f"{row.digit_prompt()}{row.digit_answer()} left the difference")
        if row.result < 0 or row.result >= TEN_THOUSAND_CAP:
            raise RuntimeError(f"{row.digit_prompt()}{row.digit_answer()} left 0..10999")


def _check_ten_thou_pairs(lattice: Lattice) -> None:
    """Two ten-thousand names. The difference stays inside 0..999.

    Their sum leaves the spelled table. Mixed units are checked on a pure
    line. Equal names cancel, and the fresh digit read of that zero is the
    nearest gauge to a remainder of zero.
    """
    spells = (
        (0, "zero"),
        (1, "one"),
        (9, "nine"),
        (24, "twenty-four"),
        (50, "fifty"),
        (500, "five hundred"),
        (999, "nine hundred ninety-nine"),
    )
    for number, spelling in spells:
        got = number_name(number)
        if got != spelling:
            raise RuntimeError(f"{number} spelled {got}")
    counts = census_ten_thousand_pairs()
    width = TEN_THOUSAND_CAP - THOUSAND_CAP
    n = width * (width + 1) // 2
    if counts["n"] != n or counts["zero"] != width:
        raise RuntimeError("ten-thousand pairs left the differences of 10000..10999")
    if counts["ones"] != sum(width - diff for diff in range(RADIX)):
        raise RuntimeError("results under ten left that band")
    if counts["place"] != sum(width - diff for diff in range(RADIX, RADIX * RADIX)):
        raise RuntimeError("results from ten through ninety-nine left that band")
    if counts["block"] != sum(width - diff for diff in range(RADIX * RADIX, width)):
        raise RuntimeError("results from one hundred up left that band")
    if counts["ones"] + counts["place"] + counts["block"] != counts["n"]:
        raise RuntimeError("ones, place, and block left the ten-thousand pair family")
    if counts["ten"] != sum(width - RADIX * step for step in range(width // RADIX)):
        raise RuntimeError("exact tens left the differences")
    if counts["hundred"] != sum(width - (RADIX * RADIX) * step for step in range(RADIX)):
        raise RuntimeError("exact hundreds left the differences")
    if counts["low"] != sum(index + 1 for index in range(RADIX)):
        raise RuntimeError("left names from ten thousand through ten thousand nine left that band")
    high_span = RADIX * RADIX
    if counts["high"] != sum((width - high_span) + 1 + index for index in range(high_span)):
        raise RuntimeError("left names from ten thousand nine hundred left that band")
    if counts["righthigh"] != high_span * (high_span + 1) // 2:
        raise RuntimeError("right names from ten thousand nine hundred left that band")
    if counts["top"] != high_span * (high_span + 1) // 2:
        raise RuntimeError("results from nine hundred up left that band")
    if counts["borrow"] + counts["noborrow"] != counts["n"]:
        raise RuntimeError("borrows left the differences")
    if counts["borrow_ten"] != 0:
        raise RuntimeError("a borrow landed on an exact ten")
    if counts["out"] != 0 or counts["cross"] != counts["n"]:
        raise RuntimeError("a ten-thousand pair stayed inside the ten-thousands")
    if counts["min_result"] != 0 or counts["max_result"] != width - 1:
        raise RuntimeError(f"ten-thousand pair results span {counts['min_result']}..{counts['max_result']}")
    if counts["hold"] * 4 != counts["train"] or counts["train"] + counts["hold"] != counts["n"]:
        raise RuntimeError(f"ten-thousand pair split is {counts['train']}/{counts['hold']} of {counts['n']}")
    expected = {
        "n": 500500,
        "train": 400400,
        "hold": 100100,
        "zero": 1000,
        "zero_train": 800,
        "zero_hold": 200,
        "borrow": 222750,
        "borrow_train": 178200,
        "borrow_hold": 44550,
        "noborrow": 277750,
        "noborrow_train": 222200,
        "noborrow_hold": 55550,
        "ten": 50500,
        "ten_train": 40400,
        "ten_hold": 10100,
        "hundred": 5500,
        "hundred_train": 4400,
        "hundred_hold": 1100,
        "ones": 9955,
        "ones_train": 7964,
        "ones_hold": 1991,
        "place": 85095,
        "place_train": 68076,
        "place_hold": 17019,
        "block": 405450,
        "block_train": 324360,
        "block_hold": 81090,
        "low": 55,
        "low_train": 44,
        "low_hold": 11,
        "high": 95050,
        "high_train": 76040,
        "high_hold": 19010,
        "righthigh": 5050,
        "righthigh_train": 4040,
        "righthigh_hold": 1010,
        "top": 5050,
        "top_train": 4040,
        "top_hold": 1010,
        "cross": 500500,
    }
    for key, value in expected.items():
        if counts[key] != value:
            raise RuntimeError(f"ten-thousand pair {key} count is {counts[key]}")
    for key in (
        "zero",
        "borrow",
        "noborrow",
        "ten",
        "hundred",
        "ones",
        "place",
        "block",
        "low",
        "high",
        "righthigh",
        "top",
    ):
        if counts[key + "_train"] == 0 or counts[key + "_hold"] == 0:
            raise RuntimeError(f"ten-thousand pair split hid every {key} on one side")
    step = K
    gauges = [index * step for index in range(RADIX)]
    ten = gauges[9] + gauges[1]
    hundred = ten * RADIX
    thousand = hundred * RADIX
    folded = 0.0
    for _ in range(RADIX):
        folded = lattice.consensus_quantity(folded, thousand, 1)
    if abs(folded - RADIX * thousand) > 1e-9:
        raise RuntimeError("the synthetic ten-thousand left the thousand counted ten times")

    def _syn_read(left: int, right: int) -> tuple[int, float, float]:
        left_q = lattice.compose_ten_thousand(left, folded, thousand, hundred, ten, gauges)
        right_q = lattice.compose_ten_thousand(right, folded, thousand, hundred, ten, gauges)
        quantity = lattice.consensus_quantity(left_q, right_q, -1)
        named, remainder, margin, _dist = lattice.read_place(
            quantity, "digit", ten, gauges, hundred, thousand, folded
        )
        return named, remainder, margin

    for number in (10000, 10001, 10024, 10050, 10100, 10500, 10990, 10999):
        built = lattice.compose_ten_thousand(number, folded, thousand, hundred, ten, gauges)
        if abs(built - number * step) > 1e-9:
            raise RuntimeError(f"synthetic {number} left the integer line")
    cases = (
        (10000, 10000, 0),
        (10001, 10000, 1),
        (10024, 10000, 24),
        (10050, 10000, 50),
        (10100, 10050, 50),
        (10500, 10000, 500),
        (10999, 10000, 999),
        (10999, 10990, 9),
        (10999, 10999, 0),
    )
    for left, right, result in cases:
        named, remainder, margin = _syn_read(left, right)
        if named != result:
            raise RuntimeError(f"rebuilt {left}-{right} read {named}")
        if margin <= DROP:
            raise RuntimeError(f"rebuilt {left}-{right} sat inside the drop")
        if result % RADIX == 0 and abs(remainder) >= DROP:
            raise RuntimeError(f"exact ten {left}-{right} left a remainder")
    fresh = tuple(TenThousandPair(left, right, result) for left, right, result in cases)
    for row in fresh:
        if abs(
            lattice.ten_thousand_pair_quantity(row, "digit")
            - lattice.algebraic_ten_thousand_pair(row, "digit")
        ) > 1e-9:
            raise RuntimeError(f"fresh {row.digit_prompt()}{row.digit_answer()} left the algebraic difference")
        got = lattice.predict_ten_thousand_pair(row, "digit")
        if got != row.digit_answer():
            raise RuntimeError(f"fresh {row.digit_prompt()}{row.digit_answer()} read {got}")
        if not (
            THOUSAND_CAP <= row.right <= row.left < TEN_THOUSAND_CAP
            and row.result == row.left - row.right
            and row.result < THOUSAND_SPAN
        ):
            raise RuntimeError(f"{row.digit_prompt()}{row.digit_answer()} left the ten-thousand pair family")


def _score_labels(pairs: list[tuple[str, str, str]]) -> tuple[float, int, list[str]]:
    if not pairs:
        return 0.0, 0, []
    misses: list[str] = []
    correct = 0
    for shown, got, want in pairs:
        if got == want:
            correct += 1
        elif len(misses) < 8:
            misses.append(f"{shown} -> {got} (want {want})")
    return correct / len(pairs), correct, misses


def _order_block(lattice: Lattice, claims: list[Claim], surface: str) -> dict:
    pairs = []
    same_abs: list[float] = []
    diff_abs: list[float] = []
    for claim in claims:
        got = lattice.predict_order(claim, surface)
        want = order_label(claim)
        shown = order_digit(claim) if surface == "digit" else order_word(claim)
        pairs.append((shown, got, want))
        remainder = abs(lattice.equality_remainder(claim, surface))
        if want == "same":
            same_abs.append(remainder)
        else:
            diff_abs.append(remainder)
    acc, correct, misses = _score_labels(pairs)
    return {
        "acc": acc,
        "correct": correct,
        "n": len(claims),
        "misses": misses,
        "same_max_abs": max(same_abs) if same_abs else 0.0,
        "diff_min_abs": min(diff_abs) if diff_abs else 0.0,
    }


def _quantity_block(lattice: Lattice, rows, surface: str, kind: str) -> dict:
    pairs = []
    gap = 0.0
    for row in rows:
        if kind == "product":
            got = lattice.predict_product(row, surface)
            gap += abs(lattice.product_quantity(row, surface) - lattice.algebraic_product(row, surface))
            want = row.digit_answer() if surface == "digit" else row.word_answer()
            shown = row.digit_prompt() + row.digit_answer() if surface == "digit" else row.word_prompt()
        else:
            got = lattice.predict_span(row, surface)
            gap += abs(lattice.span_quantity(row, surface) - lattice.algebraic_span(row, surface))
            want = row.digit_answer() if surface == "digit" else row.word_answer()
            shown = row.digit_prompt() + row.digit_answer() if surface == "digit" else row.word_prompt()
        pairs.append((shown, got, want))
    acc, correct, misses = _score_labels(pairs)
    return {
        "acc": acc,
        "correct": correct,
        "n": len(rows),
        "misses": misses,
        "gap": gap / len(rows) if rows else 0.0,
    }


def _lexeme_block(lattice: Lattice, rows: list[Lexeme], surface: str) -> dict:
    pairs = []
    gap = 0.0
    margins: list[float] = []
    ten_abs: list[float] = []
    units_abs: list[float] = []
    hundred = lattice.hundred_quantity(surface)
    for row in rows:
        got = lattice.predict_lexeme(row, surface)
        want = row.digit_answer() if surface == "digit" else row.word_answer()
        shown = row.digit_prompt() + row.digit_answer() if surface == "digit" else row.word_prompt()
        pairs.append((shown, got, want))
        quantity = lattice.lexeme_quantity(row, surface)
        gap += abs(quantity - lattice.algebraic_lexeme(row, surface))
        _named, remainder, margin, units_dist = lattice.read_place(quantity, surface, hundred=hundred)
        margins.append(margin)
        units_abs.append(units_dist)
        if row.exact_ten():
            ten_abs.append(abs(remainder))
    acc, correct, misses = _score_labels(pairs)
    return {
        "acc": acc,
        "correct": correct,
        "n": len(rows),
        "misses": misses,
        "gap": gap / len(rows) if rows else 0.0,
        "margin_min": min(margins) if margins else 0.0,
        "ten_n": len(ten_abs),
        "ten_max_abs": max(ten_abs) if ten_abs else 0.0,
        "units_max_abs": max(units_abs) if units_abs else 0.0,
    }


def _place_spell(named: int, surface: str) -> str:
    if named < 0 or named >= RADIX * RADIX:
        return "?"
    if surface == "digit":
        return decimal_name(named)
    return number_name(named)


def _blank_place() -> dict:
    return {"pairs": [], "gap": 0.0, "margins": [], "ten_abs": [], "units": []}


def _finish_place(bucket: dict) -> dict:
    pairs = bucket["pairs"]
    acc, correct, misses = _score_labels(pairs)
    ten_abs = bucket["ten_abs"]
    margins = bucket["margins"]
    units = bucket["units"]
    return {
        "acc": acc,
        "correct": correct,
        "n": len(pairs),
        "misses": misses,
        "gap": bucket["gap"] / len(pairs) if pairs else 0.0,
        "margin_min": min(margins) if margins else 0.0,
        "ten_n": len(ten_abs),
        "ten_max_abs": max(ten_abs) if ten_abs else 0.0,
        "units_max_abs": max(units) if units else 0.0,
    }


def _note_place(bucket: dict, shown: str, got: str, want: str, gap: float, margin: float, dist: float, remainder: float, exact_ten: bool) -> None:
    bucket["pairs"].append((shown, got, want))
    bucket["gap"] += gap
    bucket["margins"].append(margin)
    bucket["units"].append(dist)
    if exact_ten:
        bucket["ten_abs"].append(abs(remainder))


def _round_block(lattice: Lattice, surface: str) -> dict:
    bucket = _blank_place()
    hundred = lattice.hundred_quantity(surface)
    for place in range(RADIX, RADIX * RADIX):
        quantity = lattice.place_quantity(place, surface)
        named, remainder, margin, dist = lattice.read_place(quantity, surface, hundred=hundred)
        want = _place_spell(place, surface)
        got = _place_spell(named, surface)
        shown = want if surface == "word" else f"{want}="
        gap = abs(quantity - lattice.algebraic_place(place, surface))
        _note_place(bucket, shown, got, want, gap, margin, dist, remainder, place % RADIX == 0)
    return _finish_place(bucket)


def _score_places(lattice: Lattice, surface: str) -> dict[str, dict]:
    """One pass tags train, hold, and the structural subsets."""
    train, hold = split_place_steps()
    held = set(hold)
    hundred = lattice.hundred_quantity(surface)
    names = ("train", "hold", "closed", "zero", "borrow", "carry", "under", "ten")
    buckets = {name: _blank_place() for name in names}
    for row in train + hold:
        quantity = lattice.place_step_quantity(row, surface)
        named, remainder, margin, dist = lattice.read_place(quantity, surface, hundred=hundred)
        want = row.digit_answer() if surface == "digit" else row.word_answer()
        got = _place_spell(named, surface)
        shown = row.digit_prompt() + row.digit_answer() if surface == "digit" else row.word_prompt()
        gap = abs(quantity - lattice.algebraic_place_step(row, surface))
        tags = ["closed", "hold" if row in held else "train"]
        if row.uses_zero():
            tags.append("zero")
        if row.borrows():
            tags.append("borrow")
        if row.carries():
            tags.append("carry")
        if row.under_ten():
            tags.append("under")
        if row.exact_ten():
            tags.append("ten")
        for tag in tags:
            _note_place(buckets[tag], shown, got, want, gap, margin, dist, remainder, row.exact_ten())
    return {name: _finish_place(bucket) for name, bucket in buckets.items()}


def _score_pairs(lattice: Lattice, surface: str) -> dict[str, dict]:
    """One pass tags train, hold, and the structural subsets.

    Each place 10..99 is rebuilt once. The pair itself is one consensus pass.
    """
    train, hold = split_place_pairs()
    held = set(hold)
    ten = lattice.ten_quantity(surface)
    gauges = lattice.gauge_line(surface)
    hundred = lattice.hundred_quantity(surface)
    built = {
        place: lattice.compose_place(place, ten, gauges)
        for place in range(RADIX, RADIX * RADIX)
    }
    algebraic = {
        place: lattice.algebraic_place(place, surface)
        for place in range(RADIX, RADIX * RADIX)
    }
    sign_of = {"+": lattice.routed_sign("+"), "-": lattice.routed_sign("-")}
    names = ("train", "hold", "closed", "zero", "borrow", "carry", "under", "ten")
    buckets = {name: _blank_place() for name in names}
    for row in train + hold:
        quantity = lattice.consensus_quantity(built[row.left], built[row.right], sign_of[row.op])
        named, remainder, margin, dist = lattice.read_place(quantity, surface, ten, gauges, hundred)
        want = row.digit_answer() if surface == "digit" else row.word_answer()
        got = _place_spell(named, surface)
        shown = row.digit_prompt() + row.digit_answer() if surface == "digit" else row.word_prompt()
        sign = 1.0 if row.op == "+" else -1.0
        gap = abs(quantity - (algebraic[row.left] + sign * algebraic[row.right]))
        tags = ["closed", "hold" if row in held else "train"]
        if row.is_zero():
            tags.append("zero")
        if row.borrows():
            tags.append("borrow")
        if row.carries():
            tags.append("carry")
        if row.under_ten():
            tags.append("under")
        if row.exact_ten():
            tags.append("ten")
        for tag in tags:
            _note_place(buckets[tag], shown, got, want, gap, margin, dist, remainder, row.exact_ten())
    return {name: _finish_place(bucket) for name, bucket in buckets.items()}


def _hundred_spell(named: int, surface: str) -> str:
    if named < 0 or named >= RADIX * RADIX * RADIX:
        return "?"
    if surface == "digit":
        return decimal_name(named)
    return number_name(named)


def _score_hundreds(lattice: Lattice, surface: str) -> dict[str, dict]:
    """One pass tags train, hold, and the structural subsets.

    Each place 10..99 is rebuilt once. The sum itself is one consensus pass.
    """
    train, hold = split_hundred_sums()
    held = set(hold)
    ten = lattice.ten_quantity(surface)
    gauges = lattice.gauge_line(surface)
    hundred = lattice.hundred_quantity(surface)
    built = {
        place: lattice.compose_place(place, ten, gauges)
        for place in range(RADIX, RADIX * RADIX)
    }
    algebraic = {
        place: lattice.algebraic_place(place, surface)
        for place in range(RADIX, RADIX * RADIX)
    }
    plus = lattice.routed_sign("+")
    names = ("train", "hold", "closed", "step", "pair", "hundred", "ten", "carry", "low", "high")
    buckets = {name: _blank_place() for name in names}
    for row in train + hold:
        if row.kind == "step":
            right_q = gauges[row.right]
            right_alg = lattice._gauge(row.right, surface)
        else:
            right_q = built[row.right]
            right_alg = algebraic[row.right]
        quantity = lattice.consensus_quantity(built[row.left], right_q, plus)
        named, remainder, margin, dist = lattice.read_place(quantity, surface, ten, gauges, hundred)
        want = row.digit_answer() if surface == "digit" else row.word_answer()
        got = _hundred_spell(named, surface)
        shown = row.digit_prompt() + row.digit_answer() if surface == "digit" else row.word_prompt()
        gap = abs(quantity - (algebraic[row.left] + right_alg))
        tags = ["closed", "hold" if row in held else "train", row.kind]
        if row.exact_hundred():
            tags.append("hundred")
        if row.exact_ten():
            tags.append("ten")
        if row.carries():
            tags.append("carry")
        if row.low():
            tags.append("low")
        if row.high():
            tags.append("high")
        for tag in tags:
            _note_place(buckets[tag], shown, got, want, gap, margin, dist, remainder, row.exact_ten())
    return {name: _finish_place(bucket) for name, bucket in buckets.items()}


def _blank_running() -> dict:
    return {
        "n": 0,
        "correct": 0,
        "gap": 0.0,
        "margin_min": None,
        "units_max": 0.0,
        "ten_n": 0,
        "ten_max": 0.0,
        "misses": [],
    }


def _note_running(
    bucket: dict,
    ok: bool,
    gap: float,
    margin: float,
    dist: float,
    remainder: float,
    exact_ten: bool,
    miss: str | None,
) -> None:
    bucket["n"] += 1
    bucket["gap"] += gap
    if bucket["margin_min"] is None or margin < bucket["margin_min"]:
        bucket["margin_min"] = margin
    if dist > bucket["units_max"]:
        bucket["units_max"] = dist
    if exact_ten:
        bucket["ten_n"] += 1
        rem = abs(remainder)
        if rem > bucket["ten_max"]:
            bucket["ten_max"] = rem
    if ok:
        bucket["correct"] += 1
    elif len(bucket["misses"]) < 8 and miss is not None:
        bucket["misses"].append(miss)


def _finish_running(bucket: dict) -> dict:
    n = bucket["n"]
    correct = bucket["correct"]
    margin_min = bucket["margin_min"]
    return {
        "acc": correct / n if n else 0.0,
        "correct": correct,
        "n": n,
        "misses": bucket["misses"],
        "gap": bucket["gap"] / n if n else 0.0,
        "margin_min": 0.0 if margin_min is None else margin_min,
        "ten_n": bucket["ten_n"],
        "ten_max_abs": bucket["ten_max"],
        "units_max_abs": bucket["units_max"],
    }


def _thousand_spell(named: int, surface: str) -> str:
    if named < 0 or named >= RADIX ** 4:
        return "?"
    if surface == "digit":
        return decimal_name(named)
    return number_name(named)


def _score_thousands(lattice: Lattice, surface: str) -> dict[str, dict]:
    """One pass tags train, hold, and the structural subsets.

    Each overflowing hundred name and each place 1..99 is rebuilt once.
    The sum itself is one consensus pass.
    """
    train, hold = split_thousand_sums()
    held = set(hold)
    print(f"thousand {surface} {len(train) + len(hold)}", flush=True)
    span = RADIX * RADIX
    ten = lattice.ten_quantity(surface)
    gauges = lattice.gauge_line(surface)
    hundred = lattice.hundred_quantity(surface)
    thousand = lattice.thousand_quantity(surface)
    plus = lattice.routed_sign("+")
    consensus = lattice.consensus_quantity
    lowers = {rest: lattice.compose_place(rest, ten, gauges) for rest in range(1, span)}
    alg_lowers = {rest: lattice.algebraic_place(rest, surface) for rest in range(1, span)}
    ten_alg = lattice._gauge(9, surface) + lattice._gauge(1, surface)
    hundred_alg = RADIX * ten_alg
    acc = 0.0
    for _ in range(RADIX - 1):
        acc = consensus(acc, hundred, plus)
    base_q = (RADIX - 1) * hundred_alg
    built: dict[int, float] = {}
    alg: dict[int, float] = {}
    for rest, place_q in lowers.items():
        built[(RADIX - 1) * span + rest] = consensus(acc, place_q, plus)
        alg[(RADIX - 1) * span + rest] = base_q + alg_lowers[rest]
    names = ("train", "hold", "closed", "step", "wide", "thousand", "ten", "carry", "low", "high")
    buckets = {name: _blank_place() for name in names}
    for row in train + hold:
        if row.kind == "step":
            right_q = gauges[row.right]
            right_alg = gauges[row.right]
        else:
            right_q = lowers[row.right]
            right_alg = alg_lowers[row.right]
        quantity = consensus(built[row.left], right_q, plus)
        named, remainder, margin, dist = lattice.read_place(
            quantity, surface, ten, gauges, hundred, thousand
        )
        want = row.digit_answer() if surface == "digit" else row.word_answer()
        got = _thousand_spell(named, surface)
        shown = row.digit_prompt() + row.digit_answer() if surface == "digit" else row.word_prompt()
        gap = abs(quantity - (alg[row.left] + right_alg))
        exact = row.exact_ten()
        tags = ["closed", "hold" if row in held else "train", row.kind]
        if row.exact_thousand():
            tags.append("thousand")
        if exact:
            tags.append("ten")
        if row.carries():
            tags.append("carry")
        if row.low():
            tags.append("low")
        if row.high():
            tags.append("high")
        for tag in tags:
            _note_place(buckets[tag], shown, got, want, gap, margin, dist, remainder, exact)
    return {name: _finish_place(bucket) for name, bucket in buckets.items()}


def _score_ops(
    lattice: Lattice,
    surface: str,
    train: list[HundredOp],
    hold: list[HundredOp],
) -> tuple[dict[str, dict], dict]:
    """One pass tags train, hold, and the structural subsets.

    Each name 100..999 and each place 1..99 is rebuilt once.
    The operand itself is one consensus pass. Running totals keep the
    report off the full prompt list.
    """
    print(f"hundred operand {surface} {len(train) + len(hold)}", flush=True)
    span = RADIX * RADIX
    thousand = span * RADIX
    ten = lattice.ten_quantity(surface)
    gauges = lattice.gauge_line(surface)
    hundred = lattice.hundred_quantity(surface)
    plus = lattice.routed_sign("+")
    sign_of = {"+": plus, "-": lattice.routed_sign("-")}
    consensus = lattice.consensus_quantity
    read = lattice.read_place
    lowers = {rest: lattice.compose_place(rest, ten, gauges) for rest in range(1, span)}
    alg_lowers = {rest: lattice.algebraic_place(rest, surface) for rest in range(1, span)}
    ten_alg = lattice._gauge(9, surface) + lattice._gauge(1, surface)
    hundred_alg = RADIX * ten_alg
    built: dict[int, float] = {}
    alg: dict[int, float] = {}
    acc = 0.0
    for hundreds in range(1, RADIX):
        acc = consensus(acc, hundred, plus)
        base = hundreds * span
        base_q = hundreds * hundred_alg
        built[base] = acc
        alg[base] = base_q
        for rest, place_q in lowers.items():
            built[base + rest] = consensus(acc, place_q, plus)
            alg[base + rest] = base_q + alg_lowers[rest]
    right_q_of = [0.0] * span
    right_alg_of = [0.0] * span
    for digit in range(RADIX):
        right_q_of[digit] = gauges[digit]
        right_alg_of[digit] = gauges[digit]
    for place in range(RADIX, span):
        right_q_of[place] = lowers[place]
        right_alg_of[place] = alg_lowers[place]
    names = (
        "train",
        "hold",
        "closed",
        "zero",
        "borrow",
        "carry",
        "under",
        "ten",
        "hundred",
        "cross",
        "ones",
        "wide",
    )
    buckets = {name: _blank_running() for name in names}
    round_bucket = _blank_running()
    for number in range(span, thousand):
        quantity = built[number]
        named, remainder, margin, dist = read(quantity, surface, ten, gauges, hundred)
        ok = named == number
        miss = None
        if not ok:
            want = _hundred_spell(number, surface)
            got = _hundred_spell(named, surface)
            miss = f"{want} -> {got} (want {want})"
        _note_running(
            round_bucket,
            ok,
            abs(quantity - alg[number]),
            margin,
            dist,
            remainder,
            number % RADIX == 0,
            miss,
        )
    seen = 0
    for split_name, group in (("train", train), ("hold", hold)):
        split_bucket = buckets[split_name]
        for row in group:
            sign = sign_of[row.op]
            quantity = consensus(built[row.left], right_q_of[row.right], sign)
            named, remainder, margin, dist = read(quantity, surface, ten, gauges, hundred)
            ok = named == row.result
            gap = abs(quantity - (alg[row.left] + sign * right_alg_of[row.right]))
            exact = row.exact_ten()
            miss = None
            if not ok:
                want = row.digit_answer() if surface == "digit" else row.word_answer()
                got = _hundred_spell(named, surface)
                shown = row.digit_prompt() + row.digit_answer() if surface == "digit" else row.word_prompt()
                miss = f"{shown} -> {got} (want {want})"
            _note_running(buckets["closed"], ok, gap, margin, dist, remainder, exact, miss)
            _note_running(split_bucket, ok, gap, margin, dist, remainder, exact, miss)
            if row.uses_zero():
                _note_running(buckets["zero"], ok, gap, margin, dist, remainder, exact, miss)
            if row.borrows():
                _note_running(buckets["borrow"], ok, gap, margin, dist, remainder, exact, miss)
            if row.carries():
                _note_running(buckets["carry"], ok, gap, margin, dist, remainder, exact, miss)
            if row.under_hundred():
                _note_running(buckets["under"], ok, gap, margin, dist, remainder, exact, miss)
            if exact:
                _note_running(buckets["ten"], ok, gap, margin, dist, remainder, True, miss)
            if row.exact_hundred():
                _note_running(buckets["hundred"], ok, gap, margin, dist, remainder, True, miss)
            if row.crosses():
                _note_running(buckets["cross"], ok, gap, margin, dist, remainder, exact, miss)
            if row.ones():
                _note_running(buckets["ones"], ok, gap, margin, dist, remainder, exact, miss)
            if row.wide():
                _note_running(buckets["wide"], ok, gap, margin, dist, remainder, exact, miss)
            seen += 1
            if seen % 25000 == 0:
                print(f"hundred operand {surface} {seen}", flush=True)
    print(f"hundred operand {surface} done", flush=True)
    return {name: _finish_running(bucket) for name, bucket in buckets.items()}, _finish_running(round_bucket)


def _thou_op_cache(lattice: Lattice, surface: str) -> dict:
    """Fold each thousand once, then each lower name in 1..999 once."""
    span = THOUSAND_SPAN
    place = RADIX * RADIX
    ten = lattice.ten_quantity(surface)
    gauges = lattice.gauge_line(surface)
    hundred = lattice.hundred_quantity(surface)
    thousand = lattice.thousand_quantity(surface)
    plus = lattice.routed_sign("+")
    consensus = lattice.consensus_quantity
    ten_alg = lattice._gauge(9, surface) + lattice._gauge(1, surface)
    hundred_alg = RADIX * ten_alg
    thousand_alg = place * ten_alg
    places_q = [0.0] * place
    places_a = [0.0] * place
    for rest in range(1, place):
        places_q[rest] = lattice.compose_place(rest, ten, gauges)
        places_a[rest] = lattice.algebraic_place(rest, surface)
    lowers_q = [0.0] * span
    lowers_a = [0.0] * span
    for rest in range(1, place):
        lowers_q[rest] = places_q[rest]
        lowers_a[rest] = places_a[rest]
    acc_h = 0.0
    for hundreds in range(1, RADIX):
        acc_h = consensus(acc_h, hundred, plus)
        base = hundreds * place
        base_a = hundreds * hundred_alg
        lowers_q[base] = acc_h
        lowers_a[base] = base_a
        for rest in range(1, place):
            lowers_q[base + rest] = consensus(acc_h, places_q[rest], plus)
            lowers_a[base + rest] = base_a + places_a[rest]
    built = [0.0] * (THOUSAND_CAP - span)
    alg = [0.0] * (THOUSAND_CAP - span)
    acc = 0.0
    for thousands in range(1, RADIX):
        acc = consensus(acc, thousand, plus)
        base = thousands * span
        base_a = thousands * thousand_alg
        index = base - span
        built[index] = acc
        alg[index] = base_a
        for rest in range(1, span):
            built[index + rest] = consensus(acc, lowers_q[rest], plus)
            alg[index + rest] = base_a + lowers_a[rest]
    right_q = [0.0] * span
    right_alg = [0.0] * span
    for digit in range(RADIX):
        right_q[digit] = gauges[digit]
        right_alg[digit] = gauges[digit]
    for rest in range(RADIX, span):
        right_q[rest] = lowers_q[rest]
        right_alg[rest] = lowers_a[rest]
    return {
        "ten": ten,
        "gauges": gauges,
        "hundred": hundred,
        "thousand": thousand,
        "plus": plus,
        "minus": lattice.routed_sign("-"),
        "built": built,
        "alg": alg,
        "right_q": right_q,
        "right_alg": right_alg,
        "consensus": consensus,
        "read": lattice.read_place,
    }


def _score_thou_ops(lattice: Lattice, surface: str, counts: dict[str, int]) -> tuple[dict[str, dict], dict]:
    """One pass tags train, hold, and the structural subsets.

    Each name 1000..9999 and each name 1..999 is rebuilt once.
    The operand itself is one consensus pass. Running totals keep the
    report off the row list. Progress is the start, every 500000 rows, and the end.
    """
    print(f"thousand operand {surface} start", flush=True)
    started = time.perf_counter()
    cache = _thou_op_cache(lattice, surface)
    span = THOUSAND_SPAN
    cap = THOUSAND_CAP
    place = RADIX * RADIX
    ten = cache["ten"]
    gauges = cache["gauges"]
    hundred = cache["hundred"]
    thousand = cache["thousand"]
    built = cache["built"]
    alg = cache["alg"]
    right_q_of = cache["right_q"]
    right_alg_of = cache["right_alg"]
    consensus = cache["consensus"]
    read = cache["read"]
    plus = cache["plus"]
    minus = cache["minus"]
    names = (
        "train",
        "hold",
        "closed",
        "zero",
        "borrow",
        "carry",
        "under",
        "ten",
        "hundred",
        "thousand",
        "cross",
        "ones",
        "place",
        "block",
    )
    buckets = {name: _blank_running() for name in names}
    round_bucket = _blank_running()
    note = _note_running
    for number in range(span, cap):
        index = number - span
        quantity = built[index]
        named, remainder, margin, dist = read(quantity, surface, ten, gauges, hundred, thousand)
        ok = named == number
        miss = None
        if not ok:
            want = _thousand_spell(number, surface)
            got = _thousand_spell(named, surface)
            miss = f"{want} -> {got} (want {want})"
        note(
            round_bucket,
            ok,
            abs(quantity - alg[index]),
            margin,
            dist,
            remainder,
            number % RADIX == 0,
            miss,
        )
    seen = 0
    for left in range(span, cap):
        index = left - span
        left_q = built[index]
        left_alg = alg[index]
        left_mod = left % RADIX
        left_band = left // span
        for right in range(span):
            right_q = right_q_of[right]
            right_alg = right_alg_of[right]
            if left + right < cap:
                _thou_note(
                    note,
                    buckets,
                    consensus,
                    read,
                    surface,
                    ten,
                    gauges,
                    hundred,
                    thousand,
                    left,
                    left_q,
                    left_alg,
                    left_mod,
                    left_band,
                    right,
                    right_q,
                    right_alg,
                    0,
                    left + right,
                    plus,
                    span,
                    place,
                )
                seen += 1
            _thou_note(
                note,
                buckets,
                consensus,
                read,
                surface,
                ten,
                gauges,
                hundred,
                thousand,
                left,
                left_q,
                left_alg,
                left_mod,
                left_band,
                right,
                right_q,
                right_alg,
                1,
                left - right,
                minus,
                span,
                place,
            )
            seen += 1
            if seen % 500000 == 0:
                elapsed = time.perf_counter() - started
                print(f"thousand operand {surface} {seen} {elapsed:.0f}s", flush=True)
    print(f"thousand operand {surface} done", flush=True)
    if seen != counts["n"] or buckets["train"]["n"] != counts["train"] or buckets["hold"]["n"] != counts["hold"]:
        raise RuntimeError(
            f"thousand operand {surface} scored {seen} "
            f"({buckets['train']['n']}/{buckets['hold']['n']}) of {counts['n']}"
        )
    for key in (
        "zero",
        "borrow",
        "carry",
        "under",
        "ten",
        "hundred",
        "thousand",
        "cross",
        "ones",
        "place",
        "block",
    ):
        if buckets[key]["n"] != counts[key]:
            raise RuntimeError(f"thousand operand {surface} {key} scored {buckets[key]['n']}")
    return {name: _finish_running(bucket) for name, bucket in buckets.items()}, _finish_running(round_bucket)


def _thou_note(
    note,
    buckets: dict[str, dict],
    consensus,
    read,
    surface: str,
    ten: float,
    gauges: list[float],
    hundred: float,
    thousand: float,
    left: int,
    left_q: float,
    left_alg: float,
    left_mod: int,
    left_band: int,
    right: int,
    right_q: float,
    right_alg: float,
    op_bit: int,
    result: int,
    sign: int,
    span: int,
    place: int,
) -> None:
    quantity = consensus(left_q, right_q, sign)
    named, remainder, margin, dist = read(quantity, surface, ten, gauges, hundred, thousand)
    ok = named == result
    gap = abs(quantity - (left_alg + sign * right_alg))
    exact = result % RADIX == 0
    miss = None
    if not ok:
        want = decimal_name(result) if surface == "digit" else number_name(result)
        got = _thousand_spell(named, surface)
        op = "+" if op_bit == 0 else "-"
        if surface == "digit":
            shown = f"{decimal_name(left)}{op}{decimal_name(right)}={decimal_name(result)}"
        else:
            right_word = WORD_OF[right] if right < RADIX else number_name(right)
            shown = f"what is {number_name(left)} {OP_WORD[op]} {right_word}"
        miss = f"{shown} -> {got} (want {want})"
    held = thousand_op_held(left, right, op_bit)
    note(buckets["closed"], ok, gap, margin, dist, remainder, exact, miss)
    note(buckets["hold" if held else "train"], ok, gap, margin, dist, remainder, exact, miss)
    if right == 0:
        note(buckets["zero"], ok, gap, margin, dist, remainder, exact, miss)
    if op_bit == 1 and left_mod < right % RADIX:
        note(buckets["borrow"], ok, gap, margin, dist, remainder, exact, miss)
    if op_bit == 0 and left_mod + right % RADIX >= RADIX:
        note(buckets["carry"], ok, gap, margin, dist, remainder, exact, miss)
    if result < span:
        note(buckets["under"], ok, gap, margin, dist, remainder, exact, miss)
    if exact:
        note(buckets["ten"], ok, gap, margin, dist, remainder, True, miss)
    if result % place == 0:
        note(buckets["hundred"], ok, gap, margin, dist, remainder, True, miss)
    if result % span == 0:
        note(buckets["thousand"], ok, gap, margin, dist, remainder, True, miss)
    if left_band != result // span:
        note(buckets["cross"], ok, gap, margin, dist, remainder, exact, miss)
    if right < RADIX:
        note(buckets["ones"], ok, gap, margin, dist, remainder, exact, miss)
    elif right < place:
        note(buckets["place"], ok, gap, margin, dist, remainder, exact, miss)
    else:
        note(buckets["block"], ok, gap, margin, dist, remainder, exact, miss)


def _bound_reader(gauges, ten, hundred, thousand, ten_thousand, hundred_thousand=None):
    """One C call per row when the attend can hold this surface. Otherwise None."""
    from fsot_lattice import fast_attend

    if fast_attend.bind_read(gauges, ten, hundred, thousand, ten_thousand, hundred_thousand):
        return fast_attend.consensus_read
    return None


def _score_thou_pairs(lattice: Lattice, surface: str, counts: dict[str, int]) -> dict[str, dict]:
    """One pass tags train, hold, and the structural subsets.

    Each name 1000..9999 is rebuilt once. The pair itself is one consensus
    pass. Running totals keep the report off the row list. Progress is the
    start, every 500000 rows, and the end.
    """
    print(f"thousand pair {surface} start", flush=True)
    started = time.perf_counter()
    cache = _thou_op_cache(lattice, surface)
    span = THOUSAND_SPAN
    cap = THOUSAND_CAP
    place = RADIX * RADIX
    ten = cache["ten"]
    gauges = cache["gauges"]
    hundred = cache["hundred"]
    thousand = cache["thousand"]
    built = cache["built"]
    alg = cache["alg"]
    consensus = cache["consensus"]
    read = cache["read"]
    plus = cache["plus"]
    minus = cache["minus"]
    reader = _bound_reader(gauges, ten, hundred, thousand, thousand * len(gauges))
    names = (
        "train",
        "hold",
        "closed",
        "zero",
        "borrow",
        "carry",
        "under",
        "ten",
        "hundred",
        "thousand",
        "left1000",
        "left9000",
        "right9000",
        "top",
    )
    buckets = {name: _blank_running() for name in names}
    note = _note_running
    seen = 0
    for left in range(span, cap):
        index = left - span
        left_q = built[index]
        left_alg = alg[index]
        left_mod = left % RADIX
        low_left = left < 2 * span
        high_left = left >= 9 * span
        plus_stop = cap - left
        if plus_stop > span:
            for right in range(span, plus_stop):
                _thou_pair_note(
                    note,
                    buckets,
                    consensus,
                    read,
                    surface,
                    ten,
                    gauges,
                    hundred,
                    thousand,
                    left,
                    left_q,
                    left_alg,
                    left_mod,
                    low_left,
                    high_left,
                    right,
                    built[right - span],
                    alg[right - span],
                    0,
                    left + right,
                    plus,
                    span,
                    place,
                    reader,
                )
                seen += 1
                if seen % 500000 == 0:
                    elapsed = time.perf_counter() - started
                    print(f"thousand pair {surface} {seen} {elapsed:.0f}s", flush=True)
        for right in range(span, left + 1):
            _thou_pair_note(
                note,
                buckets,
                consensus,
                read,
                surface,
                ten,
                gauges,
                hundred,
                thousand,
                left,
                left_q,
                left_alg,
                left_mod,
                low_left,
                high_left,
                right,
                built[right - span],
                alg[right - span],
                1,
                left - right,
                minus,
                span,
                place,
                reader,
            )
            seen += 1
            if seen % 500000 == 0:
                elapsed = time.perf_counter() - started
                print(f"thousand pair {surface} {seen} {elapsed:.0f}s", flush=True)
    print(f"thousand pair {surface} done", flush=True)
    if seen != counts["n"] or buckets["train"]["n"] != counts["train"] or buckets["hold"]["n"] != counts["hold"]:
        raise RuntimeError(
            f"thousand pair {surface} scored {seen} "
            f"({buckets['train']['n']}/{buckets['hold']['n']}) of {counts['n']}"
        )
    for key in (
        "zero",
        "borrow",
        "carry",
        "under",
        "ten",
        "hundred",
        "thousand",
        "left1000",
        "left9000",
        "right9000",
        "top",
    ):
        if buckets[key]["n"] != counts[key]:
            raise RuntimeError(f"thousand pair {surface} {key} scored {buckets[key]['n']}")
    return {name: _finish_running(bucket) for name, bucket in buckets.items()}


def _thou_pair_note(
    note,
    buckets: dict[str, dict],
    consensus,
    read,
    surface: str,
    ten: float,
    gauges: list[float],
    hundred: float,
    thousand: float,
    left: int,
    left_q: float,
    left_alg: float,
    left_mod: int,
    low_left: bool,
    high_left: bool,
    right: int,
    right_q: float,
    right_alg: float,
    op_bit: int,
    result: int,
    sign: int,
    span: int,
    place: int,
    reader,
) -> None:
    got = reader(left_q, right_q, sign) if reader is not None else None
    if got is None:
        quantity = consensus(left_q, right_q, sign)
        named, remainder, margin, dist = read(quantity, surface, ten, gauges, hundred, thousand)
    else:
        quantity, named, remainder, margin, dist = got
    ok = named == result
    gap = abs(quantity - (left_alg + sign * right_alg))
    exact = result % RADIX == 0
    miss = None
    if not ok:
        want = decimal_name(result) if surface == "digit" else number_name(result)
        got = _thousand_spell(named, surface)
        op = "+" if op_bit == 0 else "-"
        if surface == "digit":
            shown = f"{decimal_name(left)}{op}{decimal_name(right)}={decimal_name(result)}"
        else:
            shown = f"what is {number_name(left)} {OP_WORD[op]} {number_name(right)}"
        miss = f"{shown} -> {got} (want {want})"
    held = thousand_pair_held(left, right, op_bit)
    note(buckets["closed"], ok, gap, margin, dist, remainder, exact, miss)
    note(buckets["hold" if held else "train"], ok, gap, margin, dist, remainder, exact, miss)
    if result == 0:
        note(buckets["zero"], ok, gap, margin, dist, remainder, True, miss)
    if op_bit == 1 and left_mod < right % RADIX:
        note(buckets["borrow"], ok, gap, margin, dist, remainder, exact, miss)
    if op_bit == 0 and left_mod + right % RADIX >= RADIX:
        note(buckets["carry"], ok, gap, margin, dist, remainder, exact, miss)
    if result < span:
        note(buckets["under"], ok, gap, margin, dist, remainder, exact, miss)
    if exact:
        note(buckets["ten"], ok, gap, margin, dist, remainder, True, miss)
    if result % place == 0:
        note(buckets["hundred"], ok, gap, margin, dist, remainder, True, miss)
    if result % span == 0:
        note(buckets["thousand"], ok, gap, margin, dist, remainder, True, miss)
    if low_left:
        note(buckets["left1000"], ok, gap, margin, dist, remainder, exact, miss)
    if high_left:
        note(buckets["left9000"], ok, gap, margin, dist, remainder, exact, miss)
    if right >= 9 * span:
        note(buckets["right9000"], ok, gap, margin, dist, remainder, exact, miss)
    if result >= 9 * span:
        note(buckets["top"], ok, gap, margin, dist, remainder, exact, miss)


def _ten_thou_spell(named: int, surface: str) -> str:
    if named < THOUSAND_CAP or named >= TEN_THOUSAND_CAP:
        return "?"
    if surface == "digit":
        return decimal_name(named)
    return number_name(named)


def _score_ten_thousands(lattice: Lattice, surface: str, counts: dict[str, int]) -> dict[str, dict]:
    """One pass tags train, hold, and the structural subsets.

    Each name 1000..9999 and each name 1..999 is rebuilt once. The sum
    itself is one consensus pass. The ten-thousand shed is the thousand-step
    counted ten times. Running totals keep the report off the row list.
    """
    print(f"ten thousand {surface} {counts['n']}", flush=True)
    started = time.perf_counter()
    cache = _thou_op_cache(lattice, surface)
    span = THOUSAND_SPAN
    cap = THOUSAND_CAP
    place = RADIX * RADIX
    ten = cache["ten"]
    gauges = cache["gauges"]
    hundred = cache["hundred"]
    thousand = cache["thousand"]
    built = cache["built"]
    alg = cache["alg"]
    right_q_of = cache["right_q"]
    right_alg_of = cache["right_alg"]
    consensus = cache["consensus"]
    read = cache["read"]
    plus = cache["plus"]
    acc = 0.0
    for _ in range(RADIX):
        acc = consensus(acc, thousand, plus)
    ten_thousand = acc
    names = (
        "train",
        "hold",
        "closed",
        "ones",
        "place",
        "block",
        "ten",
        "hundred",
        "thousand",
        "carry",
        "low",
        "high",
    )
    buckets = {name: _blank_running() for name in names}
    note = _note_running
    seen = 0
    for left in range(cap - span + 1, cap):
        index = left - span
        left_q = built[index]
        left_alg = alg[index]
        left_mod = left % RADIX
        for right in range(cap - left, span):
            result = left + right
            if result >= TEN_THOUSAND_CAP:
                continue
            kind = ten_thousand_kind(right)
            right_q = right_q_of[right]
            right_alg = right_alg_of[right]
            quantity = consensus(left_q, right_q, plus)
            named, remainder, margin, dist = read(
                quantity, surface, ten, gauges, hundred, thousand, ten_thousand
            )
            ok = named == result
            gap = abs(quantity - (left_alg + right_alg))
            exact = result % RADIX == 0
            miss = None
            if not ok:
                want = decimal_name(result) if surface == "digit" else number_name(result)
                got = _ten_thou_spell(named, surface)
                if surface == "digit":
                    shown = f"{decimal_name(left)}+{decimal_name(right)}={decimal_name(result)}"
                else:
                    right_word = WORD_OF[right] if right < RADIX else number_name(right)
                    shown = f"what is {number_name(left)} plus {right_word}"
                miss = f"{shown} -> {got} (want {want})"
            held = ten_thousand_held(left, right, kind)
            note(buckets["closed"], ok, gap, margin, dist, remainder, exact, miss)
            note(buckets["hold" if held else "train"], ok, gap, margin, dist, remainder, exact, miss)
            if kind == 0:
                note(buckets["ones"], ok, gap, margin, dist, remainder, exact, miss)
            elif kind == 1:
                note(buckets["place"], ok, gap, margin, dist, remainder, exact, miss)
            else:
                note(buckets["block"], ok, gap, margin, dist, remainder, exact, miss)
            if exact:
                note(buckets["ten"], ok, gap, margin, dist, remainder, True, miss)
            if result % place == 0:
                note(buckets["hundred"], ok, gap, margin, dist, remainder, True, miss)
            if result % span == 0:
                note(buckets["thousand"], ok, gap, margin, dist, remainder, True, miss)
            if left_mod + right % RADIX >= RADIX:
                note(buckets["carry"], ok, gap, margin, dist, remainder, exact, miss)
            if result < cap + RADIX:
                note(buckets["low"], ok, gap, margin, dist, remainder, exact, miss)
            if result >= cap + (RADIX - 1) * place:
                note(buckets["high"], ok, gap, margin, dist, remainder, exact, miss)
            seen += 1
            if seen % 100000 == 0:
                elapsed = time.perf_counter() - started
                print(f"ten thousand {surface} {seen} {elapsed:.0f}s", flush=True)
    print(f"ten thousand {surface} done", flush=True)
    if seen != counts["n"] or buckets["train"]["n"] != counts["train"] or buckets["hold"]["n"] != counts["hold"]:
        raise RuntimeError(
            f"ten thousand {surface} scored {seen} "
            f"({buckets['train']['n']}/{buckets['hold']['n']}) of {counts['n']}"
        )
    for key in ("ones", "place", "block", "ten", "hundred", "thousand", "carry", "low", "high"):
        if buckets[key]["n"] != counts[key]:
            raise RuntimeError(f"ten thousand {surface} {key} scored {buckets[key]['n']}")
    return {name: _finish_running(bucket) for name, bucket in buckets.items()}


def _ten_thou_op_cache(lattice: Lattice, surface: str) -> dict:
    """Fold the ten-thousand step once, then each rest in 1..999 once.

    Rights in 0..9999 reuse the thousand folds. The ten-thousand step is
    the thousand-step counted ten times.
    """
    base = _thou_op_cache(lattice, surface)
    span = THOUSAND_SPAN
    cap = THOUSAND_CAP
    consensus = base["consensus"]
    plus = base["plus"]
    acc = 0.0
    for _ in range(RADIX):
        acc = consensus(acc, base["thousand"], plus)
    ten_thousand = acc
    ten_alg = lattice._gauge(9, surface) + lattice._gauge(1, surface)
    ten_thousand_alg = (RADIX ** 3) * ten_alg
    lower_q = base["right_q"]
    lower_alg = base["right_alg"]
    left_q = [0.0] * span
    left_alg = [0.0] * span
    left_q[0] = consensus(0.0, ten_thousand, plus)
    left_alg[0] = ten_thousand_alg
    for rest in range(1, span):
        left_q[rest] = consensus(left_q[0], lower_q[rest], plus)
        left_alg[rest] = ten_thousand_alg + lower_alg[rest]
    right_q = [0.0] * cap
    right_alg = [0.0] * cap
    for right in range(span):
        right_q[right] = lower_q[right]
        right_alg[right] = lower_alg[right]
    built = base["built"]
    alg = base["alg"]
    for right in range(span, cap):
        right_q[right] = built[right - span]
        right_alg[right] = alg[right - span]
    return {
        "ten": base["ten"],
        "gauges": base["gauges"],
        "hundred": base["hundred"],
        "thousand": base["thousand"],
        "ten_thousand": ten_thousand,
        "plus": plus,
        "minus": base["minus"],
        "left_q": left_q,
        "left_alg": left_alg,
        "right_q": right_q,
        "right_alg": right_alg,
        "consensus": consensus,
        "read": base["read"],
    }


def _ten_op_spell(named: int, surface: str) -> str:
    if named < 0 or named >= TEN_THOUSAND_CAP:
        return "?"
    if surface == "digit":
        return decimal_name(named)
    return number_name(named)


def _score_ten_thou_ops(lattice: Lattice, surface: str, counts: dict[str, int]) -> tuple[dict[str, dict], dict]:
    """One pass tags train, hold, and the structural subsets.

    Each name 10000..10999 is one ten-thousand step plus a name in 0..999.
    The operand itself is one consensus pass. Running totals keep the
    report off the row list. Progress is the start, every 500000 rows, and the end.
    """
    print(f"ten thousand operand {surface} start", flush=True)
    started = time.perf_counter()
    cache = _ten_thou_op_cache(lattice, surface)
    span = THOUSAND_SPAN
    cap = THOUSAND_CAP
    upper = TEN_THOUSAND_CAP
    place = RADIX * RADIX
    ten = cache["ten"]
    gauges = cache["gauges"]
    hundred = cache["hundred"]
    thousand = cache["thousand"]
    ten_thousand = cache["ten_thousand"]
    left_q_of = cache["left_q"]
    left_alg_of = cache["left_alg"]
    right_q_of = cache["right_q"]
    right_alg_of = cache["right_alg"]
    consensus = cache["consensus"]
    read = cache["read"]
    plus = cache["plus"]
    minus = cache["minus"]
    reader = _bound_reader(gauges, ten, hundred, thousand, ten_thousand)
    names = (
        "train",
        "hold",
        "closed",
        "zero",
        "borrow",
        "carry",
        "under",
        "ten",
        "hundred",
        "thousand",
        "cross",
        "ones",
        "place",
        "block",
        "right1000",
    )
    buckets = {name: _blank_running() for name in names}
    round_bucket = _blank_running()
    note = _note_running
    for rest in range(span):
        number = cap + rest
        quantity = left_q_of[rest]
        named, remainder, margin, dist = read(
            quantity, surface, ten, gauges, hundred, thousand, ten_thousand
        )
        ok = named == number
        miss = None
        if not ok:
            want = _ten_op_spell(number, surface)
            got = _ten_op_spell(named, surface)
            miss = f"{want} -> {got} (want {want})"
        note(
            round_bucket,
            ok,
            abs(quantity - left_alg_of[rest]),
            margin,
            dist,
            remainder,
            number % RADIX == 0,
            miss,
        )
    seen = 0
    for left in range(cap, upper):
        rest = left - cap
        left_q = left_q_of[rest]
        left_alg = left_alg_of[rest]
        left_mod = left % RADIX
        plus_stop = upper - left
        for right in range(plus_stop):
            _ten_op_note(
                note,
                buckets,
                consensus,
                read,
                surface,
                ten,
                gauges,
                hundred,
                thousand,
                ten_thousand,
                left,
                left_q,
                left_alg,
                left_mod,
                right,
                right_q_of[right],
                right_alg_of[right],
                0,
                left + right,
                plus,
                span,
                place,
                cap,
                reader,
            )
            seen += 1
            if seen % 500000 == 0:
                elapsed = time.perf_counter() - started
                print(f"ten thousand operand {surface} {seen} {elapsed:.0f}s", flush=True)
        for right in range(cap):
            _ten_op_note(
                note,
                buckets,
                consensus,
                read,
                surface,
                ten,
                gauges,
                hundred,
                thousand,
                ten_thousand,
                left,
                left_q,
                left_alg,
                left_mod,
                right,
                right_q_of[right],
                right_alg_of[right],
                1,
                left - right,
                minus,
                span,
                place,
                cap,
                reader,
            )
            seen += 1
            if seen % 500000 == 0:
                elapsed = time.perf_counter() - started
                print(f"ten thousand operand {surface} {seen} {elapsed:.0f}s", flush=True)
    print(f"ten thousand operand {surface} done", flush=True)
    if seen != counts["n"] or buckets["train"]["n"] != counts["train"] or buckets["hold"]["n"] != counts["hold"]:
        raise RuntimeError(
            f"ten thousand operand {surface} scored {seen} "
            f"({buckets['train']['n']}/{buckets['hold']['n']}) of {counts['n']}"
        )
    for key in (
        "zero",
        "borrow",
        "carry",
        "under",
        "ten",
        "hundred",
        "thousand",
        "cross",
        "ones",
        "place",
        "block",
        "right1000",
    ):
        if buckets[key]["n"] != counts[key]:
            raise RuntimeError(f"ten thousand operand {surface} {key} scored {buckets[key]['n']}")
    return {name: _finish_running(bucket) for name, bucket in buckets.items()}, _finish_running(round_bucket)


def _ten_op_note(
    note,
    buckets: dict[str, dict],
    consensus,
    read,
    surface: str,
    ten: float,
    gauges: list[float],
    hundred: float,
    thousand: float,
    ten_thousand: float,
    left: int,
    left_q: float,
    left_alg: float,
    left_mod: int,
    right: int,
    right_q: float,
    right_alg: float,
    op_bit: int,
    result: int,
    sign: int,
    span: int,
    place: int,
    cap: int,
    reader,
) -> None:
    got = reader(left_q, right_q, sign) if reader is not None else None
    if got is None:
        quantity = consensus(left_q, right_q, sign)
        named, remainder, margin, dist = read(
            quantity, surface, ten, gauges, hundred, thousand, ten_thousand
        )
    else:
        quantity, named, remainder, margin, dist = got
    ok = named == result
    gap = abs(quantity - (left_alg + sign * right_alg))
    exact = result % RADIX == 0
    miss = None
    if not ok:
        want = decimal_name(result) if surface == "digit" else number_name(result)
        got = _ten_op_spell(named, surface)
        op = "+" if op_bit == 0 else "-"
        if surface == "digit":
            shown = f"{decimal_name(left)}{op}{decimal_name(right)}={decimal_name(result)}"
        else:
            right_word = WORD_OF[right] if right < RADIX else number_name(right)
            shown = f"what is {number_name(left)} {OP_WORD[op]} {right_word}"
        miss = f"{shown} -> {got} (want {want})"
    held = ten_thousand_op_held(left, right, op_bit)
    note(buckets["closed"], ok, gap, margin, dist, remainder, exact, miss)
    note(buckets["hold" if held else "train"], ok, gap, margin, dist, remainder, exact, miss)
    if right == 0:
        note(buckets["zero"], ok, gap, margin, dist, remainder, exact, miss)
    if op_bit == 1 and left_mod < right % RADIX:
        note(buckets["borrow"], ok, gap, margin, dist, remainder, exact, miss)
    if op_bit == 0 and left_mod + right % RADIX >= RADIX:
        note(buckets["carry"], ok, gap, margin, dist, remainder, exact, miss)
    if result < cap:
        note(buckets["under"], ok, gap, margin, dist, remainder, exact, miss)
    if exact:
        note(buckets["ten"], ok, gap, margin, dist, remainder, True, miss)
    if result % place == 0:
        note(buckets["hundred"], ok, gap, margin, dist, remainder, True, miss)
    if result % span == 0:
        note(buckets["thousand"], ok, gap, margin, dist, remainder, True, miss)
    if left // cap != result // cap:
        note(buckets["cross"], ok, gap, margin, dist, remainder, exact, miss)
    if right < RADIX:
        note(buckets["ones"], ok, gap, margin, dist, remainder, exact, miss)
    elif right < place:
        note(buckets["place"], ok, gap, margin, dist, remainder, exact, miss)
    elif right < span:
        note(buckets["block"], ok, gap, margin, dist, remainder, exact, miss)
    else:
        note(buckets["right1000"], ok, gap, margin, dist, remainder, exact, miss)


def _ten_pair_spell(named: int, surface: str) -> str:
    if named < 0 or named >= THOUSAND_SPAN:
        return "?"
    if surface == "digit":
        return decimal_name(named)
    return number_name(named)


def _score_ten_thou_pairs(lattice: Lattice, surface: str, counts: dict[str, int]) -> dict[str, dict]:
    """One pass tags train, hold, and the structural subsets.

    Each name 10000..10999 is rebuilt once. The pair itself is one consensus
    pass, and the operation is minus. Running totals keep the report off the
    row list.
    """
    print(f"ten thousand pair {surface} start", flush=True)
    started = time.perf_counter()
    cache = _ten_thou_op_cache(lattice, surface)
    cap = THOUSAND_CAP
    span = THOUSAND_SPAN
    place = RADIX * RADIX
    ten = cache["ten"]
    gauges = cache["gauges"]
    hundred = cache["hundred"]
    thousand = cache["thousand"]
    ten_thousand = cache["ten_thousand"]
    left_q_of = cache["left_q"]
    left_alg_of = cache["left_alg"]
    minus = cache["minus"]
    reader = _bound_reader(gauges, ten, hundred, thousand, ten_thousand)
    consensus = cache["consensus"]
    read = cache["read"]
    names = (
        "train",
        "hold",
        "closed",
        "zero",
        "borrow",
        "ten",
        "hundred",
        "ones",
        "place",
        "block",
        "low",
        "high",
        "righthigh",
        "top",
    )
    buckets = {name: _blank_running() for name in names}
    note = _note_running
    seen = 0
    high_cut = cap + (RADIX - 1) * place
    for left in range(cap, TEN_THOUSAND_CAP):
        rest = left - cap
        left_q = left_q_of[rest]
        left_alg = left_alg_of[rest]
        left_mod = left % RADIX
        low_left = left < cap + RADIX
        high_left = left >= high_cut
        for right in range(cap, left + 1):
            result = left - right
            right_rest = right - cap
            right_q = left_q_of[right_rest]
            right_alg = left_alg_of[right_rest]
            got = reader(left_q, right_q, minus) if reader is not None else None
            if got is None:
                quantity = consensus(left_q, right_q, minus)
                named, remainder, margin, dist = read(
                    quantity, surface, ten, gauges, hundred, thousand, ten_thousand
                )
            else:
                quantity, named, remainder, margin, dist = got
            ok = named == result
            gap = abs(quantity - (left_alg - right_alg))
            exact = result % RADIX == 0
            miss = None
            if not ok:
                want = decimal_name(result) if surface == "digit" else number_name(result)
                spelled = _ten_pair_spell(named, surface)
                if surface == "digit":
                    shown = f"{decimal_name(left)}-{decimal_name(right)}={decimal_name(result)}"
                else:
                    shown = f"what is {number_name(left)} minus {number_name(right)}"
                miss = f"{shown} -> {spelled} (want {want})"
            held = ten_thousand_pair_held(left, right)
            note(buckets["closed"], ok, gap, margin, dist, remainder, exact, miss)
            note(buckets["hold" if held else "train"], ok, gap, margin, dist, remainder, exact, miss)
            if result == 0:
                note(buckets["zero"], ok, gap, margin, dist, remainder, True, miss)
            if left_mod < right % RADIX:
                note(buckets["borrow"], ok, gap, margin, dist, remainder, exact, miss)
            if exact:
                note(buckets["ten"], ok, gap, margin, dist, remainder, True, miss)
            if result % place == 0:
                note(buckets["hundred"], ok, gap, margin, dist, remainder, True, miss)
            if result < RADIX:
                note(buckets["ones"], ok, gap, margin, dist, remainder, exact, miss)
            elif result < place:
                note(buckets["place"], ok, gap, margin, dist, remainder, exact, miss)
            else:
                note(buckets["block"], ok, gap, margin, dist, remainder, exact, miss)
            if low_left:
                note(buckets["low"], ok, gap, margin, dist, remainder, exact, miss)
            if high_left:
                note(buckets["high"], ok, gap, margin, dist, remainder, exact, miss)
            if right >= high_cut:
                note(buckets["righthigh"], ok, gap, margin, dist, remainder, exact, miss)
            if result >= 9 * place:
                note(buckets["top"], ok, gap, margin, dist, remainder, exact, miss)
            seen += 1
            if seen % 100000 == 0:
                elapsed = time.perf_counter() - started
                print(f"ten thousand pair {surface} {seen} {elapsed:.0f}s", flush=True)
    print(f"ten thousand pair {surface} done", flush=True)
    if seen != counts["n"] or buckets["train"]["n"] != counts["train"] or buckets["hold"]["n"] != counts["hold"]:
        raise RuntimeError(
            f"ten thousand pair {surface} scored {seen} "
            f"({buckets['train']['n']}/{buckets['hold']['n']}) of {counts['n']}"
        )
    for key in (
        "zero",
        "borrow",
        "ten",
        "hundred",
        "ones",
        "place",
        "block",
        "low",
        "high",
        "righthigh",
        "top",
    ):
        if buckets[key]["n"] != counts[key]:
            raise RuntimeError(f"ten thousand pair {surface} {key} scored {buckets[key]['n']}")
    return {name: _finish_running(bucket) for name, bucket in buckets.items()}


def _check_hundred_thousands(lattice: Lattice) -> None:
    """The hundred-thousand place is the ten-thousand step counted ten times.

    Mixed units are checked on a pure line. The fresh digit gauges are exact
    when the extra K on each units place cancels into that hundred-thousand,
    or when a units digit is absent on both sides. A name still inside
    0..99999 sheds zero hundred-thousands.
    """
    spells = (
        (10000, "ten thousand"),
        (10999, "ten thousand nine hundred ninety-nine"),
        (11000, "eleven thousand"),
        (12000, "twelve thousand"),
        (20000, "twenty thousand"),
        (21000, "twenty-one thousand"),
        (21024, "twenty-one thousand twenty-four"),
        (90000, "ninety thousand"),
        (90001, "ninety thousand one"),
        (99999, "ninety-nine thousand nine hundred ninety-nine"),
        (100000, "one hundred thousand"),
        (100001, "one hundred thousand one"),
        (100010, "one hundred thousand ten"),
        (100050, "one hundred thousand fifty"),
        (100100, "one hundred thousand one hundred"),
        (101000, "one hundred one thousand"),
        (109000, "one hundred nine thousand"),
        (109998, "one hundred nine thousand nine hundred ninety-eight"),
        (109999, "one hundred nine thousand nine hundred ninety-nine"),
    )
    for number, spelling in spells:
        got = number_name(number)
        if got != spelling:
            raise RuntimeError(f"{number} spelled {got}")
    if number_name(110000) != "?" or number_name(120000) != "?":
        raise RuntimeError("a name past one hundred nine thousand nine hundred ninety-nine entered the table")
    if decimal_name(100000) != "100000":
        raise RuntimeError("the decimal hundred thousand left the integer")
    counts = census_hundred_thousand_sums()
    width = THOUSAND_CAP
    if counts["n"] != (width - 1) * width // 2:
        raise RuntimeError("hundred-thousand sums left the ways to write 100000..109998")
    if counts["ones"] != sum(range(1, RADIX)):
        raise RuntimeError("hundred-thousand digit rights left 1..9")
    if counts["place"] != sum(range(RADIX, RADIX * RADIX)):
        raise RuntimeError("hundred-thousand place rights left 10..99")
    if counts["block"] != sum(range(RADIX * RADIX, THOUSAND_SPAN)):
        raise RuntimeError("hundred-thousand block rights left 100..999")
    if counts["thou"] != sum(range(THOUSAND_SPAN, THOUSAND_CAP)):
        raise RuntimeError("hundred-thousand thousand-name rights left 1000..9999")
    if counts["ones"] + counts["place"] + counts["block"] + counts["thou"] != counts["n"]:
        raise RuntimeError("hundred-thousand bands do not cover the family")
    if counts["mark"] != width - 1:
        raise RuntimeError("the exact hundred-thousands are not the ways to write 100000")
    if counts["ten"] != 5004000:
        raise RuntimeError("hundred-thousand exact tens left that band")
    if counts["hundred"] != 504900:
        raise RuntimeError("hundred-thousand exact hundreds left that band")
    if counts["thousand"] != 54990:
        raise RuntimeError("hundred-thousand exact thousands left that band")
    if counts["low"] != 99945:
        raise RuntimeError("the hundred-thousand low band left 100000..100009")
    if counts["high"] != 4950:
        raise RuntimeError("the hundred-thousand high band left 109900..109998")
    if counts["carry"] != 22522500 or counts["nocarry"] != 27472500:
        raise RuntimeError("hundred-thousand carries left the units sums")
    if counts["min_result"] != HUNDRED_THOUSAND or counts["max_result"] != 109998:
        raise RuntimeError("hundred-thousand results left 100000..109998")
    if counts["train"] + counts["hold"] != counts["n"] or counts["hold"] * 4 != counts["train"]:
        raise RuntimeError(
            f"hundred-thousand split is {counts['train']}/{counts['hold']} of {counts['n']}"
        )
    for key in (
        "ones",
        "place",
        "block",
        "thou",
        "ten",
        "hundred",
        "thousand",
        "mark",
        "carry",
        "nocarry",
        "low",
        "high",
    ):
        if counts[key + "_train"] == 0 or counts[key + "_hold"] == 0:
            raise RuntimeError(f"hundred-thousand split hid every {key} on one side")
    step = K
    gauges = [index * step for index in range(RADIX)]
    ten = gauges[9] + gauges[1]
    hundred = ten * RADIX
    thousand = hundred * RADIX
    ten_thousand = thousand * RADIX
    hundred_thousand = ten_thousand * RADIX

    def _syn_read(left: int, right: int) -> tuple[int, float, float]:
        left_q = lattice.compose_ten_thousand(left, ten_thousand, thousand, hundred, ten, gauges)
        right_q = lattice.compose_below_ten_thousand(right, thousand, hundred, ten, gauges)
        quantity = lattice.consensus_quantity(left_q, right_q, 1)
        named, remainder, margin, _dist = lattice.read_place(
            quantity, "digit", ten, gauges, hundred, thousand, ten_thousand, hundred_thousand
        )
        return named, remainder, margin

    for number in (10000, 90000, 90001, 95000, 99000, 99999):
        got = lattice.compose_ten_thousand(number, ten_thousand, thousand, hundred, ten, gauges)
        if abs(got - number * step) > 1e-9:
            raise RuntimeError(f"synthetic {number} left the integer line")
    folded = lattice.fold_steps(0.0, [(1, ten_thousand) for _ in range(RADIX)])
    if abs(folded - hundred_thousand) > 1e-9:
        raise RuntimeError("the synthetic hundred-thousand left the ten-thousand counted ten times")
    cases = (
        (99999, 1, 100000),
        (99001, 999, 100000),
        (95000, 5000, 100000),
        (90001, 9999, 100000),
        (99000, 2000, 101000),
        (99500, 500, 100000),
        (98000, 2500, 100500),
        (99999, 9999, 109998),
    )
    for left, right, result in cases:
        named, remainder, margin = _syn_read(left, right)
        if named != result:
            raise RuntimeError(f"rebuilt {left}+{right} read {named}")
        if margin <= DROP:
            raise RuntimeError(f"rebuilt {left}+{right} sat inside the drop")
        if result % RADIX == 0 and abs(remainder) >= DROP:
            raise RuntimeError(f"exact ten {left}+{right} left a remainder")
    stay = (
        (90001, 9998, 99999),
        (95000, 4999, 99999),
        (99998, 1, 99999),
    )
    for left, right, result in stay:
        named, _remainder, margin = _syn_read(left, right)
        if named != result:
            raise RuntimeError(f"rebuilt {left}+{right} shed into the hundred-thousand")
        if margin <= DROP:
            raise RuntimeError(f"rebuilt {left}+{right} sat inside the drop")
    bare = lattice.compose_ten_thousand(99999, ten_thousand, thousand, hundred, ten, gauges)
    named, _remainder, margin, _dist = lattice.read_place(
        bare, "digit", ten, gauges, hundred, thousand, ten_thousand, hundred_thousand
    )
    if named != 99999 or margin <= DROP:
        raise RuntimeError(f"synthetic 99999 read {named}")
    live_ten_thousand = lattice.ten_thousand_quantity("digit")
    live_hundred_thousand = lattice.hundred_thousand_quantity("digit")
    if abs(live_hundred_thousand - RADIX * live_ten_thousand) > 1e-9:
        raise RuntimeError("the fresh hundred-thousand left the ten-thousand counted ten times")
    if lattice.read_place(live_hundred_thousand, "digit")[0] != HUNDRED_THOUSAND:
        raise RuntimeError("the hundred-thousand step did not read as 100000")
    fresh_top = lattice.compose_ten_thousand(
        99999,
        live_ten_thousand,
        lattice.thousand_quantity("digit"),
        lattice.hundred_quantity("digit"),
        lattice.ten_quantity("digit"),
        lattice.gauge_line("digit"),
    )
    if lattice.read_place(fresh_top, "digit")[0] != 99999:
        raise RuntimeError("99999 shed a hundred-thousand on the fresh line")
    fresh = (
        HundredThousandSum("digit", 99999, 1, 100000),
        HundredThousandSum("block", 99001, 999, 100000),
        HundredThousandSum("thou", 95000, 5000, 100000),
        HundredThousandSum("thou", 90001, 9999, 100000),
        HundredThousandSum("thou", 99000, 2000, 101000),
        HundredThousandSum("block", 99500, 500, 100000),
        HundredThousandSum("thou", 98000, 2500, 100500),
    )
    for row in fresh:
        if abs(
            lattice.hundred_thousand_sum_quantity(row, "digit")
            - lattice.algebraic_hundred_thousand_sum(row, "digit")
        ) > 1e-9:
            raise RuntimeError(f"fresh {row.digit_prompt()}{row.digit_answer()} left the algebraic sum")
        got = lattice.predict_hundred_thousand(row, "digit")
        if got != row.digit_answer():
            raise RuntimeError(f"fresh {row.digit_prompt()}{row.digit_answer()} read {got}")
    demos = fresh + (HundredThousandSum("thou", 99999, 9999, 109998),)
    for row in demos:
        kind = hundred_thousand_kind(row.right)
        if row.kind != ("digit", "place", "block", "thou")[kind]:
            raise RuntimeError(f"{row.digit_prompt()}{row.digit_answer()} left its right-hand band")
        if not (HUNDRED_THOUSAND - THOUSAND_CAP + 1 <= row.left < HUNDRED_THOUSAND):
            raise RuntimeError(f"{row.digit_prompt()}{row.digit_answer()} is outside the hundred-thousand family")
        if not (0 < row.right < THOUSAND_CAP):
            raise RuntimeError(f"{row.digit_prompt()}{row.digit_answer()} put a zero or a ten-thousand on the right")
        if row.result != row.left + row.right or not (HUNDRED_THOUSAND <= row.result < HUNDRED_THOUSAND_CAP):
            raise RuntimeError(f"{row.digit_prompt()}{row.digit_answer()} left 100000..109998")


def _hund_thou_spell(named: int, surface: str) -> str:
    if named < HUNDRED_THOUSAND or named >= HUNDRED_THOUSAND_CAP:
        return "?"
    if surface == "digit":
        return decimal_name(named)
    return number_name(named)


def _score_hundred_thousands(lattice: Lattice, surface: str, counts: dict[str, int]) -> dict[str, dict]:
    """One pass tags train, hold, and the structural subsets.

    Each name 90001..99999 is nine ten-thousand steps plus a name in 1..9999.
    The sum itself is one consensus pass. The hundred-thousand shed is the
    ten-thousand step counted ten times. Running totals keep the report off
    the row list.
    """
    print(f"hundred thousand {surface} start", flush=True)
    started = time.perf_counter()
    cache = _ten_thou_op_cache(lattice, surface)
    span = THOUSAND_CAP
    cap = HUNDRED_THOUSAND
    place = RADIX * RADIX
    block = THOUSAND_SPAN
    ten = cache["ten"]
    gauges = cache["gauges"]
    hundred = cache["hundred"]
    thousand = cache["thousand"]
    ten_thousand = cache["ten_thousand"]
    right_q_of = cache["right_q"]
    right_alg_of = cache["right_alg"]
    consensus = cache["consensus"]
    read = cache["read"]
    plus = cache["plus"]
    nine = 0.0
    for _ in range(RADIX - 1):
        nine = consensus(nine, ten_thousand, plus)
    hundred_thousand = consensus(nine, ten_thousand, plus)
    ten_alg = lattice._gauge(9, surface) + lattice._gauge(1, surface)
    ten_thousand_alg = (RADIX ** 3) * ten_alg
    nine_alg = (RADIX - 1) * ten_thousand_alg
    left_q = [0.0] * span
    left_alg = [0.0] * span
    for rest in range(1, span):
        left_q[rest] = consensus(nine, right_q_of[rest], plus)
        left_alg[rest] = nine_alg + right_alg_of[rest]
    reader = _bound_reader(gauges, ten, hundred, thousand, ten_thousand, hundred_thousand)
    names = (
        "train",
        "hold",
        "closed",
        "ones",
        "place",
        "block",
        "thou",
        "ten",
        "hundred",
        "thousand",
        "mark",
        "carry",
        "low",
        "high",
    )
    buckets = {name: _blank_running() for name in names}
    note = _note_running
    seen = 0
    high_cut = cap + span - place
    for left in range(cap - span + 1, cap):
        rest = left - (cap - span)
        left_q_row = left_q[rest]
        left_alg_row = left_alg[rest]
        left_mod = left % RADIX
        for right in range(cap - left, span):
            result = left + right
            kind = 0 if right < RADIX else 1 if right < place else 2 if right < block else 3
            right_q = right_q_of[right]
            right_alg = right_alg_of[right]
            got = reader(left_q_row, right_q, plus) if reader is not None else None
            if got is None:
                quantity = consensus(left_q_row, right_q, plus)
                named, remainder, margin, dist = read(
                    quantity, surface, ten, gauges, hundred, thousand, ten_thousand, hundred_thousand
                )
            else:
                quantity, named, remainder, margin, dist = got
            ok = named == result
            gap = abs(quantity - (left_alg_row + right_alg))
            exact = result % RADIX == 0
            miss = None
            if not ok:
                want = decimal_name(result) if surface == "digit" else number_name(result)
                spelled = _hund_thou_spell(named, surface)
                if surface == "digit":
                    shown = f"{decimal_name(left)}+{decimal_name(right)}={decimal_name(result)}"
                else:
                    right_word = WORD_OF[right] if right < RADIX else number_name(right)
                    shown = f"what is {number_name(left)} plus {right_word}"
                miss = f"{shown} -> {spelled} (want {want})"
            held = (left * 3 + right * 4 + 6 * kind) % 5 == 0
            note(buckets["closed"], ok, gap, margin, dist, remainder, exact, miss)
            note(buckets["hold" if held else "train"], ok, gap, margin, dist, remainder, exact, miss)
            if kind == 0:
                note(buckets["ones"], ok, gap, margin, dist, remainder, exact, miss)
            elif kind == 1:
                note(buckets["place"], ok, gap, margin, dist, remainder, exact, miss)
            elif kind == 2:
                note(buckets["block"], ok, gap, margin, dist, remainder, exact, miss)
            else:
                note(buckets["thou"], ok, gap, margin, dist, remainder, exact, miss)
            if exact:
                note(buckets["ten"], ok, gap, margin, dist, remainder, True, miss)
            if result % place == 0:
                note(buckets["hundred"], ok, gap, margin, dist, remainder, True, miss)
            if result % block == 0:
                note(buckets["thousand"], ok, gap, margin, dist, remainder, True, miss)
            if result % span == 0:
                note(buckets["mark"], ok, gap, margin, dist, remainder, True, miss)
            if left_mod + right % RADIX >= RADIX:
                note(buckets["carry"], ok, gap, margin, dist, remainder, exact, miss)
            if result < cap + RADIX:
                note(buckets["low"], ok, gap, margin, dist, remainder, exact, miss)
            if result >= high_cut:
                note(buckets["high"], ok, gap, margin, dist, remainder, exact, miss)
            seen += 1
            if seen % 500000 == 0:
                elapsed = time.perf_counter() - started
                print(f"hundred thousand {surface} {seen} {elapsed:.0f}s", flush=True)
    print(f"hundred thousand {surface} done", flush=True)
    if seen != counts["n"] or buckets["train"]["n"] != counts["train"] or buckets["hold"]["n"] != counts["hold"]:
        raise RuntimeError(
            f"hundred thousand {surface} scored {seen} "
            f"({buckets['train']['n']}/{buckets['hold']['n']}) of {counts['n']}"
        )
    for key in (
        "ones",
        "place",
        "block",
        "thou",
        "ten",
        "hundred",
        "thousand",
        "mark",
        "carry",
        "low",
        "high",
    ):
        if buckets[key]["n"] != counts[key]:
            raise RuntimeError(f"hundred thousand {surface} {key} scored {buckets[key]['n']}")
    return {name: _finish_running(bucket) for name, bucket in buckets.items()}


def score_generated(lattice: Lattice) -> dict:
    """Score every generated family on both surfaces. Gauges stay frozen."""
    order_train, order_hold = split_claims()
    product_train, product_hold = split_products()
    span_train, span_hold = split_spans()
    lexeme_train, lexeme_hold = split_lexemes()
    place_train, place_hold = split_place_steps()
    pair_train, pair_hold = split_place_pairs()
    hundred_train, hundred_hold = split_hundred_sums()
    thou_train, thou_hold = split_thousand_sums()
    op_train, op_hold = split_hundred_ops()
    op_counts = _op_counts(op_train + op_hold)
    thou_counts = census_thousand_ops()
    thou_pair_counts = census_thousand_pairs()
    ten_thou_counts = census_ten_thousand_sums()
    ten_thou_op_counts = census_ten_thousand_ops()
    ten_thou_pair_counts = census_ten_thousand_pairs()
    hund_thou_counts = census_hundred_thousand_sums()
    products = all_products()
    spans = all_spans()
    lexemes = all_lexemes()
    pairs = all_place_pairs()
    hundreds = all_hundred_sums()
    thousands = all_thousand_sums()
    report = {
        "drop": DROP,
        "order_train_count": len(order_train),
        "order_hold_count": len(order_hold),
        "product_count": len(products),
        "product_train_count": len(product_train),
        "product_hold_count": len(product_hold),
        "span_count": len(spans),
        "span_train_count": len(span_train),
        "span_hold_count": len(span_hold),
        "product_zero_n": sum(1 for row in products if row.uses_zero()),
        "span_zero_n": sum(1 for row in spans if row.first_is_zero()),
        "lexeme_count": len(lexemes),
        "lexeme_train_count": len(lexeme_train),
        "lexeme_hold_count": len(lexeme_hold),
        "lexeme_sum_n": sum(1 for row in lexemes if row.kind == "sum"),
        "lexeme_product_n": sum(1 for row in lexemes if row.kind == "product"),
        "lexeme_ten_n": sum(1 for row in lexemes if row.exact_ten()),
        "place_count": len(place_train) + len(place_hold),
        "place_train_count": len(place_train),
        "place_hold_count": len(place_hold),
        "pair_count": len(pairs),
        "pair_train_count": len(pair_train),
        "pair_hold_count": len(pair_hold),
        "pair_plus_n": sum(1 for row in pairs if row.op == "+"),
        "pair_minus_n": sum(1 for row in pairs if row.op == "-"),
        "pair_zero_n": sum(1 for row in pairs if row.is_zero()),
        "pair_under_n": sum(1 for row in pairs if row.under_ten()),
        "pair_ten_n": sum(1 for row in pairs if row.exact_ten()),
        "pair_carry_n": sum(1 for row in pairs if row.carries()),
        "pair_borrow_n": sum(1 for row in pairs if row.borrows()),
        "hundred_count": len(hundreds),
        "hundred_train_count": len(hundred_train),
        "hundred_hold_count": len(hundred_hold),
        "hundred_pair_n": sum(1 for row in hundreds if row.kind == "pair"),
        "hundred_step_n": sum(1 for row in hundreds if row.kind == "step"),
        "hundred_exact_n": sum(1 for row in hundreds if row.exact_hundred()),
        "hundred_ten_n": sum(1 for row in hundreds if row.exact_ten()),
        "hundred_carry_n": sum(1 for row in hundreds if row.carries()),
        "hundred_low_n": sum(1 for row in hundreds if row.low()),
        "hundred_high_n": sum(1 for row in hundreds if row.high()),
        "thousand_count": len(thousands),
        "thousand_train_count": len(thou_train),
        "thousand_hold_count": len(thou_hold),
        "thousand_step_n": sum(1 for row in thousands if row.kind == "step"),
        "thousand_wide_n": sum(1 for row in thousands if row.kind == "wide"),
        "thousand_exact_n": sum(1 for row in thousands if row.exact_thousand()),
        "thousand_ten_n": sum(1 for row in thousands if row.exact_ten()),
        "thousand_carry_n": sum(1 for row in thousands if row.carries()),
        "thousand_low_n": sum(1 for row in thousands if row.low()),
        "thousand_high_n": sum(1 for row in thousands if row.high()),
        "op_count": op_counts["plus"] + op_counts["minus"],
        "op_train_count": len(op_train),
        "op_hold_count": len(op_hold),
        "op_plus_n": op_counts["plus"],
        "op_minus_n": op_counts["minus"],
        "op_zero_n": op_counts["zero"],
        "op_borrow_n": op_counts["borrow"],
        "op_carry_n": op_counts["carry"],
        "op_under_n": op_counts["under"],
        "op_ten_n": op_counts["ten"],
        "op_hundred_n": op_counts["hundred"],
        "op_cross_n": op_counts["cross"],
        "op_ones_n": op_counts["ones"],
        "op_wide_n": op_counts["wide"],
        "op_left100_n": op_counts["left100"],
        "op_left900_n": op_counts["left900"],
        "op_right90_n": op_counts["right90"],
        "op_top_n": op_counts["top"],
        "thou_op_count": thou_counts["n"],
        "thou_op_train_count": thou_counts["train"],
        "thou_op_hold_count": thou_counts["hold"],
        "thou_op_plus_n": thou_counts["plus"],
        "thou_op_minus_n": thou_counts["minus"],
        "thou_op_zero_n": thou_counts["zero"],
        "thou_op_borrow_n": thou_counts["borrow"],
        "thou_op_carry_n": thou_counts["carry"],
        "thou_op_under_n": thou_counts["under"],
        "thou_op_ten_n": thou_counts["ten"],
        "thou_op_hundred_n": thou_counts["hundred"],
        "thou_op_thousand_n": thou_counts["thousand"],
        "thou_op_cross_n": thou_counts["cross"],
        "thou_op_ones_n": thou_counts["ones"],
        "thou_op_place_n": thou_counts["place"],
        "thou_op_block_n": thou_counts["block"],
        "thou_op_left1000_n": thou_counts["left1000"],
        "thou_op_left9000_n": thou_counts["left9000"],
        "thou_op_right900_n": thou_counts["right900"],
        "thou_op_top_n": thou_counts["top"],
        "thou_pair_count": thou_pair_counts["n"],
        "thou_pair_train_count": thou_pair_counts["train"],
        "thou_pair_hold_count": thou_pair_counts["hold"],
        "thou_pair_plus_n": thou_pair_counts["plus"],
        "thou_pair_minus_n": thou_pair_counts["minus"],
        "thou_pair_zero_n": thou_pair_counts["zero"],
        "thou_pair_borrow_n": thou_pair_counts["borrow"],
        "thou_pair_carry_n": thou_pair_counts["carry"],
        "thou_pair_under_n": thou_pair_counts["under"],
        "thou_pair_ten_n": thou_pair_counts["ten"],
        "thou_pair_hundred_n": thou_pair_counts["hundred"],
        "thou_pair_thousand_n": thou_pair_counts["thousand"],
        "thou_pair_cross_n": thou_pair_counts["cross"],
        "thou_pair_left1000_n": thou_pair_counts["left1000"],
        "thou_pair_left9000_n": thou_pair_counts["left9000"],
        "thou_pair_right9000_n": thou_pair_counts["right9000"],
        "thou_pair_top_n": thou_pair_counts["top"],
        "ten_thou_count": ten_thou_counts["n"],
        "ten_thou_train_count": ten_thou_counts["train"],
        "ten_thou_hold_count": ten_thou_counts["hold"],
        "ten_thou_ones_n": ten_thou_counts["ones"],
        "ten_thou_place_n": ten_thou_counts["place"],
        "ten_thou_block_n": ten_thou_counts["block"],
        "ten_thou_ten_n": ten_thou_counts["ten"],
        "ten_thou_hundred_n": ten_thou_counts["hundred"],
        "ten_thou_thousand_n": ten_thou_counts["thousand"],
        "ten_thou_carry_n": ten_thou_counts["carry"],
        "ten_thou_low_n": ten_thou_counts["low"],
        "ten_thou_high_n": ten_thou_counts["high"],
        "ten_thou_op_count": ten_thou_op_counts["n"],
        "ten_thou_op_train_count": ten_thou_op_counts["train"],
        "ten_thou_op_hold_count": ten_thou_op_counts["hold"],
        "ten_thou_op_plus_n": ten_thou_op_counts["plus"],
        "ten_thou_op_minus_n": ten_thou_op_counts["minus"],
        "ten_thou_op_zero_n": ten_thou_op_counts["zero"],
        "ten_thou_op_borrow_n": ten_thou_op_counts["borrow"],
        "ten_thou_op_carry_n": ten_thou_op_counts["carry"],
        "ten_thou_op_under_n": ten_thou_op_counts["under"],
        "ten_thou_op_ten_n": ten_thou_op_counts["ten"],
        "ten_thou_op_hundred_n": ten_thou_op_counts["hundred"],
        "ten_thou_op_thousand_n": ten_thou_op_counts["thousand"],
        "ten_thou_op_mark_n": ten_thou_op_counts["mark"],
        "ten_thou_op_cross_n": ten_thou_op_counts["cross"],
        "ten_thou_op_ones_n": ten_thou_op_counts["ones"],
        "ten_thou_op_place_n": ten_thou_op_counts["place"],
        "ten_thou_op_block_n": ten_thou_op_counts["block"],
        "ten_thou_op_right1000_n": ten_thou_op_counts["right1000"],
        "ten_thou_op_low_n": ten_thou_op_counts["low"],
        "ten_thou_op_high_n": ten_thou_op_counts["high"],
        "ten_thou_op_right9000_n": ten_thou_op_counts["right9000"],
        "ten_thou_op_top_n": ten_thou_op_counts["top"],
        "ten_thou_pair_count": ten_thou_pair_counts["n"],
        "ten_thou_pair_train_count": ten_thou_pair_counts["train"],
        "ten_thou_pair_hold_count": ten_thou_pair_counts["hold"],
        "ten_thou_pair_zero_n": ten_thou_pair_counts["zero"],
        "ten_thou_pair_borrow_n": ten_thou_pair_counts["borrow"],
        "ten_thou_pair_noborrow_n": ten_thou_pair_counts["noborrow"],
        "ten_thou_pair_ten_n": ten_thou_pair_counts["ten"],
        "ten_thou_pair_hundred_n": ten_thou_pair_counts["hundred"],
        "ten_thou_pair_ones_n": ten_thou_pair_counts["ones"],
        "ten_thou_pair_place_n": ten_thou_pair_counts["place"],
        "ten_thou_pair_block_n": ten_thou_pair_counts["block"],
        "ten_thou_pair_low_n": ten_thou_pair_counts["low"],
        "ten_thou_pair_high_n": ten_thou_pair_counts["high"],
        "ten_thou_pair_righthigh_n": ten_thou_pair_counts["righthigh"],
        "ten_thou_pair_top_n": ten_thou_pair_counts["top"],
        "hund_thou_count": hund_thou_counts["n"],
        "hund_thou_train_count": hund_thou_counts["train"],
        "hund_thou_hold_count": hund_thou_counts["hold"],
        "hund_thou_ones_n": hund_thou_counts["ones"],
        "hund_thou_place_n": hund_thou_counts["place"],
        "hund_thou_block_n": hund_thou_counts["block"],
        "hund_thou_thou_n": hund_thou_counts["thou"],
        "hund_thou_ten_n": hund_thou_counts["ten"],
        "hund_thou_hundred_n": hund_thou_counts["hundred"],
        "hund_thou_thousand_n": hund_thou_counts["thousand"],
        "hund_thou_mark_n": hund_thou_counts["mark"],
        "hund_thou_carry_n": hund_thou_counts["carry"],
        "hund_thou_low_n": hund_thou_counts["low"],
        "hund_thou_high_n": hund_thou_counts["high"],
    }
    for surface in ("digit", "word"):
        report[f"order_{surface}_train"] = _order_block(lattice, order_train, surface)
        report[f"order_{surface}_hold"] = _order_block(lattice, order_hold, surface)
        report[f"product_{surface}_train"] = _quantity_block(lattice, product_train, surface, "product")
        report[f"product_{surface}_hold"] = _quantity_block(lattice, product_hold, surface, "product")
        report[f"product_{surface}_closed"] = _quantity_block(lattice, products, surface, "product")
        report[f"span_{surface}_train"] = _quantity_block(lattice, span_train, surface, "span")
        report[f"span_{surface}_hold"] = _quantity_block(lattice, span_hold, surface, "span")
        report[f"span_{surface}_closed"] = _quantity_block(lattice, spans, surface, "span")
        report[f"product_{surface}_zero"] = _quantity_block(
            lattice, [row for row in products if row.uses_zero()], surface, "product"
        )
        report[f"span_{surface}_zero"] = _quantity_block(
            lattice, [row for row in spans if row.first_is_zero()], surface, "span"
        )
        report[f"lexeme_{surface}_train"] = _lexeme_block(lattice, lexeme_train, surface)
        report[f"lexeme_{surface}_hold"] = _lexeme_block(lattice, lexeme_hold, surface)
        report[f"lexeme_{surface}_closed"] = _lexeme_block(lattice, lexemes, surface)
        report[f"lexeme_{surface}_ten"] = _lexeme_block(
            lattice, [row for row in lexemes if row.exact_ten()], surface
        )
        report[f"round_{surface}"] = _round_block(lattice, surface)
        for name, block in _score_places(lattice, surface).items():
            report[f"place_{surface}_{name}"] = block
        for name, block in _score_pairs(lattice, surface).items():
            report[f"pair_{surface}_{name}"] = block
        for name, block in _score_hundreds(lattice, surface).items():
            report[f"hundred_{surface}_{name}"] = block
        for name, block in _score_thousands(lattice, surface).items():
            report[f"thousand_{surface}_{name}"] = block
        blocks, rounded = _score_ops(lattice, surface, op_train, op_hold)
        for name, block in blocks.items():
            report[f"op_{surface}_{name}"] = block
        report[f"op_round_{surface}"] = rounded
        blocks, rounded = _score_thou_ops(lattice, surface, thou_counts)
        for name, block in blocks.items():
            report[f"thou_op_{surface}_{name}"] = block
        report[f"thou_op_round_{surface}"] = rounded
        for name, block in _score_thou_pairs(lattice, surface, thou_pair_counts).items():
            report[f"thou_pair_{surface}_{name}"] = block
        for name, block in _score_ten_thousands(lattice, surface, ten_thou_counts).items():
            report[f"ten_thou_{surface}_{name}"] = block
        blocks, rounded = _score_ten_thou_ops(lattice, surface, ten_thou_op_counts)
        for name, block in blocks.items():
            report[f"ten_thou_op_{surface}_{name}"] = block
        report[f"ten_thou_op_round_{surface}"] = rounded
        for name, block in _score_ten_thou_pairs(lattice, surface, ten_thou_pair_counts).items():
            report[f"ten_thou_pair_{surface}_{name}"] = block
        for name, block in _score_hundred_thousands(lattice, surface, hund_thou_counts).items():
            report[f"hund_thou_{surface}_{name}"] = block
    report["demos"] = _demos(lattice)
    report["ok"] = generated_ok(report)
    return report


def _exact(block: dict) -> bool:
    return block["acc"] >= 1.0 and block["n"] > 0


def _lexeme_ok(block: dict) -> bool:
    if not _exact(block):
        return False
    if block["gap"] >= GAUGE_FLOOR or block["margin_min"] <= DROP:
        return False
    if block["ten_n"] > 0 and block["ten_max_abs"] >= DROP:
        return False
    return True


def generated_ok(report: dict) -> bool:
    """Exact order, product, third step, place names, place pairs, hundreds, hundred operands, thousands, thousand operands, thousand pairs, ten-thousand sums, ten-thousand operands, ten-thousand differences, and hundred-thousand sums."""
    for surface in ("digit", "word"):
        order_hold = report[f"order_{surface}_hold"]
        order_train = report[f"order_{surface}_train"]
        if not (_exact(order_hold) and _exact(order_train)):
            return False
        if order_hold["same_max_abs"] >= DROP or order_train["same_max_abs"] >= DROP:
            return False
        if order_hold["diff_min_abs"] <= DROP or order_train["diff_min_abs"] <= DROP:
            return False
        for family in ("product", "span"):
            for split in ("train", "hold", "closed", "zero"):
                block = report[f"{family}_{surface}_{split}"]
                if not _exact(block):
                    return False
                if block["gap"] >= GAUGE_FLOOR:
                    return False
        for split in ("train", "hold", "closed", "ten"):
            if not _lexeme_ok(report[f"lexeme_{surface}_{split}"]):
                return False
        if not _round_ok(report[f"round_{surface}"]):
            return False
        for split in ("train", "hold", "closed", "zero", "borrow", "carry", "under", "ten"):
            if not _lexeme_ok(report[f"place_{surface}_{split}"]):
                return False
            if not _lexeme_ok(report[f"pair_{surface}_{split}"]):
                return False
        for split in ("train", "hold", "closed", "step", "pair", "hundred", "ten", "carry", "low", "high"):
            if not _lexeme_ok(report[f"hundred_{surface}_{split}"]):
                return False
        for split in ("train", "hold", "closed", "step", "wide", "thousand", "ten", "carry", "low", "high"):
            if not _lexeme_ok(report[f"thousand_{surface}_{split}"]):
                return False
        for split in (
            "train",
            "hold",
            "closed",
            "zero",
            "borrow",
            "carry",
            "under",
            "ten",
            "hundred",
            "cross",
            "ones",
            "wide",
        ):
            if not _lexeme_ok(report[f"op_{surface}_{split}"]):
                return False
        if not _round_ok(report[f"op_round_{surface}"]):
            return False
        for split in (
            "train",
            "hold",
            "closed",
            "zero",
            "borrow",
            "carry",
            "under",
            "ten",
            "hundred",
            "thousand",
            "cross",
            "ones",
            "place",
            "block",
        ):
            if not _lexeme_ok(report[f"thou_op_{surface}_{split}"]):
                return False
        if not _round_ok(report[f"thou_op_round_{surface}"]):
            return False
        for split in (
            "train",
            "hold",
            "closed",
            "zero",
            "borrow",
            "carry",
            "under",
            "ten",
            "hundred",
            "thousand",
            "left1000",
            "left9000",
            "right9000",
            "top",
        ):
            if not _lexeme_ok(report[f"thou_pair_{surface}_{split}"]):
                return False
        for split in (
            "train",
            "hold",
            "closed",
            "ones",
            "place",
            "block",
            "ten",
            "hundred",
            "thousand",
            "carry",
            "low",
            "high",
        ):
            if not _lexeme_ok(report[f"ten_thou_{surface}_{split}"]):
                return False
        for split in (
            "train",
            "hold",
            "closed",
            "zero",
            "borrow",
            "carry",
            "under",
            "ten",
            "hundred",
            "thousand",
            "cross",
            "ones",
            "place",
            "block",
            "right1000",
        ):
            if not _lexeme_ok(report[f"ten_thou_op_{surface}_{split}"]):
                return False
        if not _round_ok(report[f"ten_thou_op_round_{surface}"]):
            return False
        for split in (
            "train",
            "hold",
            "closed",
            "zero",
            "borrow",
            "ten",
            "hundred",
            "ones",
            "place",
            "block",
            "low",
            "high",
            "righthigh",
            "top",
        ):
            if not _lexeme_ok(report[f"ten_thou_pair_{surface}_{split}"]):
                return False
        for split in (
            "train",
            "hold",
            "closed",
            "ones",
            "place",
            "block",
            "thou",
            "ten",
            "hundred",
            "thousand",
            "mark",
            "carry",
            "low",
            "high",
        ):
            if not _lexeme_ok(report[f"hund_thou_{surface}_{split}"]):
                return False
    return True


def _round_ok(block: dict) -> bool:
    """The rebuilt name lands on its units gauge, inside the same floor as the additive read."""
    if not _lexeme_ok(block):
        return False
    return block["units_max_abs"] < GAUGE_FLOOR


def _demos(lattice: Lattice) -> list[dict]:
    claims = (
        Claim(2, "+", 3, 4, False),
        Claim(2, "+", 3, 6, False),
        Claim(2, "+", 3, 5, True),
        Claim(5, "-", 5, 1, False),
    )
    products = (
        Product(2, 3, 6),
        Product(4, 2, 8),
        Product(0, 5, 0),
        Product(3, 0, 0),
    )
    spans = (
        Span(2, "+", 3, "-", 1, "+", 1, 5),
        Span(5, "-", 5, "+", 3, "-", 3, 0),
        Span(0, "+", 0, "+", 4, "-", 4, 0),
    )
    lexemes = (
        Lexeme("sum", 9, 1, 10),
        Lexeme("sum", 5, 5, 10),
        Lexeme("sum", 9, 9, 18),
        Lexeme("product", 2, 5, 10),
        Lexeme("product", 4, 5, 20),
        Lexeme("product", 6, 7, 42),
        Lexeme("product", 5, 8, 40),
        Lexeme("product", 9, 9, 81),
    )
    places = (
        PlaceStep(24, "+", 3, 27),
        PlaceStep(24, "-", 3, 21),
        PlaceStep(24, "-", 7, 17),
        PlaceStep(20, "-", 9, 11),
        PlaceStep(10, "-", 9, 1),
        PlaceStep(24, "+", 0, 24),
        PlaceStep(15, "+", 5, 20),
        PlaceStep(81, "+", 9, 90),
        PlaceStep(90, "+", 9, 99),
        PlaceStep(99, "-", 0, 99),
    )
    pairs = (
        PlacePair(24, "+", 17, 41),
        PlacePair(24, "-", 17, 7),
        PlacePair(24, "-", 24, 0),
        PlacePair(20, "+", 10, 30),
        PlacePair(50, "+", 49, 99),
        PlacePair(15, "+", 15, 30),
        PlacePair(30, "-", 11, 19),
        PlacePair(99, "-", 90, 9),
        PlacePair(10, "+", 10, 20),
        PlacePair(18, "-", 11, 7),
    )
    hundreds = (
        HundredSum("pair", 50, 50, 100),
        HundredSum("step", 99, 1, 100),
        HundredSum("pair", 24, 76, 100),
        HundredSum("pair", 10, 90, 100),
        HundredSum("pair", 60, 41, 101),
        HundredSum("step", 99, 9, 108),
        HundredSum("pair", 15, 95, 110),
        HundredSum("pair", 42, 80, 122),
        HundredSum("pair", 99, 99, 198),
        HundredSum("pair", 25, 75, 100),
    )
    operands = (
        HundredOp(100, "+", 24, 124),
        HundredOp(100, "-", 24, 76),
        HundredOp(100, "+", 3, 103),
        HundredOp(100, "-", 99, 1),
        HundredOp(124, "+", 17, 141),
        HundredOp(150, "+", 50, 200),
        HundredOp(198, "+", 99, 297),
        HundredOp(200, "-", 99, 101),
        HundredOp(999, "-", 9, 990),
        HundredOp(900, "+", 99, 999),
    )
    thousands = (
        ThousandSum("step", 999, 1, 1000),
        ThousandSum("wide", 901, 99, 1000),
        ThousandSum("wide", 950, 50, 1000),
        ThousandSum("wide", 990, 10, 1000),
        ThousandSum("step", 991, 9, 1000),
        ThousandSum("step", 999, 9, 1008),
        ThousandSum("wide", 960, 41, 1001),
        ThousandSum("wide", 975, 50, 1025),
        ThousandSum("wide", 990, 20, 1010),
        ThousandSum("wide", 999, 99, 1098),
    )
    thou_ops = (
        ThousandOp(1000, "+", 24, 1024),
        ThousandOp(1000, "-", 24, 976),
        ThousandOp(1000, "+", 3, 1003),
        ThousandOp(1000, "-", 999, 1),
        ThousandOp(1024, "+", 17, 1041),
        ThousandOp(1500, "+", 500, 2000),
        ThousandOp(1998, "+", 99, 2097),
        ThousandOp(2000, "-", 999, 1001),
        ThousandOp(9999, "-", 9, 9990),
        ThousandOp(9000, "+", 999, 9999),
    )
    thou_pairs = (
        ThousandPair(1000, "+", 1000, 2000),
        ThousandPair(1000, "-", 1000, 0),
        ThousandPair(2000, "-", 1000, 1000),
        ThousandPair(1500, "+", 2500, 4000),
        ThousandPair(2500, "-", 1500, 1000),
        ThousandPair(1100, "-", 1000, 100),
        ThousandPair(4000, "+", 5000, 9000),
        ThousandPair(9999, "-", 1000, 8999),
        ThousandPair(1024, "+", 1100, 2124),
        ThousandPair(1999, "+", 1001, 3000),
    )
    ten_thousands = (
        TenThousandSum("digit", 9999, 1, 10000),
        TenThousandSum("digit", 9999, 9, 10008),
        TenThousandSum("place", 9901, 99, 10000),
        TenThousandSum("block", 9500, 500, 10000),
        TenThousandSum("block", 9001, 999, 10000),
        TenThousandSum("place", 9999, 99, 10098),
        TenThousandSum("block", 9999, 999, 10998),
        TenThousandSum("block", 9900, 200, 10100),
        TenThousandSum("place", 9950, 50, 10000),
        TenThousandSum("block", 9800, 250, 10050),
    )
    ten_thou_ops = (
        TenThousandOp(10000, "+", 0, 10000),
        TenThousandOp(10000, "+", 1, 10001),
        TenThousandOp(10000, "+", 24, 10024),
        TenThousandOp(10000, "+", 999, 10999),
        TenThousandOp(10000, "-", 0, 10000),
        TenThousandOp(10000, "-", 1, 9999),
        TenThousandOp(10000, "-", 9999, 1),
        TenThousandOp(10024, "-", 24, 10000),
        TenThousandOp(10050, "+", 50, 10100),
        TenThousandOp(10250, "-", 250, 10000),
        TenThousandOp(10999, "+", 0, 10999),
        TenThousandOp(10999, "-", 9, 10990),
        TenThousandOp(10999, "-", 999, 10000),
    )
    ten_thou_pairs = (
        TenThousandPair(10000, 10000, 0),
        TenThousandPair(10001, 10000, 1),
        TenThousandPair(10024, 10000, 24),
        TenThousandPair(10050, 10000, 50),
        TenThousandPair(10100, 10050, 50),
        TenThousandPair(10500, 10000, 500),
        TenThousandPair(10999, 10000, 999),
        TenThousandPair(10999, 10990, 9),
        TenThousandPair(10999, 10999, 0),
    )
    hund_thousands = (
        HundredThousandSum("digit", 99999, 1, 100000),
        HundredThousandSum("thou", 90001, 9999, 100000),
        HundredThousandSum("thou", 99000, 2000, 101000),
        HundredThousandSum("block", 99500, 500, 100000),
        HundredThousandSum("thou", 98000, 2500, 100500),
        HundredThousandSum("thou", 99999, 9999, 109998),
    )
    demos = []
    for claim in claims:
        demos.append(
            {
                "family": "order",
                "digit": order_digit(claim),
                "word": order_word(claim),
                "want": order_label(claim),
                "digit_got": lattice.predict_order(claim, "digit"),
                "word_got": lattice.predict_order(claim, "word"),
            }
        )
    for row in products:
        demos.append(
            {
                "family": "product",
                "digit": f"{row.digit_prompt()}{row.digit_answer()}",
                "word": row.word_prompt(),
                "want": row.word_answer(),
                "digit_got": lattice.predict_product(row, "digit"),
                "word_got": lattice.predict_product(row, "word"),
            }
        )
    for span in spans:
        demos.append(
            {
                "family": "span",
                "digit": f"{span.digit_prompt()}{span.digit_answer()}",
                "word": span.word_prompt(),
                "want": span.word_answer(),
                "digit_got": lattice.predict_span(span, "digit"),
                "word_got": lattice.predict_span(span, "word"),
            }
        )
    for row in lexemes:
        demos.append(
            {
                "family": "lexeme",
                "digit": f"{row.digit_prompt()}{row.digit_answer()}",
                "word": row.word_prompt(),
                "want": row.word_answer(),
                "digit_want": row.digit_answer(),
                "digit_got": lattice.predict_lexeme(row, "digit"),
                "word_got": lattice.predict_lexeme(row, "word"),
            }
        )
    for row in places:
        demos.append(
            {
                "family": "place",
                "digit": f"{row.digit_prompt()}{row.digit_answer()}",
                "word": row.word_prompt(),
                "want": row.word_answer(),
                "digit_want": row.digit_answer(),
                "digit_got": lattice.predict_place_step(row, "digit"),
                "word_got": lattice.predict_place_step(row, "word"),
            }
        )
    for row in pairs:
        demos.append(
            {
                "family": "pair",
                "digit": f"{row.digit_prompt()}{row.digit_answer()}",
                "word": row.word_prompt(),
                "want": row.word_answer(),
                "digit_want": row.digit_answer(),
                "digit_got": lattice.predict_place_pair(row, "digit"),
                "word_got": lattice.predict_place_pair(row, "word"),
            }
        )
    for row in hundreds:
        demos.append(
            {
                "family": "hundred",
                "digit": f"{row.digit_prompt()}{row.digit_answer()}",
                "word": row.word_prompt(),
                "want": row.word_answer(),
                "digit_want": row.digit_answer(),
                "digit_got": lattice.predict_hundred(row, "digit"),
                "word_got": lattice.predict_hundred(row, "word"),
            }
        )
    for row in operands:
        demos.append(
            {
                "family": "operand",
                "digit": f"{row.digit_prompt()}{row.digit_answer()}",
                "word": row.word_prompt(),
                "want": row.word_answer(),
                "digit_want": row.digit_answer(),
                "digit_got": lattice.predict_hundred_op(row, "digit"),
                "word_got": lattice.predict_hundred_op(row, "word"),
            }
        )
    for row in thousands:
        demos.append(
            {
                "family": "thousand",
                "digit": f"{row.digit_prompt()}{row.digit_answer()}",
                "word": row.word_prompt(),
                "want": row.word_answer(),
                "digit_want": row.digit_answer(),
                "digit_got": lattice.predict_thousand(row, "digit"),
                "word_got": lattice.predict_thousand(row, "word"),
            }
        )
    for row in thou_ops:
        demos.append(
            {
                "family": "thouop",
                "digit": f"{row.digit_prompt()}{row.digit_answer()}",
                "word": row.word_prompt(),
                "want": row.word_answer(),
                "digit_want": row.digit_answer(),
                "digit_got": lattice.predict_thousand_op(row, "digit"),
                "word_got": lattice.predict_thousand_op(row, "word"),
            }
        )
    for row in thou_pairs:
        demos.append(
            {
                "family": "thoupair",
                "digit": f"{row.digit_prompt()}{row.digit_answer()}",
                "word": row.word_prompt(),
                "want": row.word_answer(),
                "digit_want": row.digit_answer(),
                "digit_got": lattice.predict_thousand_pair(row, "digit"),
                "word_got": lattice.predict_thousand_pair(row, "word"),
            }
        )
    for row in ten_thousands:
        demos.append(
            {
                "family": "tenthous",
                "digit": f"{row.digit_prompt()}{row.digit_answer()}",
                "word": row.word_prompt(),
                "want": row.word_answer(),
                "digit_want": row.digit_answer(),
                "digit_got": lattice.predict_ten_thousand(row, "digit"),
                "word_got": lattice.predict_ten_thousand(row, "word"),
            }
        )
    for row in ten_thou_ops:
        demos.append(
            {
                "family": "tenop",
                "digit": f"{row.digit_prompt()}{row.digit_answer()}",
                "word": row.word_prompt(),
                "want": row.word_answer(),
                "digit_want": row.digit_answer(),
                "digit_got": lattice.predict_ten_thousand_op(row, "digit"),
                "word_got": lattice.predict_ten_thousand_op(row, "word"),
            }
        )
    for row in ten_thou_pairs:
        demos.append(
            {
                "family": "tenpair",
                "digit": f"{row.digit_prompt()}{row.digit_answer()}",
                "word": row.word_prompt(),
                "want": row.word_answer(),
                "digit_want": row.digit_answer(),
                "digit_got": lattice.predict_ten_thousand_pair(row, "digit"),
                "word_got": lattice.predict_ten_thousand_pair(row, "word"),
            }
        )
    for row in hund_thousands:
        demos.append(
            {
                "family": "hundthou",
                "digit": f"{row.digit_prompt()}{row.digit_answer()}",
                "word": row.word_prompt(),
                "want": row.word_answer(),
                "digit_want": row.digit_answer(),
                "digit_got": lattice.predict_hundred_thousand(row, "digit"),
                "word_got": lattice.predict_hundred_thousand(row, "word"),
            }
        )
    return demos
