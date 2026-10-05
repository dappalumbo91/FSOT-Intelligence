"""Closed arithmetic lexicon and the held-out split.

Digit surface: 2+3=5
Word surface: what is two plus three -> five
A triple held out of digit training is also held out of word training.

Equality is a separate claim set: a±b against a stated digit.
Its hold hash does not use a multiple of 5 on b, so 0+n is not forced into the holdout.

A chain is two closed steps. The second step reads the first consensus vessel.
Order, product, and a third step are generated from that same vessel.
Names past nine are place-value spellings of sums and products that leave
a single digit. The spellings are priors, the same status as the digit words.
A two-digit name can re-enter as an operand: the ten-step is folded back
up, then one digit is added or subtracted, and the same spelling reads the result.
Two such names can share one expression. The result stays inside the same table.
A sum that leaves that table sheds the ten-step counted ten times. That
hundred place names the result. The spelling is a prior.
A hundred name re-enters as an operand. The hundred-step folds back up,
the lower place is the one already in the table, and a name in 0..99
is added or subtracted. The result stays inside 0..999.
A sum that leaves 0..999 sheds the hundred-step counted ten times. That
thousand place names the result. The spelling is a prior.
A thousand name re-enters as an operand. The thousand-step folds back
up, the lower place is the one already in 0..999, and a name in 0..999
is added or subtracted. The result stays inside 0..9999.
Two such thousand names can share one expression. The result stays
inside that same table.
A sum that leaves 0..9999 sheds the thousand-step counted ten times.
That ten-thousand place names the result. The spelling is a prior.
A ten-thousand name re-enters as an operand. The ten-thousand step
folds back up, the lower place is the one already in 0..999, and a
name in 0..9999 is added or subtracted. The result stays inside 0..10999.
Two ten-thousand names share one expression. The result that stays inside
the spelled table is their difference, and that difference lands in 0..999.
A sum that leaves 0..99999 sheds the ten-thousand step counted ten times.
That hundred-thousand place names the result. The spelling is a prior.
"""

from __future__ import annotations

from dataclasses import dataclass

DIGITS = [str(i) for i in range(10)]
# Largest count whose product with another digit still fits in one digit.
REPEATED_LIMIT = 9
WORD_OF = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
}
OP_WORD = {"+": "plus", "-": "minus", "=": "equals", "*": "times"}
FUNCTION_WORDS = ("what", "is")
# Written forms of the places past nine. The place count comes from the vessel.
RADIX = len(DIGITS)
# One bound for the census, the thousand operand, and two thousand names.
THOUSAND_SPAN = RADIX ** 3
THOUSAND_CAP = RADIX ** 4
# Ten-thousand answers stay inside 10000..10999. The spelling table continues.
TEN_THOUSAND_CAP = THOUSAND_CAP + THOUSAND_SPAN
# Spellings through one hundred nine thousand nine hundred ninety-nine.
HUNDRED_THOUSAND = RADIX ** 5
HUNDRED_THOUSAND_CAP = HUNDRED_THOUSAND + THOUSAND_CAP
TEEN_WORD = {
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
}
TENS_WORD = {
    2: "twenty",
    3: "thirty",
    4: "forty",
    5: "fifty",
    6: "sixty",
    7: "seventy",
    8: "eighty",
    9: "ninety",
}
HUNDRED_WORD = "hundred"
THOUSAND_WORD = "thousand"


def number_name(n: int) -> str:
    """English spelling of a place-value count. Compounds use a hyphen."""
    span = RADIX * RADIX
    thousand = span * RADIX
    if n < 0 or n >= HUNDRED_THOUSAND_CAP:
        return "?"
    if n < RADIX:
        return WORD_OF[n]
    if n < 2 * RADIX:
        return TEEN_WORD[n]
    if n < span:
        tens, units = divmod(n, RADIX)
        stem = TENS_WORD[tens]
        if units == 0:
            return stem
        return f"{stem}-{WORD_OF[units]}"
    if n < thousand:
        hundreds, rest = divmod(n, span)
        stem = f"{WORD_OF[hundreds]} {HUNDRED_WORD}"
        if rest == 0:
            return stem
        return f"{stem} {number_name(rest)}"
    thousands, rest = divmod(n, thousand)
    stem = f"{number_name(thousands)} {THOUSAND_WORD}"
    if rest == 0:
        return stem
    return f"{stem} {number_name(rest)}"


def decimal_name(n: int) -> str:
    return str(n)


@dataclass(frozen=True)
class Triple:
    a: int
    op: str
    b: int
    c: int

    def digit_prompt(self) -> str:
        return f"{self.a}{self.op}{self.b}="

    def digit_answer(self) -> str:
        return str(self.c)

    def word_prompt(self) -> str:
        return f"what is {WORD_OF[self.a]} {OP_WORD[self.op]} {WORD_OF[self.b]}"

    def word_answer(self) -> str:
        return WORD_OF[self.c]


def all_triples() -> list[Triple]:
    rows: list[Triple] = []
    for a in range(10):
        for b in range(10):
            if a + b <= 9:
                rows.append(Triple(a, "+", b, a + b))
            if a - b >= 0:
                rows.append(Triple(a, "-", b, a - b))
    return rows


def _hold(row: Triple) -> bool:
    """Deterministic fifth of the closed set. No RNG."""
    op_bit = 0 if row.op == "+" else 1
    return (row.a * 3 + row.b * 5 + op_bit) % 5 == 0


def _covers_digits(rows: list[Triple]) -> set[int]:
    seen: set[int] = set()
    for row in rows:
        seen.add(row.a)
        seen.add(row.b)
        seen.add(row.c)
    return seen


def split_triples() -> tuple[list[Triple], list[Triple]]:
    """Train keeps every digit 0-9 at least once. Held-out triples never return."""
    rows = all_triples()
    train = [r for r in rows if not _hold(r)]
    hold = [r for r in rows if _hold(r)]
    missing = set(range(10)) - _covers_digits(train)
    if missing:
        kept: list[Triple] = []
        for row in hold:
            if missing and (row.a in missing or row.b in missing or row.c in missing):
                train.append(row)
                missing -= {row.a, row.b, row.c}
            else:
                kept.append(row)
        hold = kept
    train.sort(key=lambda r: (r.op, r.a, r.b))
    hold.sort(key=lambda r: (r.op, r.a, r.b))
    return train, hold


@dataclass(frozen=True)
class Claim:
    """A closed equation judged true or false. claimed is the stated digit."""

    a: int
    op: str
    b: int
    claimed: int
    holds: bool

    def digit_prompt(self) -> str:
        return f"{self.a}{self.op}{self.b}={self.claimed}"

    def word_prompt(self) -> str:
        return (
            f"what is {WORD_OF[self.a]} {OP_WORD[self.op]} {WORD_OF[self.b]} "
            f"equals {WORD_OF[self.claimed]}"
        )

    def answer(self) -> str:
        return "yes" if self.holds else "no"


def all_claims() -> list[Claim]:
    """Every closed triple paired with each digit 0-9 as the stated result."""
    claims: list[Claim] = []
    for row in all_triples():
        for claimed in range(10):
            claims.append(Claim(row.a, row.op, row.b, claimed, claimed == row.c))
    return claims


def _hold_claim(claim: Claim) -> bool:
    """Deterministic fifth of the claims. No RNG.

    The coefficient on b is 4. A multiple of 5 would vanish mod 5 and park
    every 0+n claim in the holdout.
    """
    op_bit = 0 if claim.op == "+" else 1
    return (claim.a * 3 + claim.b * 4 + claim.claimed * 7 + op_bit) % 5 == 0


def split_claims() -> tuple[list[Claim], list[Claim]]:
    rows = all_claims()
    train = [row for row in rows if not _hold_claim(row)]
    hold = [row for row in rows if _hold_claim(row)]
    train.sort(key=lambda row: (row.op, row.a, row.b, row.claimed))
    hold.sort(key=lambda row: (row.op, row.a, row.b, row.claimed))
    return train, hold


@dataclass(frozen=True)
class Chain:
    """Two closed steps. The second operand reads the first vessel, not a stored answer."""

    a: int
    op1: str
    b: int
    op2: str
    c: int
    result: int

    def digit_prompt(self) -> str:
        return f"{self.a}{self.op1}{self.b}{self.op2}{self.c}="

    def digit_answer(self) -> str:
        return str(self.result)

    def word_prompt(self) -> str:
        return (
            f"what is {WORD_OF[self.a]} {OP_WORD[self.op1]} {WORD_OF[self.b]} "
            f"{OP_WORD[self.op2]} {WORD_OF[self.c]}"
        )

    def word_answer(self) -> str:
        return WORD_OF[self.result]

    def intermediate_is_zero(self) -> bool:
        if self.op1 == "+":
            return self.a + self.b == 0
        return self.a - self.b == 0


def all_chains() -> list[Chain]:
    """Every closed first step, then every closed second step from that result."""
    chains: list[Chain] = []
    for row in all_triples():
        mid = row.c
        for right in range(10):
            if mid + right <= 9:
                chains.append(Chain(row.a, row.op, row.b, "+", right, mid + right))
            if mid - right >= 0:
                chains.append(Chain(row.a, row.op, row.b, "-", right, mid - right))
    return chains


def _hold_chain(chain: Chain) -> bool:
    """Deterministic fifth. No coefficient is a multiple of 5, so no operand vanishes."""
    op1 = 0 if chain.op1 == "+" else 1
    op2 = 0 if chain.op2 == "+" else 1
    return (chain.a * 3 + chain.b * 4 + chain.c * 6 + op1 + 3 * op2) % 5 == 0


def split_chains() -> tuple[list[Chain], list[Chain]]:
    rows = all_chains()
    train = [row for row in rows if not _hold_chain(row)]
    hold = [row for row in rows if _hold_chain(row)]
    train.sort(key=lambda row: (row.op1, row.op2, row.a, row.b, row.c))
    hold.sort(key=lambda row: (row.op1, row.op2, row.a, row.b, row.c))
    return train, hold


def order_label(claim: Claim) -> str:
    """More when the closed result exceeds the stated digit."""
    total = claim.a + claim.b if claim.op == "+" else claim.a - claim.b
    if total > claim.claimed:
        return "more"
    if total < claim.claimed:
        return "less"
    return "same"


def order_digit(claim: Claim) -> str:
    return f"{claim.a}{claim.op}{claim.b}?{claim.claimed}"


def order_word(claim: Claim) -> str:
    return (
        f"what is {WORD_OF[claim.a]} {OP_WORD[claim.op]} {WORD_OF[claim.b]} "
        f"beside {WORD_OF[claim.claimed]}"
    )


@dataclass(frozen=True)
class Product:
    """count times addend, kept only when the product is a single digit."""

    count: int
    addend: int
    result: int

    def digit_prompt(self) -> str:
        return f"{self.count}*{self.addend}="

    def digit_answer(self) -> str:
        return str(self.result)

    def word_prompt(self) -> str:
        return f"what is {WORD_OF[self.count]} times {WORD_OF[self.addend]}"

    def word_answer(self) -> str:
        return WORD_OF[self.result]

    def uses_zero(self) -> bool:
        return self.count == 0 or self.addend == 0


def all_products() -> list[Product]:
    rows: list[Product] = []
    for count in range(10):
        for addend in range(10):
            value = count * addend
            if value <= 9:
                rows.append(Product(count, addend, value))
    return rows


def _hold_product(row: Product) -> bool:
    """Fifth of the products. 4 is not a multiple of 5, so ×0 is not all held out."""
    return (row.count * 3 + row.addend * 4) % 5 == 0


def split_products() -> tuple[list[Product], list[Product]]:
    rows = all_products()
    train = [row for row in rows if not _hold_product(row)]
    hold = [row for row in rows if _hold_product(row)]
    train.sort(key=lambda row: (row.count, row.addend))
    hold.sort(key=lambda row: (row.count, row.addend))
    return train, hold


@dataclass(frozen=True)
class Span:
    """Three closed steps. Each step reads the running consensus vessel."""

    a: int
    op1: str
    b: int
    op2: str
    c: int
    op3: str
    d: int
    result: int

    def digit_prompt(self) -> str:
        return f"{self.a}{self.op1}{self.b}{self.op2}{self.c}{self.op3}{self.d}="

    def digit_answer(self) -> str:
        return str(self.result)

    def word_prompt(self) -> str:
        return (
            f"what is {WORD_OF[self.a]} {OP_WORD[self.op1]} {WORD_OF[self.b]} "
            f"{OP_WORD[self.op2]} {WORD_OF[self.c]} {OP_WORD[self.op3]} {WORD_OF[self.d]}"
        )

    def word_answer(self) -> str:
        return WORD_OF[self.result]

    def first_is_zero(self) -> bool:
        if self.op1 == "+":
            return self.a + self.b == 0
        return self.a - self.b == 0


def all_spans() -> list[Span]:
    """Every two-step chain extended by one more closed step."""
    spans: list[Span] = []
    for chain in all_chains():
        mid = chain.result
        for right in range(10):
            if mid + right <= 9:
                spans.append(
                    Span(
                        chain.a, chain.op1, chain.b, chain.op2, chain.c,
                        "+", right, mid + right,
                    )
                )
            if mid - right >= 0:
                spans.append(
                    Span(
                        chain.a, chain.op1, chain.b, chain.op2, chain.c,
                        "-", right, mid - right,
                    )
                )
    return spans


def _hold_span(span: Span) -> bool:
    op1 = 0 if span.op1 == "+" else 1
    op2 = 0 if span.op2 == "+" else 1
    op3 = 0 if span.op3 == "+" else 1
    return (
        span.a * 3 + span.b * 4 + span.c * 6 + span.d * 7 + op1 + 3 * op2 + 4 * op3
    ) % 5 == 0


def split_spans() -> tuple[list[Span], list[Span]]:
    rows = all_spans()
    train = [row for row in rows if not _hold_span(row)]
    hold = [row for row in rows if _hold_span(row)]
    train.sort(key=lambda row: (row.op1, row.op2, row.op3, row.a, row.b, row.c, row.d))
    hold.sort(key=lambda row: (row.op1, row.op2, row.op3, row.a, row.b, row.c, row.d))
    return train, hold


@dataclass(frozen=True)
class Lexeme:
    """A sum or product whose result needs a name past the ten digit words."""

    kind: str
    left: int
    right: int
    result: int

    def digit_prompt(self) -> str:
        op = "+" if self.kind == "sum" else "*"
        return f"{self.left}{op}{self.right}="

    def digit_answer(self) -> str:
        return decimal_name(self.result)

    def word_prompt(self) -> str:
        op = "plus" if self.kind == "sum" else OP_WORD["*"]
        return f"what is {WORD_OF[self.left]} {op} {WORD_OF[self.right]}"

    def word_answer(self) -> str:
        return number_name(self.result)

    def exact_ten(self) -> bool:
        return self.result % RADIX == 0


def all_lexemes() -> list[Lexeme]:
    """Every digit sum or product that lands past nine. Nothing below ten."""
    rows: list[Lexeme] = []
    for left in range(RADIX):
        for right in range(RADIX):
            if left + right >= RADIX:
                rows.append(Lexeme("sum", left, right, left + right))
    for left in range(RADIX):
        for right in range(RADIX):
            value = left * right
            if value >= RADIX:
                rows.append(Lexeme("product", left, right, value))
    return rows


def _hold_lexeme(row: Lexeme) -> bool:
    """Deterministic fifth. No coefficient is a multiple of 5."""
    kind = 0 if row.kind == "sum" else 1
    return (row.left * 3 + row.right * 4 + 6 * kind) % 5 == 0


def split_lexemes() -> tuple[list[Lexeme], list[Lexeme]]:
    rows = all_lexemes()
    train = [row for row in rows if not _hold_lexeme(row)]
    hold = [row for row in rows if _hold_lexeme(row)]
    train.sort(key=lambda row: (row.kind, row.left, row.right))
    hold.sort(key=lambda row: (row.kind, row.left, row.right))
    return train, hold


@dataclass(frozen=True)
class PlaceStep:
    """A two-digit name plus or minus one digit. The result stays inside 0..99."""

    place: int
    op: str
    digit: int
    result: int

    def digit_prompt(self) -> str:
        return f"{decimal_name(self.place)}{self.op}{self.digit}="

    def digit_answer(self) -> str:
        return decimal_name(self.result)

    def word_prompt(self) -> str:
        return (
            f"what is {number_name(self.place)} {OP_WORD[self.op]} {WORD_OF[self.digit]}"
        )

    def word_answer(self) -> str:
        return number_name(self.result)

    def uses_zero(self) -> bool:
        return self.digit == 0

    def borrows(self) -> bool:
        return self.op == "-" and (self.place % RADIX) < self.digit

    def carries(self) -> bool:
        return self.op == "+" and (self.place % RADIX) + self.digit >= RADIX

    def under_ten(self) -> bool:
        return self.result < RADIX

    def exact_ten(self) -> bool:
        return self.result % RADIX == 0


def all_place_steps() -> list[PlaceStep]:
    """Every place 10..99 combined with a digit when the result still fits in two digits."""
    rows: list[PlaceStep] = []
    limit = RADIX * RADIX
    for place in range(RADIX, limit):
        for digit in range(RADIX):
            if place + digit < limit:
                rows.append(PlaceStep(place, "+", digit, place + digit))
            rows.append(PlaceStep(place, "-", digit, place - digit))
    return rows


def _hold_place(row: PlaceStep) -> bool:
    """Deterministic fifth. No coefficient is a multiple of 5."""
    op_bit = 0 if row.op == "+" else 1
    return (row.place * 3 + row.digit * 4 + 6 * op_bit) % 5 == 0


def split_place_steps() -> tuple[list[PlaceStep], list[PlaceStep]]:
    rows = all_place_steps()
    train = [row for row in rows if not _hold_place(row)]
    hold = [row for row in rows if _hold_place(row)]
    train.sort(key=lambda row: (row.op, row.place, row.digit))
    hold.sort(key=lambda row: (row.op, row.place, row.digit))
    return train, hold


@dataclass(frozen=True)
class PlacePair:
    """Two two-digit names added or subtracted. The result stays inside 0..99."""

    left: int
    op: str
    right: int
    result: int

    def digit_prompt(self) -> str:
        return f"{decimal_name(self.left)}{self.op}{decimal_name(self.right)}="

    def digit_answer(self) -> str:
        return decimal_name(self.result)

    def word_prompt(self) -> str:
        return (
            f"what is {number_name(self.left)} {OP_WORD[self.op]} {number_name(self.right)}"
        )

    def word_answer(self) -> str:
        return number_name(self.result)

    def is_zero(self) -> bool:
        return self.result == 0

    def borrows(self) -> bool:
        return self.op == "-" and (self.left % RADIX) < (self.right % RADIX)

    def carries(self) -> bool:
        return self.op == "+" and (self.left % RADIX) + (self.right % RADIX) >= RADIX

    def under_ten(self) -> bool:
        return self.result < RADIX

    def exact_ten(self) -> bool:
        return self.result % RADIX == 0


def all_place_pairs() -> list[PlacePair]:
    """Every pair of places 10..99 whose sum or difference still fits in 0..99."""
    rows: list[PlacePair] = []
    limit = RADIX * RADIX
    for left in range(RADIX, limit):
        for right in range(RADIX, limit):
            if left + right < limit:
                rows.append(PlacePair(left, "+", right, left + right))
            if left >= right:
                rows.append(PlacePair(left, "-", right, left - right))
    return rows


def _hold_pair(row: PlacePair) -> bool:
    """Deterministic fifth. No coefficient is a multiple of 5."""
    op_bit = 0 if row.op == "+" else 1
    return (row.left * 3 + row.right * 4 + 6 * op_bit) % 5 == 0


def split_place_pairs() -> tuple[list[PlacePair], list[PlacePair]]:
    rows = all_place_pairs()
    train = [row for row in rows if not _hold_pair(row)]
    hold = [row for row in rows if _hold_pair(row)]
    train.sort(key=lambda row: (row.op, row.left, row.right))
    hold.sort(key=lambda row: (row.op, row.left, row.right))
    return train, hold


@dataclass(frozen=True)
class HundredSum:
    """An addition whose result leaves 0..99 and stays below 1000."""

    kind: str
    left: int
    right: int
    result: int

    def digit_prompt(self) -> str:
        return f"{decimal_name(self.left)}+{decimal_name(self.right)}="

    def digit_answer(self) -> str:
        return decimal_name(self.result)

    def word_prompt(self) -> str:
        if self.kind == "step":
            right = WORD_OF[self.right]
        else:
            right = number_name(self.right)
        return f"what is {number_name(self.left)} plus {right}"

    def word_answer(self) -> str:
        return number_name(self.result)

    def carries(self) -> bool:
        return (self.left % RADIX) + (self.right % RADIX) >= RADIX

    def exact_ten(self) -> bool:
        return self.result % RADIX == 0

    def exact_hundred(self) -> bool:
        return self.result % (RADIX * RADIX) == 0

    def low(self) -> bool:
        return self.result < RADIX * RADIX + RADIX

    def high(self) -> bool:
        return self.result >= RADIX * (2 * RADIX - 1)


def all_hundred_sums() -> list[HundredSum]:
    """Every place-plus-digit and place-plus-place whose sum is 100..999."""
    rows: list[HundredSum] = []
    limit = RADIX * RADIX
    thousand = limit * RADIX
    for place in range(RADIX, limit):
        for digit in range(RADIX):
            result = place + digit
            if limit <= result < thousand:
                rows.append(HundredSum("step", place, digit, result))
    for left in range(RADIX, limit):
        for right in range(RADIX, limit):
            result = left + right
            if limit <= result < thousand:
                rows.append(HundredSum("pair", left, right, result))
    return rows


def _hold_hundred(row: HundredSum) -> bool:
    """Deterministic fifth. No coefficient is a multiple of 5."""
    kind = 1 if row.kind == "step" else 0
    return (row.left * 3 + row.right * 4 + 6 * kind) % 5 == 0


def split_hundred_sums() -> tuple[list[HundredSum], list[HundredSum]]:
    rows = all_hundred_sums()
    train = [row for row in rows if not _hold_hundred(row)]
    hold = [row for row in rows if _hold_hundred(row)]
    train.sort(key=lambda row: (row.kind, row.left, row.right))
    hold.sort(key=lambda row: (row.kind, row.left, row.right))
    return train, hold


@dataclass(frozen=True)
class HundredOp:
    """A name in 100..999, plus or minus a name in 0..99, result inside 0..999."""

    left: int
    op: str
    right: int
    result: int

    def digit_prompt(self) -> str:
        return f"{decimal_name(self.left)}{self.op}{decimal_name(self.right)}="

    def digit_answer(self) -> str:
        return decimal_name(self.result)

    def word_prompt(self) -> str:
        right = WORD_OF[self.right] if self.right < RADIX else number_name(self.right)
        return f"what is {number_name(self.left)} {OP_WORD[self.op]} {right}"

    def word_answer(self) -> str:
        return number_name(self.result)

    def uses_zero(self) -> bool:
        return self.right == 0

    def borrows(self) -> bool:
        return self.op == "-" and (self.left % RADIX) < (self.right % RADIX)

    def carries(self) -> bool:
        return self.op == "+" and (self.left % RADIX) + (self.right % RADIX) >= RADIX

    def under_hundred(self) -> bool:
        return self.result < RADIX * RADIX

    def exact_ten(self) -> bool:
        return self.result % RADIX == 0

    def exact_hundred(self) -> bool:
        return self.result % (RADIX * RADIX) == 0

    def crosses(self) -> bool:
        span = RADIX * RADIX
        return self.left // span != self.result // span

    def ones(self) -> bool:
        return self.right < RADIX

    def wide(self) -> bool:
        return self.right >= RADIX


def all_hundred_ops() -> list[HundredOp]:
    """Every name 100..999 plus or minus a name 0..99 whose result stays in 0..999."""
    rows: list[HundredOp] = []
    span = RADIX * RADIX
    thousand = span * RADIX
    for left in range(span, thousand):
        for right in range(span):
            if left + right < thousand:
                rows.append(HundredOp(left, "+", right, left + right))
            rows.append(HundredOp(left, "-", right, left - right))
    return rows


def _hold_op(row: HundredOp) -> bool:
    """Deterministic fifth. No coefficient is a multiple of 5."""
    op_bit = 0 if row.op == "+" else 1
    return (row.left * 3 + row.right * 4 + 6 * op_bit) % 5 == 0


def split_hundred_ops() -> tuple[list[HundredOp], list[HundredOp]]:
    rows = all_hundred_ops()
    train = [row for row in rows if not _hold_op(row)]
    hold = [row for row in rows if _hold_op(row)]
    train.sort(key=lambda row: (row.op, row.left, row.right))
    hold.sort(key=lambda row: (row.op, row.left, row.right))
    return train, hold


@dataclass(frozen=True)
class ThousandSum:
    """An addition of a name in 100..999 and a name in 0..99 whose result leaves 0..999."""

    kind: str
    left: int
    right: int
    result: int

    def digit_prompt(self) -> str:
        return f"{decimal_name(self.left)}+{decimal_name(self.right)}="

    def digit_answer(self) -> str:
        return decimal_name(self.result)

    def word_prompt(self) -> str:
        right = WORD_OF[self.right] if self.right < RADIX else number_name(self.right)
        return f"what is {number_name(self.left)} plus {right}"

    def word_answer(self) -> str:
        return number_name(self.result)

    def carries(self) -> bool:
        return (self.left % RADIX) + (self.right % RADIX) >= RADIX

    def exact_ten(self) -> bool:
        return self.result % RADIX == 0

    def exact_thousand(self) -> bool:
        return self.result % (RADIX * RADIX * RADIX) == 0

    def low(self) -> bool:
        return self.result < RADIX * RADIX * RADIX + RADIX

    def high(self) -> bool:
        return self.result >= RADIX * RADIX * RADIX + (RADIX - 1) * RADIX


def all_thousand_sums() -> list[ThousandSum]:
    """Every name 100..999 plus a name 0..99 whose sum is 1000..9999."""
    rows: list[ThousandSum] = []
    span = RADIX * RADIX
    thousand = span * RADIX
    cap = thousand * RADIX
    for left in range(span, thousand):
        for right in range(span):
            result = left + right
            if thousand <= result < cap:
                kind = "step" if right < RADIX else "wide"
                rows.append(ThousandSum(kind, left, right, result))
    return rows


def _hold_thousand(row: ThousandSum) -> bool:
    """Deterministic fifth. No coefficient is a multiple of 5."""
    kind = 1 if row.kind == "step" else 0
    return (row.left * 3 + row.right * 4 + 6 * kind) % 5 == 0


def split_thousand_sums() -> tuple[list[ThousandSum], list[ThousandSum]]:
    rows = all_thousand_sums()
    train = [row for row in rows if not _hold_thousand(row)]
    hold = [row for row in rows if _hold_thousand(row)]
    train.sort(key=lambda row: (row.kind, row.left, row.right))
    hold.sort(key=lambda row: (row.kind, row.left, row.right))
    return train, hold


def thousand_op_held(left: int, right: int, op_bit: int) -> bool:
    """Deterministic fifth. No coefficient is a multiple of 5."""
    return (left * 3 + right * 4 + 6 * op_bit) % 5 == 0


@dataclass(frozen=True)
class ThousandOp:
    """A name in 1000..9999, plus or minus a name in 0..999, result inside 0..9999."""

    left: int
    op: str
    right: int
    result: int

    def digit_prompt(self) -> str:
        return f"{decimal_name(self.left)}{self.op}{decimal_name(self.right)}="

    def digit_answer(self) -> str:
        return decimal_name(self.result)

    def word_prompt(self) -> str:
        right = WORD_OF[self.right] if self.right < RADIX else number_name(self.right)
        return f"what is {number_name(self.left)} {OP_WORD[self.op]} {right}"

    def word_answer(self) -> str:
        return number_name(self.result)

    def uses_zero(self) -> bool:
        return self.right == 0

    def borrows(self) -> bool:
        return self.op == "-" and (self.left % RADIX) < (self.right % RADIX)

    def carries(self) -> bool:
        return self.op == "+" and (self.left % RADIX) + (self.right % RADIX) >= RADIX

    def under_thousand(self) -> bool:
        return self.result < THOUSAND_SPAN

    def exact_ten(self) -> bool:
        return self.result % RADIX == 0

    def exact_hundred(self) -> bool:
        return self.result % (RADIX * RADIX) == 0

    def exact_thousand(self) -> bool:
        return self.result % THOUSAND_SPAN == 0

    def crosses(self) -> bool:
        return self.left // THOUSAND_SPAN != self.result // THOUSAND_SPAN

    def ones(self) -> bool:
        return self.right < RADIX

    def place(self) -> bool:
        return RADIX <= self.right < RADIX * RADIX

    def block(self) -> bool:
        return self.right >= RADIX * RADIX


def census_thousand_ops() -> dict[str, int]:
    """Integer scan of every thousand operand. No consensus and no stored rows."""
    print("thousand operand census", flush=True)
    span = THOUSAND_SPAN
    cap = THOUSAND_CAP
    place = RADIX * RADIX
    keys = (
        "n",
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
        "borrow_ten",
        "out",
        "zero_result",
    )
    total = {key: 0 for key in keys}
    hold = {key: 0 for key in keys}
    min_result = cap
    max_result = -1
    held_row = False

    def mark(key: str) -> None:
        total[key] += 1
        if held_row:
            hold[key] += 1

    def tally(op_bit: int, result: int) -> None:
        nonlocal min_result, max_result, held_row
        held_row = thousand_op_held(left, right, op_bit)
        if result < min_result:
            min_result = result
        if result > max_result:
            max_result = result
        mark("n")
        if result < 0 or result >= cap:
            mark("out")
        if result == 0:
            mark("zero_result")
        if op_bit == 0:
            mark("plus")
            if left_mod + right_mod >= RADIX:
                mark("carry")
            else:
                mark("nocarry")
        else:
            mark("minus")
            if left_mod < right_mod:
                mark("borrow")
                if result % RADIX == 0:
                    mark("borrow_ten")
            else:
                mark("noborrow")
        if right == 0:
            mark("zero")
        if result < span:
            mark("under")
        if result % RADIX == 0:
            mark("ten")
        if result % place == 0:
            mark("hundred")
        if result % span == 0:
            mark("thousand")
        if left_band != result // span:
            mark("cross")
        if right < RADIX:
            mark("ones")
        elif right < place:
            mark("place")
        else:
            mark("block")
        if low_left:
            mark("left1000")
        if high_left:
            mark("left9000")
        if high_right:
            mark("right900")
        if result >= 9 * span:
            mark("top")

    for left in range(span, cap):
        left_mod = left % RADIX
        left_band = left // span
        low_left = left < 2 * span
        high_left = left >= 9 * span
        for right in range(span):
            right_mod = right % RADIX
            high_right = right >= 9 * place
            if left + right < cap:
                tally(0, left + right)
            tally(1, left - right)
    out: dict[str, int] = {}
    for key, value in total.items():
        out[key] = value
        out[key + "_train"] = value - hold[key]
        out[key + "_hold"] = hold[key]
    out["train"] = out["n_train"]
    out["hold"] = out["n_hold"]
    out["min_result"] = min_result
    out["max_result"] = max_result
    print(f"thousand operand census {out['n']}", flush=True)
    return out


def thousand_pair_held(left: int, right: int, op_bit: int) -> bool:
    """Deterministic fifth. No coefficient is a multiple of 5."""
    return (left * 3 + right * 4 + 6 * op_bit) % 5 == 0


@dataclass(frozen=True)
class ThousandPair:
    """Two names in 1000..9999 added or subtracted. The result stays inside 0..9999."""

    left: int
    op: str
    right: int
    result: int

    def digit_prompt(self) -> str:
        return f"{decimal_name(self.left)}{self.op}{decimal_name(self.right)}="

    def digit_answer(self) -> str:
        return decimal_name(self.result)

    def word_prompt(self) -> str:
        return (
            f"what is {number_name(self.left)} {OP_WORD[self.op]} {number_name(self.right)}"
        )

    def word_answer(self) -> str:
        return number_name(self.result)

    def is_zero(self) -> bool:
        return self.result == 0

    def borrows(self) -> bool:
        return self.op == "-" and (self.left % RADIX) < (self.right % RADIX)

    def carries(self) -> bool:
        return self.op == "+" and (self.left % RADIX) + (self.right % RADIX) >= RADIX

    def under_thousand(self) -> bool:
        return self.result < THOUSAND_SPAN

    def exact_ten(self) -> bool:
        return self.result % RADIX == 0

    def exact_hundred(self) -> bool:
        return self.result % (RADIX * RADIX) == 0

    def exact_thousand(self) -> bool:
        return self.result % THOUSAND_SPAN == 0

    def crosses(self) -> bool:
        return self.left // THOUSAND_SPAN != self.result // THOUSAND_SPAN


def census_thousand_pairs() -> dict[str, int]:
    """Integer scan of every pair of thousand names. No consensus and no stored rows."""
    print("thousand pair census", flush=True)
    span = THOUSAND_SPAN
    cap = THOUSAND_CAP
    place = RADIX * RADIX
    keys = (
        "n",
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
        "left1000",
        "left9000",
        "right9000",
        "top",
        "borrow_ten",
        "out",
    )
    total = {key: 0 for key in keys}
    hold = {key: 0 for key in keys}
    min_result = cap
    max_result = -1
    held_row = False

    def mark(key: str) -> None:
        total[key] += 1
        if held_row:
            hold[key] += 1

    def tally(op_bit: int, result: int) -> None:
        nonlocal min_result, max_result, held_row
        held_row = thousand_pair_held(left, right, op_bit)
        if result < min_result:
            min_result = result
        if result > max_result:
            max_result = result
        mark("n")
        if result < 0 or result >= cap:
            mark("out")
        if result == 0:
            mark("zero")
        if op_bit == 0:
            mark("plus")
            if left_mod + right_mod >= RADIX:
                mark("carry")
            else:
                mark("nocarry")
        else:
            mark("minus")
            if left_mod < right_mod:
                mark("borrow")
                if result % RADIX == 0:
                    mark("borrow_ten")
            else:
                mark("noborrow")
        if result < span:
            mark("under")
        if result % RADIX == 0:
            mark("ten")
        if result % place == 0:
            mark("hundred")
        if result % span == 0:
            mark("thousand")
        if left_band != result // span:
            mark("cross")
        if low_left:
            mark("left1000")
        if high_left:
            mark("left9000")
        if high_right:
            mark("right9000")
        if result >= 9 * span:
            mark("top")

    for left in range(span, cap):
        left_mod = left % RADIX
        left_band = left // span
        low_left = left < 2 * span
        high_left = left >= 9 * span
        plus_stop = cap - left
        if plus_stop > span:
            for right in range(span, plus_stop):
                right_mod = right % RADIX
                high_right = right >= 9 * span
                tally(0, left + right)
        for right in range(span, left + 1):
            right_mod = right % RADIX
            high_right = right >= 9 * span
            tally(1, left - right)
        if left_band != (left - span) // span and (left - span) % span == 0:
            print(f"thousand pair census left {left}", flush=True)
    out: dict[str, int] = {}
    for key, value in total.items():
        out[key] = value
        out[key + "_train"] = value - hold[key]
        out[key + "_hold"] = hold[key]
    out["train"] = out["n_train"]
    out["hold"] = out["n_hold"]
    out["min_result"] = min_result
    out["max_result"] = max_result
    print(f"thousand pair census {out['n']}", flush=True)
    return out


def ten_thousand_kind(right: int) -> int:
    """0 digit, 1 place, 2 block. The coefficient on this bit is not a multiple of 5."""
    if right < RADIX:
        return 0
    if right < RADIX * RADIX:
        return 1
    return 2


def ten_thousand_held(left: int, right: int, kind: int) -> bool:
    """Deterministic fifth. No coefficient is a multiple of 5."""
    return (left * 3 + right * 4 + 6 * kind) % 5 == 0


@dataclass(frozen=True)
class TenThousandSum:
    """An addition of a name in 1000..9999 and a name in 0..999 whose result leaves 0..9999."""

    kind: str
    left: int
    right: int
    result: int

    def digit_prompt(self) -> str:
        return f"{decimal_name(self.left)}+{decimal_name(self.right)}="

    def digit_answer(self) -> str:
        return decimal_name(self.result)

    def word_prompt(self) -> str:
        right = WORD_OF[self.right] if self.right < RADIX else number_name(self.right)
        return f"what is {number_name(self.left)} plus {right}"

    def word_answer(self) -> str:
        return number_name(self.result)


def census_ten_thousand_sums() -> dict[str, int]:
    """Integer scan of sums that leave 0..9999. No consensus and no stored rows."""
    print("ten thousand census", flush=True)
    span = THOUSAND_SPAN
    cap = THOUSAND_CAP
    place = RADIX * RADIX
    keys = (
        "n",
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
    )
    total = {key: 0 for key in keys}
    hold = {key: 0 for key in keys}
    min_result = TEN_THOUSAND_CAP
    max_result = -1
    held_row = False

    def mark(key: str) -> None:
        total[key] += 1
        if held_row:
            hold[key] += 1

    def tally(kind: int, result: int) -> None:
        nonlocal min_result, max_result, held_row
        held_row = ten_thousand_held(left, right, kind)
        if result < min_result:
            min_result = result
        if result > max_result:
            max_result = result
        mark("n")
        if kind == 0:
            mark("ones")
        elif kind == 1:
            mark("place")
        else:
            mark("block")
        if result % RADIX == 0:
            mark("ten")
        if result % place == 0:
            mark("hundred")
        if result % span == 0:
            mark("thousand")
        if left_mod + right_mod >= RADIX:
            mark("carry")
        else:
            mark("nocarry")
        if result < cap + RADIX:
            mark("low")
        if result >= cap + (RADIX - 1) * place:
            mark("high")

    for left in range(cap - span + 1, cap):
        left_mod = left % RADIX
        for right in range(cap - left, span):
            result = left + right
            if result >= TEN_THOUSAND_CAP:
                continue
            right_mod = right % RADIX
            tally(ten_thousand_kind(right), result)
    out: dict[str, int] = {}
    for key, value in total.items():
        out[key] = value
        out[key + "_train"] = value - hold[key]
        out[key + "_hold"] = hold[key]
    out["train"] = out["n_train"]
    out["hold"] = out["n_hold"]
    out["min_result"] = min_result
    out["max_result"] = max_result
    print(f"ten thousand census {out['n']}", flush=True)
    return out


def ten_thousand_op_held(left: int, right: int, op_bit: int) -> bool:
    """Deterministic fifth. No coefficient is a multiple of 5."""
    return (left * 3 + right * 4 + 6 * op_bit) % 5 == 0


@dataclass(frozen=True)
class TenThousandOp:
    """A name in 10000..10999, plus or minus a name in 0..9999. The result stays inside 0..10999."""

    left: int
    op: str
    right: int
    result: int

    def digit_prompt(self) -> str:
        return f"{decimal_name(self.left)}{self.op}{decimal_name(self.right)}="

    def digit_answer(self) -> str:
        return decimal_name(self.result)

    def word_prompt(self) -> str:
        right = WORD_OF[self.right] if self.right < RADIX else number_name(self.right)
        return f"what is {number_name(self.left)} {OP_WORD[self.op]} {right}"

    def word_answer(self) -> str:
        return number_name(self.result)

    def uses_zero(self) -> bool:
        return self.right == 0

    def borrows(self) -> bool:
        return self.op == "-" and (self.left % RADIX) < (self.right % RADIX)

    def carries(self) -> bool:
        return self.op == "+" and (self.left % RADIX) + (self.right % RADIX) >= RADIX

    def under_ten_thousand(self) -> bool:
        return self.result < THOUSAND_CAP

    def exact_ten(self) -> bool:
        return self.result % RADIX == 0

    def exact_hundred(self) -> bool:
        return self.result % (RADIX * RADIX) == 0

    def exact_thousand(self) -> bool:
        return self.result % THOUSAND_SPAN == 0

    def exact_mark(self) -> bool:
        return self.result == THOUSAND_CAP

    def crosses(self) -> bool:
        return self.left // THOUSAND_CAP != self.result // THOUSAND_CAP

    def ones(self) -> bool:
        return self.right < RADIX

    def place(self) -> bool:
        return RADIX <= self.right < RADIX * RADIX

    def block(self) -> bool:
        return RADIX * RADIX <= self.right < THOUSAND_SPAN

    def thousand_right(self) -> bool:
        return self.right >= THOUSAND_SPAN


def census_ten_thousand_ops() -> dict[str, int]:
    """Integer scan of every ten-thousand operand. No consensus and no stored rows."""
    print("ten thousand operand census", flush=True)
    cap = THOUSAND_CAP
    upper = TEN_THOUSAND_CAP
    span = THOUSAND_SPAN
    place = RADIX * RADIX
    keys = (
        "n",
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
        "borrow_ten",
        "out",
        "zero_result",
    )
    total = {key: 0 for key in keys}
    hold = {key: 0 for key in keys}
    min_result = upper
    max_result = -1
    held_row = False

    def mark(key: str) -> None:
        total[key] += 1
        if held_row:
            hold[key] += 1

    def tally(op_bit: int, result: int) -> None:
        nonlocal min_result, max_result, held_row
        held_row = ten_thousand_op_held(left, right, op_bit)
        if result < min_result:
            min_result = result
        if result > max_result:
            max_result = result
        mark("n")
        if result < 0 or result >= upper:
            mark("out")
        if result == 0:
            mark("zero_result")
        if op_bit == 0:
            mark("plus")
            if left_mod + right_mod >= RADIX:
                mark("carry")
            else:
                mark("nocarry")
        else:
            mark("minus")
            if left_mod < right_mod:
                mark("borrow")
                if result % RADIX == 0:
                    mark("borrow_ten")
            else:
                mark("noborrow")
        if right == 0:
            mark("zero")
        if result < cap:
            mark("under")
        if result % RADIX == 0:
            mark("ten")
        if result % place == 0:
            mark("hundred")
        if result % span == 0:
            mark("thousand")
        if result == cap:
            mark("mark")
        if left // cap != result // cap:
            mark("cross")
        if right < RADIX:
            mark("ones")
        elif right < place:
            mark("place")
        elif right < span:
            mark("block")
        else:
            mark("right1000")
        if low_left:
            mark("low")
        if high_left:
            mark("high")
        if high_right:
            mark("right9000")
        if result >= cap + (RADIX - 1) * place:
            mark("top")

    for left in range(cap, upper):
        left_mod = left % RADIX
        low_left = left < cap + RADIX
        high_left = left >= cap + (RADIX - 1) * place
        plus_stop = upper - left
        for right in range(plus_stop):
            right_mod = right % RADIX
            high_right = right >= 9 * span
            tally(0, left + right)
        for right in range(cap):
            right_mod = right % RADIX
            high_right = right >= 9 * span
            tally(1, left - right)
        if left % 250 == 0:
            print(f"ten thousand operand census left {left}", flush=True)
    out: dict[str, int] = {}
    for key, value in total.items():
        out[key] = value
        out[key + "_train"] = value - hold[key]
        out[key + "_hold"] = hold[key]
    out["train"] = out["n_train"]
    out["hold"] = out["n_hold"]
    out["min_result"] = min_result
    out["max_result"] = max_result
    print(f"ten thousand operand census {out['n']}", flush=True)
    return out


def ten_thousand_pair_held(left: int, right: int) -> bool:
    """Deterministic fifth. No coefficient is a multiple of 5."""
    return (left * 3 + right * 4) % 5 == 0


@dataclass(frozen=True)
class TenThousandPair:
    """Two names in 10000..10999. The difference stays inside 0..999."""

    left: int
    right: int
    result: int

    def digit_prompt(self) -> str:
        return f"{decimal_name(self.left)}-{decimal_name(self.right)}="

    def digit_answer(self) -> str:
        return decimal_name(self.result)

    def word_prompt(self) -> str:
        return f"what is {number_name(self.left)} minus {number_name(self.right)}"

    def word_answer(self) -> str:
        return number_name(self.result)

    def is_zero(self) -> bool:
        return self.result == 0

    def borrows(self) -> bool:
        return (self.left % RADIX) < (self.right % RADIX)

    def exact_ten(self) -> bool:
        return self.result % RADIX == 0

    def exact_hundred(self) -> bool:
        return self.result % (RADIX * RADIX) == 0

    def ones(self) -> bool:
        return self.result < RADIX

    def place(self) -> bool:
        return RADIX <= self.result < RADIX * RADIX

    def block(self) -> bool:
        return self.result >= RADIX * RADIX


def census_ten_thousand_pairs() -> dict[str, int]:
    """Integer scan of every ten-thousand difference. No consensus and no stored rows."""
    print("ten thousand pair census", flush=True)
    cap = THOUSAND_CAP
    upper = TEN_THOUSAND_CAP
    span = THOUSAND_SPAN
    place = RADIX * RADIX
    keys = (
        "n",
        "zero",
        "borrow",
        "noborrow",
        "borrow_ten",
        "ten",
        "hundred",
        "ones",
        "place",
        "block",
        "low",
        "high",
        "righthigh",
        "top",
        "cross",
        "out",
    )
    total = {key: 0 for key in keys}
    hold = {key: 0 for key in keys}
    min_result = upper
    max_result = -1
    held_row = False

    def mark(key: str) -> None:
        total[key] += 1
        if held_row:
            hold[key] += 1

    def tally(result: int) -> None:
        nonlocal min_result, max_result, held_row
        held_row = ten_thousand_pair_held(left, right)
        if result < min_result:
            min_result = result
        if result > max_result:
            max_result = result
        mark("n")
        if result < 0 or result >= span:
            mark("out")
        if result == 0:
            mark("zero")
        if left_mod < right_mod:
            mark("borrow")
            if result % RADIX == 0:
                mark("borrow_ten")
        else:
            mark("noborrow")
        if result % RADIX == 0:
            mark("ten")
        if result % place == 0:
            mark("hundred")
        if result < RADIX:
            mark("ones")
        elif result < place:
            mark("place")
        else:
            mark("block")
        if low_left:
            mark("low")
        if high_left:
            mark("high")
        if high_right:
            mark("righthigh")
        if result >= 9 * place:
            mark("top")
        if left // cap != result // cap:
            mark("cross")

    for left in range(cap, upper):
        left_mod = left % RADIX
        low_left = left < cap + RADIX
        high_left = left >= cap + (RADIX - 1) * place
        for right in range(cap, left + 1):
            right_mod = right % RADIX
            high_right = right >= cap + (RADIX - 1) * place
            tally(left - right)
    out: dict[str, int] = {}
    for key, value in total.items():
        out[key] = value
        out[key + "_train"] = value - hold[key]
        out[key + "_hold"] = hold[key]
    out["train"] = out["n_train"]
    out["hold"] = out["n_hold"]
    out["min_result"] = min_result
    out["max_result"] = max_result
    print(f"ten thousand pair census {out['n']}", flush=True)
    return out


def hundred_thousand_kind(right: int) -> int:
    """0 digit, 1 place, 2 block, 3 thousand name. The coefficient on this bit is not a multiple of 5."""
    if right < RADIX:
        return 0
    if right < RADIX * RADIX:
        return 1
    if right < THOUSAND_SPAN:
        return 2
    return 3


def hundred_thousand_held(left: int, right: int, kind: int) -> bool:
    """Deterministic fifth. No coefficient is a multiple of 5."""
    return (left * 3 + right * 4 + 6 * kind) % 5 == 0


@dataclass(frozen=True)
class HundredThousandSum:
    """An addition of a name in 90001..99999 and a name in 1..9999 whose result leaves 0..99999."""

    kind: str
    left: int
    right: int
    result: int

    def digit_prompt(self) -> str:
        return f"{decimal_name(self.left)}+{decimal_name(self.right)}="

    def digit_answer(self) -> str:
        return decimal_name(self.result)

    def word_prompt(self) -> str:
        right = WORD_OF[self.right] if self.right < RADIX else number_name(self.right)
        return f"what is {number_name(self.left)} plus {right}"

    def word_answer(self) -> str:
        return number_name(self.result)


def census_hundred_thousand_sums() -> dict[str, int]:
    """Integer scan of sums that leave 0..99999. No consensus and no stored rows."""
    print("hundred thousand census", flush=True)
    span = THOUSAND_CAP
    cap = HUNDRED_THOUSAND
    place = RADIX * RADIX
    block = THOUSAND_SPAN
    keys = (
        "n",
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
    )
    total = {key: 0 for key in keys}
    hold = {key: 0 for key in keys}
    min_result = HUNDRED_THOUSAND_CAP
    max_result = -1
    for left in range(cap - span + 1, cap):
        if (left - (cap - span)) % 1000 == 0:
            print(f"hundred thousand census left {left}", flush=True)
        left_mod = left % RADIX
        for right in range(cap - left, span):
            result = left + right
            kind = 0 if right < RADIX else 1 if right < place else 2 if right < block else 3
            held_row = (left * 3 + right * 4 + 6 * kind) % 5 == 0
            if result < min_result:
                min_result = result
            if result > max_result:
                max_result = result
            total["n"] += 1
            if held_row:
                hold["n"] += 1
            band = "ones" if kind == 0 else "place" if kind == 1 else "block" if kind == 2 else "thou"
            total[band] += 1
            if held_row:
                hold[band] += 1
            if result % RADIX == 0:
                total["ten"] += 1
                if held_row:
                    hold["ten"] += 1
            if result % place == 0:
                total["hundred"] += 1
                if held_row:
                    hold["hundred"] += 1
            if result % block == 0:
                total["thousand"] += 1
                if held_row:
                    hold["thousand"] += 1
            if result % span == 0:
                total["mark"] += 1
                if held_row:
                    hold["mark"] += 1
            hand = "carry" if left_mod + right % RADIX >= RADIX else "nocarry"
            total[hand] += 1
            if held_row:
                hold[hand] += 1
            if result < cap + RADIX:
                total["low"] += 1
                if held_row:
                    hold["low"] += 1
            if result >= cap + span - place:
                total["high"] += 1
                if held_row:
                    hold["high"] += 1
    out: dict[str, int] = {}
    for key, value in total.items():
        out[key] = value
        out[key + "_train"] = value - hold[key]
        out[key + "_hold"] = hold[key]
    out["train"] = out["n_train"]
    out["hold"] = out["n_hold"]
    out["min_result"] = min_result
    out["max_result"] = max_result
    print(f"hundred thousand census {out['n']}", flush=True)
    return out


def epoch_order(rows: list[Triple], epoch: int, phi: float) -> list[Triple]:
    """Phi-hash order. Stable across machines, changes each epoch."""

    def key(row: Triple) -> float:
        op_bit = 0 if row.op == "+" else 1
        return math_mod((row.a * 10 + row.b + op_bit + epoch * 3) * phi)

    return sorted(rows, key=key)


def math_mod(x: float) -> float:
    return x - math_floor(x)


def math_floor(x: float) -> int:
    i = int(x)
    if x < 0 and i != x:
        return i - 1
    return i
