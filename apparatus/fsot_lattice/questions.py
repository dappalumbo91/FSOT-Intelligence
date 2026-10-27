"""Questions whose operands are gauges the lattice is already holding.

A kind head is read as a kind. Any other stated word is read as an individual.
A second home sits beside the resting kind and is selected by a situation clock.
A missing gauge stays open. The read does not move a settled gauge.
A quotient names a count and a leftover. A dividend past the place span
is divided a thousand at a time. Euclid reads the common measure
from that leftover, and the reduced ratio is the same split in lowest terms.
Those quotients also name the two coefficients that rebuild the measure,
and exact division names the prime factors. A shared leftover is a
congruence. The inverse is the count that leaves one, and two congruences
name one count when their leftovers agree. A product past nine is repeated
addition. When that long sum misses the drop, the smaller count is read
one digit at a time. A product past the place span is read a thousand at
a time. Rest energy is that product: the mass, times the speed of light,
times the speed of light. A speed below light multiplies that energy
by the exact Lorentz factor when the two squares meet. The total minus
the rest is the kinetic energy. Momentum is the mass times that speed,
and the energy and the momentum meet on the square of mass times the speed
of light. A power repeats that product. The totient counts residues whose
common measure is one, and the order is the least power that leaves one.
A generator is an order that meets the totient. The least power that hits
a leftover is the logarithm. A square leftover meets Euler's count on an
odd prime, and a choice is an exact product of falling counts.
The Euclid quotients name the continued fraction. A square root lists the
counts that leave that leftover. Two squares rebuild a count, and an ordered
choice is the falling product before the choice divides.
Each of those quotients builds a convergent. Neighboring convergents differ
by one in their cross product. Fibonacci and Lucas counts are that same
two-term sum, and consecutive Fibonacci counts are the continued fraction
of ones that ends in two.
A whole square root repeats. One period of that fraction names the Pell
pair: the first square is one above the count times the second square.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fsot_lattice.apply import DROP
from fsot_lattice.lattice import Lattice, content_offset
from fsot_lattice.layouts import LayoutScore
from fsot_lattice.tasks import (
    ACQUIRED_NOUNS,
    CONTENT_WORDS,
    KIND_HEADS,
    BILLION_WORD,
    MILLION_WORD,
    THOUSAND_SPAN,
    THOUSAND_WORD,
    TRILLION_CAP,
    TRILLION_WORD,
    Product,
    acquired_kind,
    decimal_name,
    number_name,
)


def _kind_head(word: str) -> bool:
    return word in KIND_HEADS


def _known(lattice: Lattice, word: str) -> bool:
    return word in lattice.content_value or word in lattice.acquired_value


def _place_apart(gauge: float) -> float:
    """Read an acquired individual one content-line higher.

    Both lines are K times a count, so the first of each line sits on the
    same place. The content line already occupies ten places. Adding that
    width puts the acquired individual on the next free place. The gap is
    one K, wider than the drop, so a shared place does not read as the
    same individual.
    """
    return gauge + content_offset(len(CONTENT_WORDS) - 1)


def _identity(lattice: Lattice, word: str) -> float | None:
    if word in lattice.acquired_value:
        return _place_apart(lattice.acquired_self[word])
    if word in lattice.content_value:
        return lattice.content_value[word]
    return None


def _kind_vessel(lattice: Lattice, word: str) -> float | None:
    if word in lattice.content_value:
        return lattice.kind_step(lattice.content_value[word])
    if word in lattice.acquired_value:
        return lattice.kind_step(lattice.acquired_value[word])
    return None


def ask_is(lattice: Lattice, subject: str, stated: str) -> str:
    """Yes when the stated word is this subject's kind, or the same individual.

    A kind head uses the kind step. Wolf stays on the animal gauge, so a
    plant stays no. An individual uses its own gauge. An acquired individual
    is read one content-line higher, so a cat and a wolf do not share a place.
    """
    if not _known(lattice, subject) or not _known(lattice, stated):
        return "open"
    if _kind_head(stated) and subject in lattice.content_value:
        return lattice.predict_statement(subject, stated)
    if _kind_head(stated) and subject in lattice.acquired_value:
        return lattice.predict_acquired(subject, stated)
    left = _identity(lattice, subject)
    right = _identity(lattice, stated)
    if left is None or right is None:
        return "open"
    gap = lattice.consensus_quantity(left, right, lattice.routed_sign("-"))
    return lattice.remainder_answer(gap)


def ask_same_kind(lattice: Lattice, left: str, right: str) -> str:
    """Yes when both nouns step to the same kind head."""
    first = _kind_vessel(lattice, left)
    second = _kind_vessel(lattice, right)
    if first is None or second is None:
        return "open"
    gap = lattice.consensus_quantity(first, second, lattice.routed_sign("-"))
    return lattice.remainder_answer(gap)


def _between_half(lattice: Lattice, residual: float, surface: str) -> int | None:
    """The lower digit when the residual sits in the open band between two places.

    That band is the middle of the gap: both neighbors are farther than the
    drop. The middle of the gap is the half.
    """
    gauges = lattice.gauge_line(surface)
    marks = gauges + [lattice.ten_quantity(surface)]
    for index in range(len(marks) - 1):
        low, high = marks[index], marks[index + 1]
        if low <= residual <= high:
            if residual - low >= DROP and high - residual >= DROP:
                return index
            return None
    return None


def _half_name(whole: int, surface: str) -> str:
    piece = "1/2"
    if whole == 0:
        return piece
    stem = decimal_name(whole) if surface == "digit" else number_name(whole)
    if stem == "?":
        return "open"
    return f"{stem} {piece}"


def _name_quantity(lattice: Lattice, quantity: float, surface: str) -> str:
    """Name a gauge on the place reader. A negative magnitude keeps the minus.

    A units residual in the open band between two places is the half of that gap.
    """
    sign = ""
    if quantity < -DROP:
        quantity = -quantity
        sign = "-" if surface == "digit" else "minus "
    named, residual, _margin, dist = lattice.read_place(quantity, surface)
    if dist < DROP:
        text = decimal_name(named) if surface == "digit" else number_name(named)
        if text == "?":
            return "open"
        return f"{sign}{text}"
    lower = _between_half(lattice, residual, surface)
    if lower is None:
        return "open"
    units = named % 10
    if units == lower:
        whole = named
    elif lower < 9 and units == lower + 1:
        whole = named - 1
    elif lower == 9 and units == 0 and named >= 10:
        whole = named - 1
    else:
        return "open"
    text = _half_name(whole, surface)
    if text == "open":
        return "open"
    return f"{sign}{text}"


def ask_held_total(lattice: Lattice) -> str:
    """The consensus of the count clocks the paragraph is holding."""
    clocks = [
        value
        for value in (
            lattice.opening_count,
            lattice.earliest_count,
            lattice.earlier_count,
            lattice.counted,
        )
        if value != 0.0
    ]
    if not clocks:
        return "open"
    total = clocks[0]
    plus = lattice.routed_sign("+")
    for value in clocks[1:]:
        total = lattice.consensus_quantity(total, value, plus)
    return _name_quantity(lattice, total, "word")


def ask_product(lattice: Lattice, count: int, addend: int, surface: str = "digit") -> str:
    """A product named through the place reader.

    Counts through nine keep the closed product read. A wider count is the
    gauge added onto itself.
    """
    if count < 0 or addend < 0:
        return "open"
    if count <= 9 and addend <= 9:
        if count * addend >= TRILLION_CAP:
            return "open"
        named = lattice.predict_product(Product(count, addend, count * addend), surface)
        if named == "?":
            return "open"
        return named
    found = _multiply(lattice, count, addend, surface)
    if found is not None:
        return _whole_name(found[0], surface)
    if (
        count < _PLACE_SPAN
        and addend < _PLACE_SPAN
        and count * addend < _PLACE_SPAN
    ):
        return "open"
    wide = _wide_product(lattice, count, addend, surface)
    if wide is None:
        return "open"
    return _wide_name(wide[0], surface)


def ask_difference(lattice: Lattice, left: int, right: int, surface: str = "digit") -> str:
    """A difference of two place names. The minus names a result below zero."""
    if left < 0 or right < 0:
        return "open"
    steps = lattice.place_steps(surface)
    gauges = lattice.gauge_line(surface)
    quantity = lattice.consensus_quantity(
        lattice.compose_integer(left, steps, gauges),
        lattice.compose_integer(right, steps, gauges),
        lattice.routed_sign("-"),
    )
    return _name_quantity(lattice, quantity, surface)


def _count_steps(lattice: Lattice, quantity: float, step: float, limit: int) -> tuple[int, float] | None:
    """Count divisor steps past the single digit. The place shed stays capped at nine."""
    if step == 0.0 or limit < 1:
        return None
    remainder = quantity
    count = 0
    minus = lattice.routed_sign("-")
    while count < limit:
        nxt = lattice.consensus_quantity(remainder, step, minus)
        if nxt < -DROP:
            break
        if abs(nxt - remainder) <= 1e-15:
            return None
        remainder = nxt
        count += 1
    else:
        return None
    return count, remainder


def ask_quotient(lattice: Lattice, dividend: int, divisor: int, surface: str = "digit") -> str:
    """How many divisor steps sit in the dividend.

    An exact fit names the count. A leftover is named on the place reader,
    including a place past nine and a half sitting between two places.
    """
    if dividend < 0 or divisor <= 0:
        return "open"
    if dividend >= _PLACE_SPAN:
        found = _wide_divide(lattice, dividend, divisor, surface)
        if found is None:
            return "open"
        limbs, leftover, _gap = found
        count_name = _wide_name(limbs, surface)
        if count_name == "open":
            return "open"
        if leftover == 0:
            return count_name
        rem_name = _whole_name(leftover, surface)
        if rem_name == "open":
            return "open"
        return f"{count_name} remainder {rem_name}"
    steps = lattice.place_steps(surface)
    gauges = lattice.gauge_line(surface)
    quantity = lattice.compose_integer(dividend, steps, gauges)
    step = lattice.compose_integer(divisor, steps, gauges)
    counted = _count_steps(lattice, quantity, step, dividend + 1)
    if counted is None:
        return "open"
    count, remainder = counted
    count_name = decimal_name(count) if surface == "digit" else number_name(count)
    if count_name == "?":
        return "open"
    rem_name = _name_quantity(lattice, remainder, surface)
    if rem_name == "open":
        return "open"
    if rem_name in ("0", "zero"):
        return count_name
    return f"{count_name} remainder {rem_name}"


# Twelve place rungs name a count below 10^13. A wider count stays open.
_PLACE_SPAN = 10**13
# A thousand matches the spelling group. A limb times a limb stays inside the span.
_LIMB = THOUSAND_SPAN
_LIMB_CAP = 8
# SI metre. This many metres per second. It is not a fitted coefficient, and it is not K.
_LIGHT_METRES = 299_792_458
_SCALE_WORD = (
    "",
    THOUSAND_WORD,
    MILLION_WORD,
    BILLION_WORD,
    TRILLION_WORD,
    "quadrillion",
    "quintillion",
    "sextillion",
    "septillion",
)


def _whole_name(number: int, surface: str) -> str:
    text = decimal_name(number) if surface == "digit" else number_name(number)
    if text == "?":
        return "open"
    return text


def _held_integer(lattice: Lattice, number: int, surface: str) -> int | None:
    """The place reader names this count back. A miss stays open."""
    if number < 0 or number >= _PLACE_SPAN:
        return None
    steps = lattice.place_steps(surface)
    gauges = lattice.gauge_line(surface)
    named, _residual, _margin, dist = lattice.read_place(
        lattice.compose_integer(number, steps, gauges),
        surface,
    )
    if dist >= DROP or named != number:
        return None
    return named


def _division_gap(
    lattice: Lattice,
    dividend: int,
    divisor: int,
    count: int,
    leftover: int,
    surface: str,
) -> float | None:
    """How far compose(count × divisor) + compose(leftover) sits from the dividend.

    The product and the leftover are composed on their own, then one consensus
    add. A zero leftover is the product alone. The gap is that sum against
    the dividend.
    """
    if (
        dividend < 0
        or divisor <= 0
        or count < 0
        or leftover < 0
        or dividend >= _PLACE_SPAN
        or leftover >= _PLACE_SPAN
    ):
        return None
    product = count * divisor
    if product >= _PLACE_SPAN:
        return None
    steps = lattice.place_steps(surface)
    gauges = lattice.gauge_line(surface)
    left = lattice.compose_integer(dividend, steps, gauges)
    right = lattice.compose_integer(product, steps, gauges)
    if leftover:
        right = lattice.consensus_quantity(
            right,
            lattice.compose_integer(leftover, steps, gauges),
            lattice.routed_sign("+"),
        )
    return abs(lattice.consensus_quantity(left, right, lattice.routed_sign("-")))


def _divide(
    lattice: Lattice, dividend: int, divisor: int, surface: str
) -> tuple[int, int] | None:
    """Integer count and whole leftover. A half between two places stays open.

    The count is divisor steps on the gauges. The leftover has to land on a
    place, the names have to rebuild the dividend, and the composed product
    plus the composed leftover has to meet the dividend inside the drop.
    """
    if dividend < 0 or divisor <= 0 or dividend >= _PLACE_SPAN:
        return None
    steps = lattice.place_steps(surface)
    gauges = lattice.gauge_line(surface)
    quantity = lattice.compose_integer(dividend, steps, gauges)
    step = lattice.compose_integer(divisor, steps, gauges)
    counted = _count_steps(lattice, quantity, step, dividend + 1)
    if counted is None:
        return None
    count, remainder = counted
    named, _residual, _margin, dist = lattice.read_place(remainder, surface)
    if dist >= DROP:
        return None
    if count * divisor + named != dividend:
        return None
    gap = _division_gap(lattice, dividend, divisor, count, named, surface)
    if gap is None or gap >= DROP:
        return None
    return count, named


def _common(lattice: Lattice, left: int, right: int, surface: str) -> int | None:
    """Euclid on the gauge leftover. The remainder replaces the larger count.

    A zero partner is the other count, read back on the place reader. Two
    zeros stay open. The loop stops when a divisor lands on an empty leftover.
    """
    if left < 0 or right < 0:
        return None
    if left == 0 and right == 0:
        return None
    if left == 0:
        return _held_integer(lattice, right, surface)
    if right == 0:
        return _held_integer(lattice, left, surface)
    for _step in range(64):
        if right > left:
            left, right = right, left
        split = _divide(lattice, left, right, surface)
        if split is None:
            return None
        _count, leftover = split
        if leftover == 0:
            return _held_integer(lattice, right, surface)
        left, right = right, leftover
    return None


def ask_common_measure(
    lattice: Lattice, left: int, right: int, surface: str = "digit"
) -> str:
    """The greatest count that measures both. Order does not matter."""
    found = _common(lattice, left, right, surface)
    if found is None:
        return "open"
    return _whole_name(found, surface)


def ask_ratio(
    lattice: Lattice, dividend: int, divisor: int, surface: str = "digit"
) -> str:
    """The dividend in lowest terms over the divisor.

    An exact fit names the whole count. A leftover names the whole count and
    the reduced fraction. The quotient reader keeps its own remainder spelling.
    """
    if dividend < 0 or divisor <= 0:
        return "open"
    if dividend == 0:
        return _whole_name(0, surface)
    measure = _common(lattice, dividend, divisor, surface)
    if measure is None or measure <= 0:
        return "open"
    into_dividend = _divide(lattice, dividend, measure, surface)
    into_divisor = _divide(lattice, divisor, measure, surface)
    if (
        into_dividend is None
        or into_divisor is None
        or into_dividend[1] != 0
        or into_divisor[1] != 0
        or into_divisor[0] <= 0
    ):
        return "open"
    reduced_dividend = into_dividend[0]
    reduced_divisor = into_divisor[0]
    split = _divide(lattice, dividend, divisor, surface)
    if split is None:
        return "open"
    whole, leftover = split
    if leftover == 0:
        if whole != reduced_dividend or reduced_divisor != 1:
            return "open"
        return _whole_name(whole, surface)
    reduced_left = _divide(lattice, leftover, measure, surface)
    if reduced_left is None or reduced_left[1] != 0 or reduced_left[0] <= 0:
        return "open"
    numerator = reduced_left[0]
    if numerator >= reduced_divisor:
        return "open"
    if whole * reduced_divisor + numerator != reduced_dividend:
        return "open"
    piece = f"{numerator}/{reduced_divisor}"
    if whole == 0:
        return piece
    stem = _whole_name(whole, surface)
    if stem == "open":
        return "open"
    return f"{stem} {piece}"


def _coeff_name(number: int, surface: str) -> str:
    """A coefficient. A negative one keeps the minus in front of the place name."""
    if number < 0:
        stem = _whole_name(-number, surface)
        if stem == "open":
            return "open"
        if surface == "digit":
            return f"-{stem}"
        return f"minus {stem}"
    return _whole_name(number, surface)


def _compose_sum(lattice: Lattice, parts: list[int], surface: str) -> float | None:
    """Consensus of the composed counts. A zero part drops out."""
    steps = lattice.place_steps(surface)
    gauges = lattice.gauge_line(surface)
    total = 0.0
    plus = lattice.routed_sign("+")
    for number in parts:
        if number < 0 or number >= _PLACE_SPAN:
            return None
        if number == 0:
            continue
        total = lattice.consensus_quantity(
            total,
            lattice.compose_integer(number, steps, gauges),
            plus,
        )
    return total


def _bezout_gap(
    lattice: Lattice,
    left: int,
    coeff_left: int,
    right: int,
    coeff_right: int,
    measure: int,
    surface: str,
) -> float | None:
    """How far the positive combination sits from the measure plus the negative part.

    A negative coefficient moves its term beside the measure, so both sides
    of the consensus are sums of counts the place reader can compose.
    """
    if measure < 0 or measure >= _PLACE_SPAN:
        return None
    positive: list[int] = []
    negative: list[int] = [measure]
    for base, coeff in ((left, coeff_left), (right, coeff_right)):
        if coeff == 0:
            continue
        if base < 0:
            return None
        product = base * abs(coeff)
        if product >= _PLACE_SPAN:
            return None
        if coeff > 0:
            positive.append(product)
        else:
            negative.append(product)
    got = _compose_sum(lattice, positive, surface)
    want = _compose_sum(lattice, negative, surface)
    if got is None or want is None:
        return None
    return abs(lattice.consensus_quantity(got, want, lattice.routed_sign("-")))


def _bezout(
    lattice: Lattice, left: int, right: int, surface: str
) -> tuple[int, int, int] | None:
    """Measure and the two coefficients that rebuild it.

    Each quotient is a divisor step on the gauges. Walking those quotients
    backward names the coefficients. The combination has to equal the common
    measure, and the gauges have to agree inside the drop.
    """
    if left < 0 or right < 0:
        return None
    if left == 0 and right == 0:
        return None
    quotients: list[int] = []
    current, divisor = left, right
    for _step in range(64):
        if divisor == 0:
            break
        split = _divide(lattice, current, divisor, surface)
        if split is None:
            return None
        count, leftover = split
        quotients.append(count)
        current, divisor = divisor, leftover
    else:
        return None
    if current <= 0 or _held_integer(lattice, current, surface) is None:
        return None
    coeff_left, coeff_right = 1, 0
    for count in reversed(quotients):
        coeff_left, coeff_right = coeff_right, coeff_left - count * coeff_right
    if left * coeff_left + right * coeff_right != current:
        return None
    if _common(lattice, left, right, surface) != current:
        return None
    gap = _bezout_gap(
        lattice, left, coeff_left, right, coeff_right, current, surface
    )
    if gap is None or gap >= DROP:
        return None
    return current, coeff_left, coeff_right


def ask_bezout(lattice: Lattice, left: int, right: int, surface: str = "digit") -> str:
    """The coefficients for the two counts, then the measure they rebuild."""
    found = _bezout(lattice, left, right, surface)
    if found is None:
        return "open"
    measure, coeff_left, coeff_right = found
    first = _coeff_name(coeff_left, surface)
    second = _coeff_name(coeff_right, surface)
    made = _whole_name(measure, surface)
    if first == "open" or second == "open" or made == "open":
        return "open"
    return f"{first} and {second} make {made}"


def _least_factor(lattice: Lattice, number: int, surface: str) -> int | None:
    """The least divisor greater than one. A prime returns itself.

    Once the quotient drops below the trial divisor, that divisor squared
    has passed the count, so the count has no smaller factor.
    """
    if number <= 1 or number >= _PLACE_SPAN:
        return None
    divisor = 2
    while divisor <= number:
        split = _divide(lattice, number, divisor, surface)
        if split is None:
            return None
        count, leftover = split
        if leftover == 0:
            return divisor
        if count < divisor:
            return number
        divisor += 1
    return None


def _prime_factors(
    lattice: Lattice, number: int, surface: str
) -> tuple[int, ...] | None:
    """Prime factors, least first. Each exact division is a gauge identity."""
    if number <= 1 or number >= _PLACE_SPAN:
        return None
    factors: list[int] = []
    rest = number
    for _step in range(64):
        if rest == 1:
            break
        factor = _least_factor(lattice, rest, surface)
        if factor is None or factor <= 1 or factor > rest:
            return None
        split = _divide(lattice, rest, factor, surface)
        if split is None or split[1] != 0 or split[0] * factor != rest:
            return None
        factors.append(factor)
        rest = split[0]
    else:
        return None
    product = 1
    for factor in factors:
        product *= factor
    if product != number or not factors:
        return None
    return tuple(factors)


def ask_factors(lattice: Lattice, number: int, surface: str = "digit") -> str:
    """Prime factors least first. A prime names itself. Zero and one stay open."""
    factors = _prime_factors(lattice, number, surface)
    if factors is None:
        return "open"
    names = [_whole_name(factor, surface) for factor in factors]
    if any(name == "open" for name in names):
        return "open"
    separator = " * " if surface == "digit" else " times "
    return separator.join(names)


def ask_prime(lattice: Lattice, number: int, surface: str = "digit") -> str:
    """Yes when the count's only prime factor is itself. Zero and one are no."""
    if number < 0:
        return "open"
    if number <= 1:
        return "no"
    factors = _prime_factors(lattice, number, surface)
    if factors is None:
        return "open"
    if factors == (number,):
        return "yes"
    return "no"


def _residue(lattice: Lattice, number: int, modulus: int, surface: str) -> int | None:
    """The least leftover of this count grouped by the modulus.

    A negative count uses the modulus minus the leftover of its magnitude.
    That difference has to read back on the place reader.
    """
    if modulus <= 0 or modulus >= _PLACE_SPAN:
        return None
    if number >= _PLACE_SPAN or number <= -_PLACE_SPAN:
        return None
    if number >= 0:
        split = _divide(lattice, number, modulus, surface)
        if split is None:
            return None
        return split[1]
    split = _divide(lattice, -number, modulus, surface)
    if split is None:
        return None
    _count, leftover = split
    if leftover == 0:
        return 0
    if leftover >= modulus:
        return None
    steps = lattice.place_steps(surface)
    gauges = lattice.gauge_line(surface)
    difference = lattice.consensus_quantity(
        lattice.compose_integer(modulus, steps, gauges),
        lattice.compose_integer(leftover, steps, gauges),
        lattice.routed_sign("-"),
    )
    named, _residual, _margin, dist = lattice.read_place(difference, surface)
    residue = modulus - leftover
    if dist >= DROP or named != residue:
        return None
    return residue


def _positive_inverse(
    lattice: Lattice, number: int, modulus: int, surface: str
) -> int | None:
    """The least positive count whose multiple leaves one.

    The Bézout coefficient may be negative. Its leftover under the modulus
    is the inverse, and the product has to leave one on the gauges.
    """
    if number <= 0 or modulus <= 1 or number >= _PLACE_SPAN:
        return None
    found = _bezout(lattice, number, modulus, surface)
    if found is None or found[0] != 1:
        return None
    residue = _residue(lattice, found[1], modulus, surface)
    if residue is None or residue <= 0:
        return None
    product = number * residue
    if product >= _PLACE_SPAN:
        return None
    split = _divide(lattice, product, modulus, surface)
    if split is None or split[1] != 1:
        return None
    return residue


def _linear_solutions(
    lattice: Lattice, coeff: int, target: int, modulus: int, surface: str
) -> tuple[int, ...] | None:
    """Least leftovers of coeff × count ≡ target, grouped by the modulus.

    The common measure has to divide the target. The equation is then one
    inverse on the reduced modulus. More than sixty-four solutions stay open.
    """
    if coeff < 0 or target < 0 or modulus <= 0:
        return None
    if modulus >= _PLACE_SPAN or coeff >= _PLACE_SPAN or target >= _PLACE_SPAN:
        return None
    reduced_target = _residue(lattice, target, modulus, surface)
    if reduced_target is None:
        return None
    if modulus == 1:
        if reduced_target != 0:
            return None
        return (0,)
    measure = _common(lattice, coeff, modulus, surface)
    if measure is None or measure <= 0:
        return None
    into_target = _divide(lattice, reduced_target, measure, surface)
    if into_target is None or into_target[1] != 0:
        return None
    into_coeff = _divide(lattice, coeff, measure, surface)
    into_modulus = _divide(lattice, modulus, measure, surface)
    if (
        into_coeff is None
        or into_modulus is None
        or into_coeff[1] != 0
        or into_modulus[1] != 0
        or into_modulus[0] <= 0
    ):
        return None
    reduced_coeff = into_coeff[0]
    reduced_modulus = into_modulus[0]
    if measure > 64:
        return None
    if reduced_modulus == 1:
        solutions = list(range(modulus))
    else:
        inverse = _positive_inverse(lattice, reduced_coeff, reduced_modulus, surface)
        if inverse is None:
            return None
        product = into_target[0] * inverse
        if product >= _PLACE_SPAN:
            return None
        start = _residue(lattice, product, reduced_modulus, surface)
        if start is None:
            return None
        solutions = [start + step * reduced_modulus for step in range(measure)]
    if len(solutions) > 64 or len(solutions) != measure:
        return None
    for candidate in solutions:
        if candidate < 0 or candidate >= modulus:
            return None
        product = coeff * candidate
        if product >= _PLACE_SPAN:
            return None
        check = _divide(lattice, product, modulus, surface)
        if check is None or check[1] != reduced_target:
            return None
    return tuple(solutions)


def _counts_answer(numbers: tuple[int, ...], surface: str) -> str:
    names = [_whole_name(number, surface) for number in numbers]
    if any(name == "open" for name in names):
        return "open"
    if len(names) == 1:
        return names[0]
    return " and ".join(names)


def ask_same_remainder(
    lattice: Lattice, left: int, right: int, modulus: int, surface: str = "digit"
) -> str:
    """Yes when both counts leave the same leftover under the modulus."""
    if left < 0 or right < 0 or modulus <= 0:
        return "open"
    first = _residue(lattice, left, modulus, surface)
    second = _residue(lattice, right, modulus, surface)
    if first is None or second is None:
        return "open"
    if first == second:
        return "yes"
    return "no"


def ask_inverse(
    lattice: Lattice, number: int, modulus: int, surface: str = "digit"
) -> str:
    """The count that multiplies this one and leaves one."""
    found = _positive_inverse(lattice, number, modulus, surface)
    if found is None:
        return "open"
    return _whole_name(found, surface)


def ask_linear(
    lattice: Lattice, coeff: int, target: int, modulus: int, surface: str = "digit"
) -> str:
    """Counts whose multiple leaves the target leftover under the modulus."""
    found = _linear_solutions(lattice, coeff, target, modulus, surface)
    if found is None:
        return "open"
    return _counts_answer(found, surface)


def _paired_remainder(
    lattice: Lattice,
    left: int,
    left_modulus: int,
    right: int,
    right_modulus: int,
    surface: str,
) -> int | None:
    """The least count that leaves both leftovers.

    The two moduli share a measure, and that measure has to divide the gap
    between the leftovers. One linear step then names the count.
    """
    if left < 0 or right < 0 or left_modulus <= 0 or right_modulus <= 0:
        return None
    first = _residue(lattice, left, left_modulus, surface)
    second = _residue(lattice, right, right_modulus, surface)
    if first is None or second is None:
        return None
    measure = _common(lattice, left_modulus, right_modulus, surface)
    if measure is None or measure <= 0:
        return None
    gap = first - second
    if gap < 0:
        gap = -gap
    split = _divide(lattice, gap, measure, surface)
    if split is None or split[1] != 0:
        return None
    target = _residue(lattice, second - first, right_modulus, surface)
    if target is None:
        return None
    steps = _linear_solutions(lattice, left_modulus, target, right_modulus, surface)
    if steps is None or not steps:
        return None
    product = left_modulus * steps[0]
    if product >= _PLACE_SPAN:
        return None
    count = first + product
    into = _divide(lattice, left_modulus, measure, surface)
    if into is None or into[1] != 0:
        return None
    span = into[0] * right_modulus
    if span <= 0 or span >= _PLACE_SPAN:
        return None
    answer = _residue(lattice, count, span, surface)
    if answer is None:
        return None
    if _residue(lattice, answer, left_modulus, surface) != first:
        return None
    if _residue(lattice, answer, right_modulus, surface) != second:
        return None
    return answer


def ask_paired(
    lattice: Lattice,
    left: int,
    left_modulus: int,
    right: int,
    right_modulus: int,
    surface: str = "digit",
) -> str:
    """The least count that leaves the first leftover and the second leftover."""
    found = _paired_remainder(
        lattice, left, left_modulus, right, right_modulus, surface
    )
    if found is None:
        return "open"
    return _whole_name(found, surface)


def _repeated_product(
    lattice: Lattice, left: int, right: int, surface: str
) -> tuple[int, float] | None:
    """The product, and how far the repeated sum sits from its place composition.

    The larger count is added the smaller number of times. The sum has to
    name the product, and dividing by the larger count has to return the
    smaller one with an empty leftover.
    """
    if left < 0 or right < 0 or left >= _PLACE_SPAN or right >= _PLACE_SPAN:
        return None
    product = left * right
    if product >= _PLACE_SPAN:
        return None
    if product == 0:
        if _held_integer(lattice, 0, surface) != 0:
            return None
        return 0, 0.0
    times = left if left <= right else right
    piece_value = right if left <= right else left
    steps = lattice.place_steps(surface)
    gauges = lattice.gauge_line(surface)
    piece = lattice.compose_integer(piece_value, steps, gauges)
    total = 0.0
    plus = lattice.routed_sign("+")
    for _step in range(times):
        total = lattice.consensus_quantity(total, piece, plus)
    named, _residual, _margin, dist = lattice.read_place(total, surface)
    goal = lattice.compose_integer(product, steps, gauges)
    gap = abs(lattice.consensus_quantity(total, goal, lattice.routed_sign("-")))
    if dist >= DROP or gap >= DROP or named != product:
        return None
    larger = left if left >= right else right
    smaller = right if left >= right else left
    split = _divide(lattice, named, larger, surface)
    if split is None or split != (smaller, 0):
        return None
    return named, gap


def _shift_ten(
    lattice: Lattice, value: int, places: int, surface: str
) -> tuple[int, float] | None:
    """Move a count up by this many tens. Each step is ten additions."""
    if places < 0 or value < 0:
        return None
    if places == 0:
        if _held_integer(lattice, value, surface) != value:
            return None
        return value, 0.0
    gap = 0.0
    current = value
    for _step in range(places):
        stepped = _repeated_product(lattice, current, 10, surface)
        if stepped is None:
            return None
        current = stepped[0]
        if stepped[1] > gap:
            gap = stepped[1]
    return current, gap


def _grouped_product(
    lattice: Lattice, left: int, right: int, surface: str
) -> tuple[int, float] | None:
    """The product, one digit of the smaller count at a time.

    A long repeated sum can land one count off. Each digit is at most nine
    additions of the larger count, shifted by tens, and removing those pieces
    from the product has to leave zero.
    """
    if left < 0 or right < 0 or left >= _PLACE_SPAN or right >= _PLACE_SPAN:
        return None
    product = left * right
    if product >= _PLACE_SPAN or product == 0:
        return None
    smaller = left if left <= right else right
    larger = right if left <= right else left
    if smaller < 10:
        return None
    total: int | None = None
    gap = 0.0
    place = 0
    rest = smaller
    while rest:
        digit = rest % 10
        rest //= 10
        if digit:
            part = _repeated_product(lattice, larger, digit, surface)
            if part is None:
                return None
            shifted = _shift_ten(lattice, part[0], place, surface)
            if shifted is None:
                return None
            part_gap = part[1] if part[1] > shifted[1] else shifted[1]
            if total is None:
                total = shifted[0]
            else:
                added = _add_counts(lattice, total, shifted[0], surface)
                if added is None:
                    return None
                total = added[0]
                if added[1] > part_gap:
                    part_gap = added[1]
            if part_gap > gap:
                gap = part_gap
        place += 1
    if total is None or total != product:
        return None
    rest_value = total
    place = 0
    rest = smaller
    while rest:
        digit = rest % 10
        rest //= 10
        if digit:
            part = _repeated_product(lattice, larger, digit, surface)
            if part is None:
                return None
            shifted = _shift_ten(lattice, part[0], place, surface)
            if shifted is None or shifted[0] > rest_value:
                return None
            if shifted[0] == rest_value:
                rest_value = 0
            else:
                subtracted = _subtract_counts(lattice, rest_value, shifted[0], surface)
                if subtracted is None:
                    return None
                rest_value = subtracted[0]
                if subtracted[1] > gap:
                    gap = subtracted[1]
        place += 1
    if rest_value != 0:
        return None
    return total, gap


def _multiply(
    lattice: Lattice, left: int, right: int, surface: str
) -> tuple[int, float] | None:
    """The product on the gauges.

    The long repeated sum is the first read. When that sum misses the drop,
    the smaller count is taken one digit at a time.
    """
    found = _repeated_product(lattice, left, right, surface)
    if found is not None:
        return found
    return _grouped_product(lattice, left, right, surface)


def _see(worst: list[float], gap: float) -> None:
    if gap > worst[0]:
        worst[0] = gap


def _limb_split(number: int) -> tuple[int, ...] | None:
    """Groups of a thousand, least first. More than eight groups stay open."""
    if number < 0:
        return None
    if number == 0:
        return (0,)
    limbs: list[int] = []
    rest = number
    while rest:
        limbs.append(rest % _LIMB)
        rest //= _LIMB
        if len(limbs) > _LIMB_CAP:
            return None
    return tuple(limbs)


def _limb_value(limbs: tuple[int, ...] | list[int]) -> int:
    value = 0
    scale = 1
    for limb in limbs:
        value += limb * scale
        scale *= _LIMB
    return value


def _limb_borrow(
    lattice: Lattice, limbs: list[int], index: int, surface: str, worst: list[float]
) -> bool:
    """Move one thousand down from the next limb. A zero limb borrows first."""
    donor = index + 1
    if donor >= len(limbs):
        return False
    if limbs[donor] == 0 and not _limb_borrow(lattice, limbs, donor, surface, worst):
        return False
    found = _subtract_counts(lattice, limbs[donor], 1, surface)
    if found is None:
        return False
    limbs[donor] = found[0]
    _see(worst, found[1])
    moved = _multiply(lattice, 1, _LIMB, surface)
    if moved is None or moved[0] != _LIMB:
        return False
    _see(worst, moved[1])
    added = _add_counts(lattice, limbs[index], _LIMB, surface)
    if added is None:
        return False
    limbs[index] = added[0]
    _see(worst, added[1])
    return True


def _limb_subtract(
    lattice: Lattice,
    limbs: list[int],
    index: int,
    digit: int,
    surface: str,
    worst: list[float],
) -> bool:
    """Take this group off one limb. A short limb borrows one thousand."""
    if digit == 0:
        return True
    if digit < 0 or digit >= _LIMB or index >= len(limbs):
        return False
    if limbs[index] < digit and not _limb_borrow(lattice, limbs, index, surface, worst):
        return False
    found = _subtract_counts(lattice, limbs[index], digit, surface)
    if found is None:
        return False
    limbs[index] = found[0]
    _see(worst, found[1])
    return True


def _wide_remove(
    lattice: Lattice,
    limbs: list[int],
    part: int,
    place: int,
    surface: str,
    worst: list[float],
) -> list[int] | None:
    """Take one partial product off the limbs. The partial sits at this group."""
    if part < 0 or place < 0:
        return None
    if part == 0:
        return limbs
    rest = part
    offset = place
    while True:
        if rest < _LIMB:
            digit = rest
            rest = 0
        else:
            split = _divide(lattice, rest, _LIMB, surface)
            if split is None:
                return None
            rest, digit = split
            gap = _division_gap(lattice, digit + rest * _LIMB, _LIMB, rest, digit, surface)
            if gap is None:
                return None
            _see(worst, gap)
        if digit and not _limb_subtract(lattice, limbs, offset, digit, surface, worst):
            return None
        offset += 1
        if rest == 0:
            break
        if offset > len(limbs) + _LIMB_CAP:
            return None
    return limbs


def _wide_product(
    lattice: Lattice, left: int, right: int, surface: str
) -> tuple[tuple[int, ...], float] | None:
    """The product as groups of a thousand.

    Each group product is a gauge product. Carries are exact divisions by a
    thousand. Removing every partial product leaves zero.
    """
    if left < 0 or right < 0:
        return None
    if left == 0 or right == 0:
        if _held_integer(lattice, 0, surface) != 0:
            return None
        return (0,), 0.0
    left_limbs = _limb_split(left)
    right_limbs = _limb_split(right)
    if left_limbs is None or right_limbs is None:
        return None
    worst = [0.0]
    for limb in (*left_limbs, *right_limbs):
        if limb and _held_integer(lattice, limb, surface) != limb:
            return None
    columns = [0] * (len(left_limbs) + len(right_limbs))
    partials: list[tuple[int, int]] = []
    for index, left_limb in enumerate(left_limbs):
        if left_limb == 0:
            continue
        for offset, right_limb in enumerate(right_limbs):
            if right_limb == 0:
                continue
            part = _multiply(lattice, left_limb, right_limb, surface)
            if part is None:
                return None
            _see(worst, part[1])
            column = index + offset
            added = _add_counts(lattice, columns[column], part[0], surface)
            if added is None:
                return None
            columns[column] = added[0]
            _see(worst, added[1])
            partials.append((part[0], column))
    limbs: list[int] = []
    carry = 0
    for column in columns:
        if carry:
            added = _add_counts(lattice, column, carry, surface)
            if added is None:
                return None
            column = added[0]
            _see(worst, added[1])
        split = _divide(lattice, column, _LIMB, surface)
        if split is None:
            return None
        carry, leftover = split
        gap = _division_gap(lattice, column, _LIMB, carry, leftover, surface)
        if gap is None:
            return None
        _see(worst, gap)
        limbs.append(leftover)
    while carry:
        split = _divide(lattice, carry, _LIMB, surface)
        if split is None:
            return None
        carry, leftover = split
        gap = _division_gap(
            lattice, leftover + carry * _LIMB, _LIMB, carry, leftover, surface
        )
        if gap is None:
            return None
        _see(worst, gap)
        limbs.append(leftover)
        if len(limbs) > _LIMB_CAP * 2 + 1:
            return None
    while len(limbs) > 1 and limbs[-1] == 0:
        limbs.pop()
    if _limb_value(limbs) != left * right:
        return None
    acc = list(limbs)
    for part, place in partials:
        removed = _wide_remove(lattice, acc, part, place, surface, worst)
        if removed is None:
            return None
        acc = removed
    if any(limb != 0 for limb in acc):
        return None
    return tuple(limbs), worst[0]


def _wide_name(limbs: tuple[int, ...], surface: str) -> str:
    """Digits past the span, or the same groups in words."""
    value = _limb_value(limbs)
    if surface == "digit":
        if value < _PLACE_SPAN:
            return _whole_name(value, surface)
        return decimal_name(value)
    if value < TRILLION_CAP:
        return _whole_name(value, surface)
    words: list[str] = []
    for index in range(len(limbs) - 1, -1, -1):
        group = limbs[index]
        if group == 0:
            continue
        if index >= len(_SCALE_WORD):
            return "open"
        spelled = number_name(group)
        if spelled == "?":
            return "open"
        scale = _SCALE_WORD[index]
        words.append(spelled if not scale else f"{spelled} {scale}")
    if not words:
        return _whole_name(0, surface)
    return " ".join(words)


def _times(
    lattice: Lattice, left: int, right: int, surface: str
) -> tuple[tuple[int, ...], float] | None:
    """The product. A result inside the span keeps the place reader."""
    if left < 0 or right < 0:
        return None
    if left == 0 or right == 0:
        if _held_integer(lattice, 0, surface) != 0:
            return None
        return (0,), 0.0
    if left >= _PLACE_SPAN or right >= _PLACE_SPAN or left * right >= _PLACE_SPAN:
        return _wide_product(lattice, left, right, surface)
    found = _multiply(lattice, left, right, surface)
    if found is None:
        return None
    limbs = _limb_split(found[0])
    if limbs is None or _limb_value(limbs) != found[0]:
        return None
    return limbs, found[1]


def _take_limbs(
    lattice: Lattice, number: int, surface: str
) -> tuple[int, ...] | None:
    """Groups of a thousand, each nonzero group named on the place reader."""
    if number < 0:
        return None
    limbs = _limb_split(number)
    if limbs is None or _limb_value(limbs) != number:
        return None
    for limb in limbs:
        if limb and _held_integer(lattice, limb, surface) != limb:
            return None
    return limbs


def _wide_add(
    lattice: Lattice, left: int, right: int, surface: str
) -> tuple[tuple[int, ...], float] | None:
    """The sum, one thousand at a time. A zero addend keeps the other count."""
    if left < 0 or right < 0:
        return None
    if left == 0 or right == 0:
        held = right if left == 0 else left
        if held < _PLACE_SPAN and _held_integer(lattice, held, surface) != held:
            return None
        limbs = _take_limbs(lattice, held, surface)
        if limbs is None:
            return None
        return limbs, 0.0
    left_limbs = _take_limbs(lattice, left, surface)
    right_limbs = _take_limbs(lattice, right, surface)
    if left_limbs is None or right_limbs is None:
        return None
    worst = [0.0]
    carry = 0
    total: list[int] = []
    width = len(left_limbs) if len(left_limbs) >= len(right_limbs) else len(right_limbs)
    for index in range(width):
        part = left_limbs[index] if index < len(left_limbs) else 0
        other = right_limbs[index] if index < len(right_limbs) else 0
        if other:
            added = _add_counts(lattice, part, other, surface)
            if added is None:
                return None
            part = added[0]
            _see(worst, added[1])
        if carry:
            added = _add_counts(lattice, part, carry, surface)
            if added is None:
                return None
            part = added[0]
            _see(worst, added[1])
        split = _divide(lattice, part, _LIMB, surface)
        if split is None:
            return None
        carry, leftover = split
        gap = _division_gap(lattice, part, _LIMB, carry, leftover, surface)
        if gap is None:
            return None
        _see(worst, gap)
        total.append(leftover)
    while carry:
        split = _divide(lattice, carry, _LIMB, surface)
        if split is None:
            return None
        carry, leftover = split
        gap = _division_gap(
            lattice, leftover + carry * _LIMB, _LIMB, carry, leftover, surface
        )
        if gap is None:
            return None
        _see(worst, gap)
        total.append(leftover)
        if len(total) > _LIMB_CAP * 2 + 1:
            return None
    while len(total) > 1 and total[-1] == 0:
        total.pop()
    if _limb_value(total) != left + right:
        return None
    return tuple(total), worst[0]


def _wide_divide(
    lattice: Lattice, dividend: int, divisor: int, surface: str
) -> tuple[tuple[int, ...], int, float] | None:
    """Quotient groups and the leftover.

    A dividend below one thousand keeps the place division. A wider dividend
    is divided from the highest thousand, so each quotient digit stays inside
    one group. The leftover sits strictly below the divisor, and the quotient
    times the divisor rebuilds the dividend.
    """
    if dividend < 0 or divisor <= 0 or divisor >= _PLACE_SPAN:
        return None
    if _held_integer(lattice, divisor, surface) != divisor:
        return None
    if dividend < _LIMB:
        split = _divide(lattice, dividend, divisor, surface)
        if split is None:
            return None
        count, leftover = split
        gap = _division_gap(lattice, dividend, divisor, count, leftover, surface)
        if gap is None:
            return None
        limbs = _limb_split(count)
        if limbs is None or _limb_value(limbs) != count:
            return None
        return limbs, leftover, gap
    limbs = _take_limbs(lattice, dividend, surface)
    if limbs is None:
        return None
    worst = [0.0]
    remainder = 0
    quotients = [0] * len(limbs)
    for index in range(len(limbs) - 1, -1, -1):
        if remainder == 0:
            piece = limbs[index]
        else:
            scaled = _multiply(lattice, remainder, _LIMB, surface)
            if scaled is None:
                return None
            _see(worst, scaled[1])
            added = _add_counts(lattice, scaled[0], limbs[index], surface)
            if added is None:
                return None
            piece = added[0]
            _see(worst, added[1])
        if piece >= _PLACE_SPAN:
            return None
        split = _divide(lattice, piece, divisor, surface)
        if split is None:
            return None
        count, leftover = split
        if count >= _LIMB:
            return None
        gap = _division_gap(lattice, piece, divisor, count, leftover, surface)
        if gap is None:
            return None
        _see(worst, gap)
        quotients[index] = count
        remainder = leftover
    below = _subtract_counts(lattice, divisor, remainder, surface)
    if below is None or below[0] == 0:
        return None
    _see(worst, below[1])
    while len(quotients) > 1 and quotients[-1] == 0:
        quotients.pop()
    quotient = _limb_value(quotients)
    product = _times(lattice, quotient, divisor, surface)
    if product is None:
        return None
    _see(worst, product[1])
    rebuilt = _wide_add(lattice, _limb_value(product[0]), remainder, surface)
    if rebuilt is None or _limb_value(rebuilt[0]) != dividend:
        return None
    _see(worst, rebuilt[1])
    if quotient * divisor + remainder != dividend:
        return None
    return tuple(quotients), remainder, worst[0]


def _wide_subtract(
    lattice: Lattice, left: int, right: int, surface: str
) -> tuple[tuple[int, ...], float] | None:
    """left minus right. Adding the difference back meets the left count.

    A pair inside the span keeps the place subtraction. A wider pair takes
    each thousand off and borrows when a limb is short.
    """
    if left < 0 or right < 0 or right > left:
        return None
    if right == 0:
        if left < _PLACE_SPAN and _held_integer(lattice, left, surface) != left:
            return None
        limbs = _take_limbs(lattice, left, surface)
        if limbs is None:
            return None
        return limbs, 0.0
    if left < _PLACE_SPAN:
        found = _subtract_counts(lattice, left, right, surface)
        if found is None:
            return None
        limbs = _limb_split(found[0])
        if limbs is None or _limb_value(limbs) != found[0]:
            return None
        return limbs, found[1]
    left_limbs = _take_limbs(lattice, left, surface)
    right_limbs = _take_limbs(lattice, right, surface)
    if left_limbs is None or right_limbs is None or len(right_limbs) > len(left_limbs):
        return None
    acc = list(left_limbs)
    worst = [0.0]
    for index, digit in enumerate(right_limbs):
        if not _limb_subtract(lattice, acc, index, digit, surface, worst):
            return None
    while len(acc) > 1 and acc[-1] == 0:
        acc.pop()
    if _limb_value(acc) != left - right:
        return None
    rebuilt = _wide_add(lattice, _limb_value(acc), right, surface)
    if rebuilt is None or _limb_value(rebuilt[0]) != left:
        return None
    _see(worst, rebuilt[1])
    return tuple(acc), worst[0]


def _rest_joules(
    lattice: Lattice, mass: int, surface: str
) -> tuple[tuple[int, ...], float] | None:
    """Joules in this many kilograms. Both orders have to meet.

    One kilogram is the speed of light squared. A different mass multiplies
    that square, and multiplying by the speed twice has to name the same count.
    """
    if mass < 0:
        return None
    if _held_integer(lattice, _LIGHT_METRES, surface) != _LIGHT_METRES:
        return None
    square = _times(lattice, _LIGHT_METRES, _LIGHT_METRES, surface)
    if square is None:
        return None
    direct = _times(lattice, mass, _limb_value(square[0]), surface)
    stepped = _times(lattice, mass, _LIGHT_METRES, surface)
    if direct is None or stepped is None:
        return None
    again = _times(lattice, _limb_value(stepped[0]), _LIGHT_METRES, surface)
    if again is None or again[0] != direct[0]:
        return None
    if _limb_value(direct[0]) != mass * _LIGHT_METRES * _LIGHT_METRES:
        return None
    gap = square[1]
    if direct[1] > gap:
        gap = direct[1]
    if stepped[1] > gap:
        gap = stepped[1]
    if again[1] > gap:
        gap = again[1]
    return direct[0], gap


def ask_energy(lattice: Lattice, mass: int, surface: str = "digit") -> str:
    """Rest energy in joules of this many kilograms.

    The speed of light is the exact metre count. The mass is not a stored pair.
    """
    found = _rest_joules(lattice, mass, surface)
    if found is None:
        return "open"
    return _wide_name(found[0], surface)


def _light_factor(
    lattice: Lattice, rise: int, run: int, surface: str
) -> tuple[int, float] | None:
    """The denominator of the Lorentz factor. The numerator is the run.

    The speed is rise/run of light, strictly below light. The factor is
    exact when the rise's square and this denominator's square meet the run.
    """
    if rise < 0 or run <= 0 or rise >= run or run >= _LIMB:
        return None
    rise_part = _times(lattice, rise, rise, surface)
    run_part = _times(lattice, run, run, surface)
    if rise_part is None or run_part is None:
        return None
    diff = _subtract_counts(
        lattice, _limb_value(run_part[0]), _limb_value(rise_part[0]), surface
    )
    if diff is None:
        return None
    floor = _square_floor(lattice, diff[0], surface)
    if floor is None or not floor[1] or floor[0] <= 0:
        return None
    side = floor[0]
    if side * side != diff[0]:
        return None
    gap = rise_part[1]
    if run_part[1] > gap:
        gap = run_part[1]
    if diff[1] > gap:
        gap = diff[1]
    return side, gap


def _motion_joules(
    lattice: Lattice, mass: int, rise: int, run: int, surface: str
) -> tuple[tuple[int, ...], float] | None:
    """Joules at this fraction of light. Both products have to divide exactly.

    Rest energy times the run, divided by the Lorentz denominator, meets
    the square of the speed of light times the run and then the mass.
    """
    if mass < 0:
        return None
    factor = _light_factor(lattice, rise, run, surface)
    if factor is None:
        return None
    side, gap = factor
    rest = _rest_joules(lattice, mass, surface)
    square = _times(lattice, _LIGHT_METRES, _LIGHT_METRES, surface)
    if rest is None or square is None:
        return None
    grown = _times(lattice, _limb_value(rest[0]), run, surface)
    if grown is None:
        return None
    direct = _wide_divide(lattice, _limb_value(grown[0]), side, surface)
    if direct is None or direct[1] != 0:
        return None
    along = _times(lattice, _limb_value(square[0]), run, surface)
    if along is None:
        return None
    with_mass = _times(lattice, _limb_value(along[0]), mass, surface)
    if with_mass is None:
        return None
    other = _wide_divide(lattice, _limb_value(with_mass[0]), side, surface)
    if other is None or other[1] != 0 or other[0] != direct[0]:
        return None
    joules = _limb_value(direct[0])
    if side == 0 or mass * _LIGHT_METRES * _LIGHT_METRES * run != joules * side:
        return None
    for extra in (rest[1], square[1], grown[1], direct[2], along[1], with_mass[1], other[2]):
        if extra > gap:
            gap = extra
    return direct[0], gap


def ask_motion(
    lattice: Lattice, mass: int, rise: int, run: int, surface: str = "digit"
) -> str:
    """Joules of this mass at rise/run of the speed of light.

    A zero rise is the rest energy. The mass and the speed are not a stored pair.
    """
    found = _motion_joules(lattice, mass, rise, run, surface)
    if found is None:
        return "open"
    return _wide_name(found[0], surface)


def _kinetic_joules(
    lattice: Lattice, mass: int, rise: int, run: int, surface: str
) -> tuple[tuple[int, ...], float] | None:
    """Total energy minus rest energy.

    The same count is the rest energy times the run minus the Lorentz
    denominator, divided by that denominator.
    """
    if mass < 0:
        return None
    factor = _light_factor(lattice, rise, run, surface)
    if factor is None:
        return None
    side, gap = factor
    motion = _motion_joules(lattice, mass, rise, run, surface)
    rest = _rest_joules(lattice, mass, surface)
    if motion is None or rest is None:
        return None
    motion_count = _limb_value(motion[0])
    rest_count = _limb_value(rest[0])
    if motion_count < rest_count or run < side:
        return None
    direct = _wide_subtract(lattice, motion_count, rest_count, surface)
    if direct is None:
        return None
    extra = run - side
    grown = _times(lattice, rest_count, extra, surface)
    if grown is None:
        return None
    other = _wide_divide(lattice, _limb_value(grown[0]), side, surface)
    if other is None or other[1] != 0 or other[0] != direct[0]:
        return None
    joules = _limb_value(direct[0])
    if joules * side != rest_count * extra:
        return None
    for piece in (motion[1], rest[1], direct[1], grown[1], other[2]):
        if piece > gap:
            gap = piece
    return direct[0], gap


def ask_kinetic(
    lattice: Lattice, mass: int, rise: int, run: int, surface: str = "digit"
) -> str:
    """Kinetic joules of this mass at rise/run of the speed of light.

    A zero rise is zero. The total and the rest are not stored.
    """
    found = _kinetic_joules(lattice, mass, rise, run, surface)
    if found is None:
        return "open"
    return _wide_name(found[0], surface)


def _momentum_count(
    lattice: Lattice, mass: int, rise: int, run: int, surface: str
) -> tuple[tuple[int, ...], float] | None:
    """Mass times rise times the speed of light, over the Lorentz denominator.

    The mass-then-speed product and the speed-then-mass product both divide
    exactly and name the same count.
    """
    if mass < 0:
        return None
    factor = _light_factor(lattice, rise, run, surface)
    if factor is None:
        return None
    side, gap = factor
    if rise == 0:
        if _held_integer(lattice, 0, surface) != 0:
            return None
        return (0,), gap
    by_mass = _times(lattice, mass, rise, surface)
    by_light = _times(lattice, _LIGHT_METRES, mass, surface)
    if by_mass is None or by_light is None:
        return None
    first = _times(lattice, _limb_value(by_mass[0]), _LIGHT_METRES, surface)
    second = _times(lattice, _limb_value(by_light[0]), rise, surface)
    if first is None or second is None:
        return None
    direct = _wide_divide(lattice, _limb_value(first[0]), side, surface)
    other = _wide_divide(lattice, _limb_value(second[0]), side, surface)
    if (
        direct is None
        or other is None
        or direct[1] != 0
        or other[1] != 0
        or other[0] != direct[0]
    ):
        return None
    count = _limb_value(direct[0])
    if count * side != mass * rise * _LIGHT_METRES:
        return None
    for piece in (by_mass[1], by_light[1], first[1], second[1], direct[2], other[2]):
        if piece > gap:
            gap = piece
    return direct[0], gap


def ask_momentum(
    lattice: Lattice, mass: int, rise: int, run: int, surface: str = "digit"
) -> str:
    """Momentum of this mass at rise/run of the speed of light.

    A zero rise is zero. A count that does not divide evenly stays open.
    """
    found = _momentum_count(lattice, mass, rise, run, surface)
    if found is None:
        return "open"
    return _wide_name(found[0], surface)


def _shared_square(
    lattice: Lattice, mass: int, rise: int, run: int, surface: str
) -> tuple[tuple[int, ...], float] | None:
    """(energy / c) squared, minus momentum squared, is (mass * c) squared.

    The energy-over-light count is mass times the run times the speed,
    divided by the Lorentz denominator. Multiplying that count by the speed
    has to meet the energy. Dividing the joule count by the speed in one
    span-sized quotient is not the read.
    """
    if mass < 0:
        return None
    factor = _light_factor(lattice, rise, run, surface)
    motion = _motion_joules(lattice, mass, rise, run, surface)
    momentum = _momentum_count(lattice, mass, rise, run, surface)
    if factor is None or motion is None or momentum is None:
        return None
    side = factor[0]
    by_mass = _times(lattice, mass, _LIGHT_METRES, surface)
    if by_mass is None:
        return None
    grown = _times(lattice, _limb_value(by_mass[0]), run, surface)
    if grown is None:
        return None
    per_light = _wide_divide(lattice, _limb_value(grown[0]), side, surface)
    if per_light is None or per_light[1] != 0:
        return None
    energy_over = _limb_value(per_light[0])
    restored = _times(lattice, energy_over, _LIGHT_METRES, surface)
    if restored is None or restored[0] != motion[0]:
        return None
    moved = _limb_value(momentum[0])
    mass_light = mass * _LIGHT_METRES
    left = _times(lattice, energy_over, energy_over, surface)
    right = _times(lattice, moved, moved, surface)
    goal = _times(lattice, mass_light, mass_light, surface)
    if left is None or right is None or goal is None:
        return None
    left_count = _limb_value(left[0])
    right_count = _limb_value(right[0])
    goal_count = _limb_value(goal[0])
    if left_count < right_count:
        return None
    diff = _wide_subtract(lattice, left_count, right_count, surface)
    if diff is None or _limb_value(diff[0]) != goal_count:
        return None
    rebuilt = _wide_add(lattice, goal_count, right_count, surface)
    if rebuilt is None or _limb_value(rebuilt[0]) != left_count:
        return None
    if energy_over * energy_over - moved * moved != mass_light * mass_light:
        return None
    gap = factor[1]
    for piece in (
        motion[1],
        momentum[1],
        by_mass[1],
        grown[1],
        per_light[2],
        restored[1],
        left[1],
        right[1],
        goal[1],
        diff[1],
        rebuilt[1],
    ):
        if piece > gap:
            gap = piece
    return goal[0], gap


def ask_balance(
    lattice: Lattice, mass: int, rise: int, run: int, surface: str = "digit"
) -> str:
    """The square of mass times the speed of light, rebuilt from energy and momentum.

    Energy over the speed of light, squared, minus the momentum squared,
    names this count. A momentum that stays open keeps the balance open.
    """
    found = _shared_square(lattice, mass, rise, run, surface)
    if found is None:
        return "open"
    return _wide_name(found[0], surface)


def _power(lattice: Lattice, base: int, exponent: int, surface: str) -> int | None:
    """The base multiplied onto itself. Zero to the zero stays open."""
    if base < 0 or exponent < 0:
        return None
    if exponent == 0:
        if base == 0:
            return None
        return _held_integer(lattice, 1, surface)
    if base == 0:
        return _held_integer(lattice, 0, surface)
    if exponent == 1:
        return _held_integer(lattice, base, surface)
    value = base
    for _step in range(exponent - 1):
        found = _multiply(lattice, value, base, surface)
        if found is None:
            return None
        value = found[0]
    if value != base**exponent:
        return None
    return value


def _power_residue(
    lattice: Lattice, base: int, exponent: int, modulus: int, surface: str
) -> int | None:
    """The leftover of a power. Each factor is reduced before the next product."""
    if base < 0 or exponent < 0 or modulus <= 0:
        return None
    if exponent == 0:
        if base == 0:
            return None
        return _residue(lattice, 1, modulus, surface)
    acc = _residue(lattice, 1, modulus, surface)
    factor = _residue(lattice, base, modulus, surface)
    if acc is None or factor is None:
        return None
    for _step in range(exponent):
        found = _multiply(lattice, acc, factor, surface)
        if found is None:
            return None
        acc = _residue(lattice, found[0], modulus, surface)
        if acc is None:
            return None
    if acc != pow(base, exponent, modulus):
        return None
    return acc


def _totient(lattice: Lattice, modulus: int, surface: str) -> int | None:
    """How many residues share the common measure one with the modulus."""
    if modulus <= 0 or modulus >= _PLACE_SPAN:
        return None
    count = 0
    for candidate in range(modulus):
        measure = _common(lattice, candidate, modulus, surface)
        if measure is None:
            return None
        if measure == 1:
            count += 1
    if _held_integer(lattice, count, surface) != count:
        return None
    return count


def _order(lattice: Lattice, number: int, modulus: int, surface: str) -> int | None:
    """The least positive power that leaves one.

    The count and the modulus share measure one. The exponent has to divide
    the totient.
    """
    if number < 0 or modulus <= 1:
        return None
    if _common(lattice, number, modulus, surface) != 1:
        return None
    totient = _totient(lattice, modulus, surface)
    if totient is None or totient <= 0:
        return None
    acc = _residue(lattice, 1, modulus, surface)
    factor = _residue(lattice, number, modulus, surface)
    if acc is None or factor is None:
        return None
    for exponent in range(1, totient + 1):
        found = _multiply(lattice, acc, factor, surface)
        if found is None:
            return None
        acc = _residue(lattice, found[0], modulus, surface)
        if acc is None:
            return None
        if acc != 1:
            continue
        split = _divide(lattice, totient, exponent, surface)
        if split is None or split[1] != 0:
            return None
        return exponent
    return None


def ask_power(lattice: Lattice, base: int, exponent: int, surface: str = "digit") -> str:
    """The power of a count. Zero to the zero stays open."""
    found = _power(lattice, base, exponent, surface)
    if found is None:
        return "open"
    return _whole_name(found, surface)


def ask_power_remainder(
    lattice: Lattice, base: int, exponent: int, modulus: int, surface: str = "digit"
) -> str:
    """The leftover of a power grouped by the modulus."""
    found = _power_residue(lattice, base, exponent, modulus, surface)
    if found is None:
        return "open"
    return _whole_name(found, surface)


def ask_totient(lattice: Lattice, modulus: int, surface: str = "digit") -> str:
    """How many residues have an inverse under the modulus."""
    found = _totient(lattice, modulus, surface)
    if found is None:
        return "open"
    return _whole_name(found, surface)


def ask_order(
    lattice: Lattice, number: int, modulus: int, surface: str = "digit"
) -> str:
    """The least power of this count that leaves one."""
    found = _order(lattice, number, modulus, surface)
    if found is None:
        return "open"
    return _whole_name(found, surface)


def ask_euler(
    lattice: Lattice, number: int, modulus: int, surface: str = "digit"
) -> str:
    """Yes when the totient power leaves the same leftover as one.

    A count whose common measure with the modulus is wider than one stays open.
    """
    if number < 0 or modulus <= 0:
        return "open"
    measure = _common(lattice, number, modulus, surface)
    if measure is None:
        return "open"
    if measure != 1:
        return "open"
    totient = _totient(lattice, modulus, surface)
    if totient is None:
        return "open"
    got = _power_residue(lattice, number, totient, modulus, surface)
    want = _residue(lattice, 1, modulus, surface)
    if got is None or want is None:
        return "open"
    if got == want:
        return "yes"
    return "no"


def _log(lattice: Lattice, base: int, target: int, modulus: int, surface: str) -> int | None:
    """The least power of the base that leaves the target.

    The zero power is allowed when the base is positive, because that power
    leaves one. Zero to the zero stays open, so a zero base begins at one.
    """
    if base < 0 or target < 0 or modulus <= 0 or modulus >= _PLACE_SPAN:
        return None
    reduced = _residue(lattice, target, modulus, surface)
    factor = _residue(lattice, base, modulus, surface)
    if reduced is None or factor is None:
        return None
    if base == 0:
        if reduced != 0:
            return None
        if _power_residue(lattice, 0, 1, modulus, surface) != 0:
            return None
        return 1
    one = _residue(lattice, 1, modulus, surface)
    if one is None:
        return None
    if one == reduced:
        if _power_residue(lattice, base, 0, modulus, surface) != reduced:
            return None
        return 0
    if factor == 0:
        if reduced != 0:
            return None
        if _power_residue(lattice, base, 1, modulus, surface) != 0:
            return None
        return 1
    acc = one
    for exponent in range(1, modulus + 1):
        found = _multiply(lattice, acc, factor, surface)
        if found is None:
            return None
        acc = _residue(lattice, found[0], modulus, surface)
        if acc is None:
            return None
        if acc != reduced:
            continue
        if _power_residue(lattice, base, exponent, modulus, surface) != reduced:
            return None
        return exponent
    return None


def _square(lattice: Lattice, number: int, modulus: int, surface: str) -> bool | None:
    """Whether some square leaves this leftover.

    On an odd prime the power of half of one-less-than-the-prime has to
    agree: one when a square exists, one less than the prime when it does
    not, and zero when the prime divides the leftover.
    """
    if number < 0 or modulus <= 0 or modulus >= _PLACE_SPAN:
        return None
    target = _residue(lattice, number, modulus, surface)
    if target is None:
        return None
    found = False
    for candidate in range(modulus):
        product = _multiply(lattice, candidate, candidate, surface)
        if product is None:
            return None
        got = _residue(lattice, product[0], modulus, surface)
        if got is None:
            return None
        if got == target:
            found = True
            break
    if ask_prime(lattice, modulus, surface) != "yes" or modulus <= 2:
        return found
    split = _divide(lattice, modulus - 1, 2, surface)
    if split is None or split[1] != 0:
        return None
    symbol = _power_residue(lattice, target, split[0], modulus, surface)
    if symbol is None:
        return None
    if symbol == 0:
        if target == 0 and found:
            return True
        return None
    if symbol == 1:
        if found:
            return True
        return None
    if symbol == modulus - 1:
        if found:
            return None
        return False
    return None


def _choose(lattice: Lattice, total: int, pick: int, surface: str) -> int | None:
    """How many ways to choose. Each growing product divides exactly."""
    if total < 0 or pick < 0 or pick > total or total >= _PLACE_SPAN:
        return None
    if pick > total - pick:
        pick = total - pick
    if pick == 0:
        return _held_integer(lattice, 1, surface)
    value = 1
    expect = 1
    for index in range(1, pick + 1):
        factor = total - pick + index
        found = _multiply(lattice, value, factor, surface)
        if found is None:
            return None
        split = _divide(lattice, found[0], index, surface)
        if split is None or split[1] != 0:
            return None
        value = split[0]
        expect = expect * factor // index
        if value != expect:
            return None
    return value


def ask_primitive(
    lattice: Lattice, number: int, modulus: int, surface: str = "digit"
) -> str:
    """Yes when the order of this count is the totient."""
    if number < 0 or modulus <= 1:
        return "open"
    order = _order(lattice, number, modulus, surface)
    if order is None:
        return "open"
    totient = _totient(lattice, modulus, surface)
    if totient is None:
        return "open"
    if order == totient:
        return "yes"
    return "no"


def ask_log(
    lattice: Lattice, base: int, target: int, modulus: int, surface: str = "digit"
) -> str:
    """The least power of the base that leaves the target leftover."""
    found = _log(lattice, base, target, modulus, surface)
    if found is None:
        return "open"
    return _whole_name(found, surface)


def ask_square(
    lattice: Lattice, number: int, modulus: int, surface: str = "digit"
) -> str:
    """Yes when some square leaves this leftover under the modulus."""
    found = _square(lattice, number, modulus, surface)
    if found is None:
        return "open"
    if found:
        return "yes"
    return "no"


def ask_choose(lattice: Lattice, total: int, pick: int, surface: str = "digit") -> str:
    """How many ways to choose the pick from the total."""
    found = _choose(lattice, total, pick, surface)
    if found is None:
        return "open"
    return _whole_name(found, surface)


def _add_counts(
    lattice: Lattice, left: int, right: int, surface: str
) -> tuple[int, float] | None:
    """The sum, and how far the consensus sits from the composed total."""
    if left < 0 or right < 0 or left >= _PLACE_SPAN or right >= _PLACE_SPAN:
        return None
    total_count = left + right
    if total_count >= _PLACE_SPAN:
        return None
    if left == 0:
        if _held_integer(lattice, right, surface) != right:
            return None
        return right, 0.0
    if right == 0:
        if _held_integer(lattice, left, surface) != left:
            return None
        return left, 0.0
    steps = lattice.place_steps(surface)
    gauges = lattice.gauge_line(surface)
    total = lattice.consensus_quantity(
        lattice.compose_integer(left, steps, gauges),
        lattice.compose_integer(right, steps, gauges),
        lattice.routed_sign("+"),
    )
    named, _residual, _margin, dist = lattice.read_place(total, surface)
    goal = lattice.compose_integer(total_count, steps, gauges)
    gap = abs(lattice.consensus_quantity(total, goal, lattice.routed_sign("-")))
    if dist >= DROP or gap >= DROP or named != total_count:
        return None
    return named, gap


def _scale_add(
    lattice: Lattice, coeff: int, current: int, previous: int, surface: str
) -> int | None:
    """coeff × current + previous. A zero factor keeps the previous count."""
    if coeff < 0 or current < 0 or previous < 0:
        return None
    if coeff == 0 or current == 0:
        product = 0
    else:
        found = _multiply(lattice, coeff, current, surface)
        if found is None:
            return None
        product = found[0]
    if previous == 0:
        if _held_integer(lattice, product, surface) != product:
            return None
        return product
    added = _add_counts(lattice, product, previous, surface)
    if added is None:
        return None
    return added[0]


def _continued(
    lattice: Lattice, left: int, right: int, surface: str
) -> tuple[int, ...] | None:
    """Euclid's quotients. The last convergent meets the original counts.

    Each quotient is a divisor step. Rebuilding the convergent multiplies
    and adds on the gauges, and the two cross products have to agree.
    """
    if left < 0 or right <= 0 or left >= _PLACE_SPAN or right >= _PLACE_SPAN:
        return None
    quotients: list[int] = []
    current, divisor = left, right
    for _step in range(64):
        if divisor == 0:
            break
        split = _divide(lattice, current, divisor, surface)
        if split is None:
            return None
        count, leftover = split
        quotients.append(count)
        current, divisor = divisor, leftover
    else:
        return None
    if not quotients:
        return None
    prev_h, curr_h = 0, 1
    prev_k, curr_k = 1, 0
    for quotient in quotients:
        next_h = _scale_add(lattice, quotient, curr_h, prev_h, surface)
        next_k = _scale_add(lattice, quotient, curr_k, prev_k, surface)
        if next_h is None or next_k is None:
            return None
        prev_h, curr_h = curr_h, next_h
        prev_k, curr_k = curr_k, next_k
    if curr_k <= 0 or _common(lattice, curr_h, curr_k, surface) != 1:
        return None
    if curr_h == 0:
        cross_left = 0
    else:
        cross = _multiply(lattice, curr_h, right, surface)
        if cross is None:
            return None
        cross_left = cross[0]
    if left == 0:
        cross_right = 0
    else:
        cross = _multiply(lattice, curr_k, left, surface)
        if cross is None:
            return None
        cross_right = cross[0]
    if cross_left != cross_right or cross_left != curr_h * right:
        return None
    return tuple(quotients)


def _continued_name(quotients: tuple[int, ...], surface: str) -> str:
    names = [_whole_name(quotient, surface) for quotient in quotients]
    if any(name == "open" for name in names):
        return "open"
    if len(names) == 1:
        return names[0]
    return names[0] + "; " + ", ".join(names[1:])


def _square_roots(
    lattice: Lattice, number: int, modulus: int, surface: str
) -> tuple[int, ...] | None:
    """The residues whose square leaves this leftover. None when there are none."""
    if number < 0 or modulus <= 0 or modulus >= _PLACE_SPAN:
        return None
    square = _square(lattice, number, modulus, surface)
    if square is None:
        return None
    target = _residue(lattice, number, modulus, surface)
    if target is None:
        return None
    roots: list[int] = []
    for candidate in range(modulus):
        product = _multiply(lattice, candidate, candidate, surface)
        if product is None:
            return None
        got = _residue(lattice, product[0], modulus, surface)
        if got is None:
            return None
        if got == target:
            roots.append(candidate)
    if square != bool(roots) or len(roots) > 64:
        return None
    if not roots:
        return None
    return tuple(roots)


def _two_squares(
    lattice: Lattice, number: int, surface: str
) -> tuple[tuple[int, int], ...] | None:
    """Pairs of counts, smaller first, whose squares sum to the number.

    An odd prime that leaves 1 when grouped by 4 has a pair. An odd prime
    that leaves 3 does not. Two itself is 1 and 1.
    """
    if number < 0 or number >= _PLACE_SPAN:
        return None
    pairs: list[tuple[int, int]] = []
    for left in range(number + 1):
        left_square = _multiply(lattice, left, left, surface)
        if left_square is None:
            return None
        if left_square[0] > number:
            break
        rest = number - left_square[0]
        for right in range(left, number + 1):
            right_square = _multiply(lattice, right, right, surface)
            if right_square is None:
                return None
            if right_square[0] > rest:
                break
            if right_square[0] != rest:
                continue
            added = _add_counts(lattice, left_square[0], right_square[0], surface)
            if added is None or added[0] != number:
                return None
            pairs.append((left, right))
            break
    prime = ask_prime(lattice, number, surface)
    if prime == "yes" and number > 2:
        residue = _residue(lattice, number, 4, surface)
        if residue == 1 and not pairs:
            return None
        if residue == 3 and pairs:
            return None
        if residue not in (1, 3):
            return None
    if number == 2 and pairs != [(1, 1)]:
        return None
    if not pairs or len(pairs) > 64:
        return None
    return tuple(pairs)


def _permute(lattice: Lattice, total: int, pick: int, surface: str) -> int | None:
    """The falling product. It has to equal the choice times the pick factorial."""
    if total < 0 or pick < 0 or pick > total or total >= _PLACE_SPAN:
        return None
    if pick == 0:
        return _held_integer(lattice, 1, surface)
    value = 1
    expect = 1
    for index in range(pick):
        factor = total - index
        found = _multiply(lattice, value, factor, surface)
        if found is None:
            return None
        value = found[0]
        expect *= factor
        if value != expect:
            return None
    chosen = _choose(lattice, total, pick, surface)
    if chosen is None:
        return None
    factorial = 1
    for index in range(1, pick + 1):
        found = _multiply(lattice, factorial, index, surface)
        if found is None:
            return None
        factorial = found[0]
    product = _multiply(lattice, chosen, factorial, surface)
    if product is None or product[0] != value:
        return None
    return value


def _pair_list(pairs: tuple[tuple[int, int], ...], surface: str) -> str:
    parts: list[str] = []
    for left, right in pairs:
        first = _whole_name(left, surface)
        second = _whole_name(right, surface)
        if first == "open" or second == "open":
            return "open"
        parts.append(f"{first} and {second}")
    return " or ".join(parts)


def ask_continued(
    lattice: Lattice, left: int, right: int, surface: str = "digit"
) -> str:
    """The Euclid quotients of the two counts. The first stands before the rest."""
    found = _continued(lattice, left, right, surface)
    if found is None:
        return "open"
    return _continued_name(found, surface)


def ask_square_root(
    lattice: Lattice, number: int, modulus: int, surface: str = "digit"
) -> str:
    """The counts whose square leaves this leftover. No square stays open."""
    found = _square_roots(lattice, number, modulus, surface)
    if found is None:
        return "open"
    return _counts_answer(found, surface)


def ask_two_squares(lattice: Lattice, number: int, surface: str = "digit") -> str:
    """Pairs whose squares sum to the count. The smaller count in each pair is first."""
    found = _two_squares(lattice, number, surface)
    if found is None:
        return "open"
    return _pair_list(found, surface)


def ask_permute(lattice: Lattice, total: int, pick: int, surface: str = "digit") -> str:
    """How many ordered ways to choose the pick from the total."""
    found = _permute(lattice, total, pick, surface)
    if found is None:
        return "open"
    return _whole_name(found, surface)


def _product_count(
    lattice: Lattice, left: int, right: int, surface: str
) -> int | None:
    """The product on the gauges. A zero factor is the held zero."""
    if left < 0 or right < 0:
        return None
    if left == 0 or right == 0:
        if _held_integer(lattice, 0, surface) != 0:
            return None
        return 0
    found = _multiply(lattice, left, right, surface)
    if found is None:
        return None
    return found[0]


def _apart_by_one(lattice: Lattice, smaller: int, larger: int, surface: str) -> bool:
    """The larger count is the smaller one plus one."""
    if smaller < 0 or larger < 0 or larger != smaller + 1:
        return False
    if smaller == 0:
        return _held_integer(lattice, larger, surface) == larger
    added = _add_counts(lattice, smaller, 1, surface)
    return added is not None and added[0] == larger


def _extend_pair(lattice: Lattice, series: list[int], surface: str) -> int | None:
    """The next count is the sum of the two before it."""
    if len(series) < 2:
        return None
    added = _add_counts(lattice, series[-1], series[-2], surface)
    if added is None or added[0] != series[-1] + series[-2]:
        return None
    return added[0]


def _pairs_from_quotients(
    lattice: Lattice, quotients: tuple[int, ...] | list[int], surface: str
) -> tuple[tuple[int, int], ...] | None:
    """The convergents of these quotients.

    Neighboring cross products differ by one, and each pair shares the
    measure one.
    """
    if not quotients:
        return None
    prev_h, curr_h = 0, 1
    prev_k, curr_k = 1, 0
    pairs: list[tuple[int, int]] = []
    for index, quotient in enumerate(quotients):
        next_h = _scale_add(lattice, quotient, curr_h, prev_h, surface)
        next_k = _scale_add(lattice, quotient, curr_k, prev_k, surface)
        if next_h is None or next_k is None or next_k <= 0:
            return None
        left_prod = _product_count(lattice, next_h, curr_k, surface)
        right_prod = _product_count(lattice, curr_h, next_k, surface)
        if left_prod is None or right_prod is None:
            return None
        if index % 2 == 0:
            smaller, larger = left_prod, right_prod
        else:
            smaller, larger = right_prod, left_prod
        if not _apart_by_one(lattice, smaller, larger, surface):
            return None
        if _common(lattice, next_h, next_k, surface) != 1:
            return None
        pairs.append((next_h, next_k))
        prev_h, curr_h = curr_h, next_h
        prev_k, curr_k = curr_k, next_k
    if not pairs:
        return None
    return tuple(pairs)


def _convergents(
    lattice: Lattice, left: int, right: int, surface: str
) -> tuple[tuple[int, int], ...] | None:
    """Each convergent the Euclid quotients build.

    The cross product of one convergent with the next differs by one, and
    each pair shares the measure one. The last pair meets the original counts.
    """
    if left < 0 or right <= 0 or left >= _PLACE_SPAN or right >= _PLACE_SPAN:
        return None
    quotients: list[int] = []
    current, divisor = left, right
    for _step in range(64):
        if divisor == 0:
            break
        split = _divide(lattice, current, divisor, surface)
        if split is None:
            return None
        count, leftover = split
        quotients.append(count)
        current, divisor = divisor, leftover
    else:
        return None
    if not quotients:
        return None
    pairs = _pairs_from_quotients(lattice, quotients, surface)
    if pairs is None:
        return None
    curr_h, curr_k = pairs[-1]
    if curr_h == 0:
        cross_left = 0
    else:
        cross = _multiply(lattice, curr_h, right, surface)
        if cross is None:
            return None
        cross_left = cross[0]
    if left == 0:
        cross_right = 0
    else:
        cross = _multiply(lattice, curr_k, left, surface)
        if cross is None:
            return None
        cross_right = cross[0]
    if cross_left != cross_right or cross_left != curr_h * right:
        return None
    if not pairs:
        return None
    return tuple(pairs)


def _convergent_name(pairs: tuple[tuple[int, int], ...], surface: str) -> str:
    parts: list[str] = []
    for numerator, denominator in pairs:
        top = _whole_name(numerator, surface)
        bottom = _whole_name(denominator, surface)
        if top == "open" or bottom == "open":
            return "open"
        parts.append(f"{top}/{bottom}")
    return ", ".join(parts)


def ask_convergents(
    lattice: Lattice, left: int, right: int, surface: str = "digit"
) -> str:
    """The convergents of the two counts, each numerator over its denominator."""
    found = _convergents(lattice, left, right, surface)
    if found is None:
        return "open"
    return _convergent_name(found, surface)


def _series(
    lattice: Lattice, index: int, surface: str
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """Fibonacci and Lucas counts through this index.

    Each later count is the sum of the two before it. Neighbors share the
    measure one. From index two on, the square of a Fibonacci count and the
    product of the counts on either side differ by one. From index one on,
    the Lucas count is the sum of the Fibonacci neighbor on each side. The
    sum of the Fibonacci counts through this index, plus one, is the count
    two places ahead. From index two on, the continued fraction of a count
    over the one before it is a run of ones that ends in two.
    """
    if index < 0 or index > 32:
        return None
    if _held_integer(lattice, 0, surface) != 0:
        return None
    if _held_integer(lattice, 1, surface) != 1:
        return None
    if _held_integer(lattice, 2, surface) != 2:
        return None
    if index == 0:
        return (0,), (2,)
    if _common(lattice, 1, 0, surface) != 1:
        return None
    fib: list[int] = [0, 1]
    while len(fib) <= index:
        nxt = _extend_pair(lattice, fib, surface)
        if nxt is None:
            return None
        step = len(fib)
        if _common(lattice, nxt, fib[-1], surface) != 1:
            return None
        prod = _product_count(lattice, nxt, fib[-2], surface)
        square = _product_count(lattice, fib[-1], fib[-1], surface)
        if prod is None or square is None:
            return None
        if step % 2 == 0:
            smaller, larger = prod, square
        else:
            smaller, larger = square, prod
        if not _apart_by_one(lattice, smaller, larger, surface):
            return None
        fib.append(nxt)
    if index >= 2:
        quotients = _continued(lattice, fib[index], fib[index - 1], surface)
        if index == 2:
            expect: tuple[int, ...] = (1,)
        elif index == 3:
            expect = (2,)
        else:
            expect = (1,) + (1,) * (index - 4) + (2,)
        if quotients != expect:
            return None
    while len(fib) <= index + 2:
        nxt = _extend_pair(lattice, fib, surface)
        if nxt is None:
            return None
        fib.append(nxt)
    lucas: list[int] = [2, 1]
    for step in range(1, index + 1):
        if step >= 2:
            nxt = _extend_pair(lattice, lucas, surface)
            if nxt is None:
                return None
            lucas.append(nxt)
        joined = _add_counts(lattice, fib[step - 1], fib[step + 1], surface)
        if joined is None or joined[0] != lucas[step]:
            return None
    total = 0
    for value in fib[1 : index + 1]:
        if total == 0:
            if _held_integer(lattice, value, surface) != value:
                return None
            total = value
            continue
        added = _add_counts(lattice, total, value, surface)
        if added is None:
            return None
        total = added[0]
    closed = _add_counts(lattice, total, 1, surface)
    if closed is None or closed[0] != fib[index + 2]:
        return None
    return tuple(fib[: index + 1]), tuple(lucas[: index + 1])


def ask_fibonacci(lattice: Lattice, index: int, surface: str = "digit") -> str:
    """The Fibonacci count at this index. The count begins at zero, then one."""
    found = _series(lattice, index, surface)
    if found is None:
        return "open"
    return _whole_name(found[0][index], surface)


def ask_lucas(lattice: Lattice, index: int, surface: str = "digit") -> str:
    """The Lucas count at this index. The count begins at two, then one."""
    found = _series(lattice, index, surface)
    if found is None:
        return "open"
    return _whole_name(found[1][index], surface)


def ask_fibonacci_common(
    lattice: Lattice, left: int, right: int, surface: str = "digit"
) -> str:
    """The common measure of two Fibonacci counts.

    It is the Fibonacci count at the common measure of the two indices.
    """
    if left < 0 or right < 0 or (left == 0 and right == 0):
        return "open"
    span = left if left >= right else right
    found = _series(lattice, span, surface)
    if found is None:
        return "open"
    fib = found[0]
    measure = _common(lattice, left, right, surface)
    if measure is None or measure > span:
        return "open"
    shared = _common(lattice, fib[left], fib[right], surface)
    if shared is None or shared != fib[measure]:
        return "open"
    return _whole_name(shared, surface)


def _subtract_counts(
    lattice: Lattice, left: int, right: int, surface: str
) -> tuple[int, float] | None:
    """left minus right, when the result stays at or above zero.

    Adding the result back to the right count has to meet the left count.
    """
    if left < 0 or right < 0 or right > left or left >= _PLACE_SPAN:
        return None
    diff = left - right
    if right == 0:
        if _held_integer(lattice, left, surface) != left:
            return None
        return left, 0.0
    if diff == 0:
        if _held_integer(lattice, 0, surface) != 0:
            return None
        return 0, 0.0
    steps = lattice.place_steps(surface)
    gauges = lattice.gauge_line(surface)
    total = lattice.consensus_quantity(
        lattice.compose_integer(left, steps, gauges),
        lattice.compose_integer(right, steps, gauges),
        lattice.routed_sign("-"),
    )
    named, _residual, _margin, dist = lattice.read_place(total, surface)
    goal = lattice.compose_integer(diff, steps, gauges)
    gap = abs(lattice.consensus_quantity(total, goal, lattice.routed_sign("-")))
    if dist >= DROP or gap >= DROP or named != diff:
        return None
    added = _add_counts(lattice, named, right, surface)
    if added is None or added[0] != left:
        return None
    return named, gap


def _bounded_square(lattice: Lattice, root: int, surface: str) -> int | None:
    """The square, when the product still fits on the place reader."""
    if root < 0 or root >= _PLACE_SPAN:
        return None
    if root > 1 and root > (_PLACE_SPAN - 1) // root:
        return None
    return _product_count(lattice, root, root, surface)


def _square_floor(lattice: Lattice, number: int, surface: str) -> tuple[int, bool] | None:
    """The greatest count whose square does not exceed the number.

    The flag is true when that square meets the number.
    """
    if number < 0 or number >= _PLACE_SPAN:
        return None
    if _held_integer(lattice, 0, surface) != 0:
        return None
    if number == 0:
        return 0, True
    one = _bounded_square(lattice, 1, surface)
    if one is None:
        return None
    if one == number:
        return 1, True
    if one > number:
        return None
    best = 1
    best_square = one
    root = 1
    over: int | None = None
    while root < _PLACE_SPAN:
        if root > (_PLACE_SPAN - 1) // 2:
            break
        root *= 2
        square = _bounded_square(lattice, root, surface)
        if square is None:
            over = root
            break
        if square == number:
            return root, True
        if square > number:
            over = root
            break
        best = root
        best_square = square
    left = best + 1
    right = over - 1 if over is not None else best
    while left <= right:
        mid = (left + right) // 2
        square = _bounded_square(lattice, mid, surface)
        if square is None or square > number:
            right = mid - 1
            continue
        best = mid
        best_square = square
        if square == number:
            return mid, True
        left = mid + 1
    return best, best_square == number


def _root_step(
    lattice: Lattice,
    number: int,
    head: int,
    state: int,
    span: int,
    quotient: int,
    surface: str,
) -> tuple[int, int, int] | None:
    """The next state of a whole square root's continued fraction."""
    product = _product_count(lattice, quotient, span, surface)
    if product is None:
        return None
    nxt_state = _subtract_counts(lattice, product, state, surface)
    if nxt_state is None:
        return None
    state = nxt_state[0]
    square = _bounded_square(lattice, state, surface)
    if square is None:
        return None
    numer = _subtract_counts(lattice, number, square, surface)
    if numer is None:
        return None
    split = _divide(lattice, numer[0], span, surface)
    if split is None or split[1] != 0 or split[0] <= 0:
        return None
    span = split[0]
    summed = _add_counts(lattice, head, state, surface)
    if summed is None:
        return None
    split = _divide(lattice, summed[0], span, surface)
    if split is None or split[0] <= 0:
        return None
    return state, span, split[0]


def _root_fraction(
    lattice: Lattice, number: int, surface: str
) -> tuple[int, tuple[int, ...]] | None:
    """The head and one period of the square root.

    A perfect square has an empty period. A repeating tail ends when the
    quotient is twice the head, and the next quotient has to be the first
    of that tail.
    """
    found = _square_floor(lattice, number, surface)
    if found is None:
        return None
    head, exact = found
    if exact:
        return head, ()
    twice = _add_counts(lattice, head, head, surface)
    if twice is None:
        return None
    state, span, quotient = 0, 1, head
    period: list[int] = []
    for _step in range(64):
        stepped = _root_step(lattice, number, head, state, span, quotient, surface)
        if stepped is None:
            return None
        state, span, quotient = stepped
        period.append(quotient)
        if quotient == twice[0]:
            break
    else:
        return None
    if not period or period[-1] != twice[0]:
        return None
    again = _root_step(lattice, number, head, state, span, quotient, surface)
    if again is None or again[2] != period[0]:
        return None
    return head, tuple(period)


def _root_fraction_name(head: int, period: tuple[int, ...], surface: str) -> str:
    head_name = _whole_name(head, surface)
    if head_name == "open":
        return "open"
    if not period:
        return head_name
    rest = [_whole_name(quotient, surface) for quotient in period]
    if any(name == "open" for name in rest):
        return "open"
    return head_name + "; [" + ", ".join(rest) + "]"


def _root_terms(head: int, period: tuple[int, ...], index: int) -> tuple[int, ...] | None:
    """Quotients from the head through this index of the repeating tail."""
    if index < 0 or not period:
        return None
    terms = [head]
    while len(terms) <= index:
        terms.append(period[(len(terms) - 1) % len(period)])
    return tuple(terms)


def _square_gap(
    lattice: Lattice, left: int, right: int, number: int, surface: str
) -> int | None:
    """left squared, minus number times right squared."""
    if left < 0 or right < 0 or number < 0:
        return None
    left_square = _bounded_square(lattice, left, surface)
    right_square = _bounded_square(lattice, right, surface)
    if left_square is None or right_square is None:
        return None
    scaled = _product_count(lattice, number, right_square, surface)
    if scaled is None:
        return None
    if left_square == scaled:
        return 0
    if left_square > scaled:
        diff = left_square - scaled
        if diff == 1:
            if not _apart_by_one(lattice, scaled, left_square, surface):
                return None
            return 1
        added = _add_counts(lattice, scaled, diff, surface)
        if added is None or added[0] != left_square:
            return None
        return diff
    diff = scaled - left_square
    if diff == 1:
        if not _apart_by_one(lattice, left_square, scaled, surface):
            return None
        return -1
    added = _add_counts(lattice, left_square, diff, surface)
    if added is None or added[0] != scaled:
        return None
    return -diff


def _pell_solution(
    lattice: Lattice, number: int, target: int, surface: str
) -> tuple[int, int] | None:
    """The least positive pair with left squared minus number times right squared equal to the target.

    The target is one, or minus one. An even period has no minus-one pair.
    An odd period meets minus one after one period and one after two.
    An even period meets one after a single period. No earlier convergent
    meets that same target.
    """
    if target not in (1, -1):
        return None
    fraction = _root_fraction(lattice, number, surface)
    if fraction is None:
        return None
    head, period = fraction
    if not period:
        return None
    length = len(period)
    if target == -1 and length % 2 == 0:
        return None
    if target == 1 and length % 2 == 0:
        index = length - 1
    else:
        index = length - 1 if target == -1 else 2 * length - 1
    terms = _root_terms(head, period, index)
    if terms is None:
        return None
    pairs = _pairs_from_quotients(lattice, terms, surface)
    if pairs is None or len(pairs) != index + 1:
        return None
    for earlier, _denom in pairs[:-1]:
        gap = _square_gap(lattice, earlier, _denom, number, surface)
        if gap is None or gap == target:
            return None
    left, right = pairs[-1]
    if _square_gap(lattice, left, right, number, surface) != target:
        return None
    if left <= 0 or right <= 0:
        return None
    return left, right


def _joined_pair(left: int, right: int, surface: str) -> str:
    first = _whole_name(left, surface)
    second = _whole_name(right, surface)
    if first == "open" or second == "open":
        return "open"
    return f"{first} and {second}"


def ask_floor_root(lattice: Lattice, number: int, surface: str = "digit") -> str:
    """The greatest count whose square does not exceed this number."""
    found = _square_floor(lattice, number, surface)
    if found is None:
        return "open"
    return _whole_name(found[0], surface)


def ask_square_count(lattice: Lattice, number: int, surface: str = "digit") -> str:
    """The count whose square is this number. A non-square stays open."""
    found = _square_floor(lattice, number, surface)
    if found is None or not found[1]:
        return "open"
    return _whole_name(found[0], surface)


def ask_root_fraction(lattice: Lattice, number: int, surface: str = "digit") -> str:
    """One period of the square root. A square names its own count."""
    found = _root_fraction(lattice, number, surface)
    if found is None:
        return "open"
    return _root_fraction_name(found[0], found[1], surface)


def ask_pell(lattice: Lattice, number: int, surface: str = "digit") -> str:
    """The least pair whose first square is one above this count times the second."""
    found = _pell_solution(lattice, number, 1, surface)
    if found is None:
        return "open"
    return _joined_pair(found[0], found[1], surface)


def ask_negative_pell(lattice: Lattice, number: int, surface: str = "digit") -> str:
    """The least pair whose squares differ by one less than this count.

    The first square is one below the count times the second square.
    An even period stays open.
    """
    found = _pell_solution(lattice, number, -1, surface)
    if found is None:
        return "open"
    return _joined_pair(found[0], found[1], surface)


def _home_on(lattice: Lattice, score: LayoutScore, name: str) -> str | None:
    """The one home whose gauge cancels this name. Two homes, or none, stay open."""
    if not score.chewable or name not in dict(score.name_gauge):
        return None
    found = [
        home
        for home, _gauge in score.home_gauge
        if ask_layout(lattice, score, name, home) == "yes"
    ]
    if len(found) != 1:
        return None
    return found[0]


def ask_amino_acid(
    lattice: Lattice, codons: LayoutScore, letters: LayoutScore, codon: str
) -> str:
    """The English name reached through the codon letter. A stop stays open."""
    letter = _home_on(lattice, codons, codon)
    if letter is None:
        return "open"
    english = _home_on(lattice, letters, letter)
    if english is None:
        return "open"
    return english


@dataclass
class Beside:
    """A second home for a taught noun. The resting kind gauge is not this field."""

    home: dict[str, str] = field(default_factory=dict)
    situation: dict[str, str] = field(default_factory=dict)
    clock: float = 0.0


def teach_beside(beside: Beside, noun: str, home: str, situation: str) -> None:
    """Record a second home selected by a situation word. The kind gauge stays."""
    if noun not in ACQUIRED_NOUNS:
        raise RuntimeError("a second sense needs a taught noun")
    if home not in KIND_HEADS or situation not in CONTENT_WORDS:
        raise RuntimeError("a second sense needs a kind and a situation word")
    if home == acquired_kind(noun):
        raise RuntimeError("the resting kind is already that home")
    beside.home[noun] = home
    beside.situation[noun] = situation


def deposit_situation(lattice: Lattice, beside: Beside, word: str) -> None:
    """Leave this content gauge on the situation clock. Noun gauges stay put."""
    beside.clock = lattice.content_value[word]


def clear_situation(beside: Beside) -> None:
    beside.clock = 0.0


def ask_beside(lattice: Lattice, beside: Beside, noun: str, stated: str) -> str:
    """The sense the situation clock selects. An empty clock keeps the resting kind.

    A clock that matches the stored situation answers the second home. Any
    other clock answers the resting kind. The resting gauge is not written.
    """
    if noun not in lattice.acquired_value or stated not in lattice.content_value:
        return "open"
    resting = lattice.predict_acquired(noun, stated)
    second = beside.home.get(noun)
    scene = beside.situation.get(noun)
    if second is None or scene is None:
        return resting
    gap = lattice.consensus_quantity(
        beside.clock,
        lattice.content_value[scene],
        lattice.routed_sign("-"),
    )
    if lattice.remainder_answer(gap) != "yes":
        return resting
    if not _kind_head(stated):
        return resting
    if stated == second:
        return "yes"
    return "no"


def ask_layout(lattice: Lattice, score: LayoutScore, name: str, home: str) -> str:
    """Yes when a kept name sits on that home. A missing name stays open."""
    if not score.chewable:
        return "open"
    names = dict(score.name_gauge)
    homes = dict(score.home_gauge)
    if name not in names or home not in homes:
        return "open"
    gap = lattice.consensus_quantity(names[name], homes[home], lattice.routed_sign("-"))
    return lattice.remainder_answer(gap)
