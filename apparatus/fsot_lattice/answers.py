"""A question that states its amounts answers with those amounts.

The problems are the single-equation questions of Koncel-Kedziorski,
Hajishirzi, Sabharwal, Etzioni, and Ang, Parsing Algebraic Word Problems
into Equations, TACL 2015, leaving out any question that also sits in
the AddSub file of Hosseini, Hajishirzi, Etzioni, and Kushman, EMNLP
2014. A row whose equation does not rebuild the recorded solution is
left out as well. Every fifth remaining problem, starting at the first,
is held out. The other problems teach the read.

https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/dataset/SingleEq/SingleEq.json

The words of the sentence that holds an amount, and the words of the
asking sentence, name one of three homes: the amount is dropped, added,
or subtracted. A word seen fewer than five times shares one gauge. The
composed value is the mean of those gauges. The fluid step moves the
word gauges. The stop watches the teaching signs. The held-out problems
are scored once after that stop.

The answer is the added amounts minus the subtracted amounts. A question
that does not state two amounts stays open. An amount whose nearest home
sits outside the drop stays open.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from fsot_lattice.apply import DROP
from fsot_lattice.engine import K, fluid_lr
from fsot_lattice.lattice import GAUGE_GAIN, Lattice

MIN_SEEN = 5
RISE = 0.001
PLATEAU = 20
MIN_EPOCH = 30
MAX_EPOCH = 80
OTHER = "*"
HOLD_STRIDE = 5
LABELS = ("drop", "plus", "minus")
SOURCE = (
    "https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/dataset/SingleEq/SingleEq.json"
)
_DATA = Path(__file__).resolve().parents[1] / "data" / "amounts"
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_TERM = re.compile(r"[+-]?(?:[Xx]|\d+(?:\.\d+)?)")


@dataclass(frozen=True)
class AnswerScore:
    words: int
    epochs: int
    train_count: int
    hold_count: int
    hold_hits: int
    baseline_hits: int
    sign_hits: int
    sign_count: int
    landed: bool
    reason: str
    last_loss: float
    widest_gap: float
    opened: int
    word_gauge: tuple[tuple[str, float], ...] = ()
    home_gauge: tuple[tuple[str, float], ...] = ()


def plain_question(text: str) -> str:
    """Join digits that a thousands comma split apart."""
    return re.sub(r"(?<=\d),(?=\d)", "", text)


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _number_tokens(text: str) -> list[str]:
    return _NUMBER.findall(plain_question(text))


def _sentence_at(question: str, start: int) -> str:
    offset = 0
    for part in question.split("."):
        end = offset + len(part)
        if offset <= start < end:
            return part
        offset = end + 1
    return question


def _asking_words(question: str) -> list[str]:
    parts = [part.strip() for part in question.split(".") if part.strip()]
    if not parts:
        return _words(question)
    return _words(parts[-1])


def amount_contexts(question: str) -> list[tuple[str, ...]]:
    """The sentence around each amount, then the asking sentence."""
    question = plain_question(question)
    ask = _asking_words(question)
    found = []
    for match in _NUMBER.finditer(question):
        bag = _words(_sentence_at(question, match.start())) + ask
        found.append(tuple(bag) if bag else (OTHER,))
    return found


def _side_terms(side: str) -> list[tuple[int, str]]:
    found = []
    for part in _TERM.findall(side):
        sign = 1
        if part[0] == "+":
            part = part[1:]
        elif part[0] == "-":
            sign = -1
            part = part[1:]
        found.append((sign, part))
    return found


def _gold_signs(question: str, equation: str) -> list[str] | None:
    """Signs once x is isolated. A stated amount the equation skips is dropped."""
    raw = equation.replace(" ", "")
    if raw.count("=") != 1 or re.search(r"[*/()]", raw):
        return None
    left, right = raw.split("=")
    coeff = 0
    moved: list[tuple[int, str]] = []
    for sign, token in _side_terms(left):
        if token.lower() == "x":
            coeff += sign
        else:
            moved.append((sign, token))
    for sign, token in _side_terms(right):
        if token.lower() == "x":
            coeff -= sign
        else:
            moved.append((-sign, token))
    if abs(coeff) != 1:
        return None
    wanted: list[tuple[float, str]] = []
    for sign, token in moved:
        contrib = -sign / coeff
        wanted.append((float(token), "plus" if contrib > 0 else "minus"))
    numbers = [float(token) for token in _number_tokens(question)]
    labels = ["drop"] * len(numbers)
    used = [False] * len(wanted)
    for index, number in enumerate(numbers):
        for slot, (value, sign) in enumerate(wanted):
            if not used[slot] and abs(value - number) < 1e-6:
                labels[index] = sign
                used[slot] = True
                break
    if not all(used):
        return None
    return labels


def _scale(tokens: list[str]) -> int:
    places = 0
    for token in tokens:
        if "." in token:
            places = max(places, len(token.split(".", 1)[1]))
    return places


def _scaled(token: str, scale: int) -> int:
    if "." in token:
        whole, frac = token.split(".", 1)
    else:
        whole, frac = token, ""
    frac = (frac + ("0" * scale))[:scale]
    magnitude = 10**scale
    return int(whole) * magnitude + (int(frac) if frac else 0)


def format_amount(scaled: int, scale: int) -> str:
    """The count, with the decimal point the amounts in the question used."""
    if scaled < 0:
        return "open"
    if scale == 0:
        return str(scaled)
    magnitude = 10**scale
    whole, frac = divmod(scaled, magnitude)
    return f"{whole}.{frac:0{scale}d}"


def _signed_units(question: str, signs: list[str]) -> tuple[int, int, int] | None:
    tokens = _number_tokens(question)
    if len(tokens) != len(signs) or "plus" not in signs:
        return None
    scale = _scale(tokens)
    added = subtracted = 0
    for token, sign in zip(tokens, signs):
        amount = _scaled(token, scale)
        if sign == "plus":
            added += amount
        elif sign == "minus":
            subtracted += amount
    if subtracted > added:
        return None
    return added, subtracted, scale


def _rebuilds(question: str, equation: str, solution: float) -> bool:
    signs = _gold_signs(question, equation)
    if signs is None:
        return False
    units = _signed_units(question, signs)
    if units is None:
        return False
    added, subtracted, scale = units
    got = (added - subtracted) / 10**scale
    return abs(got - solution) <= 1e-4


def eligible_rows(single_path: Path, add_path: Path) -> list[dict]:
    """Single-equation rows whose stated amounts rebuild the solution.

    A question that also appears in the AddSub file is excluded.
    """
    single = json.loads(single_path.read_text(encoding="utf-8"))
    add = json.loads(add_path.read_text(encoding="utf-8"))
    known = {_norm(row["sQuestion"]) for row in add}
    rows = []
    for row in single:
        question = plain_question(row["sQuestion"])
        equation = row["lEquations"][0]
        raw = equation.replace(" ", "")
        if _norm(question) in known:
            continue
        if raw.count("=") != 1 or re.search(r"[*/()]", raw):
            continue
        solution = float(row["lSolutions"][0])
        if not _rebuilds(question, equation, solution):
            continue
        kept = dict(row)
        kept["sQuestion"] = question
        rows.append(kept)
    return rows


def _sum_baseline(question: str, solution: float) -> bool:
    tokens = _number_tokens(question)
    if not tokens:
        return False
    total = sum(float(token) for token in tokens)
    return abs(total - solution) <= 1e-4


def read_answers(
    single_path: Path | None = None,
    add_path: Path | None = None,
    observer=None,
) -> AnswerScore:
    """Teach the sign of each amount, then score the held-out problems once."""
    single_path = _DATA / "SingleEq.json" if single_path is None else single_path
    add_path = _DATA / "AddSub.json" if add_path is None else add_path
    rows = eligible_rows(single_path, add_path)
    teach = [index for index in range(len(rows)) if index % HOLD_STRIDE != 0]
    hold = [index for index in range(len(rows)) if index % HOLD_STRIDE == 0]
    counts: Counter[str] = Counter()
    examples: list[tuple[tuple[str, ...], str]] = []
    for index in teach:
        row = rows[index]
        signs = _gold_signs(row["sQuestion"], row["lEquations"][0])
        bags = amount_contexts(row["sQuestion"])
        if signs is None or len(bags) != len(signs):
            continue
        for bag, sign in zip(bags, signs):
            examples.append((bag, sign))
            counts.update(bag)
    if not examples:
        raise RuntimeError("the teaching amounts were empty")
    kept = {word for word, count in counts.items() if count >= MIN_SEEN}
    vocab = sorted(kept | {OTHER})
    gauge = {word: K for word in vocab}
    home = {label: K * (index + 1) for index, label in enumerate(LABELS)}
    lattice = Lattice.fresh()

    def slot(word: str) -> str:
        return word if word in gauge else OTHER

    def composed(bag: tuple[str, ...]) -> float:
        keys = [slot(word) for word in bag]
        return sum(gauge[key] for key in keys) / len(keys)

    best = 0.0
    still = 0
    used = 0
    loss = 0.0
    sign_hits = 0
    for epoch in range(1, MAX_EPOCH + 1):
        grad = {word: 0.0 for word in vocab}
        seen = {word: 0 for word in vocab}
        align = 0.0
        hits = 0
        for bag, sign in examples:
            value = composed(bag)
            err = home[sign] - value
            align += err * err
            nearest = min(LABELS, key=lambda label: abs(value - home[label]))
            if nearest == sign:
                hits += 1
            for key in (slot(word) for word in bag):
                grad[key] += err
                seen[key] += 1
        loss = align / len(examples)
        sign_acc = hits / len(examples)
        step = fluid_lr(loss, sign_acc) * GAUGE_GAIN
        for word in vocab:
            if seen[word]:
                gauge[word] += step * grad[word] / seen[word]
        used = epoch
        sign_hits = hits
        if sign_acc > best + RISE:
            best = sign_acc
            still = 0
        else:
            still += 1
        if observer is not None:
            observer(epoch, loss, sign_acc)
        if still >= PLATEAU and epoch > MIN_EPOCH:
            break

    word_gauge = tuple(sorted(gauge.items()))
    home_gauge = tuple((label, home[label]) for label in LABELS)
    score = AnswerScore(
        words=len(word_gauge),
        epochs=used,
        train_count=len(teach),
        hold_count=len(hold),
        hold_hits=0,
        baseline_hits=0,
        sign_hits=sign_hits,
        sign_count=len(examples),
        landed=False,
        reason="",
        last_loss=loss,
        widest_gap=0.0,
        opened=0,
        word_gauge=word_gauge,
        home_gauge=home_gauge,
    )
    hold_hits = 0
    baseline_hits = 0
    opened = 0
    widest = 0.0
    stored = dict(word_gauge)
    for index in hold:
        row = rows[index]
        question = row["sQuestion"]
        solution = float(row["lSolutions"][0])
        if _sum_baseline(question, solution):
            baseline_hits += 1
        terms = predict_terms(lattice, score, question, stored)
        if terms is None:
            opened += 1
            continue
        added, subtracted, scale, gap = terms
        widest = max(widest, gap)
        got = (sum(added) - sum(subtracted)) / 10**scale
        if abs(got - solution) <= 1e-4:
            hold_hits += 1
    landed = hold_hits > baseline_hits
    reason = (
        "the signed amounts beat the sum of every amount"
        if landed
        else "the signed amounts missed the sum of every amount"
    )
    return AnswerScore(
        words=score.words,
        epochs=score.epochs,
        train_count=score.train_count,
        hold_count=score.hold_count,
        hold_hits=hold_hits,
        baseline_hits=baseline_hits,
        sign_hits=sign_hits,
        sign_count=len(examples),
        landed=landed,
        reason=reason,
        last_loss=loss,
        widest_gap=widest,
        opened=opened,
        word_gauge=word_gauge,
        home_gauge=home_gauge,
    )


def predict_terms(
    lattice: Lattice,
    score: AnswerScore,
    question: str,
    gauges: dict[str, float] | None = None,
) -> tuple[list[int], list[int], int, float] | None:
    """Added units, subtracted units, the decimal scale, and the widest sign gap."""
    if gauges is None:
        if not score.landed:
            return None
        gauges = dict(score.word_gauge)
    tokens = _number_tokens(question)
    bags = amount_contexts(question)
    if len(tokens) != len(bags) or len(tokens) < 2 or OTHER not in gauges:
        return None
    homes = dict(score.home_gauge) if score.home_gauge else {
        label: K * (index + 1) for index, label in enumerate(LABELS)
    }
    minus = lattice.routed_sign("-")
    signs: list[str] = []
    widest = 0.0
    for bag in bags:
        keys = [word if word in gauges else OTHER for word in bag]
        value = sum(gauges[key] for key in keys) / len(keys)
        nearest = min(LABELS, key=lambda label: abs(value - homes[label]))
        gap = abs(lattice.consensus_quantity(value, homes[nearest], minus))
        if gap >= DROP:
            return None
        widest = max(widest, gap)
        signs.append(nearest)
    if "plus" not in signs:
        return None
    scale = _scale(tokens)
    added: list[int] = []
    subtracted: list[int] = []
    for token, sign in zip(tokens, signs):
        amount = _scaled(token, scale)
        if sign == "plus":
            added.append(amount)
        elif sign == "minus":
            subtracted.append(amount)
    if not added or sum(subtracted) > sum(added):
        return None
    return added, subtracted, scale, widest


def save_answers(path: Path, score: AnswerScore, pin: str, source: str) -> None:
    """Write a landed amount read. A miss is not kept."""
    if not score.landed:
        raise RuntimeError("a missed amount read is not kept")
    payload = {
        "pin": pin,
        "source": source,
        "min_seen": MIN_SEEN,
        "other": OTHER,
        "hold_stride": HOLD_STRIDE,
        "words": score.words,
        "epochs": score.epochs,
        "train_count": score.train_count,
        "hold_count": score.hold_count,
        "hold_hits": score.hold_hits,
        "baseline_hits": score.baseline_hits,
        "sign_hits": score.sign_hits,
        "sign_count": score.sign_count,
        "last_loss": score.last_loss,
        "widest_gap": score.widest_gap,
        "opened": score.opened,
        "reason": score.reason,
        "word_gauge": {word: gauge for word, gauge in score.word_gauge},
        "home_gauge": {home: gauge for home, gauge in score.home_gauge},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_answers(path: Path) -> AnswerScore:
    """Read a kept amount read back into gauges the asker can use."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload["min_seen"]) != MIN_SEEN or payload["other"] != OTHER:
        raise RuntimeError("the kept amount read uses a different sign read")
    if int(payload["hold_stride"]) != HOLD_STRIDE:
        raise RuntimeError("the kept amount read holds out a different slice")
    words = tuple(sorted((key, float(value)) for key, value in payload["word_gauge"].items()))
    homes = tuple((label, float(payload["home_gauge"][label])) for label in LABELS)
    return AnswerScore(
        words=len(words),
        epochs=int(payload["epochs"]),
        train_count=int(payload["train_count"]),
        hold_count=int(payload["hold_count"]),
        hold_hits=int(payload["hold_hits"]),
        baseline_hits=int(payload["baseline_hits"]),
        sign_hits=int(payload["sign_hits"]),
        sign_count=int(payload["sign_count"]),
        landed=True,
        reason=str(payload["reason"]),
        last_loss=float(payload["last_loss"]),
        widest_gap=float(payload["widest_gap"]),
        opened=int(payload["opened"]),
        word_gauge=words,
        home_gauge=homes,
    )
