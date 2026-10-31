"""The opening two words of a question name its coarse type.

The teaching file and the held-out file are the CogComp question
classification split (Xin Li and Dan Roth, Learning Question Classifiers,
COLING 2002):

https://cogcomp.seas.upenn.edu/Data/QA/QC/

The coarse label is the text before the colon. The files are latin-1.
A word seen fewer than five times in the teaching questions shares one
gauge. The composed value is the consensus sum of the two word gauges.
Each label's home sits at K times its place. The fluid step moves the
word gauges. The stop watches the teaching accuracy. The held-out
questions are scored once after that stop. A question shorter than two
words stays open. A nearest home outside the drop stays open.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from fsot_lattice.apply import DROP
from fsot_lattice.engine import fluid_lr
from fsot_lattice.lattice import GAUGE_GAIN, K, Lattice

# Five occurrences in the teaching questions keep a word's own gauge.
MIN_SEEN = 5
# The teaching accuracy must rise by more than this, or the pass has plateaued.
RISE = 0.001
PLATEAU = 20
# The plateau is read only after this many passes.
MIN_EPOCH = 30
MAX_EPOCH = 200
OTHER = "*"
SOURCE = "https://cogcomp.seas.upenn.edu/Data/QA/QC/"


@dataclass(frozen=True)
class SpeechScore:
    words: int
    homes: int
    epochs: int
    train_count: int
    hold_count: int
    train_hits: int
    hold_hits: int
    nearest_hits: int
    train_acc: float
    hold_acc: float
    majority_label: str
    majority_hits: int
    majority_acc: float
    landed: bool
    reason: str
    last_loss: float
    widest_gap: float
    opened: int
    word_gauge: tuple[tuple[str, float], ...] = ()
    home_gauge: tuple[tuple[str, float], ...] = ()


def question_words(text: str) -> tuple[str, ...]:
    """Lowercase letter-and-digit words, in order."""
    found: list[str] = []
    token: list[str] = []
    for ch in text.lower():
        if ch.isalnum():
            token.append(ch)
        elif token:
            found.append("".join(token))
            token = []
    if token:
        found.append("".join(token))
    return tuple(found)


@dataclass(frozen=True)
class _Question:
    left: str
    right: str
    label: str
    short: bool


def _load_rows(path: Path) -> list[_Question]:
    """Every nonempty question. A question shorter than two words is marked short."""
    rows: list[_Question] = []
    for line in path.read_text(encoding="latin-1").splitlines():
        if not line.strip():
            continue
        label, question = line.split(" ", 1)
        tokens = question_words(question)
        coarse = label.split(":", 1)[0]
        if len(tokens) < 2:
            rows.append(_Question("", "", coarse, True))
        else:
            rows.append(_Question(tokens[0], tokens[1], coarse, False))
    return rows


def _nearest(
    lattice: Lattice,
    composed: float,
    home_gauge: tuple[tuple[str, float], ...],
) -> tuple[str, float]:
    """The closest home. An equal gap keeps the earlier label."""
    minus = lattice.routed_sign("-")
    best_label = ""
    best_gap = 0.0
    found = False
    for label, home in home_gauge:
        gap = abs(lattice.consensus_quantity(composed, home, minus))
        if not found or gap < best_gap:
            best_label = label
            best_gap = gap
            found = True
    if not found:
        return "open", DROP
    return best_label, best_gap


def opening_read(
    lattice: Lattice,
    word_gauge: dict[str, float],
    home_gauge: tuple[tuple[str, float], ...],
    left: str,
    right: str,
) -> tuple[str, str, float]:
    """The named type, the nearest label, and the gap.

    Outside the drop the named type is open. The nearest label is still
    returned so a score can count it apart from the named read.
    """
    if OTHER not in word_gauge or not home_gauge:
        return "open", "open", DROP
    left_key = left if left in word_gauge else OTHER
    right_key = right if right in word_gauge else OTHER
    composed = lattice.consensus_quantity(
        word_gauge[left_key],
        word_gauge[right_key],
        lattice.routed_sign("+"),
    )
    label, gap = _nearest(lattice, composed, home_gauge)
    if gap >= DROP:
        return "open", label, gap
    return label, label, gap


def name_opening(
    lattice: Lattice,
    word_gauge: dict[str, float],
    home_gauge: tuple[tuple[str, float], ...],
    left: str,
    right: str,
) -> tuple[str, float]:
    """The coarse type, and the gap to that home. Outside the drop, the type is open."""
    named, _nearest_label, gap = opening_read(lattice, word_gauge, home_gauge, left, right)
    return named, gap


def read_questions(
    train_path: Path,
    test_path: Path,
    observer=None,
) -> SpeechScore:
    """Teach the opening two words, then score the held-out questions once.

    The stop uses the teaching accuracy. It does not read the held-out labels.
    """
    train_rows = _load_rows(train_path)
    test = _load_rows(test_path)
    train = [row for row in train_rows if not row.short]
    if not train:
        raise RuntimeError("the teaching questions were empty")
    counts = Counter(word for row in train for word in (row.left, row.right))
    kept = {word for word, count in counts.items() if count >= MIN_SEEN}
    labels = sorted({row.label for row in train})
    home = {label: K * (index + 1) for index, label in enumerate(labels)}
    home_gauge = tuple((label, home[label]) for label in labels)
    vocab = sorted(kept | {OTHER})
    gauge = {word: K for word in vocab}
    lattice = Lattice.fresh()
    majority_label = Counter(row.label for row in train).most_common(1)[0][0]

    def slot(word: str) -> str:
        return word if word in gauge else OTHER

    def composed(left: str, right: str) -> float:
        return lattice.consensus_quantity(
            gauge[slot(left)],
            gauge[slot(right)],
            lattice.routed_sign("+"),
        )

    best_train = 0.0
    still = 0
    used = 0
    loss = 0.0
    train_hits = 0
    for epoch in range(1, MAX_EPOCH + 1):
        grad = {word: 0.0 for word in vocab}
        seen = {word: 0 for word in vocab}
        align = 0.0
        hits = 0
        for row in train:
            value = composed(row.left, row.right)
            err = home[row.label] - value
            align += err * err
            if _nearest(lattice, value, home_gauge)[0] == row.label:
                hits += 1
            for word in (slot(row.left), slot(row.right)):
                grad[word] += err
                seen[word] += 1
        loss = align / len(train)
        train_acc = hits / len(train)
        # The fluid step takes this pass's teaching accuracy. The layout step
        # takes a hit count. This read keeps the accuracy.
        step = fluid_lr(loss, train_acc) * GAUGE_GAIN
        for word in vocab:
            if seen[word]:
                gauge[word] += step * grad[word] / seen[word]
        used = epoch
        train_hits = hits
        if train_acc > best_train + RISE:
            best_train = train_acc
            still = 0
        else:
            still += 1
        if observer is not None:
            observer(epoch, loss, train_acc)
        if still >= PLATEAU and epoch > MIN_EPOCH:
            break

    word_gauge = tuple(sorted(gauge.items()))
    stored = dict(word_gauge)
    hold_hits = 0
    nearest_hits = 0
    majority_hits = 0
    opened = 0
    widest = 0.0
    for row in test:
        if row.label == majority_label:
            majority_hits += 1
        if row.short:
            opened += 1
            continue
        got, nearest_label, gap = opening_read(
            lattice, stored, home_gauge, row.left, row.right
        )
        if gap > widest:
            widest = gap
        if gap >= DROP:
            opened += 1
        if got == row.label:
            hold_hits += 1
        if nearest_label == row.label:
            nearest_hits += 1
    # A short held-out question is open, so it counts in the denominator.
    hold_count = len(test)
    hold_acc = hold_hits / hold_count if hold_count else 0.0
    majority_acc = majority_hits / hold_count if hold_count else 0.0
    train_acc = train_hits / len(train)
    landed = hold_count > 0 and hold_hits > majority_hits
    reason = (
        "the opening words beat the teaching majority"
        if landed
        else "the opening words missed the teaching majority"
    )
    return SpeechScore(
        words=len(word_gauge),
        homes=len(home_gauge),
        epochs=used,
        train_count=len(train),
        hold_count=hold_count,
        train_hits=train_hits,
        hold_hits=hold_hits,
        nearest_hits=nearest_hits,
        train_acc=train_acc,
        hold_acc=hold_acc,
        majority_label=majority_label,
        majority_hits=majority_hits,
        majority_acc=majority_acc,
        landed=landed,
        reason=reason,
        last_loss=loss,
        widest_gap=widest,
        opened=opened,
        word_gauge=word_gauge,
        home_gauge=home_gauge,
    )


def save_speech(path: Path, score: SpeechScore, pin: str, source: str) -> None:
    """Write a landed question-type read. A miss is not kept."""
    if not score.landed:
        raise RuntimeError("a missed question type is not kept")
    payload = {
        "pin": pin,
        "source": source,
        "min_seen": MIN_SEEN,
        "other": OTHER,
        "words": score.words,
        "homes": score.homes,
        "epochs": score.epochs,
        "train_count": score.train_count,
        "hold_count": score.hold_count,
        "train_hits": score.train_hits,
        "hold_hits": score.hold_hits,
        "nearest_hits": score.nearest_hits,
        "train_acc": score.train_acc,
        "hold_acc": score.hold_acc,
        "majority_label": score.majority_label,
        "majority_hits": score.majority_hits,
        "majority_acc": score.majority_acc,
        "last_loss": score.last_loss,
        "widest_gap": score.widest_gap,
        "opened": score.opened,
        "reason": score.reason,
        "word_gauge": {word: gauge for word, gauge in score.word_gauge},
        "home_gauge": {home: gauge for home, gauge in score.home_gauge},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_speech(path: Path) -> SpeechScore:
    """Read a kept question-type read back into gauges the asker can use."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload["min_seen"]) != MIN_SEEN or payload["other"] != OTHER:
        raise RuntimeError("the kept question type uses a different opening read")
    words = tuple(sorted((key, float(value)) for key, value in payload["word_gauge"].items()))
    homes = tuple(sorted((key, float(value)) for key, value in payload["home_gauge"].items()))
    return SpeechScore(
        words=len(words),
        homes=len(homes),
        epochs=int(payload["epochs"]),
        train_count=int(payload["train_count"]),
        hold_count=int(payload["hold_count"]),
        train_hits=int(payload["train_hits"]),
        hold_hits=int(payload["hold_hits"]),
        nearest_hits=int(payload["nearest_hits"]),
        train_acc=float(payload["train_acc"]),
        hold_acc=float(payload["hold_acc"]),
        majority_label=str(payload["majority_label"]),
        majority_hits=int(payload["majority_hits"]),
        majority_acc=float(payload["majority_acc"]),
        landed=True,
        reason=str(payload["reason"]),
        last_loss=float(payload["last_loss"]),
        widest_gap=float(payload["widest_gap"]),
        opened=int(payload["opened"]),
        word_gauge=words,
        home_gauge=homes,
    )
