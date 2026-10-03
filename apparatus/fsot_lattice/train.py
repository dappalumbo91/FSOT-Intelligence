"""Train the lattice, consolidate, then bind the word surface.

Promotion requires the whole package:
digit holdout and train do not fall after the word phase,
digit gauges do not move after the freeze,
the consensus read stays within 1e-4 of v[a] ± v[b],
word holdout clears the coherence gate,
the idle bonds still support the digit pathways,
equality is the collapse of the consensus remainder,
a two-step chain reads the first vessel as the next operand,
and the generated pass scores order, product, a third step, names past nine,
two-digit names re-entering as operands, two places in one expression,
sums that leave 0..99, a hundred name used as an operand whose
result stays inside 0..999, a sum that leaves 0..999, a thousand
name used as an operand whose result stays inside 0..9999, two
thousand names in one expression, a sum that leaves 0..9999, and a
ten-thousand name used as an operand whose result stays inside 0..10999.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from fsot_lattice.apply import check_generated, score_generated
from fsot_lattice.consensus import (
    consensus_attend,
    consensus_states,
    position_coherence,
    trit_similarity,
)
from fsot_lattice.engine import (
    COHERENCE_GATE,
    COLLAPSE,
    K,
    PHI,
    PIN,
    POOF,
    SUCTION,
    require_live_pin,
    scalar,
)
from fsot_lattice.lattice import (
    AMP,
    WIDTH,
    Lattice,
    apply_digit_epoch,
    apply_word_epoch,
    chain_gap,
    clock_means,
    digit_drift,
    gauge_residual,
    mean_confidence,
    read_gap,
    reload_frozen_digits,
    score_chains,
    score_digits,
    score_equality,
    score_words,
    write_engrams,
)
from fsot_lattice.tasks import (
    WORD_OF,
    Chain,
    Claim,
    Triple,
    REPEATED_LIMIT,
    all_chains,
    all_claims,
    all_triples,
    epoch_order,
    split_chains,
    split_claims,
    split_triples,
)

APPARATUS = Path(__file__).resolve().parents[1]
RESULTS = APPARATUS / "results" / "lattice_run.json"
ENGRAMS = APPARATUS / "data" / "engrams.json"
DIGIT_EPOCHS = 500
WORD_EPOCHS = 500
EXACT_FLOOR = 1.0
GAUGE_FLOOR = 1e-4


def self_check(lattice: Lattice, train, hold) -> None:
    require_live_pin()
    if lattice.routed_sign("+") != 1 or lattice.routed_sign("-") != -1:
        raise RuntimeError("operator trit route does not separate plus and minus")
    hot = [2.0] * WIDTH
    cold = [-2.0] * WIDTH
    if trit_similarity(hot, hot) != 1.0 or trit_similarity(hot, cold) != -1.0:
        raise RuntimeError("trit similarity failed the saturated-vector check")
    short = consensus_states([hot, cold])
    longer = consensus_states([hot, cold, [3.0] * WIDTH])
    if short[0] != longer[0] or short[1] != longer[1]:
        raise RuntimeError("consensus read a future token")
    second = lattice.phrase_clocks(["0"], "digit")
    solo = lattice.symbol_vector("0", "digit")
    if abs(second.trace_norm - (SUCTION * _norm(solo))) > 1e-9:
        raise RuntimeError("trace leaked across the state flush")
    universe = {(r.a, r.op, r.b) for r in all_triples()}
    seen = {(r.a, r.op, r.b) for r in train} | {(r.a, r.op, r.b) for r in hold}
    if seen != universe or {(r.a, r.op, r.b) for r in train} & {(r.a, r.op, r.b) for r in hold}:
        raise RuntimeError("train/hold split is not a partition of the closed set")
    covered = set()
    for row in train:
        covered.update((row.a, row.b, row.c))
    if set(range(10)) - covered:
        raise RuntimeError(f"train split misses digits {set(range(10)) - covered}")
    plus = lattice.carrier_plus
    minus = lattice.carrier_minus
    ortho = lattice.carrier_ortho
    if abs(sum(a * b for a, b in zip(plus, minus))) > 1e-9:
        raise RuntimeError("plus and minus carriers are not orthogonal")
    if trit_similarity(plus, minus) != 0.0 or trit_similarity(plus, ortho) != 0.0:
        raise RuntimeError("quantity carriers are trit-visible to each other")
    _, dropped = consensus_attend(plus, [[1e-6] * WIDTH])
    if dropped != 0:
        raise RuntimeError("sub-threshold key entered the consensus read")
    # At init every digit gauge is K*(d+1), above the drop, so the consensus
    # state equals the algebraic sum. A self-key on the query would add 1 on plus.
    probes = (
        Triple(2, "+", 3, 5),
        Triple(5, "-", 2, 3),
        Triple(4, "+", 0, 4),
        Triple(6, "-", 0, 6),
    )
    for row in probes:
        sign = lattice.routed_sign(row.op)
        left = lattice.digit_value[str(row.a)]
        right = lattice.digit_value[str(row.b)]
        got = lattice.consensus_quantity(left, right, sign)
        if abs(got - (left + sign * right)) > 1e-9:
            raise RuntimeError(
                f"consensus read missed {row.digit_prompt()}{row.digit_answer()}"
            )
    bare = lattice.consensus_quantity(lattice.digit_value["5"], 1e-8, 1)
    if abs(bare - lattice.digit_value["5"]) > 1e-9:
        raise RuntimeError("sub-threshold operand was kept in the consensus read")
    cancelled = lattice.consensus_quantity(lattice.digit_value["4"], lattice.digit_value["4"], -1)
    if lattice.remainder_answer(cancelled) != "yes":
        raise RuntimeError("equal gauges left a live equality remainder")
    one_step = lattice.consensus_quantity(lattice.digit_value["4"], lattice.digit_value["3"], -1)
    if lattice.remainder_answer(one_step) != "no":
        raise RuntimeError("a one-step remainder collapsed to yes")
    if lattice.remainder_answer(0.0) != "yes" or lattice.remainder_answer(K) != "no":
        raise RuntimeError("equality gate does not separate a zero remainder from K")
    dead = [1e-6] * WIDTH
    if position_coherence(dead) > COHERENCE_GATE:
        raise RuntimeError("a null vessel cleared the equality gate")
    eq_train, eq_hold = split_claims()
    eq_all = set(all_claims())
    eq_train_set = set(eq_train)
    eq_hold_set = set(eq_hold)
    if eq_train_set | eq_hold_set != eq_all or eq_train_set & eq_hold_set:
        raise RuntimeError("equality split is not a partition of the claim set")
    zero_plus = [row for row in eq_all if row.a == 0 and row.op == "+" and row.holds]
    if not any(row in eq_train_set for row in zero_plus):
        raise RuntimeError("equality split held out every true 0+n claim")
    if not any(row in eq_hold_set for row in zero_plus):
        raise RuntimeError("equality split kept every true 0+n claim in train")
    for bunch in (eq_train, eq_hold):
        if not any(row.holds for row in bunch) or not any(not row.holds for row in bunch):
            raise RuntimeError("equality split is missing a true or a false claim")
    cancel_then = Chain(4, "-", 4, "+", 3, 3)
    if abs(lattice.compose_quantity(cancel_then, "digit") - lattice.digit_value["3"]) > 1e-9:
        raise RuntimeError("a cancelled vessel did not re-enter the next addition")
    if lattice.predict_chain(cancel_then, "digit") != "3":
        raise RuntimeError("nearest gauge missed the re-read vessel")
    ch_train, ch_hold = split_chains()
    ch_all = set(all_chains())
    ch_train_set = set(ch_train)
    ch_hold_set = set(ch_hold)
    if ch_train_set | ch_hold_set != ch_all or ch_train_set & ch_hold_set:
        raise RuntimeError("chain split is not a partition of the two-step set")
    zeros = [row for row in ch_all if row.intermediate_is_zero()]
    if not any(row in ch_train_set for row in zeros) or not any(row in ch_hold_set for row in zeros):
        raise RuntimeError("chain split hid every zero intermediate on one side")
    check_generated(lattice)


def _composition_ok(
    digit_train,
    digit_hold,
    word_train,
    word_hold,
    digit_gap: float,
    word_gap: float,
    digit_gap_closed: float,
    word_gap_closed: float,
    digit_zero,
    word_zero,
) -> bool:
    """Exact two-step read on both splits, including chains whose first vessel is zero."""
    return (
        digit_train[0] >= EXACT_FLOOR
        and digit_hold[0] >= EXACT_FLOOR
        and word_train[0] >= EXACT_FLOOR
        and word_hold[0] >= EXACT_FLOOR
        and digit_gap < GAUGE_FLOOR
        and word_gap < GAUGE_FLOOR
        and digit_gap_closed < GAUGE_FLOOR
        and word_gap_closed < GAUGE_FLOOR
        and digit_zero[0] >= EXACT_FLOOR
        and word_zero[0] >= EXACT_FLOOR
        and digit_zero[4] > 0
    )


def _chain_demo(lattice: Lattice, chain: Chain) -> dict:
    return {
        "digit": f"{chain.digit_prompt()}{chain.digit_answer()}",
        "word": chain.word_prompt(),
        "want": chain.word_answer(),
        "digit_got": lattice.predict_chain(chain, "digit"),
        "word_got": lattice.predict_chain(chain, "word"),
        "digit_quantity": lattice.compose_quantity(chain, "digit"),
        "word_quantity": lattice.compose_quantity(chain, "word"),
        "digit_algebra": lattice.algebraic_chain(chain, "digit"),
        "intermediate_zero": chain.intermediate_is_zero(),
    }


def _equal_demo(lattice: Lattice, claim: Claim) -> dict:
    return {
        "digit": claim.digit_prompt(),
        "word": claim.word_prompt(),
        "want": claim.answer(),
        "digit_got": lattice.predict_equal(claim, "digit"),
        "word_got": lattice.predict_equal(claim, "word"),
        "digit_remainder": lattice.equality_remainder(claim, "digit"),
        "word_remainder": lattice.equality_remainder(claim, "word"),
    }


def _equality_ok(train_score, hold_score, full_score, drop: float) -> bool:
    """Exact yes/no on both splits, with the remainder on the correct side of collapse."""
    full = full_score[3]
    return (
        train_score[0] >= EXACT_FLOOR
        and hold_score[0] >= EXACT_FLOOR
        and full["true_max_abs"] < drop
        and full["false_min_abs"] > drop
        and full["zero_true_acc"] >= EXACT_FLOOR
        and full["zero_plus_true_acc"] >= EXACT_FLOOR
    )


def _norm(vec: list[float]) -> float:
    return sum(v * v for v in vec) ** 0.5


def _run_phase(lattice: Lattice, train, hold, epochs: int, surface: str) -> list[dict]:
    history = []
    hits = 0.0
    for epoch in range(epochs):
        ordered = epoch_order(train, epoch, PHI)
        if surface == "digit":
            mse, lr = apply_digit_epoch(lattice, ordered, hits)
            train_acc, train_n, _ = score_digits(lattice, train)
            hold_acc, hold_n, _ = score_digits(lattice, hold)
            residual = gauge_residual(lattice.digit_value)
        else:
            mse, lr = apply_word_epoch(lattice, ordered, hits)
            train_acc, train_n, _ = score_words(lattice, train)
            hold_acc, hold_n, _ = score_words(lattice, hold)
            residual = gauge_residual({str(i): lattice.word_value[WORD_OF[i]] for i in range(10)})
        gap = read_gap(lattice, hold, surface)
        closed_gap = read_gap(lattice, all_triples(), surface)
        # Nine additions of the zero gauge are the widest closed product.
        # Consensus drops that gauge, so the fold gap is this multiple.
        zero_fold = 0.0
        if surface == "word":
            zero_fold = REPEATED_LIMIT * abs(lattice.word_value["zero"])
        hits = train_acc * len(train)
        row = {
            "epoch": epoch,
            "mse": mse,
            "lr": lr,
            "train_acc": train_acc,
            "hold_acc": hold_acc,
            "train_correct": train_n,
            "hold_correct": hold_n,
            "gauge_residual": residual,
            "read_gap": gap,
            "read_gap_closed": closed_gap,
            "zero_fold": zero_fold,
        }
        history.append(row)
        settled = residual < GAUGE_FLOOR and gap < GAUGE_FLOOR and closed_gap < GAUGE_FLOOR and zero_fold < GAUGE_FLOOR
        if epoch % 10 == 0 or settled:
            print(
                f"{surface} epoch {epoch:03d}  mse {mse:.6e}  lr {lr:.4f}  "
                f"train {train_acc:.3f}  hold {hold_acc:.3f}  "
                f"gauge {residual:.3e}  read {gap:.3e}  zero_fold {zero_fold:.3e}",
                flush=True,
            )
        if (
            train_acc >= EXACT_FLOOR
            and hold_acc >= EXACT_FLOOR
            and settled
        ):
            break
    return history


def run() -> dict:
    started = time.perf_counter()
    pin = require_live_pin()
    lattice = Lattice.fresh()
    train, hold = split_triples()
    self_check(lattice, train, hold)

    s_home = scalar(0.0, observed=False)
    s_obs = scalar(0.0, observed=True)
    print(
        f"pin {pin[:12]}  K {K:.16f}  collapse {COLLAPSE:.16f}  "
        f"width {WIDTH}  train {len(train)}  hold {len(hold)}",
        flush=True,
    )
    print(f"S(D=25, unobserved) {s_home:.6f}  S(D=25, observed) {s_obs:.6f}", flush=True)

    digit_history = _run_phase(lattice, train, hold, DIGIT_EPOCHS, "digit")
    digit_train_before, _, _ = score_digits(lattice, train)
    digit_hold_before, _, digit_hold_misses = score_digits(lattice, hold)
    clocks_before = clock_means(lattice, train, "digit")

    consolidate_lr = digit_history[-1]["lr"] * POOF
    for row in train:
        lattice.hebbian([str(row.a), row.op, str(row.b), "=", str(row.c)], consolidate_lr)
    bond_meta = lattice.consolidate(train, consolidate_lr)
    confidence = mean_confidence(lattice, train)
    write_engrams(
        ENGRAMS,
        lattice,
        pin,
        {"phase": "digit_idle", "confidence": confidence, **bond_meta},
    )
    reloaded = reload_frozen_digits(ENGRAMS)
    reload_drift = digit_drift(lattice.frozen_digits or {}, reloaded)

    # Lock the consolidated gauges for the word surface.
    lattice.digit_value = dict(lattice.frozen_digits or {})
    word_history = _run_phase(lattice, train, hold, WORD_EPOCHS, "word")
    lattice.digit_value = dict(lattice.frozen_digits or {})

    digit_train_after, _, _ = score_digits(lattice, train)
    digit_hold_after, _, _ = score_digits(lattice, hold)
    word_train, _, _ = score_words(lattice, train)
    word_hold, _, word_misses = score_words(lattice, hold)
    drift = digit_drift(lattice.digit_value, lattice.frozen_digits or {})
    clocks_after = clock_means(lattice, train, "word")
    confidence_after = mean_confidence(lattice, train)
    closed = all_triples()
    digit_read_gap = read_gap(lattice, hold, "digit")
    word_read_gap = read_gap(lattice, hold, "word")
    digit_read_gap_closed = read_gap(lattice, closed, "digit")
    word_read_gap_closed = read_gap(lattice, closed, "word")
    eq_train, eq_hold = split_claims()
    eq_drop = COLLAPSE / AMP
    digit_eq_train = score_equality(lattice, eq_train, "digit")
    digit_eq_hold = score_equality(lattice, eq_hold, "digit")
    word_eq_train = score_equality(lattice, eq_train, "word")
    word_eq_hold = score_equality(lattice, eq_hold, "word")
    digit_eq_all = score_equality(lattice, all_claims(), "digit")
    word_eq_all = score_equality(lattice, all_claims(), "word")
    ch_train, ch_hold = split_chains()
    ch_all = all_chains()
    digit_ch_train = score_chains(lattice, ch_train, "digit")
    digit_ch_hold = score_chains(lattice, ch_hold, "digit")
    word_ch_train = score_chains(lattice, ch_train, "word")
    word_ch_hold = score_chains(lattice, ch_hold, "word")
    digit_chain_gap = chain_gap(lattice, ch_hold, "digit")
    word_chain_gap = chain_gap(lattice, ch_hold, "word")
    digit_chain_gap_closed = chain_gap(lattice, ch_all, "digit")
    word_chain_gap_closed = chain_gap(lattice, ch_all, "word")
    zero_chains = [row for row in ch_all if row.intermediate_is_zero()]
    digit_zero_chain = score_chains(lattice, zero_chains, "digit")
    word_zero_chain = score_chains(lattice, zero_chains, "word")
    applied = score_generated(lattice)

    retention_ok = (
        digit_hold_after + 1e-12 >= digit_hold_before
        and digit_train_after + 1e-12 >= digit_train_before
        and drift == 0.0
        and reload_drift == 0.0
    )
    word_ok = (
        word_hold >= COHERENCE_GATE
        and word_train >= COHERENCE_GATE
        and word_read_gap < GAUGE_FLOOR
        and word_read_gap_closed < GAUGE_FLOOR
    )
    digit_ok = (
        digit_hold_before >= EXACT_FLOOR
        and digit_train_before >= EXACT_FLOOR
        and gauge_residual(lattice.digit_value) < GAUGE_FLOOR
        and digit_read_gap < GAUGE_FLOOR
        and digit_read_gap_closed < GAUGE_FLOOR
    )
    bonds_ok = confidence >= COHERENCE_GATE and confidence_after >= COHERENCE_GATE
    equality_ok = _equality_ok(digit_eq_train, digit_eq_hold, digit_eq_all, eq_drop) and _equality_ok(
        word_eq_train, word_eq_hold, word_eq_all, eq_drop
    )
    composition_ok = _composition_ok(
        digit_ch_train,
        digit_ch_hold,
        word_ch_train,
        word_ch_hold,
        digit_chain_gap,
        word_chain_gap,
        digit_chain_gap_closed,
        word_chain_gap_closed,
        digit_zero_chain,
        word_zero_chain,
    )
    promoted = (
        retention_ok and word_ok and digit_ok and bonds_ok and equality_ok and composition_ok and applied["ok"]
    )

    demos = []
    for row in (train[:2] + hold[:2]):
        demos.append(
            {
                "digit": f"{row.digit_prompt()}{row.digit_answer()}",
                "digit_got": lattice.predict_digit(row),
                "word": row.word_prompt(),
                "word_want": row.word_answer(),
                "word_got": lattice.predict_word(row),
            }
        )

    report = {
        "promoted": promoted,
        "pin": pin,
        "K": K,
        "collapse": COLLAPSE,
        "suction": SUCTION,
        "poof": POOF,
        "phi": PHI,
        "width": WIDTH,
        "learned_parameters": lattice.learned_parameter_count(),
        "train_count": len(train),
        "hold_count": len(hold),
        "S_home_unobserved": s_home,
        "S_home_observed": s_obs,
        "digit_epochs": len(digit_history),
        "word_epochs": len(word_history),
        "digit_train_before": digit_train_before,
        "digit_hold_before": digit_hold_before,
        "digit_train_after": digit_train_after,
        "digit_hold_after": digit_hold_after,
        "digit_gauge_residual": gauge_residual(lattice.digit_value),
        "read_gap": digit_read_gap,
        "read_gap_closed": digit_read_gap_closed,
        "word_read_gap": word_read_gap,
        "word_read_gap_closed": word_read_gap_closed,
        "digit_values": {key: lattice.digit_value[key] for key in [str(i) for i in range(10)]},
        "word_values": {WORD_OF[i]: lattice.word_value[WORD_OF[i]] for i in range(10)},
        "word_train": word_train,
        "word_hold": word_hold,
        "digit_drift_after_words": drift,
        "engram_reload_drift": reload_drift,
        "pathway_confidence_after_idle": confidence,
        "pathway_confidence_after_words": confidence_after,
        "kept_bonds": bond_meta["kept_bonds"],
        "clocks_digit": clocks_before,
        "clocks_word": clocks_after,
        "retention_ok": retention_ok,
        "word_ok": word_ok,
        "digit_ok": digit_ok,
        "bonds_ok": bonds_ok,
        "equality_ok": equality_ok,
        "composition_ok": composition_ok,
        "apply_ok": applied["ok"],
        "apply": applied,
        "chain_train_count": len(ch_train),
        "chain_hold_count": len(ch_hold),
        "chain_digit_train": digit_ch_train[0],
        "chain_digit_hold": digit_ch_hold[0],
        "chain_word_train": word_ch_train[0],
        "chain_word_hold": word_ch_hold[0],
        "chain_digit_gap": digit_chain_gap,
        "chain_word_gap": word_chain_gap,
        "chain_digit_gap_closed": digit_chain_gap_closed,
        "chain_word_gap_closed": word_chain_gap_closed,
        "chain_zero_n": len(zero_chains),
        "chain_zero_digit": digit_zero_chain[0],
        "chain_zero_word": word_zero_chain[0],
        "chain_digit_hold_misses": digit_ch_hold[2],
        "chain_word_hold_misses": word_ch_hold[2],
        "equality_drop": eq_drop,
        "equal_train_count": len(eq_train),
        "equal_hold_count": len(eq_hold),
        "equal_digit_train": digit_eq_train[0],
        "equal_digit_hold": digit_eq_hold[0],
        "equal_word_train": word_eq_train[0],
        "equal_word_hold": word_eq_hold[0],
        "equal_digit_margin": digit_eq_all[3],
        "equal_word_margin": word_eq_all[3],
        "equal_digit_hold_misses": digit_eq_hold[2],
        "equal_word_hold_misses": word_eq_hold[2],
        "digit_hold_misses": digit_hold_misses,
        "word_hold_misses": word_misses,
        "demos": demos,
        "equal_demos": [
            _equal_demo(lattice, claim)
            for claim in (
                Claim(2, "+", 3, 5, True),
                Claim(2, "+", 3, 4, False),
                Claim(0, "+", 0, 0, True),
                Claim(0, "+", 5, 5, True),
                Claim(5, "-", 5, 0, True),
                Claim(5, "-", 5, 1, False),
            )
        ],
        "chain_demos": [
            _chain_demo(lattice, chain)
            for chain in (
                Chain(2, "+", 3, "-", 1, 4),
                Chain(2, "+", 3, "+", 1, 6),
                Chain(5, "-", 5, "+", 3, 3),
                Chain(0, "+", 0, "+", 4, 4),
                Chain(9, "-", 9, "+", 0, 0),
            )
        ],
        "wall_seconds": time.perf_counter() - started,
        "digit_history_tail": digit_history[-3:],
        "word_history_tail": word_history[-3:],
    }
    write_engrams(ENGRAMS, lattice, pin, {"phase": "final", "promoted": promoted})
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"promoted {promoted}  digit hold {digit_hold_before:.3f}->{digit_hold_after:.3f}  "
        f"word hold {word_hold:.3f}  read {digit_read_gap:.3e}  "
        f"equal {digit_eq_hold[0]:.3f}/{word_eq_hold[0]:.3f}  "
        f"rem {digit_eq_all[3]['true_max_abs']:.3e}/{digit_eq_all[3]['false_min_abs']:.3f}  "
        f"chain {digit_ch_hold[0]:.3f}/{word_ch_hold[0]:.3f}  "
        f"chain_gap {digit_chain_gap:.3e}  "
        f"apply {applied['order_digit_hold']['acc']:.3f}/"
        f"{applied['product_digit_hold']['acc']:.3f}/"
        f"{applied['span_digit_hold']['acc']:.3f}  "
        f"lex {applied['lexeme_digit_hold']['acc']:.3f}  "
        f"ten {applied['lexeme_digit_ten']['ten_max_abs']:.3e}  "
        f"step {applied['place_digit_hold']['acc']:.3f}  "
        f"pair {applied['pair_digit_hold']['acc']:.3f}  "
        f"hund {applied['hundred_digit_hold']['acc']:.3f}  "
        f"op {applied['op_digit_hold']['acc']:.3f}  "
        f"thou {applied['thousand_digit_hold']['acc']:.3f}  "
        f"thouop {applied['thou_op_digit_hold']['acc']:.3f}  "
        f"thoupair {applied['thou_pair_digit_hold']['acc']:.3f}  "
        f"tenthous {applied['ten_thou_digit_hold']['acc']:.3f}  "
        f"tenop {applied['ten_thou_op_digit_hold']['acc']:.3f}  "
        f"bonds {int(bond_meta['kept_bonds'])}  confidence {confidence:.3f}  "
        f"wall {time.perf_counter() - started:.2f}s",
        flush=True,
    )
    return report
