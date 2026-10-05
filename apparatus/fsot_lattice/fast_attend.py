"""Compiled attend for the scalar quantity read.

The DLL is the same loops as consensus.py. If it is missing, or a lattice
carries a different axis, the caller keeps the Python read.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

_LIB = None
_READY = False
_EXTRA = False
_READ_BOUND = False
_ENABLED = True
_PLUS = None
_MINUS = None
_COLLAPSE = 0.0
_GATE = 0.0
_DROP = 0.0
_MINUS_SIGN = -1
_WIDTH = 0


def _load():
    global _LIB, _EXTRA
    if _LIB is not None:
        return _LIB
    path = None
    for name in ("attend_core.dll", "attend_core.so"):
        candidate = Path(__file__).with_name(name)
        if candidate.exists():
            path = candidate
            break
    if path is None:
        return None
    lib = ctypes.CDLL(str(path))
    lib.fsot_consensus_quantity.argtypes = [
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
    ]
    lib.fsot_consensus_quantity.restype = ctypes.c_double
    lib.fsot_read_place.argtypes = [
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.fsot_read_place.restype = None
    try:
        lib.fsot_fold_steps.argtypes = [
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_double,
        ]
        lib.fsot_fold_steps.restype = ctypes.c_double
        lib.fsot_bind_read.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int,
        ]
        lib.fsot_bind_read.restype = ctypes.c_int
        lib.fsot_consensus_read.argtypes = [
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        lib.fsot_consensus_read.restype = None
        _EXTRA = True
    except AttributeError:
        _EXTRA = False
    _LIB = lib
    return lib


def bind(plus: list[float], minus: list[float], collapse: float, gate: float, drop: float, minus_sign: int) -> bool:
    """Copy the carriers into stable buffers. Later lattices must match them."""
    global _READY, _PLUS, _MINUS, _COLLAPSE, _GATE, _DROP, _MINUS_SIGN, _WIDTH
    lib = _load()
    if lib is None:
        _READY = False
        return False
    width = len(plus)
    if width == 0 or width != len(minus) or width > 64:
        _READY = False
        return False
    _PLUS = (ctypes.c_double * width)(*plus)
    _MINUS = (ctypes.c_double * width)(*minus)
    _COLLAPSE = float(collapse)
    _GATE = float(gate)
    _DROP = float(drop)
    _MINUS_SIGN = int(minus_sign)
    _WIDTH = width
    _READY = True
    return True


def ensure(lattice: object) -> None:
    """Bind once from the live carriers. A missing DLL leaves the Python read."""
    if _READY or _LIB is False:
        return
    from fsot_lattice.engine import COHERENCE_GATE, COLLAPSE
    from fsot_lattice.lattice import AMP

    bind(
        list(lattice.carrier_plus),
        list(lattice.carrier_minus),
        COLLAPSE,
        COHERENCE_GATE,
        COLLAPSE / AMP,
        int(lattice.routed_sign("-")),
    )


def enabled() -> bool:
    return _ENABLED and _READY


def set_enabled(flag: bool) -> None:
    global _ENABLED
    _ENABLED = bool(flag)


def consensus_quantity(left: float, right: float, sign: int) -> float | None:
    if not enabled() or _LIB is None or _PLUS is None or _MINUS is None:
        return None
    return float(
        _LIB.fsot_consensus_quantity(
            float(left),
            float(right),
            int(sign),
            _PLUS,
            _MINUS,
            _WIDTH,
            _COLLAPSE,
            _GATE,
        )
    )


def read_place(
    quantity: float,
    gauges: list[float],
    ten: float,
    hundred: float,
    thousand: float,
    ten_thousand: float,
    hundred_thousand: float,
) -> tuple[int, float, float, float] | None:
    if not enabled() or _LIB is None or _PLUS is None or _MINUS is None:
        return None
    n = len(gauges)
    buf = (ctypes.c_double * n)(*gauges)
    named = ctypes.c_int()
    remainder = ctypes.c_double()
    margin = ctypes.c_double()
    dist = ctypes.c_double()
    _LIB.fsot_read_place(
        float(quantity),
        buf,
        n,
        float(ten),
        float(hundred),
        float(thousand),
        float(ten_thousand),
        float(hundred_thousand),
        _PLUS,
        _MINUS,
        _WIDTH,
        _COLLAPSE,
        _GATE,
        _DROP,
        _MINUS_SIGN,
        ctypes.byref(named),
        ctypes.byref(remainder),
        ctypes.byref(margin),
        ctypes.byref(dist),
    )
    return int(named.value), float(remainder.value), float(margin.value), float(dist.value)


def fold_steps(left: float, steps: list[tuple[int, float]]) -> float | None:
    """One call for a chain of consensus steps. A missing symbol keeps the Python loop."""
    if not enabled() or not _EXTRA or _LIB is None or _PLUS is None or _MINUS is None:
        return None
    n = len(steps)
    if n == 0:
        return float(left)
    signs = (ctypes.c_int * n)(*(int(sign) for sign, _right in steps))
    rights = (ctypes.c_double * n)(*(float(right) for _sign, right in steps))
    return float(
        _LIB.fsot_fold_steps(
            float(left),
            signs,
            rights,
            n,
            _PLUS,
            _MINUS,
            _WIDTH,
            _COLLAPSE,
            _GATE,
        )
    )


def bind_read(
    gauges: list[float],
    ten: float,
    hundred: float,
    thousand: float,
    ten_thousand: float,
    hundred_thousand: float | None = None,
) -> bool:
    """Copy one surface into the attend. Later rows skip the gauge copy."""
    global _READ_BOUND
    _READ_BOUND = False
    if not enabled() or not _EXTRA or _LIB is None or _PLUS is None or _MINUS is None:
        return False
    n = len(gauges)
    if n <= 0 or n > 16:
        return False
    if hundred_thousand is None:
        hundred_thousand = float(ten_thousand) * n
    buf = (ctypes.c_double * n)(*gauges)
    ok = _LIB.fsot_bind_read(
        buf,
        n,
        float(ten),
        float(hundred),
        float(thousand),
        float(ten_thousand),
        float(hundred_thousand),
        _PLUS,
        _MINUS,
        _WIDTH,
        _COLLAPSE,
        _GATE,
        _DROP,
        _MINUS_SIGN,
    )
    _READ_BOUND = bool(ok)
    return _READ_BOUND


def consensus_read(left: float, right: float, sign: int):
    """Consensus and the place read in one call. None keeps the two Python crossings."""
    if not _READ_BOUND or not enabled() or not _EXTRA or _LIB is None:
        return None
    quantity = ctypes.c_double()
    named = ctypes.c_int()
    remainder = ctypes.c_double()
    margin = ctypes.c_double()
    dist = ctypes.c_double()
    _LIB.fsot_consensus_read(
        float(left),
        float(right),
        int(sign),
        ctypes.byref(quantity),
        ctypes.byref(named),
        ctypes.byref(remainder),
        ctypes.byref(margin),
        ctypes.byref(dist),
    )
    return (
        float(quantity.value),
        int(named.value),
        float(remainder.value),
        float(margin.value),
        float(dist.value),
    )
