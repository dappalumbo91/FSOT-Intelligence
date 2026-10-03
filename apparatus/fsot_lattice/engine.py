"""Live FSOT engine bind for this project.

Constants come from the Lean tree's vendor/fsot_compute.py.
The SHA-256 prefix of that file is the pin.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LEAN_ENGINE = PROJECT_ROOT / "fsot-2.1-lean" / "vendor" / "fsot_compute.py"
_VENDOR_ENGINE = PROJECT_ROOT / "vendor" / "fsot_compute.py"
# A local theory checkout wins. A published clone ships the pinned file.
ENGINE_PATH = _LEAN_ENGINE if _LEAN_ENGINE.is_file() else _VENDOR_ENGINE

# Same causal coherence cut the GPU consensus operator uses (k_coh > 0.5).
COHERENCE_GATE = 0.5


def _load():
    if not ENGINE_PATH.is_file():
        raise FileNotFoundError(f"missing live engine: {ENGINE_PATH}")
    raw = ENGINE_PATH.read_bytes()
    pin = hashlib.sha256(raw).hexdigest()
    spec = importlib.util.spec_from_file_location("fsot_intelligence_engine", ENGINE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {ENGINE_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return pin, mod


PIN, _ENG = _load()


def _f(name: str) -> float:
    return float(getattr(_ENG, name))


PI = _f("PI")
E = _f("E")
PHI = _f("PHI")
GAMMA = _f("GAMMA")
ALPHA = _f("ALPHA")
PSI_CON = _f("PSI_CON")
ETA_EFF = _f("ETA_EFF")
BETA = _f("BETA")
THETA_S = _f("THETA_S")
POOF = _f("POOF")
C_EFF = _f("C_EFF")
P_VAR = _f("P_VAR")
SUCTION = _f("SUCTION")
K = _f("K")
COLLAPSE = _f("C_EFF") * _f("P_VAR")


def require_live_pin() -> str:
    """This folder's engine copy is pin AEB2AD."""
    if not PIN.startswith("aeb2ad"):
        raise RuntimeError(f"engine pin {PIN[:12]} is not aeb2ad ({ENGINE_PATH})")
    return PIN


def scalar(recent_hits: float, observed: bool, d_eff: float = 25.0) -> float:
    """S = K(T1+T2+T3) at the given fold. recent_hits is a fraction in [0, 1]."""
    eng = _ENG
    fold = eng.ScalarInput(
        N=eng.mpf(1),
        P=eng.mpf(1),
        D_eff=eng.mpf(d_eff),
        recent_hits=eng.mpf(recent_hits),
        observed=observed,
        delta_psi=eng.mpf(1),
        delta_theta=eng.mpf(1),
    )
    return float(eng.compute_scalar(fold))


def fluid_lr(loss: float, recent_hits: float) -> float:
    """Suction–poof step on the raw fluid scale.

    The 135M host damped this by alpha/phi^2 and a golden epoch envelope.
    Those dampers stall a 10-gauge fit before the additive residual closes.
    """
    raw = SUCTION * (1.0 - POOF * math.tanh(loss)) * math.exp(-ALPHA * recent_hits)
    return raw * K
