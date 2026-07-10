"""
test_reproducibility.py — the committed data/results.json is reproducible.

run_experiments.run_all() drives every scenario from fixed per-run seeds, so all
stored curves are deterministic; only the speedup timings and the generated_at
timestamp legitimately vary between runs (see the run_all docstring). This test
regenerates the results in-memory and asserts they match the committed file,
guarding against silent changes to the model or the experiment harness.

Comparison policy:
- The stochastic CA curves are compared bit-for-bit. Every transition in
  model.seiqr_advance is a PCG64 draw against a constant probability plus an
  exact integer infected-neighbour test, so the curves reproduce exactly across
  numpy versions and platforms (no libm pow on the CA path).
- The ode_validation block is derived from scipy.integrate.solve_ivp, whose
  RK45 step sequence can differ in the low bits across scipy versions, so it is
  compared with a relative tolerance instead.
- speedup (unseeded wall-clock timings) and generated_at (timestamp) are excluded.
"""

import json
from pathlib import Path

import numpy as np
import pytest

import run_experiments

RESULTS_PATH = Path(__file__).resolve().parent.parent / "data" / "results.json"

# Legitimately non-deterministic, so never compared.
EXCLUDED_KEYS = {"speedup", "generated_at"}
# solve_ivp-derived: portable to a tight tolerance, not bit-for-bit.
TOLERANCE_KEY = "ode_validation"
ODE_RTOL = 1e-6
ODE_ATOL = 1e-9

# Loaded at import so the bit-identity check can be parametrised per scenario.
_COMMITTED = json.loads(RESULTS_PATH.read_text())
_STRICT_KEYS = sorted(set(_COMMITTED) - EXCLUDED_KEYS - {TOLERANCE_KEY})


def _maxdiff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).max()) if a.size else 0.0


def _compare(regen, ref, path, rtol):
    """Recursively assert ``regen`` matches ``ref``.

    ``rtol is None`` requires exact (bit-for-bit) equality; a float ``rtol``
    compares numeric leaves with numpy at that relative tolerance (atol
    ``ODE_ATOL``, NaNs treated as equal).
    """
    if isinstance(ref, dict):
        assert isinstance(regen, dict), f"{path}: expected dict, got {type(regen).__name__}"
        assert set(regen) == set(ref), (
            f"{path}: key mismatch "
            f"(regen-only={set(regen) - set(ref)}, ref-only={set(ref) - set(regen)})"
        )
        for key in ref:
            _compare(regen[key], ref[key], f"{path}.{key}", rtol)
    elif isinstance(ref, list) and ref and isinstance(ref[0], (bool, int, float)):
        a = np.asarray(regen, dtype=float)
        b = np.asarray(ref, dtype=float)
        assert a.shape == b.shape, f"{path}: shape {a.shape} != {b.shape}"
        if rtol is None:
            assert np.array_equal(a, b), f"{path}: not bit-identical (max abs diff {_maxdiff(a, b)})"
        else:
            assert np.allclose(a, b, rtol=rtol, atol=ODE_ATOL, equal_nan=True), (
                f"{path}: exceeds tolerance rtol={rtol} (max abs diff {_maxdiff(a, b)})"
            )
    elif isinstance(ref, list):
        assert len(regen) == len(ref), f"{path}: length {len(regen)} != {len(ref)}"
        for idx, (r, e) in enumerate(zip(regen, ref)):
            _compare(r, e, f"{path}[{idx}]", rtol)
    elif isinstance(ref, bool):
        assert regen == ref, f"{path}: {regen!r} != {ref!r}"
    elif isinstance(ref, (int, float)):
        if rtol is None:
            assert regen == ref, f"{path}: {regen!r} != {ref!r} (expected bit-identical)"
        else:
            assert np.isclose(regen, ref, rtol=rtol, atol=ODE_ATOL, equal_nan=True), (
                f"{path}: {regen!r} != {ref!r} (rtol={rtol})"
            )
    else:
        assert regen == ref, f"{path}: {regen!r} != {ref!r}"


@pytest.fixture(scope="module")
def regenerated():
    """Regenerate the full results dict once and share it across the tests."""
    return run_experiments.run_all()


def test_scenarios_present(regenerated):
    """The regenerated run exposes exactly the committed scenarios."""
    assert set(regenerated) - EXCLUDED_KEYS == set(_COMMITTED) - EXCLUDED_KEYS


@pytest.mark.parametrize("key", _STRICT_KEYS)
def test_scenario_is_bit_identical(key, regenerated):
    """Each stochastic scenario reproduces the committed curves exactly."""
    _compare(regenerated[key], _COMMITTED[key], key, rtol=None)


def test_ode_validation_matches_within_tolerance(regenerated):
    """The solve_ivp-derived validation block matches to a tight tolerance."""
    _compare(regenerated[TOLERANCE_KEY], _COMMITTED[TOLERANCE_KEY], TOLERANCE_KEY, rtol=ODE_RTOL)
