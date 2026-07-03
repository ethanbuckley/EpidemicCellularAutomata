"""Tests for the analytical ODE and the mean-field validation (ode_reference.py)."""

import numpy as np
import pytest

import ode_reference as oref
from config import SimConfig

CFG = SimConfig()
N = oref.interior_n(CFG.n)


def test_interior_population():
    assert oref.interior_n(50) == 2304
    assert oref.interior_n(10) == 64


def test_p_to_rate():
    assert oref.p_to_rate(0.0) == 0.0
    assert oref.p_to_rate(0.5) == pytest.approx(np.log(2))
    # The log form exceeds the naive p ~= rate shortcut when p is not small.
    assert oref.p_to_rate(0.5) > 0.5
    assert oref.p_to_rate(0.1) < oref.p_to_rate(0.2)


def test_rates_use_combined_exit_hazard():
    r = oref.seiqr_rates(0.50, 0.10, 0.05, 0.10)
    assert r["sigma"] == pytest.approx(-np.log(0.5))
    assert r["delta"] == pytest.approx(-np.log(0.9))
    # Total I-exit hazard is the combined -ln(1 - (p_q + p_ri)), not the sum of
    # the two independent conversions.
    assert r["gamma_Q"] + r["gamma_R"] == pytest.approx(-np.log(1 - 0.15))
    # Branching ratio matches the CA's per-step probabilities p_q : p_ri.
    assert r["gamma_Q"] / r["gamma_R"] == pytest.approx(0.10 / 0.05)


def test_foi_saturating_and_low_prevalence_slope():
    p = 0.30
    i = 0.20
    assert oref.foi(i, p) == pytest.approx(-np.log1p(-(p * (1 - (1 - i) ** 8))))
    # Low-prevalence limit: lambda ~= beta * i with beta = 8 * p_expose.
    small = 1e-4
    assert oref.foi(small, p) == pytest.approx(oref.beta_lowprev(p) * small, rel=1e-2)
    assert oref.foi(0.1, p) < oref.foi(0.3, p)


def test_R0_matches_definition():
    r = oref.seiqr_rates(CFG.p_infect, CFG.p_quarantine, CFG.p_recover_i, CFG.p_recover_q)
    r0 = oref.R0(CFG.p_uniform, r)
    assert r0 == pytest.approx(8 * CFG.p_uniform / (r["gamma_Q"] + r["gamma_R"]))
    assert 14 < r0 < 16


def test_ode_conserves_compartments():
    out = oref.integrate_seiqr(CFG.p_uniform, CFG.p_infect, CFG.p_quarantine,
                               CFG.p_recover_i, CFG.p_recover_q, CFG.num_steps, N=N)
    frac_total = sum(out["frac"][k] for k in "SEIQR")
    assert np.allclose(frac_total, 1.0, atol=1e-6)
    count_total = sum(out["counts"][k] for k in "SEIQR")
    assert np.allclose(count_total, N, atol=1e-3)


def test_discrete_recursion_conserves_compartments():
    rec = oref.seiqr_discrete_meanfield(CFG.p_uniform, CFG.p_infect, CFG.p_quarantine,
                                        CFG.p_recover_i, CFG.p_recover_q, CFG.num_steps, N=N)
    total = sum(rec["frac"][k] for k in "SEIQR")
    assert np.allclose(total, 1.0, atol=1e-9)


def test_well_mixed_ca_reproduces_discrete_recursion():
    """Tight check: the global-coupling CA matches the exact discrete recursion.

    This is the core validation that the discrete->continuous rate mapping and
    the combined-hazard split are correct, with no continuous-time approximation.
    """
    rec = oref.seiqr_discrete_meanfield(CFG.p_uniform, CFG.p_infect, CFG.p_quarantine,
                                        CFG.p_recover_i, CFG.p_recover_q, CFG.num_steps, N=N)
    wm = oref.run_well_mixed_ensemble(10, CFG.n, CFG.p_uniform, CFG.p_infect,
                                      CFG.p_quarantine, CFG.p_recover_i, CFG.p_recover_q,
                                      CFG.num_steps, seed=0)
    m = oref.compare_curves(wm["mean"], rec["counts"], N)
    assert m["rmse_pct_of_peak"] < 5.0


def test_well_mixed_ca_tracks_ode():
    """The global-coupling CA reproduces the continuous ODE peak and attack rate."""
    ode = oref.integrate_seiqr(CFG.p_uniform, CFG.p_infect, CFG.p_quarantine,
                               CFG.p_recover_i, CFG.p_recover_q, CFG.num_steps, N=N)
    wm = oref.run_well_mixed_ensemble(10, CFG.n, CFG.p_uniform, CFG.p_infect,
                                      CFG.p_quarantine, CFG.p_recover_i, CFG.p_recover_q,
                                      CFG.num_steps, seed=0)
    m = oref.compare_curves(wm["mean"], ode["counts"], N)
    assert m["peak_rel_err_pct"] < 10.0
    assert abs(m["attack_ca"] - m["attack_ref"]) < 0.02


def test_local_ca_departs_from_well_mixed():
    """Spatial structure suppresses and delays the peak relative to the ODE limit."""
    wm = oref.run_well_mixed_ensemble(10, CFG.n, CFG.p_uniform, CFG.p_infect,
                                      CFG.p_quarantine, CFG.p_recover_i, CFG.p_recover_q,
                                      CFG.num_steps, seed=0)
    loc = oref.run_local_ensemble(10, CFG.n, CFG.p_uniform, CFG.p_infect,
                                  CFG.p_quarantine, CFG.p_recover_i, CFG.p_recover_q,
                                  CFG.num_steps, seed=0)
    assert loc["I_mean"].max() < 0.6 * wm["I_mean"].max()
    assert loc["I_mean"].argmax() > wm["I_mean"].argmax()
