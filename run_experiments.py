"""
run_experiments.py: reproduce every scenario from the original group report.

Uses the vectorised model core (model.py). Results are saved to
data/results.json for the Streamlit dashboard to read without re-running
simulations.

Scenarios
---------
1. Uniform vs density grid          (5 runs each)
2. Zone-level wave breakdown        (5 runs, density grid)
3. Lockdown comparison              (5 runs each: none / whole-grid / centre-only)
4. Vaccination comparison           (5 runs each: none / uniform / targeted)
5. Combined strategy                (5 runs each: none / targeted / blanket / vax only)
6. Epidemic threshold sweep         (3 runs each, 20 baseline values)
7. Analytical ODE validation        (well-mixed limit vs local CA, 20 runs)

All model parameters come from a SimConfig (see config.py); the ensemble sizes
and the threshold-sweep grid below are harness settings for this suite.

Usage
-----
    python run_experiments.py

The speedup measurement times the original seiqr.py, which imports matplotlib,
so this script needs the dev requirements (requirements-dev.txt), not just
requirements.txt.

Output: data/results.json
"""

import datetime
import json
import os
import time

import numpy as np

import ode_reference as ode_ref
from config import SimConfig
from model import (
    STATE_NAMES,
    I,
    initial_grid_single_seed,
    make_density_map,
    run_ensemble,
    run_seiqr,
    vaccinate,
    zone_masks,
)

# =============================================================================
# HARNESS SETTINGS
# =============================================================================

CFG = SimConfig()  # model parameters (report Table 1 defaults)

N_RUNS_MAIN = 5
N_RUNS_SWEEP = 3
N_RUNS_ODE = 20
SPEEDUP_REPS = 3

THRESHOLD_BASELINES = np.round(np.arange(0.025, 0.525, 0.025), 3).tolist()

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "results.json")

# Keys in the results dict that are not scenario curves.
METADATA_KEYS = ("generated_at", "speedup", "zone_sizes", "lockdown_window")


# =============================================================================
# HELPERS
# =============================================================================


def _ensemble_with_zones(
    cfg,
    n_runs,
    density_map,
    seed_base,
    lockdown_map=None,
    lockdown_start=None,
    lockdown_end=None,
    initial_grid_fn=None,
):
    """Run n_runs simulations; return mean population curves and per-zone infected fractions.

    Grids are stored so the zone breakdown can be computed from the snapshots.
    Run ``k`` uses ``np.random.default_rng(seed_base + k)``; ``initial_grid_fn``,
    if given, receives that generator and returns the starting grid.
    """
    masks = zone_masks(cfg.n)
    zone_sizes = {name: int(mask.sum()) for name, mask in masks.items()}

    totals = {k: np.zeros(cfg.num_steps + 1) for k in STATE_NAMES}
    zone_I = {name: np.zeros(cfg.num_steps + 1) for name in masks}

    for run in range(n_runs):
        rng = np.random.default_rng(seed_base + run)
        init = initial_grid_fn(rng) if initial_grid_fn else None
        grids, *curves = run_seiqr(
            cfg.n,
            density_map,
            cfg.p_infect,
            cfg.p_quarantine,
            cfg.p_recover_i,
            cfg.p_recover_q,
            cfg.num_steps,
            lockdown_map=lockdown_map,
            lockdown_start=lockdown_start,
            lockdown_end=lockdown_end,
            initial_grid=init,
            rng=rng,
            store_grids=True,
        )
        for k, curve in zip(STATE_NAMES, curves, strict=True):
            totals[k] += curve

        infected = np.array([g == I for g in grids])  # (T, n, n)
        for name, mask in masks.items():
            zone_I[name] += infected[:, mask].sum(axis=1) / zone_sizes[name]

    curves = {k: (v / n_runs).tolist() for k, v in totals.items()}
    zones = {k: (v / n_runs).tolist() for k, v in zone_I.items()}
    return curves, zones


def _vaccinated_init(cfg, targeted):
    """Return an initial-grid factory: single centre seed plus pre-simulation vaccination."""

    def init(rng):
        grid = initial_grid_single_seed(cfg.n)
        return vaccinate(grid, cfg.vax_doses, targeted=targeted, efficacy=cfg.vax_efficacy, rng=rng)

    return init


def _ode_validation(cfg, n_runs=N_RUNS_ODE):
    """Scenario 7: validate the CA against the analytical SEIQR ODE.

    On the uniform grid (p_expose = p_uniform) the well-mixed limit applies, so
    a global-coupling CA ensemble should track the ODE; the exact discrete
    recursion is the tightest reference. The standard local CA is included for
    contrast: its departure is the spatial effect, not a validation failure.

    Runs that die out before a major outbreak are counted and reported
    separately (n_extinct, attack_major); the mean curves include them.
    """
    p = cfg.p_uniform
    N = ode_ref.interior_n(cfg.n)
    rate_args = (cfg.p_infect, cfg.p_quarantine, cfg.p_recover_i, cfg.p_recover_q)

    ode = ode_ref.integrate_seiqr(p, *rate_args, cfg.num_steps, N=N)
    rec = ode_ref.seiqr_discrete_meanfield(p, *rate_args, cfg.num_steps, N=N)
    wm = ode_ref.run_well_mixed_ensemble(n_runs, cfg.n, p, *rate_args, cfg.num_steps, seed=4000)
    loc = ode_ref.run_local_ensemble(n_runs, cfg.n, p, *rate_args, cfg.num_steps, seed=5000)

    rates = {k: float(v) for k, v in ode["rates"].items()}
    rates["beta_lowprev"] = float(ode_ref.beta_lowprev(p))
    rates["R0"] = float(ode["R0"])

    return {
        "params": {
            "p_expose": p,
            "p_infect": cfg.p_infect,
            "p_quarantine": cfg.p_quarantine,
            "p_recover_i": cfg.p_recover_i,
            "p_recover_q": cfg.p_recover_q,
        },
        "N_interior": N,
        "n_runs": n_runs,
        "major_outbreak_threshold": ode_ref.MAJOR_OUTBREAK_THRESHOLD,
        "rates": rates,
        "ode": {k: ode["counts"][k].tolist() for k in STATE_NAMES},
        "discrete": {"I": rec["counts"]["I"].tolist()},
        "well_mixed_ca": {
            **{k: wm["mean"][k].tolist() for k in STATE_NAMES},
            "I_std": wm["I_std"].tolist(),
            "n_extinct": wm["n_extinct"],
            "attack_all": wm["attack_all"],
            "attack_major": wm["attack_major"],
        },
        "local_ca": {"I": loc["I_mean"].tolist(), "n_extinct": loc["n_extinct"]},
        "metrics": {
            "vs_ode": ode_ref.compare_curves(wm["mean"], ode["counts"], N),
            "vs_discrete": ode_ref.compare_curves(wm["mean"], rec["counts"], N),
        },
    }


def _measure_speedup(cfg, density_map, reps=SPEEDUP_REPS):
    """Wall-clock time per run for the original loop model and the vectorised one."""
    from seiqr import run_seiqr as old_run  # imports matplotlib; dev requirement only

    def timed(fn):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        return (time.perf_counter() - t0) / reps

    args = (
        cfg.n,
        density_map,
        cfg.p_infect,
        cfg.p_quarantine,
        cfg.p_recover_i,
        cfg.p_recover_q,
        cfg.num_steps,
    )
    t_old = timed(lambda: old_run(*args))
    t_new = timed(lambda: run_seiqr(*args, rng=np.random.default_rng()))
    return {
        "original_s": round(t_old, 3),
        "vectorized_s": round(t_new, 4),
        "speedup_x": round(t_old / t_new, 0),
    }


# =============================================================================
# MAIN
# =============================================================================


def run_all(cfg: SimConfig = CFG):
    """Run every scenario and return the results dict written to results.json.

    The curve-producing scenarios use fixed per-run seeds, so the output is
    reproducible; only the speedup timings and the generated_at timestamp vary
    between runs.
    """
    results = {}
    density_map = make_density_map(cfg.n, cfg.p_centre, cfg.p_middle, cfg.p_outer)
    uniform_map = make_density_map(cfg.n, cfg.p_uniform, cfg.p_uniform, cfg.p_uniform)

    lockdown_whole = make_density_map(cfg.n, cfg.lockdown_p, cfg.lockdown_p, cfg.lockdown_p)
    lockdown_centre = make_density_map(cfg.n, cfg.lockdown_p, cfg.p_middle, cfg.p_outer)
    lockdown_window = dict(lockdown_start=cfg.lockdown_start, lockdown_end=cfg.lockdown_end)

    uniform_vax = _vaccinated_init(cfg, targeted=False)
    targeted_vax = _vaccinated_init(cfg, targeted=True)

    def ensemble(seed_base, **kwargs):
        return _ensemble_with_zones(cfg, N_RUNS_MAIN, density_map, seed_base, **kwargs)

    # 1. Uniform vs density grid, with the zone-level wave breakdown
    print("1/7  Uniform vs density grid ...")
    results["uniform_grid"], _ = _ensemble_with_zones(cfg, N_RUNS_MAIN, uniform_map, seed_base=0)
    results["density_grid"], results["zone_breakdown"] = ensemble(seed_base=100)

    # 2. Lockdown comparison (density grid, no vaccination)
    print("2/7  Lockdown comparison ...")
    results["lockdown_none"], _ = ensemble(seed_base=200)
    results["lockdown_whole"], _ = ensemble(
        seed_base=300, lockdown_map=lockdown_whole, **lockdown_window
    )
    results["lockdown_centre"], _ = ensemble(
        seed_base=400, lockdown_map=lockdown_centre, **lockdown_window
    )

    # 3. Vaccination comparison (density grid, no lockdown)
    print("3/7  Vaccination comparison ...")
    results["vax_none"], _ = ensemble(seed_base=500)
    results["vax_uniform"], _ = ensemble(seed_base=600, initial_grid_fn=uniform_vax)
    results["vax_targeted"], _ = ensemble(seed_base=700, initial_grid_fn=targeted_vax)

    # 4. Combined strategy (4 arms)
    print("4/7  Combined strategy ...")
    results["combined_none"], _ = ensemble(seed_base=800)
    # Targeted vaccination + centre-only lockdown (the report's best strategy)
    results["combined_targeted"], _ = ensemble(
        seed_base=900, lockdown_map=lockdown_centre, initial_grid_fn=targeted_vax, **lockdown_window
    )
    # Blanket: uniform vaccination + whole-grid lockdown
    results["combined_blanket"], _ = ensemble(
        seed_base=1000, lockdown_map=lockdown_whole, initial_grid_fn=uniform_vax, **lockdown_window
    )
    # Targeted vaccination alone (no lockdown)
    results["combined_vax_only"], _ = ensemble(seed_base=1100, initial_grid_fn=targeted_vax)

    # 5. Threshold sweep
    print("5/7  Threshold sweep ...")
    rate_args = (cfg.p_infect, cfg.p_quarantine, cfg.p_recover_i, cfg.p_recover_q)
    sweep_uniform_peak, sweep_density_peak = [], []
    for baseline in THRESHOLD_BASELINES:
        u_map = make_density_map(cfg.n, baseline, baseline, baseline)
        u_curves = run_ensemble(N_RUNS_SWEEP, cfg.n, u_map, *rate_args, cfg.num_steps, seed=2000)
        sweep_uniform_peak.append(float(max(u_curves["I"])))

        # Density: centre = 2x, middle = 1x, outer = 0.3x the baseline (report Fig 6)
        d_map = make_density_map(cfg.n, min(2.0 * baseline, 1.0), baseline, 0.3 * baseline)
        d_curves = run_ensemble(N_RUNS_SWEEP, cfg.n, d_map, *rate_args, cfg.num_steps, seed=3000)
        sweep_density_peak.append(float(max(d_curves["I"])))

    results["threshold_sweep"] = {
        "baselines": THRESHOLD_BASELINES,
        "uniform_peak": sweep_uniform_peak,
        "density_peak": sweep_density_peak,
    }

    # 6. Speedup measurement (stored once for dashboard display)
    print("6/7  Measuring speedup ...")
    results["speedup"] = _measure_speedup(cfg, density_map)

    # 7. Analytical ODE validation (well-mixed limit)
    print("7/7  ODE validation ...")
    results["ode_validation"] = _ode_validation(cfg)

    results["zone_sizes"] = {name: int(mask.sum()) for name, mask in zone_masks(cfg.n).items()}
    results["generated_at"] = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    results["lockdown_window"] = [cfg.lockdown_start, cfg.lockdown_end]

    return results


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Running all experiment scenarios ...\n")
    t_start = time.perf_counter()
    results = run_all()
    elapsed = time.perf_counter() - t_start

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone in {elapsed:.1f}s; results written to {OUTPUT_PATH}")
    print(f"  File size: {os.path.getsize(OUTPUT_PATH) / 1024:.1f} KB")
    print(f"  Speedup recorded: {results['speedup']['speedup_x']:.0f}x")
    print(
        f"  Extinct well-mixed runs: "
        f"{results['ode_validation']['well_mixed_ca']['n_extinct']} of "
        f"{results['ode_validation']['n_runs']}"
    )
    print(f"  Scenarios: {[k for k in results if k not in METADATA_KEYS]}")
