"""
run_experiments.py
==================
Reproduces all scenarios from the original group report using the vectorized
model core (model.py). Results are saved to data/results.json for the
Streamlit dashboard to read without re-running simulations.

Scenarios
---------
1. Uniform vs density grid          (5 runs each)
2. Zone-level wave breakdown        (5 runs, density grid)
3. Lockdown comparison              (5 runs each: none / whole-grid / centre-only)
4. Vaccination comparison           (5 runs each: none / uniform / targeted)
5. Combined strategy                (5 runs each: none / targeted / blanket)
6. Epidemic threshold sweep         (3 runs each, ~20 baseline values)

All model parameters come from a SimConfig (see config.py); the ensemble sizes
and the threshold-sweep grid below are harness settings for this suite.

Usage
-----
    python run_experiments.py

Output: data/results.json
"""

import json
import os
import time
import datetime

import numpy as np

from config import SimConfig
from model import (
    make_density_map, vaccinate, run_seiqr, run_ensemble,
    S, E, I, Q, R,
    CENTRE_RADIUS, MIDDLE_RADIUS,
)

# =============================================================================
# HARNESS SETTINGS
# =============================================================================

CFG = SimConfig()  # model parameters (report Table 1 defaults)

N_RUNS_MAIN  = 5
N_RUNS_SWEEP = 3

THRESHOLD_BASELINES = np.round(np.arange(0.025, 0.525, 0.025), 3).tolist()

OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "results.json")


# =============================================================================
# HELPER: zone masks & zone-aware ensemble
# =============================================================================

def _zone_masks(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return boolean masks for centre / middle / outer zones."""
    mid  = n // 2
    rows = np.arange(n)
    dist = np.maximum(np.abs(rows[:, None] - mid), np.abs(rows[None, :] - mid))
    return (dist <= CENTRE_RADIUS,
            (dist > CENTRE_RADIUS) & (dist <= MIDDLE_RADIUS),
            dist > MIDDLE_RADIUS)


def _zone_sizes(n: int) -> dict:
    """Return the cell count of each zone for an n×n grid."""
    centre, middle, outer = _zone_masks(n)
    return {"centre": int(centre.sum()),
            "middle": int(middle.sum()),
            "outer":  int(outer.sum())}


def _ensemble_with_zones(cfg, n_runs, density_map, seed_base,
                          lockdown_map=None, lockdown_start=None,
                          lockdown_end=None, initial_grid_fn=None):
    """
    Run n_runs simulations, returning mean population curves AND
    mean per-zone infected fractions.  store_grids=True so we can
    compute zone breakdowns from the snapshots.
    """
    centre_mask, middle_mask, outer_mask = _zone_masks(cfg.n)
    zone_sizes = _zone_sizes(cfg.n)

    totals = {k: np.zeros(cfg.num_steps + 1) for k in ('S', 'E', 'I', 'Q', 'R')}
    zone_I = {"centre": np.zeros(cfg.num_steps + 1),
              "middle": np.zeros(cfg.num_steps + 1),
              "outer":  np.zeros(cfg.num_steps + 1)}

    for run in range(n_runs):
        rng  = np.random.default_rng(seed_base + run)
        init = initial_grid_fn(rng) if initial_grid_fn else None
        grids, Sc, Ec, Ic, Qc, Rc = run_seiqr(
            cfg.n, density_map,
            cfg.p_infect, cfg.p_quarantine, cfg.p_recover_i, cfg.p_recover_q,
            cfg.num_steps,
            lockdown_map=lockdown_map,
            lockdown_start=lockdown_start,
            lockdown_end=lockdown_end,
            initial_grid=init,
            rng=rng,
            store_grids=True,
        )
        totals['S'] += Sc;  totals['E'] += Ec
        totals['I'] += Ic;  totals['Q'] += Qc;  totals['R'] += Rc

        for t, g in enumerate(grids):
            infected = (g == I)
            zone_I["centre"][t] += infected[centre_mask].sum() / zone_sizes["centre"]
            zone_I["middle"][t] += infected[middle_mask].sum() / zone_sizes["middle"]
            zone_I["outer"][t]  += infected[outer_mask].sum()  / zone_sizes["outer"]

    curves = {k: (v / n_runs).tolist() for k, v in totals.items()}
    zones  = {k: (v / n_runs).tolist() for k, v in zone_I.items()}
    return curves, zones


# =============================================================================
# MAIN
# =============================================================================

def run_all(cfg: SimConfig = CFG):
    results = {}
    density_map = make_density_map(cfg.n, cfg.p_centre, cfg.p_middle, cfg.p_outer)
    uniform_map = make_density_map(cfg.n, cfg.p_uniform, cfg.p_uniform, cfg.p_uniform)

    # ------------------------------------------------------------------
    # 1. Uniform vs density grid
    # ------------------------------------------------------------------
    print("1/6  Uniform vs density grid …")
    results["uniform_grid"], _ = _ensemble_with_zones(
        cfg, N_RUNS_MAIN, uniform_map, seed_base=0)
    results["density_grid"], results["zone_breakdown"] = _ensemble_with_zones(
        cfg, N_RUNS_MAIN, density_map, seed_base=100)

    # ------------------------------------------------------------------
    # 2. Lockdown comparison (density grid, no vaccination)
    # ------------------------------------------------------------------
    print("2/6  Lockdown comparison …")
    # Whole-grid lockdown map
    lockdown_whole  = make_density_map(cfg.n, cfg.lockdown_p, cfg.lockdown_p, cfg.lockdown_p)
    # Centre-only lockdown map: only centre zone reduced
    lockdown_centre = make_density_map(cfg.n, cfg.lockdown_p, cfg.p_middle, cfg.p_outer)

    results["lockdown_none"], _ = _ensemble_with_zones(
        cfg, N_RUNS_MAIN, density_map, seed_base=200)
    results["lockdown_whole"], _ = _ensemble_with_zones(
        cfg, N_RUNS_MAIN, density_map, seed_base=300,
        lockdown_map=lockdown_whole,
        lockdown_start=cfg.lockdown_start, lockdown_end=cfg.lockdown_end)
    results["lockdown_centre"], _ = _ensemble_with_zones(
        cfg, N_RUNS_MAIN, density_map, seed_base=400,
        lockdown_map=lockdown_centre,
        lockdown_start=cfg.lockdown_start, lockdown_end=cfg.lockdown_end)

    # ------------------------------------------------------------------
    # 3. Vaccination comparison (density grid, no lockdown)
    # ------------------------------------------------------------------
    print("3/6  Vaccination comparison …")

    def _uniform_vax(rng):
        g = np.zeros((cfg.n, cfg.n)); g[cfg.n//2, cfg.n//2] = I
        return vaccinate(g, cfg.n, cfg.vax_doses, targeted=False,
                         efficacy=cfg.vax_efficacy, rng=rng)

    def _targeted_vax(rng):
        g = np.zeros((cfg.n, cfg.n)); g[cfg.n//2, cfg.n//2] = I
        return vaccinate(g, cfg.n, cfg.vax_doses, targeted=True,
                         efficacy=cfg.vax_efficacy, rng=rng)

    results["vax_none"], _ = _ensemble_with_zones(
        cfg, N_RUNS_MAIN, density_map, seed_base=500)
    results["vax_uniform"], _ = _ensemble_with_zones(
        cfg, N_RUNS_MAIN, density_map, seed_base=600,
        initial_grid_fn=_uniform_vax)
    results["vax_targeted"], _ = _ensemble_with_zones(
        cfg, N_RUNS_MAIN, density_map, seed_base=700,
        initial_grid_fn=_targeted_vax)

    # ------------------------------------------------------------------
    # 4. Combined strategy (4 arms)
    # ------------------------------------------------------------------
    print("4/6  Combined strategy …")

    def _blanket_init(rng):
        g = np.zeros((cfg.n, cfg.n)); g[cfg.n//2, cfg.n//2] = I
        return vaccinate(g, cfg.n, cfg.vax_doses, targeted=False,
                         efficacy=cfg.vax_efficacy, rng=rng)

    def _targeted_init(rng):
        g = np.zeros((cfg.n, cfg.n)); g[cfg.n//2, cfg.n//2] = I
        return vaccinate(g, cfg.n, cfg.vax_doses, targeted=True,
                         efficacy=cfg.vax_efficacy, rng=rng)

    results["combined_none"], _ = _ensemble_with_zones(
        cfg, N_RUNS_MAIN, density_map, seed_base=800)

    # Targeted vaccination + centre-only lockdown (report's best strategy)
    results["combined_targeted"], _ = _ensemble_with_zones(
        cfg, N_RUNS_MAIN, density_map, seed_base=900,
        lockdown_map=lockdown_centre,
        lockdown_start=cfg.lockdown_start, lockdown_end=cfg.lockdown_end,
        initial_grid_fn=_targeted_init)

    # Blanket: uniform vaccination + whole-grid lockdown
    results["combined_blanket"], _ = _ensemble_with_zones(
        cfg, N_RUNS_MAIN, density_map, seed_base=1000,
        lockdown_map=lockdown_whole,
        lockdown_start=cfg.lockdown_start, lockdown_end=cfg.lockdown_end,
        initial_grid_fn=_blanket_init)

    # Targeted vaccination alone (no lockdown) — arm 4
    results["combined_vax_only"], _ = _ensemble_with_zones(
        cfg, N_RUNS_MAIN, density_map, seed_base=1100,
        initial_grid_fn=_targeted_init)

    # ------------------------------------------------------------------
    # 5. Threshold sweep
    # ------------------------------------------------------------------
    print("5/6  Threshold sweep …")
    sweep_uniform_peak = []
    sweep_density_peak = []

    for baseline in THRESHOLD_BASELINES:
        # Uniform: all cells = baseline
        u_map = make_density_map(cfg.n, baseline, baseline, baseline)
        u_curves = run_ensemble(N_RUNS_SWEEP, cfg.n, u_map,
                                cfg.p_infect, cfg.p_quarantine,
                                cfg.p_recover_i, cfg.p_recover_q,
                                cfg.num_steps, seed=2000)
        sweep_uniform_peak.append(float(max(u_curves['I'])))

        # Density: centre=2×, middle=1×, outer=0.3×  (from report Fig 6)
        d_map = make_density_map(cfg.n,
                                  min(baseline * 2.0, 1.0),
                                  min(baseline * 1.0, 1.0),
                                  min(baseline * 0.3, 1.0))
        d_curves = run_ensemble(N_RUNS_SWEEP, cfg.n, d_map,
                                cfg.p_infect, cfg.p_quarantine,
                                cfg.p_recover_i, cfg.p_recover_q,
                                cfg.num_steps, seed=3000)
        sweep_density_peak.append(float(max(d_curves['I'])))

    results["threshold_sweep"] = {
        "baselines":     THRESHOLD_BASELINES,
        "uniform_peak":  sweep_uniform_peak,
        "density_peak":  sweep_density_peak,
    }

    # ------------------------------------------------------------------
    # 6. Speedup measurement (stored once for dashboard display)
    # ------------------------------------------------------------------
    print("6/6  Measuring speedup …")
    from seiqr import run_seiqr as old_run

    REPS = 3
    t0 = time.perf_counter()
    for _ in range(REPS):
        old_run(cfg.n, density_map, cfg.p_infect, cfg.p_quarantine,
                cfg.p_recover_i, cfg.p_recover_q, cfg.num_steps)
    t_old = (time.perf_counter() - t0) / REPS

    t0 = time.perf_counter()
    for _ in range(REPS):
        run_seiqr(cfg.n, density_map, cfg.p_infect, cfg.p_quarantine,
                  cfg.p_recover_i, cfg.p_recover_q, cfg.num_steps,
                  rng=np.random.default_rng())
    t_new = (time.perf_counter() - t0) / REPS

    results["speedup"] = {
        "original_s":    round(t_old, 3),
        "vectorized_s":  round(t_new, 4),
        "speedup_x":     round(t_old / t_new, 0),
    }

    results["zone_sizes"]    = _zone_sizes(cfg.n)
    results["generated_at"]  = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    results["lockdown_window"] = [cfg.lockdown_start, cfg.lockdown_end]

    return results


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Running all experiment scenarios …\n")
    t_start = time.perf_counter()
    results = run_all()
    elapsed = time.perf_counter() - t_start

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone in {elapsed:.1f}s — results written to {OUTPUT_PATH}")
    print(f"  File size: {os.path.getsize(OUTPUT_PATH) / 1024:.1f} KB")
    print(f"  Speedup recorded: {results['speedup']['speedup_x']:.0f}×")
    print(f"  Scenarios: {[k for k in results if k not in ('generated_at','speedup','zone_sizes','lockdown_window')]}")
