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

Usage
-----
    python run_experiments.py

Output: data/results.json  (~50 KB)
"""

import json
import os
import time
import datetime

import numpy as np

from model import (
    make_density_map, vaccinate, run_seiqr, run_ensemble,
    S, E, I, Q, R,
)

# =============================================================================
# FIXED PARAMETERS (from report Table 1)
# =============================================================================

N            = 50
NUM_STEPS    = 100
P_INFECT     = 0.50
P_QUARANTINE = 0.10
P_RECOVER_I  = 0.05
P_RECOVER_Q  = 0.10

P_CENTRE = 0.50
P_MIDDLE = 0.30
P_OUTER  = 0.15
P_UNIFORM = 0.30

LOCKDOWN_P        = 0.10
LOCKDOWN_START    = 10
LOCKDOWN_END      = 40

VAX_DOSES    = 200
VAX_EFFICACY = 0.80

N_RUNS_MAIN    = 5
N_RUNS_SWEEP   = 3

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
    return dist <= 8, (dist > 8) & (dist <= 16), dist > 16


CENTRE_MASK, MIDDLE_MASK, OUTER_MASK = _zone_masks(N)
ZONE_SIZES = {
    "centre": int(CENTRE_MASK.sum()),
    "middle": int(MIDDLE_MASK.sum()),
    "outer":  int(OUTER_MASK.sum()),
}


def _ensemble_with_zones(n_runs, density_map, seed_base,
                          lockdown_map=None, lockdown_start=None,
                          lockdown_end=None, initial_grid_fn=None):
    """
    Run n_runs simulations, returning mean population curves AND
    mean per-zone infected fractions.  store_grids=True so we can
    compute zone breakdowns from the snapshots.
    """
    totals = {k: np.zeros(NUM_STEPS + 1) for k in ('S', 'E', 'I', 'Q', 'R')}
    zone_I = {"centre": np.zeros(NUM_STEPS + 1),
              "middle": np.zeros(NUM_STEPS + 1),
              "outer":  np.zeros(NUM_STEPS + 1)}

    for run in range(n_runs):
        rng  = np.random.default_rng(seed_base + run)
        init = initial_grid_fn(rng) if initial_grid_fn else None
        grids, Sc, Ec, Ic, Qc, Rc = run_seiqr(
            N, density_map, P_INFECT, P_QUARANTINE, P_RECOVER_I, P_RECOVER_Q,
            NUM_STEPS,
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
            zone_I["centre"][t] += infected[CENTRE_MASK].sum() / ZONE_SIZES["centre"]
            zone_I["middle"][t] += infected[MIDDLE_MASK].sum() / ZONE_SIZES["middle"]
            zone_I["outer"][t]  += infected[OUTER_MASK].sum()  / ZONE_SIZES["outer"]

    curves = {k: (v / n_runs).tolist() for k, v in totals.items()}
    zones  = {k: (v / n_runs).tolist() for k, v in zone_I.items()}
    return curves, zones


def _curves_to_lists(d: dict) -> dict:
    return {k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in d.items()}


# =============================================================================
# MAIN
# =============================================================================

def run_all():
    results = {}
    density_map = make_density_map(N, P_CENTRE, P_MIDDLE, P_OUTER)
    uniform_map = make_density_map(N, P_UNIFORM, P_UNIFORM, P_UNIFORM)

    # ------------------------------------------------------------------
    # 1. Uniform vs density grid
    # ------------------------------------------------------------------
    print("1/6  Uniform vs density grid …")
    results["uniform_grid"], _ = _ensemble_with_zones(
        N_RUNS_MAIN, uniform_map, seed_base=0)
    results["density_grid"], results["zone_breakdown"] = _ensemble_with_zones(
        N_RUNS_MAIN, density_map, seed_base=100)

    # ------------------------------------------------------------------
    # 2. Lockdown comparison (density grid, no vaccination)
    # ------------------------------------------------------------------
    print("2/6  Lockdown comparison …")
    # Whole-grid lockdown map
    lockdown_whole  = make_density_map(N, LOCKDOWN_P, LOCKDOWN_P, LOCKDOWN_P)
    # Centre-only lockdown map: only centre zone reduced
    lockdown_centre = make_density_map(N, LOCKDOWN_P, P_MIDDLE, P_OUTER)

    results["lockdown_none"], _ = _ensemble_with_zones(
        N_RUNS_MAIN, density_map, seed_base=200)
    results["lockdown_whole"], _ = _ensemble_with_zones(
        N_RUNS_MAIN, density_map, seed_base=300,
        lockdown_map=lockdown_whole,
        lockdown_start=LOCKDOWN_START, lockdown_end=LOCKDOWN_END)
    results["lockdown_centre"], _ = _ensemble_with_zones(
        N_RUNS_MAIN, density_map, seed_base=400,
        lockdown_map=lockdown_centre,
        lockdown_start=LOCKDOWN_START, lockdown_end=LOCKDOWN_END)

    # ------------------------------------------------------------------
    # 3. Vaccination comparison (density grid, no lockdown)
    # ------------------------------------------------------------------
    print("3/6  Vaccination comparison …")

    def _uniform_vax(rng):
        g = np.zeros((N, N)); g[N//2, N//2] = I
        return vaccinate(g, N, VAX_DOSES, targeted=False,
                         efficacy=VAX_EFFICACY, rng=rng)

    def _targeted_vax(rng):
        g = np.zeros((N, N)); g[N//2, N//2] = I
        return vaccinate(g, N, VAX_DOSES, targeted=True,
                         efficacy=VAX_EFFICACY, rng=rng)

    results["vax_none"], _ = _ensemble_with_zones(
        N_RUNS_MAIN, density_map, seed_base=500)
    results["vax_uniform"], _ = _ensemble_with_zones(
        N_RUNS_MAIN, density_map, seed_base=600,
        initial_grid_fn=_uniform_vax)
    results["vax_targeted"], _ = _ensemble_with_zones(
        N_RUNS_MAIN, density_map, seed_base=700,
        initial_grid_fn=_targeted_vax)

    # ------------------------------------------------------------------
    # 4. Combined strategy (4 arms)
    # ------------------------------------------------------------------
    print("4/6  Combined strategy …")

    def _blanket_init(rng):
        g = np.zeros((N, N)); g[N//2, N//2] = I
        return vaccinate(g, N, VAX_DOSES, targeted=False,
                         efficacy=VAX_EFFICACY, rng=rng)

    def _targeted_init(rng):
        g = np.zeros((N, N)); g[N//2, N//2] = I
        return vaccinate(g, N, VAX_DOSES, targeted=True,
                         efficacy=VAX_EFFICACY, rng=rng)

    results["combined_none"], _ = _ensemble_with_zones(
        N_RUNS_MAIN, density_map, seed_base=800)

    # Targeted vaccination + centre-only lockdown (report's best strategy)
    results["combined_targeted"], _ = _ensemble_with_zones(
        N_RUNS_MAIN, density_map, seed_base=900,
        lockdown_map=lockdown_centre,
        lockdown_start=LOCKDOWN_START, lockdown_end=LOCKDOWN_END,
        initial_grid_fn=_targeted_init)

    # Blanket: uniform vaccination + whole-grid lockdown
    results["combined_blanket"], _ = _ensemble_with_zones(
        N_RUNS_MAIN, density_map, seed_base=1000,
        lockdown_map=lockdown_whole,
        lockdown_start=LOCKDOWN_START, lockdown_end=LOCKDOWN_END,
        initial_grid_fn=_blanket_init)

    # Targeted vaccination alone (no lockdown) — arm 4
    results["combined_vax_only"], _ = _ensemble_with_zones(
        N_RUNS_MAIN, density_map, seed_base=1100,
        initial_grid_fn=_targeted_init)

    # ------------------------------------------------------------------
    # 5. Threshold sweep
    # ------------------------------------------------------------------
    print("5/6  Threshold sweep …")
    sweep_uniform_peak = []
    sweep_density_peak = []

    for baseline in THRESHOLD_BASELINES:
        # Uniform: all cells = baseline
        u_map = make_density_map(N, baseline, baseline, baseline)
        u_curves = run_ensemble(N_RUNS_SWEEP, N, u_map,
                                P_INFECT, P_QUARANTINE, P_RECOVER_I, P_RECOVER_Q,
                                NUM_STEPS, seed=2000)
        sweep_uniform_peak.append(float(max(u_curves['I'])))

        # Density: centre=2×, middle=1×, outer=0.3×  (from report Fig 6)
        d_map = make_density_map(N,
                                  min(baseline * 2.0, 1.0),
                                  min(baseline * 1.0, 1.0),
                                  min(baseline * 0.3, 1.0))
        d_curves = run_ensemble(N_RUNS_SWEEP, N, d_map,
                                P_INFECT, P_QUARANTINE, P_RECOVER_I, P_RECOVER_Q,
                                NUM_STEPS, seed=3000)
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
        old_run(N, density_map, P_INFECT, P_QUARANTINE, P_RECOVER_I, P_RECOVER_Q, NUM_STEPS)
    t_old = (time.perf_counter() - t0) / REPS

    t0 = time.perf_counter()
    for _ in range(REPS):
        run_seiqr(N, density_map, P_INFECT, P_QUARANTINE, P_RECOVER_I, P_RECOVER_Q,
                  NUM_STEPS, rng=np.random.default_rng())
    t_new = (time.perf_counter() - t0) / REPS

    results["speedup"] = {
        "original_s":    round(t_old, 3),
        "vectorized_s":  round(t_new, 4),
        "speedup_x":     round(t_old / t_new, 0),
    }

    results["zone_sizes"]    = ZONE_SIZES
    results["generated_at"]  = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    results["lockdown_window"] = [LOCKDOWN_START, LOCKDOWN_END]

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
