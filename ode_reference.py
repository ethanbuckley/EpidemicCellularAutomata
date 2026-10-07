"""
ode_reference.py: analytical SEIQR ODE and the mean-field validation of the CA.

The cellular automaton in model.py is spatial and stochastic. This module adds
the classical well-mixed SEIQR compartmental ODE that the CA rules reduce to in
the mean-field limit, and the tools to validate that reduction.

The claim this supports: the CA's local update rules reduce to the ODE when the
population is well mixed; driven by the global infected fraction rather than
local neighbours, a CA ensemble reproduces the ODE to within stochastic
scatter. The standard local (spatial) CA departs from the ODE, and that
departure (a lower, later, broader peak) is the genuine effect of spatial
structure, not a validation failure.

Key modelling choices (see the README Validation section for the full argument):
- Per-timestep probabilities map to continuous rates by rho = -ln(1 - p). The
  common rho ~= p shortcut is not used because p_infect = 0.50 is not small
  (it is 28% wrong there).
- The CA exposure rule fires on the presence of at least one infected Moore
  neighbour, so it saturates in the local infected count. Its mean-field limit
  is the force of infection lambda(i) = -ln(1 - p_expose * (1 - (1 - i) ** 8)),
  which linearises to a frequency-dependent beta = 8 * p_expose at low prevalence.
- The single-draw I -> Q / I -> R competition maps to a combined exit hazard
  rho_tot = -ln(1 - p_quarantine - p_recover_i) split in the ratio
  p_quarantine : p_recover_i, which preserves the CA's mean sojourn time and
  quarantine fraction exactly.
- Comparisons use the participating interior population N = (n - 2) ** 2 (the
  grid excluding the permanently-susceptible border), so the CA and ODE share a
  denominator.
- A stochastic run started from one infected cell can die out before the
  epidemic takes off; the deterministic references cannot. Ensemble functions
  therefore report how many runs went extinct, and the attack rate both over
  all runs and over major outbreaks only. Averaging extinct runs into the mean
  lowers the peak and attack rate for a reason unrelated to the rate mapping.
"""

import numpy as np
from scipy.integrate import solve_ivp

from model import (
    MOORE_NEIGHBOURS,
    STATE_NAMES,
    STATES,
    I,
    apply_transitions,
    initial_grid_single_seed,
    interior_mask,
    make_density_map,
    run_seiqr,
    state_counts,
)

# A run counts as a major outbreak if at least this fraction of the interior
# population is ever infected; below it the seed died out.
MAJOR_OUTBREAK_THRESHOLD = 0.10


def interior_n(n: int) -> int:
    """Participating population: the interior grid excluding the frozen border."""
    return (n - 2) ** 2


# =============================================================================
# Rate mapping
# =============================================================================


def p_to_rate(p: float) -> float:
    """Convert a per-timestep probability to a continuous hazard rate.

    Solves exp(-rate * 1) = 1 - p for a unit timestep, i.e. rate = -ln(1 - p).
    Uses log1p for accuracy as p -> 0.
    """
    return -np.log1p(-p)


def seiqr_rates(
    p_infect: float, p_quarantine: float, p_recover_i: float, p_recover_q: float
) -> dict[str, float]:
    """Map the CA per-step transition probabilities to ODE rates.

    E -> I and Q -> R are single exits (sigma, delta). I -> Q and I -> R
    compete within one CA step (a single uniform draw decides), so they share a
    combined leaving hazard split by the branching ratio; converting each
    independently and summing would mis-split the branches.
    """
    sigma = p_to_rate(p_infect)
    delta = p_to_rate(p_recover_q)

    p_tot = p_quarantine + p_recover_i
    if p_tot > 0:
        rho_tot = p_to_rate(p_tot)
        gamma_Q = rho_tot * p_quarantine / p_tot
        gamma_R = rho_tot * p_recover_i / p_tot
    else:
        gamma_Q = gamma_R = 0.0

    return {"sigma": sigma, "gamma_Q": gamma_Q, "gamma_R": gamma_R, "delta": delta}


def p_expose_well_mixed(i: float | np.ndarray, p_expose: float) -> float | np.ndarray:
    """Per-step exposure probability when a fraction ``i`` of the population is infectious.

    A susceptible has at least one infected cell among its 8 random neighbours
    with probability 1 - (1 - i) ** 8; the CA rule then exposes it with
    probability ``p_expose``. This is the exact discrete quantity the well-mixed
    CA uses; ``foi`` is its continuous-rate form.
    """
    return p_expose * (1.0 - (1.0 - i) ** MOORE_NEIGHBOURS)


def foi(i: float | np.ndarray, p_expose: float) -> float | np.ndarray:
    """Mean-field force of infection: the rate form of the saturating CA rule."""
    arg = np.minimum(p_expose_well_mixed(i, p_expose), 1.0 - 1e-12)  # keep log1p in-domain
    return -np.log1p(-arg)


def beta_lowprev(p_expose: float) -> float:
    """Low-prevalence linear slope of the force of infection: beta = 8 * p_expose."""
    return MOORE_NEIGHBOURS * p_expose


def R0(p_expose: float, rates: dict[str, float]) -> float:
    """Well-mixed basic reproduction number beta / (gamma_Q + gamma_R).

    This is the mean-field R0; the spatial CA's effective reproduction number is
    far lower because a cell's infected neighbours are quickly depleted.
    """
    return beta_lowprev(p_expose) / (rates["gamma_Q"] + rates["gamma_R"])


# =============================================================================
# Analytical references
# =============================================================================


def _initial_fractions(N: int | None, i0: float | None) -> tuple[int, float]:
    """Resolve the population and initial infected fraction defaults."""
    if N is None:
        N = interior_n(50)
    if i0 is None:
        i0 = 1.0 / N
    return N, i0


def integrate_seiqr(
    p_expose: float,
    p_infect: float,
    p_quarantine: float,
    p_recover_i: float,
    p_recover_q: float,
    num_steps: int,
    N: int | None = None,
    i0: float | None = None,
) -> dict:
    """Integrate the continuous well-mixed SEIQR ODE over [0, num_steps].

    Returns fraction and count curves sampled at each integer timestep, the
    derived rates, and R0. Counts scale the fractions by the population N.
    """
    N, i0 = _initial_fractions(N, i0)
    rates = seiqr_rates(p_infect, p_quarantine, p_recover_i, p_recover_q)
    sigma, gamma_Q, gamma_R, delta = (rates[k] for k in ("sigma", "gamma_Q", "gamma_R", "delta"))

    def rhs(t, y):
        s, e, i, q, r = y
        lam = foi(max(i, 0.0), p_expose)
        return [
            -lam * s,
            lam * s - sigma * e,
            sigma * e - (gamma_Q + gamma_R) * i,
            gamma_Q * i - delta * q,
            gamma_R * i + delta * q,
        ]

    y0 = [1.0 - i0, 0.0, i0, 0.0, 0.0]
    sol = solve_ivp(
        rhs,
        (0.0, num_steps),
        y0,
        t_eval=np.arange(num_steps + 1),
        rtol=1e-8,
        atol=1e-10,
        method="RK45",
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")

    frac = dict(zip(STATE_NAMES, sol.y, strict=True))
    drift = float(np.max(np.abs(sum(frac.values()) - 1.0)))
    if drift >= 1e-6:
        raise RuntimeError(f"ODE compartments not conserved (drift {drift:g})")

    counts = {k: v * N for k, v in frac.items()}
    return {"frac": frac, "counts": counts, "rates": rates, "R0": R0(p_expose, rates), "N": N}


def seiqr_discrete_meanfield(
    p_expose: float,
    p_infect: float,
    p_quarantine: float,
    p_recover_i: float,
    p_recover_q: float,
    num_steps: int,
    N: int | None = None,
    i0: float | None = None,
) -> dict:
    """The exact discrete recursion the CA implements in the well-mixed limit.

    This is the deterministic expectation of well_mixed_advance (below); it uses
    the raw per-step probabilities, not the continuous-time rates, so it carries
    no discretisation approximation. It is the tightest reference for the
    well-mixed CA ensemble.
    """
    N, i0 = _initial_fractions(N, i0)

    state = np.array([1.0 - i0, 0.0, i0, 0.0, 0.0])
    history = [state]

    for _ in range(num_steps):
        s, e, i, q, r = state
        new_exp = s * p_expose_well_mixed(i, p_expose)
        new_inf = e * p_infect
        i_to_q = i * p_quarantine
        i_to_r = i * p_recover_i
        q_to_r = q * p_recover_q
        state = np.array(
            [
                s - new_exp,
                e + new_exp - new_inf,
                i + new_inf - i_to_q - i_to_r,
                q + i_to_q - q_to_r,
                r + i_to_r + q_to_r,
            ]
        )
        history.append(state)

    history = np.array(history)
    frac = {k: history[:, idx] for idx, k in zip(STATES, STATE_NAMES, strict=True)}
    counts = {k: v * N for k, v in frac.items()}
    return {"frac": frac, "counts": counts, "N": N}


# =============================================================================
# Global-coupling (well-mixed) CA: the same rules driven by the global fraction
# =============================================================================


def well_mixed_advance(
    grid: np.ndarray,
    p_expose: float,
    p_infect: float,
    p_quarantine: float,
    p_recover_i: float,
    p_recover_q: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """One timestep of the CA with global coupling.

    Shares model.apply_transitions with the local model, so every transition
    and random draw is identical; only the exposure probability differs. Here it
    is the saturating rule evaluated at the grid-wide interior infected
    fraction, so a well-mixed run has no spatial structure.
    """
    interior = interior_mask(grid.shape[0])
    # Keep the numpy float64 arithmetic of the original implementation so the
    # exposure threshold is bit-identical (results.json is compared to it).
    i_frac = (grid[interior] == I).sum() / interior.sum()
    p_se = p_expose_well_mixed(i_frac, p_expose)
    return apply_transitions(grid, p_se, p_infect, p_quarantine, p_recover_i, p_recover_q, rng)


def _interior_counts(grid: np.ndarray) -> np.ndarray:
    """State counts over the interior (border-excluded) cells only."""
    return state_counts(grid[1:-1, 1:-1])


def _summarise_ensemble(count_runs: np.ndarray, n_interior: int) -> dict:
    """Ensemble statistics from per-run interior count curves of shape (runs, T, 5).

    Besides the mean curves, reports how many runs died out without a major
    outbreak and the attack rate over major outbreaks only, so that stochastic
    extinction can be separated from disagreement with the deterministic
    references (which cannot go extinct).
    """
    mean = count_runs.mean(axis=0)
    I_runs = count_runs[:, :, I]
    peak_frac = I_runs.max(axis=1) / n_interior
    major = peak_frac >= MAJOR_OUTBREAK_THRESHOLD
    final_R = count_runs[:, -1, STATES[-1]] / n_interior

    return {
        "mean": dict(zip(STATE_NAMES, mean.T, strict=True)),
        "I_mean": mean[:, I],
        "I_std": I_runs.std(axis=0),
        "n_interior": n_interior,
        "n_runs": int(count_runs.shape[0]),
        "n_extinct": int((~major).sum()),
        "attack_all": float(final_R.mean()),
        "attack_major": float(final_R[major].mean()) if major.any() else float("nan"),
    }


def run_well_mixed_ensemble(
    n_runs: int,
    n: int,
    p_expose: float,
    p_infect: float,
    p_quarantine: float,
    p_recover_i: float,
    p_recover_q: float,
    num_steps: int,
    seed: int = 0,
) -> dict:
    """Run the global-coupling CA n_runs times; return interior-only statistics.

    Run ``k`` uses ``np.random.default_rng(seed + k)``. See _summarise_ensemble
    for the returned keys.
    """
    n_int = int(interior_mask(n).sum())
    count_runs = np.zeros((n_runs, num_steps + 1, len(STATES)))

    for run in range(n_runs):
        rng = np.random.default_rng(seed + run)
        grid = initial_grid_single_seed(n)
        count_runs[run, 0] = _interior_counts(grid)
        for step in range(1, num_steps + 1):
            grid = well_mixed_advance(
                grid, p_expose, p_infect, p_quarantine, p_recover_i, p_recover_q, rng
            )
            count_runs[run, step] = _interior_counts(grid)

    return _summarise_ensemble(count_runs, n_int)


def run_local_ensemble(
    n_runs: int,
    n: int,
    p_expose: float,
    p_infect: float,
    p_quarantine: float,
    p_recover_i: float,
    p_recover_q: float,
    num_steps: int,
    seed: int = 0,
) -> dict:
    """Run the standard local (spatial) CA on a uniform grid; interior-only statistics.

    Reuses model.run_seiqr unchanged and recomputes interior counts from the
    stored grids, so the local and well-mixed curves share the same denominator.
    """
    n_int = int(interior_mask(n).sum())
    uniform_map = make_density_map(n, p_expose, p_expose, p_expose)
    count_runs = np.zeros((n_runs, num_steps + 1, len(STATES)))

    for run in range(n_runs):
        rng = np.random.default_rng(seed + run)
        grids, *_ = run_seiqr(
            n,
            uniform_map,
            p_infect,
            p_quarantine,
            p_recover_i,
            p_recover_q,
            num_steps,
            rng=rng,
            store_grids=True,
        )
        count_runs[run] = [_interior_counts(g) for g in grids]

    return _summarise_ensemble(count_runs, n_int)


# =============================================================================
# Comparison
# =============================================================================


def compare_curves(ca: dict, ref: dict, N: int) -> dict:
    """Agreement metrics between a CA ensemble mean and an analytical reference.

    ``ca`` and ``ref`` are dicts of count curves with at least 'I' and 'R'.
    Peak height, peak timing and RMSE are computed on the I curve; the attack
    rate uses the final recovered fraction.
    """
    ca_I = np.asarray(ca["I"], dtype=float)
    ref_I = np.asarray(ref["I"], dtype=float)
    peak_ca, peak_ref = float(ca_I.max()), float(ref_I.max())
    t_peak_ca, t_peak_ref = int(ca_I.argmax()), int(ref_I.argmax())
    rmse = float(np.sqrt(np.mean((ca_I - ref_I) ** 2)))

    def pct_of_peak(x: float) -> float:
        return 100.0 * x / peak_ref if peak_ref else float("nan")

    return {
        "peak_ca": peak_ca,
        "peak_ref": peak_ref,
        "peak_rel_err_pct": pct_of_peak(abs(peak_ca - peak_ref)),
        "t_peak_ca": t_peak_ca,
        "t_peak_ref": t_peak_ref,
        "t_peak_gap": abs(t_peak_ca - t_peak_ref),
        "attack_ca": float(np.asarray(ca["R"], dtype=float)[-1] / N),
        "attack_ref": float(np.asarray(ref["R"], dtype=float)[-1] / N),
        "rmse": rmse,
        "rmse_pct_of_peak": pct_of_peak(rmse),
    }


if __name__ == "__main__":
    # Quick self-check against the well-mixed limit at the report's uniform
    # baseline (p_expose = 0.30). Prints the agreement the README quotes.
    from config import SimConfig

    cfg = SimConfig()
    N = interior_n(cfg.n)
    p = cfg.p_uniform
    rates_args = (cfg.p_infect, cfg.p_quarantine, cfg.p_recover_i, cfg.p_recover_q)

    ode = integrate_seiqr(p, *rates_args, cfg.num_steps, N=N)
    rec = seiqr_discrete_meanfield(p, *rates_args, cfg.num_steps, N=N)
    wm = run_well_mixed_ensemble(10, cfg.n, p, *rates_args, cfg.num_steps, seed=0)
    loc = run_local_ensemble(10, cfg.n, p, *rates_args, cfg.num_steps, seed=0)

    print(f"N interior            = {N}")
    print(f"R0 (well-mixed)       = {ode['R0']:.1f}")
    print(f"rates                 = {ode['rates']}")
    print(f"extinct runs          = {wm['n_extinct']} of {wm['n_runs']}")
    print()
    m_ode = compare_curves(wm["mean"], ode["counts"], N)
    m_rec = compare_curves(wm["mean"], rec["counts"], N)
    print("well-mixed CA vs continuous ODE:")
    print(
        f"  peak {m_ode['peak_ca']:.1f} vs {m_ode['peak_ref']:.1f} "
        f"({m_ode['peak_rel_err_pct']:.1f}%), t_peak {m_ode['t_peak_ca']} vs "
        f"{m_ode['t_peak_ref']}, RMSE {m_ode['rmse_pct_of_peak']:.1f}% of peak"
    )
    print("well-mixed CA vs exact discrete recursion:")
    print(
        f"  RMSE {m_rec['rmse_pct_of_peak']:.1f}% of peak, "
        f"attack (all runs) {m_rec['attack_ca']:.3f}, "
        f"attack (major outbreaks) {wm['attack_major']:.3f}, "
        f"reference {m_rec['attack_ref']:.3f}"
    )
    print(
        f"local (spatial) CA peak = {loc['I_mean'].max():.1f} at "
        f"t = {int(loc['I_mean'].argmax())} (vs well-mixed "
        f"{wm['I_mean'].max():.1f} at t = {int(wm['I_mean'].argmax())})"
    )
