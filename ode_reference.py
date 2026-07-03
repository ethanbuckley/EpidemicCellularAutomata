"""
ode_reference.py: analytical SEIQR ODE and the mean-field validation of the CA.

The cellular automaton in model.py is spatial and stochastic. This module adds
the classical well-mixed SEIQR compartmental ODE that the CA rules reduce to in
the mean-field limit, and the tools to validate that reduction.

The honest claim this supports: the CA's local update rules reduce to the ODE
when the population is well mixed; driven by the global infected fraction rather
than local neighbours, a CA ensemble reproduces the ODE to within stochastic
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
"""

import numpy as np
from scipy.integrate import solve_ivp

from model import S, E, I, Q, R


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


def seiqr_rates(p_infect: float, p_quarantine: float,
                p_recover_i: float, p_recover_q: float) -> dict:
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


def foi(i, p_expose: float):
    """Mean-field force of infection: the rate form of the saturating CA rule.

    In a well-mixed population a fraction i is infectious, so a susceptible has
    at least one infected cell among 8 random neighbours with probability
    1 - (1 - i) ** 8; the per-step exposure probability is p_expose times that,
    converted to a rate. Accepts scalar or array i.
    """
    arg = p_expose * (1.0 - (1.0 - i) ** 8)
    arg = np.minimum(arg, 1.0 - 1e-12)  # keep log1p in-domain if p_expose -> 1
    return -np.log1p(-arg)


def beta_lowprev(p_expose: float) -> float:
    """Low-prevalence linear slope of the force of infection: beta = 8 * p_expose."""
    return 8.0 * p_expose


def R0(p_expose: float, rates: dict) -> float:
    """Well-mixed basic reproduction number beta / (gamma_Q + gamma_R).

    This is the mean-field R0; the spatial CA's effective reproduction number is
    far lower because a cell's infected neighbours are quickly depleted.
    """
    return beta_lowprev(p_expose) / (rates["gamma_Q"] + rates["gamma_R"])


# =============================================================================
# Analytical references
# =============================================================================

def integrate_seiqr(p_expose: float, p_infect: float, p_quarantine: float,
                    p_recover_i: float, p_recover_q: float, num_steps: int,
                    N: int = None, i0: float = None) -> dict:
    """Integrate the continuous well-mixed SEIQR ODE over [0, num_steps].

    Returns fraction and count curves sampled at each integer timestep, the
    derived rates, and R0. Counts scale the fractions by the population N.
    """
    if N is None:
        N = interior_n(50)
    if i0 is None:
        i0 = 1.0 / N

    rates = seiqr_rates(p_infect, p_quarantine, p_recover_i, p_recover_q)
    sigma = rates["sigma"]
    gamma_Q = rates["gamma_Q"]
    gamma_R = rates["gamma_R"]
    delta = rates["delta"]

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
    sol = solve_ivp(rhs, (0.0, num_steps), y0,
                    t_eval=np.arange(num_steps + 1),
                    rtol=1e-8, atol=1e-10, method="RK45")

    frac = {k: sol.y[idx] for idx, k in enumerate(("S", "E", "I", "Q", "R"))}
    total = sum(frac.values())
    drift = float(np.max(np.abs(total - 1.0)))
    assert drift < 1e-6, f"ODE compartments not conserved (drift {drift:g})"

    counts = {k: frac[k] * N for k in frac}
    return {"frac": frac, "counts": counts, "rates": rates,
            "R0": R0(p_expose, rates), "N": N}


def seiqr_discrete_meanfield(p_expose: float, p_infect: float, p_quarantine: float,
                             p_recover_i: float, p_recover_q: float, num_steps: int,
                             N: int = None, i0: float = None) -> dict:
    """The exact discrete recursion the CA implements in the well-mixed limit.

    This is the deterministic expectation of well_mixed_advance (below); it uses
    the raw per-step probabilities, not the continuous-time rates, so it carries
    no discretisation approximation. It is the tightest reference for the
    well-mixed CA ensemble.
    """
    if N is None:
        N = interior_n(50)
    if i0 is None:
        i0 = 1.0 / N

    s, e, i, q, r = 1.0 - i0, 0.0, i0, 0.0, 0.0
    curves = {k: [v] for k, v in zip("SEIQR", (s, e, i, q, r))}

    for _ in range(num_steps):
        p_se = p_expose * (1.0 - (1.0 - i) ** 8)
        new_exp = s * p_se
        new_inf = e * p_infect
        i_to_q = i * p_quarantine
        i_to_r = i * p_recover_i
        q_to_r = q * p_recover_q

        s = s - new_exp
        e = e + new_exp - new_inf
        i = i + new_inf - i_to_q - i_to_r
        q = q + i_to_q - q_to_r
        r = r + i_to_r + q_to_r

        for k, v in zip("SEIQR", (s, e, i, q, r)):
            curves[k].append(v)

    frac = {k: np.array(v) for k, v in curves.items()}
    counts = {k: frac[k] * N for k in frac}
    return {"frac": frac, "counts": counts, "N": N}


# =============================================================================
# Global-coupling (well-mixed) CA — the same rules driven by the global fraction
# =============================================================================

def well_mixed_advance(grid, n, p_expose, p_infect, p_quarantine,
                       p_recover_i, p_recover_q, rng, interior, n_interior):
    """One timestep of the CA with global coupling.

    Identical to model.seiqr_advance in every transition and in random-draw
    order, except the local infected-neighbour count is replaced by the
    grid-wide infected fraction: the exposure probability is the same saturating
    function evaluated at the global fraction, so a well-mixed run has no spatial
    structure.
    """
    is_S = grid == S
    is_E = grid == E
    is_I = grid == I
    is_Q = grid == Q

    i_frac = (is_I & interior).sum() / n_interior
    p_se = p_expose * (1.0 - (1.0 - i_frac) ** 8)

    r_s = rng.random((n, n))
    r_e = rng.random((n, n))
    r_i = rng.random((n, n))
    r_q = rng.random((n, n))

    expose_mask     = is_S & interior & (r_s < p_se)
    infect_mask     = is_E & interior & (r_e < p_infect)
    quarantine_mask = is_I & interior & (r_i < p_quarantine)
    recover_i_mask  = is_I & interior & ~quarantine_mask & (r_i < p_quarantine + p_recover_i)
    recover_q_mask  = is_Q & interior & (r_q < p_recover_q)

    new_grid = grid.copy()
    new_grid[expose_mask]     = E
    new_grid[infect_mask]     = I
    new_grid[quarantine_mask] = Q
    new_grid[recover_i_mask]  = R
    new_grid[recover_q_mask]  = R
    return new_grid


def _interior_counts(grid):
    """State counts over the interior (border-excluded) cells only."""
    inner = grid[1:-1, 1:-1]
    return {"S": int((inner == S).sum()), "E": int((inner == E).sum()),
            "I": int((inner == I).sum()), "Q": int((inner == Q).sum()),
            "R": int((inner == R).sum())}


def run_well_mixed_ensemble(n_runs, n, p_expose, p_infect, p_quarantine,
                            p_recover_i, p_recover_q, num_steps, seed=0) -> dict:
    """Run the global-coupling CA n_runs times; return interior-only curves.

    Returns mean S/E/I/Q/R count curves, the per-run I curves and their std
    (for a scatter band), and the participating population n_interior.
    """
    interior = np.zeros((n, n), dtype=bool)
    interior[1:-1, 1:-1] = True
    n_int = int(interior.sum())

    totals = {k: np.zeros(num_steps + 1) for k in "SEIQR"}
    I_runs = np.zeros((n_runs, num_steps + 1))

    for run in range(n_runs):
        rng = np.random.default_rng(seed + run)
        grid = np.zeros((n, n))
        grid[n // 2, n // 2] = I

        for step in range(num_steps + 1):
            if step > 0:
                grid = well_mixed_advance(grid, n, p_expose, p_infect,
                                          p_quarantine, p_recover_i, p_recover_q,
                                          rng, interior, n_int)
            c = _interior_counts(grid)
            for k in "SEIQR":
                totals[k][step] += c[k]
            I_runs[run, step] = c["I"]

    mean = {k: totals[k] / n_runs for k in "SEIQR"}
    return {"mean": mean, "I_mean": mean["I"], "I_std": I_runs.std(axis=0),
            "n_interior": n_int}


def run_local_ensemble(n_runs, n, p_expose, p_infect, p_quarantine,
                       p_recover_i, p_recover_q, num_steps, seed=0) -> dict:
    """Run the standard local (spatial) CA on a uniform grid, interior-only counts.

    Reuses model.run_seiqr unchanged and recomputes interior counts from the
    stored grids, so the local and well-mixed curves share the same denominator.
    """
    import model

    uniform_map = model.make_density_map(n, p_expose, p_expose, p_expose)
    totals = {k: np.zeros(num_steps + 1) for k in "SEIQR"}
    I_runs = np.zeros((n_runs, num_steps + 1))

    for run in range(n_runs):
        rng = np.random.default_rng(seed + run)
        grids, *_ = model.run_seiqr(n, uniform_map, p_infect, p_quarantine,
                                    p_recover_i, p_recover_q, num_steps,
                                    rng=rng, store_grids=True)
        for step, g in enumerate(grids):
            c = _interior_counts(g)
            for k in "SEIQR":
                totals[k][step] += c[k]
            I_runs[run, step] = c["I"]

    mean = {k: totals[k] / n_runs for k in "SEIQR"}
    return {"mean": mean, "I_mean": mean["I"], "I_std": I_runs.std(axis=0)}


# =============================================================================
# Comparison
# =============================================================================

def compare_curves(ca: dict, ref: dict, N: int) -> dict:
    """Agreement metrics between a CA ensemble mean and an analytical reference.

    ca and ref are dicts of count curves with at least 'I' and 'R'. Peak height,
    peak timing and RMSE are computed on the I curve; the attack rate uses the
    final recovered fraction.
    """
    ca_I = np.asarray(ca["I"], dtype=float)
    ref_I = np.asarray(ref["I"], dtype=float)
    peak_ref = float(ref_I.max())

    rmse = float(np.sqrt(np.mean((ca_I - ref_I) ** 2)))
    return {
        "peak_ca": float(ca_I.max()),
        "peak_ref": peak_ref,
        "peak_rel_err_pct": 100.0 * abs(ca_I.max() - peak_ref) / peak_ref if peak_ref else float("nan"),
        "t_peak_ca": int(ca_I.argmax()),
        "t_peak_ref": int(ref_I.argmax()),
        "t_peak_gap": int(abs(int(ca_I.argmax()) - int(ref_I.argmax()))),
        "attack_ca": float(np.asarray(ca["R"], dtype=float)[-1] / N),
        "attack_ref": float(np.asarray(ref["R"], dtype=float)[-1] / N),
        "rmse": rmse,
        "rmse_pct_of_peak": 100.0 * rmse / peak_ref if peak_ref else float("nan"),
    }


if __name__ == "__main__":
    # Quick self-check against the well-mixed limit at the report's uniform
    # baseline (p_expose = 0.30). Prints the agreement the README quotes.
    from config import SimConfig

    cfg = SimConfig()
    N = interior_n(cfg.n)
    p = cfg.p_uniform

    ode = integrate_seiqr(p, cfg.p_infect, cfg.p_quarantine,
                          cfg.p_recover_i, cfg.p_recover_q, cfg.num_steps, N=N)
    rec = seiqr_discrete_meanfield(p, cfg.p_infect, cfg.p_quarantine,
                                   cfg.p_recover_i, cfg.p_recover_q, cfg.num_steps, N=N)
    wm = run_well_mixed_ensemble(10, cfg.n, p, cfg.p_infect, cfg.p_quarantine,
                                 cfg.p_recover_i, cfg.p_recover_q, cfg.num_steps, seed=0)
    loc = run_local_ensemble(10, cfg.n, p, cfg.p_infect, cfg.p_quarantine,
                             cfg.p_recover_i, cfg.p_recover_q, cfg.num_steps, seed=0)

    print(f"N interior            = {N}")
    print(f"R0 (well-mixed)       = {ode['R0']:.1f}")
    print(f"rates                 = {ode['rates']}")
    print()
    m_ode = compare_curves(wm["mean"], ode["counts"], N)
    m_rec = compare_curves(wm["mean"], rec["counts"], N)
    print("well-mixed CA vs continuous ODE:")
    print(f"  peak {m_ode['peak_ca']:.1f} vs {m_ode['peak_ref']:.1f} "
          f"({m_ode['peak_rel_err_pct']:.1f}%), t_peak {m_ode['t_peak_ca']} vs "
          f"{m_ode['t_peak_ref']}, RMSE {m_ode['rmse_pct_of_peak']:.1f}% of peak")
    print("well-mixed CA vs exact discrete recursion:")
    print(f"  RMSE {m_rec['rmse_pct_of_peak']:.1f}% of peak, "
          f"attack {m_rec['attack_ca']:.3f} vs {m_rec['attack_ref']:.3f}")
    print(f"local (spatial) CA peak = {loc['I_mean'].max():.1f} at "
          f"t = {int(loc['I_mean'].argmax())} (vs well-mixed "
          f"{wm['I_mean'].max():.1f} at t = {int(wm['I_mean'].argmax())})")
