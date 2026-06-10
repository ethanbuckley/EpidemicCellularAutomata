"""
model.py — Vectorized SEIQR cellular automaton core
=====================================================
Individually-extended version of the UCL group project (seiqr.py).
All per-cell Python loops replaced with NumPy array operations and a
single scipy.signal.convolve2d call for Moore-neighbour counting.

States: S=0, E=1, I=2, Q=3, R=4
"""

import numpy as np
from scipy.signal import convolve2d

# 3×3 Moore-neighbourhood kernel with centre zeroed so a cell doesn't
# count itself as its own neighbour.
_NEIGHBOUR_KERNEL = np.array([[1, 1, 1],
                               [1, 0, 1],
                               [1, 1, 1]], dtype=np.float32)

# State constants
S, E, I, Q, R = 0, 1, 2, 3, 4


def make_density_map(n: int,
                     p_centre: float,
                     p_middle: float,
                     p_outer: float) -> np.ndarray:
    """
    Create an n×n density map with three concentric square zones.
    Vectorized: builds the map with two np.maximum calls, no Python loops.
    """
    mid = n // 2
    rows = np.arange(n)
    # Chebyshev distance from centre for every cell
    dist = np.maximum(np.abs(rows[:, None] - mid), np.abs(rows[None, :] - mid))

    density = np.where(dist <= 8, p_centre,
               np.where(dist <= 16, p_middle, p_outer))
    return density.astype(float)


def vaccinate(grid: np.ndarray,
              n: int,
              num_to_vaccinate: int,
              targeted: bool,
              efficacy: float = 0.80,
              rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Vaccinate susceptible cells before the simulation starts.
    Targeted mode prioritises the centre zone (dist <= 8) first.
    Vectorized: candidate selection uses boolean masking + np.random.choice.
    """
    if rng is None:
        rng = np.random.default_rng()

    mid = n // 2
    rows = np.arange(n)
    dist = np.maximum(np.abs(rows[:, None] - mid), np.abs(rows[None, :] - mid))

    susceptible = grid == S

    if targeted:
        centre_mask = susceptible & (dist <= 8)
        other_mask  = susceptible & (dist > 8)
        centre_idx  = np.argwhere(centre_mask)
        other_idx   = np.argwhere(other_mask)
        rng.shuffle(centre_idx)
        rng.shuffle(other_idx)
        candidates = np.concatenate([centre_idx, other_idx], axis=0)
    else:
        candidates = np.argwhere(susceptible)
        rng.shuffle(candidates)

    count = min(num_to_vaccinate, len(candidates))
    chosen = candidates[:count]

    # Vectorized efficacy: generate one random float per chosen cell
    successes = rng.random(count) < efficacy
    immune_cells = chosen[successes]

    grid = grid.copy()
    if len(immune_cells):
        grid[immune_cells[:, 0], immune_cells[:, 1]] = R
    return grid


def seiqr_advance(grid: np.ndarray,
                  n: int,
                  density_map: np.ndarray,
                  p_infect: float,
                  p_quarantine: float,
                  p_recover_i: float,
                  p_recover_q: float,
                  rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Apply one timestep of SEIQR rules to the full n×n grid.

    Vectorized implementation:
    - Infected-neighbour counts: single convolve2d over the I-state binary
      array, which simultaneously computes the 8-neighbour sum for every
      cell with no Python loop.
    - Stochastic transitions: one rng.random(n, n) draw per transition
      group, compared against probability arrays via boolean masking.
    - Boundary cells (row/col 0 and n-1) are excluded from all transitions
      via an interior mask, preserving the original model's behaviour of
      permanent boundary susceptibles.

    Quarantined cells (state Q) are NOT included in the infected-neighbour
    count, matching the original model.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Boolean arrays for each state
    is_S = grid == S
    is_E = grid == E
    is_I = grid == I
    is_Q = grid == Q

    # Interior mask: boundary cells are never updated
    interior = np.zeros((n, n), dtype=bool)
    interior[1:-1, 1:-1] = True

    # --- Infected-neighbour count (only I cells contribute, not Q) ---
    infected_neighbours = convolve2d(is_I.astype(np.float32),
                                     _NEIGHBOUR_KERNEL,
                                     mode='same', boundary='fill',
                                     fillvalue=0)

    # --- Draw random arrays once per transition group ---
    r_s   = rng.random((n, n))   # S → E
    r_e   = rng.random((n, n))   # E → I
    r_i   = rng.random((n, n))   # I → Q or I → R  (single draw, same as original)
    r_q   = rng.random((n, n))   # Q → R

    # --- Transition masks ---
    # S → E: susceptible, interior, at least one infected neighbour, random < p_expose
    expose_mask      = is_S & interior & (infected_neighbours > 0) & (r_s < density_map)

    # E → I
    infect_mask      = is_E & interior & (r_e < p_infect)

    # I → Q (quarantine checked first, matching original single-draw logic)
    quarantine_mask  = is_I & interior & (r_i < p_quarantine)

    # I → R (only if not quarantined — r_i in [p_quarantine, p_quarantine+p_recover_i))
    recover_i_mask   = is_I & interior & ~quarantine_mask & (r_i < p_quarantine + p_recover_i)

    # Q → R
    recover_q_mask   = is_Q & interior & (r_q < p_recover_q)

    # --- Apply all transitions simultaneously ---
    new_grid = grid.copy()
    new_grid[expose_mask]     = E
    new_grid[infect_mask]     = I
    new_grid[quarantine_mask] = Q
    new_grid[recover_i_mask]  = R
    new_grid[recover_q_mask]  = R

    return new_grid


def run_seiqr(n: int,
              density_map: np.ndarray,
              p_infect: float,
              p_quarantine: float,
              p_recover_i: float,
              p_recover_q: float,
              num_steps: int,
              lockdown_map: np.ndarray | None = None,
              lockdown_start: int | None = None,
              lockdown_end: int | None = None,
              initial_grid: np.ndarray | None = None,
              rng: np.random.Generator | None = None,
              store_grids: bool = True):
    """
    Run a full SEIQR simulation and return population curves (and optionally
    all grid snapshots).

    Parameters
    ----------
    store_grids : bool
        If True (default), return all n×n grid snapshots at each timestep.
        Set False for ensemble runs where only the curves are needed — avoids
        storing 101 × 50 × 50 arrays per run.

    Returns
    -------
    all_grids : list[np.ndarray] or None
    S_counts, E_counts, I_counts, Q_counts, R_counts : list[int]
    """
    if rng is None:
        rng = np.random.default_rng()

    if initial_grid is not None:
        grid = initial_grid.copy()
    else:
        grid = np.zeros((n, n))
        grid[n // 2, n // 2] = I

    all_grids = [grid.copy()] if store_grids else None
    S_counts = [int(np.sum(grid == S))]
    E_counts = [int(np.sum(grid == E))]
    I_counts = [int(np.sum(grid == I))]
    Q_counts = [int(np.sum(grid == Q))]
    R_counts = [int(np.sum(grid == R))]

    for step in range(num_steps):
        current_map = (lockdown_map
                       if lockdown_map is not None
                          and lockdown_start <= step < lockdown_end
                       else density_map)

        grid = seiqr_advance(grid, n, current_map,
                             p_infect, p_quarantine,
                             p_recover_i, p_recover_q,
                             rng=rng)

        if store_grids:
            all_grids.append(grid.copy())
        S_counts.append(int(np.sum(grid == S)))
        E_counts.append(int(np.sum(grid == E)))
        I_counts.append(int(np.sum(grid == I)))
        Q_counts.append(int(np.sum(grid == Q)))
        R_counts.append(int(np.sum(grid == R)))

    return all_grids, S_counts, E_counts, I_counts, Q_counts, R_counts


def run_ensemble(n_runs: int,
                 n: int,
                 density_map: np.ndarray,
                 p_infect: float,
                 p_quarantine: float,
                 p_recover_i: float,
                 p_recover_q: float,
                 num_steps: int,
                 lockdown_map: np.ndarray | None = None,
                 lockdown_start: int | None = None,
                 lockdown_end: int | None = None,
                 initial_grid: np.ndarray | None = None,
                 seed: int | None = None) -> dict:
    """
    Run n_runs independent simulations and return mean population curves.
    Grid snapshots are not stored (store_grids=False) to keep memory low.

    Returns a dict with keys 'S','E','I','Q','R', each a 1D array of length
    num_steps+1 (mean counts over all runs).
    """
    totals = {k: np.zeros(num_steps + 1) for k in ('S', 'E', 'I', 'Q', 'R')}

    for run in range(n_runs):
        rng = np.random.default_rng(None if seed is None else seed + run)
        init = initial_grid.copy() if initial_grid is not None else None
        _, S, E, I_c, Q, Rc = run_seiqr(
            n, density_map, p_infect, p_quarantine,
            p_recover_i, p_recover_q, num_steps,
            lockdown_map=lockdown_map,
            lockdown_start=lockdown_start,
            lockdown_end=lockdown_end,
            initial_grid=init,
            rng=rng,
            store_grids=False,
        )
        totals['S'] += S
        totals['E'] += E
        totals['I'] += I_c
        totals['Q'] += Q
        totals['R'] += Rc

    return {k: v / n_runs for k, v in totals.items()}
