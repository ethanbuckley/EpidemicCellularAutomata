"""
model.py: vectorised SEIQR cellular automaton core.

Individually-extended version of the UCL group project (seiqr.py). All per-cell
Python loops are replaced with NumPy array operations and a single
scipy.signal.convolve2d call for Moore-neighbour counting.

States: S=0, E=1, I=2, Q=3, R=4

Every stochastic transition is one rng.random((n, n)) draw compared against a
probability, in the fixed order S->E, E->I, I->Q/R, Q->R. That order is part of
the model's contract: tests/test_reproducibility.py asserts the committed
data/results.json bit-for-bit, so a change to the draw order or count is a
deliberate model change, not a refactor.
"""

from functools import cache

import numpy as np
from scipy.signal import convolve2d

# State constants
S, E, I, Q, R = 0, 1, 2, 3, 4
STATES = (S, E, I, Q, R)
STATE_NAMES = ("S", "E", "I", "Q", "R")

# Grids hold only the five state codes, so one signed byte per cell is enough.
# A stored 100-step run of 101 snapshots is 250 KB at int8 against 2 MB at the
# float64 that np.zeros would give. Every function accepts any integer-valued
# grid; run_seiqr and initial_grid_single_seed produce this dtype.
GRID_DTYPE = np.int8

# 3x3 Moore-neighbourhood kernel with the centre zeroed so a cell does not
# count itself as its own neighbour.
_NEIGHBOUR_KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.float32)
MOORE_NEIGHBOURS = int(_NEIGHBOUR_KERNEL.sum())  # 8

# Zone radii (Chebyshev distance from the grid centre) that define the three
# concentric density zones. Single source of truth for make_density_map,
# vaccinate and zone_masks.
CENTRE_RADIUS = 8
MIDDLE_RADIUS = 16


# =============================================================================
# Grid geometry
# =============================================================================


def chebyshev_distance(n: int) -> np.ndarray:
    """Chebyshev (king-move) distance of every cell from the centre of an n x n grid."""
    mid = n // 2
    offsets = np.abs(np.arange(n) - mid)
    return np.maximum(offsets[:, None], offsets[None, :])


def zone_masks(
    n: int, centre_radius: int = CENTRE_RADIUS, middle_radius: int = MIDDLE_RADIUS
) -> dict[str, np.ndarray]:
    """Boolean masks for the centre, middle and outer zones of an n x n grid.

    The zones partition the grid: centre is dist <= centre_radius, middle is
    centre_radius < dist <= middle_radius, outer is everything beyond.
    """
    dist = chebyshev_distance(n)
    centre = dist <= centre_radius
    middle = ~centre & (dist <= middle_radius)
    return {"centre": centre, "middle": middle, "outer": ~centre & ~middle}


@cache
def interior_mask(n: int) -> np.ndarray:
    """Boolean mask of the cells that take part in the dynamics.

    The border (row/col 0 and n-1) is never updated, so it stays permanently
    susceptible, matching the original model. Cached per n and never mutated.
    """
    mask = np.zeros((n, n), dtype=bool)
    mask[1:-1, 1:-1] = True
    mask.setflags(write=False)
    return mask


def make_density_map(
    n: int,
    p_centre: float,
    p_middle: float,
    p_outer: float,
    centre_radius: int = CENTRE_RADIUS,
    middle_radius: int = MIDDLE_RADIUS,
) -> np.ndarray:
    """Create an n x n exposure-probability map with three concentric square zones."""
    dist = chebyshev_distance(n)
    density = np.where(
        dist <= centre_radius, p_centre, np.where(dist <= middle_radius, p_middle, p_outer)
    )
    return density.astype(float)


def state_counts(grid: np.ndarray) -> np.ndarray:
    """Number of cells in each state, indexed by state constant (length 5)."""
    return np.bincount(grid.astype(np.intp).ravel(), minlength=len(STATES))


# =============================================================================
# Interventions
# =============================================================================


def vaccinate(
    grid: np.ndarray,
    num_to_vaccinate: int,
    targeted: bool,
    efficacy: float = 0.80,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return a copy of ``grid`` with up to ``num_to_vaccinate`` susceptibles immunised.

    Each dose succeeds independently with probability ``efficacy``; a failed
    dose leaves the cell susceptible. Targeted mode offers doses to the centre
    zone first, then to the rest of the grid. Candidate order is shuffled so
    which cells receive doses is random within each priority group.
    """
    if rng is None:
        rng = np.random.default_rng()

    susceptible = grid == S

    if targeted:
        centre = chebyshev_distance(grid.shape[0]) <= CENTRE_RADIUS
        centre_idx = np.argwhere(susceptible & centre)
        other_idx = np.argwhere(susceptible & ~centre)
        rng.shuffle(centre_idx)
        rng.shuffle(other_idx)
        candidates = np.concatenate([centre_idx, other_idx], axis=0)
    else:
        candidates = np.argwhere(susceptible)
        rng.shuffle(candidates)

    count = min(num_to_vaccinate, len(candidates))
    chosen = candidates[:count]

    # One efficacy draw per dose, vectorised.
    immune_cells = chosen[rng.random(count) < efficacy]

    grid = grid.copy()
    if len(immune_cells):
        grid[immune_cells[:, 0], immune_cells[:, 1]] = R
    return grid


# =============================================================================
# Dynamics
# =============================================================================


def apply_transitions(
    grid: np.ndarray,
    p_expose: np.ndarray | float,
    p_infect: float,
    p_quarantine: float,
    p_recover_i: float,
    p_recover_q: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply one step of the SEIQR transition rules given a per-cell exposure probability.

    ``p_expose`` is the probability that each susceptible cell becomes exposed
    this step: an n x n array for the local (spatial) model, or a scalar for
    the well-mixed model in ode_reference.py. Everything else about the step is
    shared, so the two models differ only in how ``p_expose`` is computed.

    Draw order is S->E, E->I, I->Q/R (one draw decides between Q and R, as in
    the original model), Q->R. Border cells never change state.
    """
    n = grid.shape[0]
    interior = interior_mask(n)

    is_S = grid == S
    is_E = grid == E
    is_I = grid == I
    is_Q = grid == Q

    r_s = rng.random((n, n))
    r_e = rng.random((n, n))
    r_i = rng.random((n, n))
    r_q = rng.random((n, n))

    expose_mask = is_S & interior & (r_s < p_expose)
    infect_mask = is_E & interior & (r_e < p_infect)
    quarantine_mask = is_I & interior & (r_i < p_quarantine)
    # I -> R only if not quarantined: r_i in [p_quarantine, p_quarantine + p_recover_i)
    recover_i_mask = is_I & interior & ~quarantine_mask & (r_i < p_quarantine + p_recover_i)
    recover_q_mask = is_Q & interior & (r_q < p_recover_q)

    new_grid = grid.copy()
    new_grid[expose_mask] = E
    new_grid[infect_mask] = I
    new_grid[quarantine_mask] = Q
    new_grid[recover_i_mask] = R
    new_grid[recover_q_mask] = R
    return new_grid


def seiqr_advance(
    grid: np.ndarray,
    density_map: np.ndarray,
    p_infect: float,
    p_quarantine: float,
    p_recover_i: float,
    p_recover_q: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply one timestep of the local SEIQR rules to the full n x n grid.

    A susceptible cell can only be exposed if at least one of its 8 Moore
    neighbours is infected (state I; quarantined cells do not transmit), in
    which case it is exposed with probability ``density_map[row, col]``. The
    neighbour count comes from one convolve2d call over the I-state array.
    """
    if rng is None:
        rng = np.random.default_rng()

    is_I = grid == I
    infected_neighbours = convolve2d(
        is_I.astype(np.float32), _NEIGHBOUR_KERNEL, mode="same", boundary="fill", fillvalue=0
    )
    p_expose = np.where(infected_neighbours > 0, density_map, 0.0)

    return apply_transitions(grid, p_expose, p_infect, p_quarantine, p_recover_i, p_recover_q, rng)


def initial_grid_single_seed(n: int) -> np.ndarray:
    """An all-susceptible n x n grid with one infected cell at the centre."""
    grid = np.zeros((n, n), dtype=GRID_DTYPE)
    grid[n // 2, n // 2] = I
    return grid


def run_seiqr(
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
    rng: np.random.Generator | None = None,
    store_grids: bool = True,
):
    """Run a full SEIQR simulation and return population curves.

    Parameters
    ----------
    lockdown_map, lockdown_start, lockdown_end
        If given, ``lockdown_map`` replaces ``density_map`` for steps in the
        half-open window [lockdown_start, lockdown_end).
    initial_grid
        Starting state (for example after vaccinate). Defaults to a single
        infected cell at the centre. The array is copied (and cast to
        GRID_DTYPE), not modified.
    store_grids
        If True (default), also return every grid snapshot. Set False for
        ensemble runs where only the curves are needed.

    Returns
    -------
    all_grids : list[np.ndarray] or None
    S_counts, E_counts, I_counts, Q_counts, R_counts : list[int]
        Each of length num_steps + 1 (the initial state plus one entry per step).
    """
    if rng is None:
        rng = np.random.default_rng()

    # A lockdown map is only meaningful with a window. Fail loudly on a
    # half-specified lockdown rather than indexing a None bound in the loop.
    if lockdown_map is not None and (lockdown_start is None or lockdown_end is None):
        raise ValueError(
            "lockdown_start and lockdown_end must both be set when lockdown_map is provided."
        )

    if initial_grid is not None:
        grid = np.array(initial_grid, dtype=GRID_DTYPE)  # copies
    else:
        grid = initial_grid_single_seed(n)

    all_grids = [grid.copy()] if store_grids else None
    counts = [state_counts(grid)]

    for step in range(num_steps):
        locked_down = lockdown_map is not None and lockdown_start <= step < lockdown_end
        current_map = lockdown_map if locked_down else density_map

        grid = seiqr_advance(
            grid, current_map, p_infect, p_quarantine, p_recover_i, p_recover_q, rng=rng
        )

        if store_grids:
            all_grids.append(grid.copy())
        counts.append(state_counts(grid))

    counts = np.array(counts)  # shape (num_steps + 1, 5)
    S_counts, E_counts, I_counts, Q_counts, R_counts = (counts[:, k].tolist() for k in STATES)
    return all_grids, S_counts, E_counts, I_counts, Q_counts, R_counts


def run_ensemble(
    n_runs: int,
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
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Run n_runs independent simulations and return mean population curves.

    Run ``k`` uses ``np.random.default_rng(seed + k)`` (or an unseeded generator
    if ``seed`` is None). Grid snapshots are not stored, to keep memory low.

    Returns a dict with keys 'S', 'E', 'I', 'Q', 'R', each a 1D array of length
    num_steps + 1 holding the mean count over all runs.
    """
    totals = np.zeros((num_steps + 1, len(STATES)))

    for run in range(n_runs):
        rng = np.random.default_rng(None if seed is None else seed + run)
        _, *curves = run_seiqr(
            n,
            density_map,
            p_infect,
            p_quarantine,
            p_recover_i,
            p_recover_q,
            num_steps,
            lockdown_map=lockdown_map,
            lockdown_start=lockdown_start,
            lockdown_end=lockdown_end,
            initial_grid=initial_grid,
            rng=rng,
            store_grids=False,
        )
        totals += np.array(curves).T

    means = totals / n_runs
    return {name: means[:, k] for k, name in zip(STATES, STATE_NAMES, strict=True)}
