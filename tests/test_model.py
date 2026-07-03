"""Tests for the vectorised SEIQR cellular automaton core (model.py)."""

import numpy as np
import pytest

import model
import seiqr
from model import S, E, I, Q, R


def test_density_map_zone_counts():
    """Three concentric zones with the documented cell counts and probabilities."""
    dm = model.make_density_map(50, 0.5, 0.3, 0.15)
    assert dm.shape == (50, 50)
    assert int((dm == 0.5).sum()) == 289    # centre: dist <= 8
    assert int((dm == 0.3).sum()) == 800    # middle: 8 < dist <= 16
    assert int((dm == 0.15).sum()) == 1411  # outer:  dist > 16


def test_density_map_uses_shared_radii():
    """Zone boundaries fall exactly at the shared radius constants."""
    dm = model.make_density_map(50, 0.5, 0.3, 0.15)
    mid = 25
    assert dm[mid, mid + model.CENTRE_RADIUS] == 0.5
    assert dm[mid, mid + model.CENTRE_RADIUS + 1] == 0.3
    assert dm[mid, mid + model.MIDDLE_RADIUS] == 0.3
    assert dm[mid, mid + model.MIDDLE_RADIUS + 1] == 0.15


def test_boundary_cells_stay_susceptible():
    """The grid border is permanently susceptible, even at maximum exposure."""
    dm = model.make_density_map(20, 1.0, 1.0, 1.0)
    grids, *_ = model.run_seiqr(20, dm, 1.0, 0.5, 0.2, 0.2, 40,
                                rng=np.random.default_rng(0), store_grids=True)
    final = grids[-1]
    assert np.all(final[0, :] == S)
    assert np.all(final[-1, :] == S)
    assert np.all(final[:, 0] == S)
    assert np.all(final[:, -1] == S)


def test_reproducibility_same_seed():
    """The same seed gives identical curves; a different seed differs."""
    dm = model.make_density_map(30, 0.5, 0.3, 0.15)

    def peak(seed):
        _, _, _, Ic, _, _ = model.run_seiqr(
            30, dm, 0.5, 0.1, 0.05, 0.1, 30,
            rng=np.random.default_rng(seed), store_grids=False)
        return Ic

    assert peak(123) == peak(123)
    assert peak(123) != peak(456)


def test_susceptible_needs_infected_neighbour():
    """S -> E fires next to an infected cell but not for a distant susceptible."""
    n = 9
    dm = np.ones((n, n))                 # exposure certain wherever eligible
    grid = np.zeros((n, n))
    grid[4, 4] = I                       # single infected at centre
    nxt = model.seiqr_advance(grid, n, dm, 0.0, 0.0, 0.0, 0.0,
                              rng=np.random.default_rng(0))
    assert nxt[3, 3] == E                # a Moore neighbour of the seed
    assert nxt[5, 4] == E                # another neighbour
    assert nxt[1, 1] == S                # interior but far from any infected
    assert nxt[4, 4] == I                # seed persists (no recovery this step)


def test_exposed_to_infected_is_gated_by_p_infect():
    """E -> I fires iff the draw is below p_infect; the extremes are deterministic."""
    n = 6
    dm = np.zeros((n, n))                # suppress new exposures
    grid = np.zeros((n, n))
    grid[2, 2] = E
    became_i = model.seiqr_advance(grid, n, dm, 1.0, 0.0, 0.0, 0.0,
                                   rng=np.random.default_rng(0))
    stayed_e = model.seiqr_advance(grid, n, dm, 0.0, 0.0, 0.0, 0.0,
                                   rng=np.random.default_rng(0))
    assert became_i[2, 2] == I
    assert stayed_e[2, 2] == E


def test_quarantined_do_not_transmit():
    """Quarantined cells are excluded from the infected-neighbour count."""
    n = 5
    dm = np.ones((n, n))
    grid = np.zeros((n, n))
    grid[2, 2] = Q                       # only a quarantined cell, no I
    nxt = model.seiqr_advance(grid, n, dm, 1.0, 0.0, 0.0, 0.0,
                              rng=np.random.default_rng(0))
    assert not np.any(nxt == E)          # nothing gets exposed


def test_infected_quarantine_threshold():
    """With p_quarantine = 1 every interior infected cell moves to Q."""
    n = 6
    dm = np.zeros((n, n))
    grid = np.zeros((n, n))
    grid[1:-1, 1:-1] = I
    nxt = model.seiqr_advance(grid, n, dm, 0.0, 1.0, 0.0, 0.0,
                              rng=np.random.default_rng(0))
    assert np.all(nxt[1:-1, 1:-1] == Q)


def test_lockdown_window_is_half_open():
    """The lockdown map applies on [start, end): active at start, inactive at end."""
    n = 5
    density = np.ones((n, n))            # exposure certain
    lockdown = np.zeros((n, n))          # exposure impossible
    grid = np.zeros((n, n))
    grid[2, 1] = I                       # infected next to interior S at (2, 2)
    grids, *_ = model.run_seiqr(
        n, density, 0.0, 0.0, 0.0, 0.0, 2,
        lockdown_map=lockdown, lockdown_start=0, lockdown_end=1,
        initial_grid=grid, rng=np.random.default_rng(0), store_grids=True)
    assert grids[1][2, 2] == S           # step 0 locked down -> no exposure
    assert grids[2][2, 2] == E           # step 1 not locked down (end exclusive)


def test_lockdown_guard_rejects_half_specified_window():
    dm = model.make_density_map(10, 0.5, 0.3, 0.15)
    with pytest.raises(ValueError):
        model.run_seiqr(10, dm, 0.5, 0.1, 0.05, 0.1, 5,
                        lockdown_map=dm, lockdown_start=None, lockdown_end=5)


def test_vaccinate_dose_count_and_input_untouched():
    """vaccinate immunises efficacy*doses cells and leaves the input grid unchanged."""
    n = 30
    grid = np.zeros((n, n))
    out = model.vaccinate(grid, n, 100, targeted=False, efficacy=1.0,
                          rng=np.random.default_rng(0))
    assert int((grid == R).sum()) == 0       # copy semantics: input unchanged
    assert int((out == R).sum()) == 100      # efficacy 1.0 -> exactly 100 immune
    out_half = model.vaccinate(grid, n, 200, targeted=False, efficacy=0.5,
                               rng=np.random.default_rng(1))
    assert 60 <= int((out_half == R).sum()) <= 140  # efficacy 0.5 -> roughly half


def test_vaccinate_targeted_prioritises_centre():
    n = 50
    grid = np.zeros((n, n))
    out = model.vaccinate(grid, n, 100, targeted=True, efficacy=1.0,
                          rng=np.random.default_rng(0))
    mid = 25
    rows = np.arange(n)
    dist = np.maximum(np.abs(rows[:, None] - mid), np.abs(rows[None, :] - mid))
    immune = out == R
    # 100 doses fit inside the 289-cell centre, so all immune cells are central.
    assert np.all(dist[immune] <= model.CENTRE_RADIUS)


def test_model_matches_seiqr_statistically():
    """The vectorised model.py and the original seiqr.py agree in distribution.

    They are not bit-identical (they consume random numbers in different orders),
    so this compares ensemble-mean peak infected and final recovered counts.
    """
    n = 50
    dm = model.make_density_map(n, 0.5, 0.3, 0.15)
    seeds = range(8)

    def new_run(seed):
        _, _, _, Ic, _, Rc = model.run_seiqr(
            n, dm, 0.5, 0.1, 0.05, 0.1, 100,
            rng=np.random.default_rng(seed), store_grids=False)
        return max(Ic), Rc[-1]

    def old_run(seed):
        np.random.seed(seed)
        _, _, _, Ic, _, Rc = seiqr.run_seiqr(n, dm, 0.5, 0.1, 0.05, 0.1, 100)
        return max(Ic), Rc[-1]

    new = np.array([new_run(s) for s in seeds])
    old = np.array([old_run(s) for s in seeds])

    # Peak infected means within one ensemble standard deviation or ~30 cells.
    assert abs(new[:, 0].mean() - old[:, 0].mean()) < 30
    # Final recovered (attack size) means within ~50 of ~1950-2050 cells.
    assert abs(new[:, 1].mean() - old[:, 1].mean()) < 50
