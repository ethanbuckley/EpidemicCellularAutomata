"""Tests for the vectorised SEIQR cellular automaton core (model.py)."""

import numpy as np
import pytest

import model
import seiqr
from model import E, I, Q, R, S


def test_density_map_zone_counts():
    """Three concentric zones with the documented cell counts and probabilities."""
    dm = model.make_density_map(50, 0.5, 0.3, 0.15)
    assert dm.shape == (50, 50)
    assert int((dm == 0.5).sum()) == 289  # centre: dist <= 8
    assert int((dm == 0.3).sum()) == 800  # middle: 8 < dist <= 16
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
    grids, *_ = model.run_seiqr(
        20, dm, 1.0, 0.5, 0.2, 0.2, 40, rng=np.random.default_rng(0), store_grids=True
    )
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
            30, dm, 0.5, 0.1, 0.05, 0.1, 30, rng=np.random.default_rng(seed), store_grids=False
        )
        return Ic

    assert peak(123) == peak(123)
    assert peak(123) != peak(456)


def test_susceptible_needs_infected_neighbour():
    """S -> E fires next to an infected cell but not for a distant susceptible."""
    n = 9
    dm = np.ones((n, n))  # exposure certain wherever eligible
    grid = np.zeros((n, n))
    grid[4, 4] = I  # single infected at centre
    nxt = model.seiqr_advance(grid, dm, 0.0, 0.0, 0.0, 0.0, rng=np.random.default_rng(0))
    assert nxt[3, 3] == E  # a Moore neighbour of the seed
    assert nxt[5, 4] == E  # another neighbour
    assert nxt[1, 1] == S  # interior but far from any infected
    assert nxt[4, 4] == I  # seed persists (no recovery this step)


def test_exposed_to_infected_is_gated_by_p_infect():
    """E -> I fires iff the draw is below p_infect; the extremes are deterministic."""
    n = 6
    dm = np.zeros((n, n))  # suppress new exposures
    grid = np.zeros((n, n))
    grid[2, 2] = E
    became_i = model.seiqr_advance(grid, dm, 1.0, 0.0, 0.0, 0.0, rng=np.random.default_rng(0))
    stayed_e = model.seiqr_advance(grid, dm, 0.0, 0.0, 0.0, 0.0, rng=np.random.default_rng(0))
    assert became_i[2, 2] == I
    assert stayed_e[2, 2] == E


def test_quarantined_do_not_transmit():
    """Quarantined cells are excluded from the infected-neighbour count."""
    n = 5
    dm = np.ones((n, n))
    grid = np.zeros((n, n))
    grid[2, 2] = Q  # only a quarantined cell, no I
    nxt = model.seiqr_advance(grid, dm, 1.0, 0.0, 0.0, 0.0, rng=np.random.default_rng(0))
    assert not np.any(nxt == E)  # nothing gets exposed


def test_infected_quarantine_threshold():
    """With p_quarantine = 1 every interior infected cell moves to Q."""
    n = 6
    dm = np.zeros((n, n))
    grid = np.zeros((n, n))
    grid[1:-1, 1:-1] = I
    nxt = model.seiqr_advance(grid, dm, 0.0, 1.0, 0.0, 0.0, rng=np.random.default_rng(0))
    assert np.all(nxt[1:-1, 1:-1] == Q)


def test_lockdown_window_is_half_open():
    """The lockdown map applies on [start, end): active at start, inactive at end."""
    n = 5
    density = np.ones((n, n))  # exposure certain
    lockdown = np.zeros((n, n))  # exposure impossible
    grid = np.zeros((n, n))
    grid[2, 1] = I  # infected next to interior S at (2, 2)
    grids, *_ = model.run_seiqr(
        n,
        density,
        0.0,
        0.0,
        0.0,
        0.0,
        2,
        lockdown_map=lockdown,
        lockdown_start=0,
        lockdown_end=1,
        initial_grid=grid,
        rng=np.random.default_rng(0),
        store_grids=True,
    )
    assert grids[1][2, 2] == S  # step 0 locked down -> no exposure
    assert grids[2][2, 2] == E  # step 1 not locked down (end exclusive)


def test_lockdown_guard_rejects_half_specified_window():
    dm = model.make_density_map(10, 0.5, 0.3, 0.15)
    with pytest.raises(ValueError):
        model.run_seiqr(
            10, dm, 0.5, 0.1, 0.05, 0.1, 5, lockdown_map=dm, lockdown_start=None, lockdown_end=5
        )


def test_vaccinate_dose_count_and_input_untouched():
    """vaccinate immunises efficacy*doses cells and leaves the input grid unchanged."""
    n = 30
    grid = np.zeros((n, n))
    out = model.vaccinate(grid, 100, targeted=False, efficacy=1.0, rng=np.random.default_rng(0))
    assert int((grid == R).sum()) == 0  # copy semantics: input unchanged
    assert int((out == R).sum()) == 100  # efficacy 1.0 -> exactly 100 immune
    out_half = model.vaccinate(
        grid, 200, targeted=False, efficacy=0.5, rng=np.random.default_rng(1)
    )
    assert 60 <= int((out_half == R).sum()) <= 140  # efficacy 0.5 -> roughly half


def test_vaccinate_targeted_prioritises_centre():
    n = 50
    grid = np.zeros((n, n))
    out = model.vaccinate(grid, 100, targeted=True, efficacy=1.0, rng=np.random.default_rng(0))
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
            n, dm, 0.5, 0.1, 0.05, 0.1, 100, rng=np.random.default_rng(seed), store_grids=False
        )
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


def test_infected_recover_branch_uses_same_draw():
    """With p_quarantine = 0 and p_recover_i = 1 every interior infected cell recovers."""
    n = 6
    dm = np.zeros((n, n))
    grid = np.zeros((n, n))
    grid[1:-1, 1:-1] = I
    nxt = model.seiqr_advance(grid, dm, 0.0, 0.0, 1.0, 0.0, rng=np.random.default_rng(0))
    assert np.all(nxt[1:-1, 1:-1] == R)


def test_apply_transitions_is_the_shared_step():
    """seiqr_advance equals apply_transitions with the density map masked by neighbours."""
    n = 12
    dm = model.make_density_map(n, 0.9, 0.5, 0.2)
    grid = np.zeros((n, n))
    grid[5:8, 5:8] = I
    grid[3, 3] = E
    grid[8, 2] = Q
    via_advance = model.seiqr_advance(grid, dm, 0.5, 0.1, 0.05, 0.1, rng=np.random.default_rng(7))
    is_I = grid == I
    neighbours = sum(
        np.roll(np.roll(is_I, di, 0), dj, 1)
        for di in (-1, 0, 1)
        for dj in (-1, 0, 1)
        if (di, dj) != (0, 0)
    )
    p_expose = np.where(neighbours > 0, dm, 0.0)
    via_shared = model.apply_transitions(
        grid, p_expose, 0.5, 0.1, 0.05, 0.1, np.random.default_rng(7)
    )
    assert np.array_equal(via_advance, via_shared)


def test_state_counts_partition_the_grid():
    grid = np.zeros((7, 7))
    grid[1, 1] = E
    grid[2, 2:5] = I
    grid[3, 3] = Q
    grid[4:6, 4] = R
    counts = model.state_counts(grid)
    assert counts.tolist() == [49 - 7, 1, 3, 1, 2]
    assert counts.sum() == 49


def test_zone_masks_partition_the_grid():
    masks = model.zone_masks(50)
    assert set(masks) == {"centre", "middle", "outer"}
    stacked = np.stack(list(masks.values()))
    assert np.all(stacked.sum(axis=0) == 1)  # every cell in exactly one zone
    assert [int(m.sum()) for m in masks.values()] == [289, 800, 1411]


def test_run_seiqr_curves_sum_to_grid_size():
    n = 20
    dm = model.make_density_map(n, 0.5, 0.3, 0.15)
    _, *curves = model.run_seiqr(
        n, dm, 0.5, 0.1, 0.05, 0.1, 25, rng=np.random.default_rng(3), store_grids=False
    )
    totals = np.array(curves).sum(axis=0)
    assert totals.shape == (26,)
    assert np.all(totals == n * n)


def test_run_ensemble_is_seeded_per_run():
    """run_ensemble(seed=s) is the mean of run_seiqr with rng seeds s, s+1, ..."""
    n, steps = 15, 10
    dm = model.make_density_map(n, 0.5, 0.3, 0.15)
    args = (n, dm, 0.5, 0.1, 0.05, 0.1, steps)
    ens = model.run_ensemble(3, *args, seed=11)
    manual = np.mean(
        [
            model.run_seiqr(*args, rng=np.random.default_rng(11 + k), store_grids=False)[3]
            for k in range(3)
        ],
        axis=0,
    )
    assert np.array_equal(ens["I"], manual)
    assert set(ens) == {"S", "E", "I", "Q", "R"}


def test_vaccinate_caps_doses_at_available_susceptibles():
    n = 6
    grid = np.zeros((n, n))
    grid[2, 2] = I
    out = model.vaccinate(grid, 10_000, targeted=True, efficacy=1.0, rng=np.random.default_rng(0))
    assert int((out == R).sum()) == n * n - 1  # every susceptible immunised
    assert out[2, 2] == I  # the infected seed is untouched


def test_grids_use_compact_integer_dtype():
    """Snapshots are int8, and a float initial grid is cast rather than kept as float64."""
    n = 10
    dm = model.make_density_map(n, 0.5, 0.3, 0.15)
    grids, *_ = model.run_seiqr(
        n, dm, 0.5, 0.1, 0.05, 0.1, 3, rng=np.random.default_rng(0), store_grids=True
    )
    assert all(g.dtype == model.GRID_DTYPE for g in grids)
    float_init = np.zeros((n, n))
    float_init[n // 2, n // 2] = I
    grids2, *_ = model.run_seiqr(
        n,
        dm,
        0.5,
        0.1,
        0.05,
        0.1,
        3,
        initial_grid=float_init,
        rng=np.random.default_rng(0),
        store_grids=True,
    )
    assert grids2[0].dtype == model.GRID_DTYPE
    assert float_init.dtype == np.float64  # caller's array untouched
    assert np.array_equal(grids2[-1], grids[-1])  # same run, same seed
