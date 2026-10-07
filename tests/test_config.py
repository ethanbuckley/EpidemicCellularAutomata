"""Tests for SimConfig validation (config.py)."""

import dataclasses

import pytest

from config import SimConfig


def test_defaults_are_valid_and_frozen():
    cfg = SimConfig()
    assert cfg.n == 50 and cfg.num_steps == 100
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.n = 10


@pytest.mark.parametrize(
    "field, value",
    [
        ("p_infect", 1.5),
        ("p_outer", -0.1),
        ("lockdown_p", 2.0),
        ("vax_efficacy", 1.01),
    ],
)
def test_probabilities_must_lie_in_unit_interval(field, value):
    with pytest.raises(ValueError, match=field):
        SimConfig(**{field: value})


def test_single_draw_exits_from_i_must_not_exceed_one():
    """I -> Q and I -> R share one draw, so their probabilities cannot sum past 1."""
    SimConfig(p_quarantine=0.6, p_recover_i=0.4)  # exactly 1 is allowed
    with pytest.raises(ValueError, match="p_quarantine \\+ p_recover_i"):
        SimConfig(p_quarantine=0.6, p_recover_i=0.5)


def test_grid_needs_an_interior():
    with pytest.raises(ValueError, match="n must be at least 3"):
        SimConfig(n=2)


def test_lockdown_window_must_be_ordered():
    with pytest.raises(ValueError, match="lockdown window"):
        SimConfig(lockdown_start=40, lockdown_end=10)
