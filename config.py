"""
config.py: tunable parameters for the SEIQR cellular automaton.

SimConfig bundles the experiment parameters that were previously scattered as
module-level constants in run_experiments.py. The defaults reproduce the values
from the original group report (Table 1), so an unmodified SimConfig() gives the
published scenarios; construct SimConfig(p_infect=..., ...) to explore variants.

The structural zone radii are deliberately not held here. They live in model.py
(CENTRE_RADIUS, MIDDLE_RADIUS) because they define the density-map geometry
rather than a tunable rate.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SimConfig:
    """Parameters for a single SEIQR simulation and the report experiment suite."""

    # Grid and time
    n: int = 50
    num_steps: int = 100

    # Per-timestep disease transition probabilities
    p_infect: float = 0.50      # E -> I
    p_quarantine: float = 0.10  # I -> Q
    p_recover_i: float = 0.05   # I -> R (unquarantined)
    p_recover_q: float = 0.10   # Q -> R

    # Zone exposure probabilities for the density map
    p_centre: float = 0.50
    p_middle: float = 0.30
    p_outer: float = 0.15
    p_uniform: float = 0.30     # single value used for the uniform-grid control

    # Lockdown: exposure probability applied within affected zones during
    # the half-open window [lockdown_start, lockdown_end)
    lockdown_p: float = 0.10
    lockdown_start: int = 10
    lockdown_end: int = 40

    # Vaccination applied before the simulation starts
    vax_doses: int = 200
    vax_efficacy: float = 0.80
