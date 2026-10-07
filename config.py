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

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class SimConfig:
    """Parameters for a single SEIQR simulation and the report experiment suite."""

    # Grid and time
    n: int = 50
    num_steps: int = 100

    # Per-timestep disease transition probabilities
    p_infect: float = 0.50  # E -> I
    p_quarantine: float = 0.10  # I -> Q
    p_recover_i: float = 0.05  # I -> R (unquarantined)
    p_recover_q: float = 0.10  # Q -> R

    # Zone exposure probabilities for the density map
    p_centre: float = 0.50
    p_middle: float = 0.30
    p_outer: float = 0.15
    p_uniform: float = 0.30  # single value used for the uniform-grid control

    # Lockdown: exposure probability applied within affected zones during
    # the half-open window [lockdown_start, lockdown_end)
    lockdown_p: float = 0.10
    lockdown_start: int = 10
    lockdown_end: int = 40

    # Vaccination applied before the simulation starts
    vax_doses: int = 200
    vax_efficacy: float = 0.80

    def __post_init__(self):
        """Reject parameter sets the model cannot represent.

        The model needs an interior, so n >= 3. Every probability must lie in
        [0, 1], and because I -> Q and I -> R are decided by a single draw,
        their probabilities must also sum to at most 1.
        """
        if self.n < 3:
            raise ValueError(f"n must be at least 3 (got {self.n}); the border never updates")
        if self.num_steps < 0:
            raise ValueError(f"num_steps must be non-negative (got {self.num_steps})")
        for f in fields(self):
            if f.name.startswith(("p_", "lockdown_p")) or f.name == "vax_efficacy":
                value = getattr(self, f.name)
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"{f.name} must be in [0, 1] (got {value})")
        if self.p_quarantine + self.p_recover_i > 1.0:
            raise ValueError(
                "p_quarantine + p_recover_i must not exceed 1 (got "
                f"{self.p_quarantine + self.p_recover_i}); one draw decides both exits"
            )
        if not 0 <= self.lockdown_start <= self.lockdown_end:
            raise ValueError(
                f"lockdown window must satisfy 0 <= start <= end "
                f"(got {self.lockdown_start}, {self.lockdown_end})"
            )
        if self.vax_doses < 0:
            raise ValueError(f"vax_doses must be non-negative (got {self.vax_doses})")
