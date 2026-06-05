"""
Cellular Automaton Simulation of Epidemic Spread
SEIQR model on a 2D grid with Moore neighbourhood and density zones.

States: S=0, E=1, I=2, Q=3, R=4
"""

import numpy as np
import matplotlib.pyplot as plt
import time

# State colours for grid plots: S=blue, E=orange, I=red, Q=purple, R=green
STATE_COLORS = np.array([
    [0, 0, 1],       # S = blue
    [1, 0.65, 0],    # E = orange
    [1, 0, 0],       # I = red
    [0.5, 0, 0.5],   # Q = purple
    [0, 0.5, 0]      # R = green
])

# Core simulation

def seiqr_advance(grid, n, density_map, p_infect, p_quarantine,
                  p_recover_i, p_recover_q):
    """
    Apply one timestep of SEIQR rules.
    p_expose is read from density_map[i,j] for each cell.
    Quarantined cells do NOT spread infection.
    """
    new_grid = grid.copy()

    for i in range(1, n - 1):
        for j in range(1, n - 1):
            num_infected = np.sum(grid[i-1:i+2, j-1:j+2] == 2)
            if grid[i, j] == 2:
                num_infected -= 1

            p_expose = density_map[i, j]

            if grid[i, j] == 0:  # Susceptible
                if num_infected > 0 and np.random.rand() < p_expose:
                    new_grid[i, j] = 1

            elif grid[i, j] == 1:  # Exposed
                if np.random.rand() < p_infect:
                    new_grid[i, j] = 2

            elif grid[i, j] == 2:  # Infected
                r = np.random.rand()
                if r < p_quarantine:
                    new_grid[i, j] = 3
                elif r < p_quarantine + p_recover_i:
                    new_grid[i, j] = 4

            elif grid[i, j] == 3:  # Quarantined
                if np.random.rand() < p_recover_q:
                    new_grid[i, j] = 4

    return new_grid


def run_seiqr(n, density_map, p_infect, p_quarantine,
              p_recover_i, p_recover_q, num_steps,
              lockdown_map=None, lockdown_start=None, lockdown_end=None,
              initial_grid=None):
    """
    Run SEIQR simulation with density-dependent p_expose.
    If lockdown_map is provided, switches to it during [lockdown_start, lockdown_end).
    If initial_grid is provided (e.g. with vaccination), uses it.
    Otherwise starts with one infected cell in the centre.
    """
    if initial_grid is not None:
        grid = initial_grid.copy()
    else:
        grid = np.zeros((n, n))
        grid[n // 2, n // 2] = 2

    all_grids = [grid.copy()]
    S_counts = [np.sum(grid == 0)]
    E_counts = [np.sum(grid == 1)]
    I_counts = [np.sum(grid == 2)]
    Q_counts = [np.sum(grid == 3)]
    R_counts = [np.sum(grid == 4)]

    for step in range(num_steps):
        if lockdown_map is not None and lockdown_start <= step < lockdown_end:
            current_map = lockdown_map
        else:
            current_map = density_map

        grid = seiqr_advance(grid, n, current_map, p_infect,
                             p_quarantine, p_recover_i, p_recover_q)
        all_grids.append(grid.copy())
        S_counts.append(np.sum(grid == 0))
        E_counts.append(np.sum(grid == 1))
        I_counts.append(np.sum(grid == 2))
        Q_counts.append(np.sum(grid == 3))
        R_counts.append(np.sum(grid == 4))

    return all_grids, S_counts, E_counts, I_counts, Q_counts, R_counts


def make_density_map(n, p_centre, p_middle, p_outer):
    """
    Create n x n density map with 3 concentric square zones.
    """
    density = np.zeros((n, n))
    mid = n // 2

    for i in range(n):
        for j in range(n):
            dist = max(abs(i - mid), abs(j - mid))
            if dist <= 8:
                density[i, j] = p_centre
            elif dist <= 16:
                density[i, j] = p_middle
            else:
                density[i, j] = p_outer

    return density


def vaccinate(grid, n, num_to_vaccinate, targeted, efficacy=0.80):
    """
    Vaccinate susceptible cells.
    efficacy: The real-world success rate of the vaccine (e.g., 0.80 = 80%).
    If the vaccine 'fails' for an individual, they remain susceptible.
    """
    mid = n // 2
    if targeted:
        centre_cells = []
        other_cells = []
        for i in range(n):
            for j in range(n):
                if grid[i, j] == 0:
                    dist = max(abs(i - mid), abs(j - mid))
                    if dist <= 8:
                        centre_cells.append((i, j))
                    else:
                        other_cells.append((i, j))
        np.random.shuffle(centre_cells)
        np.random.shuffle(other_cells)
        candidates = centre_cells + other_cells
    else:
        candidates = []
        for i in range(n):
            for j in range(n):
                if grid[i, j] == 0:
                    candidates.append((i, j))
        np.random.shuffle(candidates)

    count = min(num_to_vaccinate, len(candidates))
    for k in range(count):
        i, j = candidates[k]

        if np.random.rand() < efficacy:
            grid[i, j] = 4
        # Otherwise, they stay State 0 (Susceptible) and can cause breakthrough infections

    return grid
