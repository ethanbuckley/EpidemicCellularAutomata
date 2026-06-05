# SEIQR Epidemic Spread Simulator

A stochastic cellular automaton model of epidemic spread across a spatially heterogeneous population, built to investigate how population density gradients affect disease dynamics and the efficiency of public health interventions.

Built as a computing project during my Physics and Physical Chemistry degree at UCL.

---

## What It Does

The model simulates a population of 2,500 individuals on a 50x50 grid, divided into three concentric density zones representing an urban core, suburban ring, and rural outskirts. Each individual occupies one cell and can be in one of five states: Susceptible, Exposed, Infected, Quarantined, or Recovered (SEIQR).

At each timestep, state transitions are governed by probabilistic rules based on a Moore neighbourhood (the 8 surrounding cells). Transmission probability varies spatially according to the density zone, so the model captures how disease spreads locally rather than assuming a well-mixed population.

The simulation was used to test four questions:

- Does a density gradient change epidemic dynamics compared to a uniform population?
- Does infection spread outward in a wave-like pattern from dense regions?
- Do interventions targeted at the dense centre outperform uniform interventions?
- Does combining targeted lockdown and vaccination outperform either measure alone?

---

## Key Findings

**Density gradient vs uniform population**

The density grid produces a lower, earlier infection peak (around 210 at t=45) compared to the uniform grid (around 300 at t=60). The low-exposure outer zone acts as a natural brake on transmission, capping the worst-case peak at higher transmission rates.

**Wave-like propagation**

Infection spreads outward from the dense centre in a measurable wave. The centre zone peaks first (34% infected at t=21), the middle zone follows (25% at t=45), and the outer zone peaks last (17% at t=75). The centre drives early exponential growth.

**Targeted lockdown**

Locking down only the 289-cell centre zone (12% of the grid) produces nearly the same suppression as a whole-grid lockdown covering all 2,500 cells. The centre-only lockdown is significantly more resource efficient.

**Targeted vaccination**

200 doses concentrated in the centre zone dramatically outperform the same 200 doses distributed uniformly across the grid. Uniform distribution gives negligible protection per zone; targeted distribution removes approximately 55% of susceptible individuals from the zone driving transmission.

**Combined strategy**

Centre vaccination combined with centre-only lockdown achieves the best outcome overall, keeping peak infection below 60 during the lockdown window with a small post-lockdown rebound. This outperforms the blanket approach (uniform vaccination plus whole-grid lockdown) while using fewer resources.

---

## Model Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Grid size | 50x50 | 2,500 cells total |
| Timesteps | 100 | Simulation duration |
| p_infect | 0.50 | E to I transition probability |
| p_quarantine | 0.10 | I to Q transition probability |
| p_recover_i | 0.05 | I to R transition probability |
| p_recover_q | 0.10 | Q to R transition probability |
| p_expose (centre) | 0.50 | Transmission probability, urban core |
| p_expose (middle) | 0.30 | Transmission probability, suburban ring |
| p_expose (outer) | 0.15 | Transmission probability, rural outskirts |
| Vaccine doses | 200 | Applied before simulation starts |
| Vaccine efficacy | 80% | Probability of full immunity per dose |
| Lockdown window | t=10 to t=40 | p_expose reduced to 0.1 in affected zones |
| Runs per scenario | 5 | Stochastic averaging (3 for threshold sweep) |

---

## Intervention Strategies Tested

**Lockdown** reduces p_expose to 0.1 in affected zones during t=10 to t=40, applied either to the whole grid or to the centre zone only.

**Vaccination** sets 200 randomly selected susceptible cells to Recovered before the simulation starts, with 80% efficacy. Applied either uniformly across the grid or targeted at the centre zone first.

**Combined** pairs targeted centre vaccination with centre-only lockdown.

---

## Technical Stack

| Component | Technology |
|-----------|-----------|
| Simulation | Python, NumPy |
| Visualisation | Matplotlib |
| Stochastic averaging | 5 independent runs per scenario |
| Neighbourhood | Moore (8-cell) |

---

## Installation

```bash
git clone https://github.com/ethanbuckley/epidemic-seiqr.git
cd epidemic-seiqr
pip install numpy matplotlib
```

---

## Usage

```bash
python seiqr.py
```

The script runs all simulation scenarios and outputs grid snapshots, population curves, and comparison plots.

---

## Project Structure

```
epidemic-seiqr/
├── seiqr.py         # Full simulation and plotting pipeline
└── README.md
```

---

## Limitations

The 50x50 grid is much smaller than a real city population, which compresses the timescale of wave propagation. Zone boundaries are sharp squares rather than gradual transitions. Individuals are fixed to cells with no movement between zones, which overstates the advantage of centre-targeted interventions. Vaccine efficacy is binary rather than graded. Future extensions could introduce agent movement, real population density maps, age-stratified contact rates, and waning immunity.

---

## Reference

Ghosh, S. and Bhattacharya, S. (2021). Computational Model on COVID-19 Pandemic Using Probabilistic Cellular Automata. SN Computer Science, 2(3).

---

## Author

Ethan Buckley, MSci Natural Sciences (Physics and Physical Chemistry), UCL
[ethan.buckley.24@ucl.ac.uk](mailto:ethan.buckley.24@ucl.ac.uk)
