"""
make_demo_gif.py: render a short GIF of the SEIQR epidemic spreading across the
density grid, for embedding in the README.

GitHub cannot render the live Plotly dashboard, so this produces a static
preview. The output is reproducible (fixed seed).

    python make_demo_gif.py   # writes assets/demo.gif

Requires the dev extras (matplotlib, pillow); see requirements-dev.txt.
"""

import os

import numpy as np

from config import SimConfig
from model import make_density_map, run_seiqr

STATE_COLOURS = ["#2563EB", "#F97316", "#DC2626", "#7C3AED", "#16A34A"]
STATE_LABELS = ["Susceptible", "Exposed", "Infected", "Quarantined", "Recovered"]

OUTPUT = os.path.join(os.path.dirname(__file__), "assets", "demo.gif")


def main():
    # Import matplotlib inside main so the headless backend is selected before
    # pyplot loads, without tripping the import-order lint at module level.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.colors import ListedColormap

    cfg = SimConfig()
    density = make_density_map(cfg.n, cfg.p_centre, cfg.p_middle, cfg.p_outer)
    grids, *_ = run_seiqr(
        cfg.n, density, cfg.p_infect, cfg.p_quarantine,
        cfg.p_recover_i, cfg.p_recover_q, cfg.num_steps,
        rng=np.random.default_rng(42), store_grids=True)

    frames = list(range(0, len(grids), 2))  # every second timestep
    cmap = ListedColormap(STATE_COLOURS)

    fig, ax = plt.subplots(figsize=(4.4, 4.6), dpi=100)
    im = ax.imshow(grids[0], cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    title = ax.set_title("SEIQR spread on the density grid  (t = 0)", fontsize=10)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in STATE_COLOURS]
    ax.legend(handles, STATE_LABELS, loc="upper center",
              bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=7, frameon=False)
    fig.tight_layout()

    def update(t):
        im.set_data(grids[t])
        title.set_text(f"SEIQR spread on the density grid  (t = {t})")
        return im, title

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    anim.save(OUTPUT, writer=PillowWriter(fps=12))
    plt.close(fig)
    print(f"wrote {OUTPUT}  "
          f"({os.path.getsize(OUTPUT) / 1024:.0f} KB, {len(frames)} frames)")


if __name__ == "__main__":
    main()
