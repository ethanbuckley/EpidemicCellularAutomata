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

# Palette matches the portfolio site (ethanbuckley.github.io, September 2026
# tokens): ordered by value, not hue, so the states separate for readers who
# cannot rely on colour alone. S = site --line, E/I = amber and red, Q = the
# site's neutral grey, R = the site accent.
STATE_COLOURS = ["#E4E7EB", "#E0A03D", "#A52F22", "#8D97A3", "#2440B3"]
STATE_LABELS = ["Susceptible", "Exposed", "Infected", "Quarantined", "Recovered"]
INK = "#14181d"          # site --ink
FONTS = ["Helvetica Neue", "Arial", "DejaVu Sans"]   # the site's system sans, then matplotlib's default

OUTPUT = os.path.join(os.path.dirname(__file__), "assets", "demo.gif")


def main():
    # Import matplotlib inside main so the headless backend is selected before
    # pyplot loads, without tripping the import-order lint at module level.
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = FONTS
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
    # Ink, not pure black: the susceptible state is now light, so the frame has to
    # hold the grid edge against the white canvas without shouting.
    for spine in ax.spines.values():
        spine.set_edgecolor(INK)
        spine.set_linewidth(0.8)
    title = ax.set_title("SEIQR spread on the density grid  (t = 0)",
                         fontsize=10, color=INK)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in STATE_COLOURS]
    legend = ax.legend(handles, STATE_LABELS, loc="upper center",
                       bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=7, frameon=False)
    for text in legend.get_texts():
        text.set_color(INK)
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
