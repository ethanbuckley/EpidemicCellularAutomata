"""
app.py — SEIQR Epidemic Cellular Automaton Dashboard
Reads data/results.json for ensemble scenarios (produced by run_experiments.py).
Runs model.py live for the interactive single-simulation view.
"""

import json
import os

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from model import make_density_map, vaccinate, run_seiqr, I

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="SEIQR Epidemic Simulator",
    page_icon="🦠",
    layout="wide",
)

# =============================================================================
# CONSTANTS
# =============================================================================

N = 50
NUM_STEPS = 100

# Discrete colour scale mapping states 0–4 to S/E/I/Q/R colours
# Each state occupies a 0.2-wide band in [0,1]
_STATE_COLORSCALE = [
    [0.00, "#2563EB"], [0.20, "#2563EB"],   # S — blue
    [0.20, "#F97316"], [0.40, "#F97316"],   # E — orange
    [0.40, "#DC2626"], [0.60, "#DC2626"],   # I — red
    [0.60, "#7C3AED"], [0.80, "#7C3AED"],   # Q — purple
    [0.80, "#16A34A"], [1.00, "#16A34A"],   # R — green
]

_CURVE_COLORS = {
    "S": "#2563EB", "E": "#F97316",
    "I": "#DC2626", "Q": "#7C3AED", "R": "#16A34A",
}

# =============================================================================
# DATA LOADING
# =============================================================================

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "results.json")


@st.cache_data
def load_results():
    if not os.path.exists(DATA_PATH):
        return None
    with open(DATA_PATH) as f:
        return json.load(f)


results = load_results()

# =============================================================================
# HEADER + DISCLAIMER
# =============================================================================

st.title("SEIQR Epidemic Cellular Automaton")
st.caption(
    "Individual extension of a UCL group project. "
    "Original model by Ethan Buckley and team — "
    "see the README for full credits and methodology."
)

st.info(
    "**Educational project only.** This model is a simplified 50×50 grid "
    "simulation for academic exploration of epidemic dynamics. "
    "It does not constitute public-health advice.",
    icon="ℹ️",
)

# =============================================================================
# TABS
# =============================================================================

tab_interactive, tab_report, tab_perf = st.tabs([
    "Interactive Simulation", "Report Scenarios", "Performance",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — INTERACTIVE SINGLE RUN
# ─────────────────────────────────────────────────────────────────────────────

def _make_grid_fig(grid: np.ndarray, step: int) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=grid,
        colorscale=_STATE_COLORSCALE,
        zmin=0, zmax=4,
        showscale=False,
        hoverongaps=False,
        hovertemplate="Row %{y}, Col %{x}<br>State %{z}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"Grid at t = {step}", font_size=14),
        xaxis=dict(showticklabels=False, scaleanchor="y"),
        yaxis=dict(showticklabels=False, autorange="reversed"),
        margin=dict(l=0, r=0, t=40, b=0),
        height=380,
    )
    return fig


def _make_curve_fig(S_c, E_c, I_c, Q_c, R_c,
                    highlight_step=None,
                    lockdown_window=None) -> go.Figure:
    t = list(range(len(S_c)))
    fig = go.Figure()

    if lockdown_window:
        ls, le = lockdown_window
        fig.add_vrect(x0=ls, x1=le, fillcolor="gray",
                      opacity=0.15, line_width=0,
                      annotation_text="lockdown", annotation_position="top left")

    for label, counts, color in [
        ("S", S_c, _CURVE_COLORS["S"]),
        ("E", E_c, _CURVE_COLORS["E"]),
        ("I", I_c, _CURVE_COLORS["I"]),
        ("Q", Q_c, _CURVE_COLORS["Q"]),
        ("R", R_c, _CURVE_COLORS["R"]),
    ]:
        fig.add_trace(go.Scatter(
            x=t, y=counts, name=label, line=dict(color=color, width=2),
            mode="lines",
        ))

    if highlight_step is not None:
        fig.add_vline(x=highlight_step, line_dash="dot",
                      line_color="black", opacity=0.4)

    fig.update_layout(
        title="Population curves",
        xaxis_title="Timestep",
        yaxis_title="Cell count",
        legend=dict(orientation="h", y=1.12),
        margin=dict(l=0, r=0, t=50, b=0),
        height=380,
    )
    return fig


with tab_interactive:
    st.subheader("Configure and run a single simulation")

    # ── Sidebar-style controls in two columns ────────────────────────────────
    col_ctrl, col_vis = st.columns([1, 2], gap="large")

    with col_ctrl:
        st.markdown("**Disease parameters**")
        p_infect     = st.slider("p_infect (E→I)",     0.0, 1.0, 0.50, 0.01)
        p_quarantine = st.slider("p_quarantine (I→Q)", 0.0, 1.0, 0.10, 0.01)
        p_recover_i  = st.slider("p_recover_i (I→R)",  0.0, 1.0, 0.05, 0.01)
        p_recover_q  = st.slider("p_recover_q (Q→R)",  0.0, 1.0, 0.10, 0.01)

        st.markdown("**Density map**")
        p_centre = st.slider("p_expose (centre)", 0.0, 1.0, 0.50, 0.05)
        p_middle = st.slider("p_expose (middle)", 0.0, 1.0, 0.30, 0.05)
        p_outer  = st.slider("p_expose (outer)",  0.0, 1.0, 0.15, 0.05)

        st.markdown("**Lockdown**")
        lockdown_scope = st.radio(
            "Scope", ["None", "Centre only", "Whole grid"], horizontal=True)
        if lockdown_scope != "None":
            ld_start, ld_end = st.slider(
                "Window (timesteps)", 0, NUM_STEPS, (10, 40))
        else:
            ld_start, ld_end = None, None

        st.markdown("**Vaccination**")
        vax_scope  = st.radio("Target", ["None", "Uniform", "Centre-first"],
                               horizontal=True)
        vax_doses  = st.slider("Doses", 0, 500, 200, 10,
                                disabled=(vax_scope == "None"))
        vax_eff    = st.slider("Efficacy", 0.0, 1.0, 0.80, 0.05,
                                disabled=(vax_scope == "None"))

        seed_val   = st.number_input("Random seed", 0, 9999, 42, 1)
        run_btn    = st.button("Run simulation", type="primary",
                               width="stretch")

    # ── Run simulation ────────────────────────────────────────────────────────
    if run_btn or "sim_grids" not in st.session_state:
        density_map = make_density_map(N, p_centre, p_middle, p_outer)

        if lockdown_scope == "Whole grid":
            lockdown_map = make_density_map(N, 0.1, 0.1, 0.1)
        elif lockdown_scope == "Centre only":
            lockdown_map = make_density_map(N, 0.1, p_middle, p_outer)
        else:
            lockdown_map = None

        rng = np.random.default_rng(seed_val)
        init_grid = np.zeros((N, N))
        init_grid[N // 2, N // 2] = I

        if vax_scope != "None":
            targeted = (vax_scope == "Centre-first")
            init_grid = vaccinate(init_grid, N, vax_doses, targeted,
                                  efficacy=vax_eff, rng=rng)

        grids, Sc, Ec, Ic, Qc, Rc = run_seiqr(
            N, density_map, p_infect, p_quarantine, p_recover_i, p_recover_q,
            NUM_STEPS,
            lockdown_map=lockdown_map,
            lockdown_start=ld_start,
            lockdown_end=ld_end,
            initial_grid=init_grid,
            rng=rng,
            store_grids=True,
        )
        st.session_state.update(dict(
            sim_grids=grids, sim_S=Sc, sim_E=Ec, sim_I=Ic, sim_Q=Qc, sim_R=Rc,
            sim_lockdown=(ld_start, ld_end) if lockdown_scope != "None" else None,
        ))

    # ── Visualisation ─────────────────────────────────────────────────────────
    with col_vis:
        step = st.slider("Timestep", 0, NUM_STEPS, 0, key="step_slider")
        lw   = st.session_state.get("sim_lockdown")

        left, right = st.columns(2)
        with left:
            st.plotly_chart(
                _make_grid_fig(st.session_state["sim_grids"][step], step),
                width="stretch",
            )
        with right:
            st.plotly_chart(
                _make_curve_fig(
                    st.session_state["sim_S"], st.session_state["sim_E"],
                    st.session_state["sim_I"], st.session_state["sim_Q"],
                    st.session_state["sim_R"],
                    highlight_step=step, lockdown_window=lw,
                ),
                width="stretch",
            )

        # Legend
        st.markdown(
            "<small>🔵 Susceptible &nbsp;🟠 Exposed &nbsp;"
            "🔴 Infected &nbsp;🟣 Quarantined &nbsp;🟢 Recovered</small>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — REPORT SCENARIOS
# ─────────────────────────────────────────────────────────────────────────────

def _line(fig, t, y, name, color, dash="solid", width=2):
    fig.add_trace(go.Scatter(
        x=t, y=y, name=name,
        line=dict(color=color, width=width, dash=dash),
    ))


def _base_fig(title, yaxis="Cell count", height=360):
    fig = go.Figure()
    fig.update_layout(
        title=title, xaxis_title="Timestep", yaxis_title=yaxis,
        legend=dict(orientation="h", y=1.15),
        margin=dict(l=0, r=0, t=60, b=0), height=height,
    )
    return fig


def _add_lockdown_shade(fig, window):
    if window:
        fig.add_vrect(x0=window[0], x1=window[1], fillcolor="gray",
                      opacity=0.15, line_width=0,
                      annotation_text="lockdown", annotation_position="top left")


with tab_report:
    if results is None:
        st.warning(
            "No precomputed results found. "
            "Run `python run_experiments.py` locally to generate `data/results.json`."
        )
        st.stop()

    T = list(range(NUM_STEPS + 1))
    LW = results.get("lockdown_window", [10, 40])
    st.caption(f"Precomputed results — generated at {results['generated_at']} UTC")

    # ── Fig 4a: Uniform vs density grid ──────────────────────────────────────
    st.subheader("Uniform vs density grid")
    fig4a = _base_fig("Total infected over time (5-run mean)")
    _line(fig4a, T, results["uniform_grid"]["I"], "Uniform grid", "#6366F1")
    _line(fig4a, T, results["density_grid"]["I"], "Density grid", "#DC2626")
    st.plotly_chart(fig4a, width="stretch")

    st.caption(
        "The density grid produces a lower, earlier infection peak (~210 at t≈45) "
        "vs the uniform grid (~300 at t≈60). The low-exposure outer zone acts as a "
        "natural brake on transmission."
    )

    # ── Fig 4b: Zone-level wave ───────────────────────────────────────────────
    st.subheader("Wave-like propagation by zone")
    fig4b = _base_fig("Fraction of each zone infected (5-run mean)", yaxis="Fraction infected")
    _line(fig4b, T, results["zone_breakdown"]["centre"], "Centre", "#DC2626")
    _line(fig4b, T, results["zone_breakdown"]["middle"], "Middle", "#F97316")
    _line(fig4b, T, results["zone_breakdown"]["outer"],  "Outer",  "#2563EB")
    st.plotly_chart(fig4b, width="stretch")

    st.caption(
        "Infection spreads outward in a measurable wave: the centre peaks first and "
        "highest, the middle zone follows, and the outer zone peaks last."
    )

    st.divider()

    # ── Fig 5a + 5b: Interventions ────────────────────────────────────────────
    st.subheader("Intervention strategies")
    col5a, col5b = st.columns(2)

    with col5a:
        fig5a = _base_fig("Lockdown comparison (5-run mean)")
        _add_lockdown_shade(fig5a, LW)
        _line(fig5a, T, results["lockdown_none"]["I"],   "No lockdown",      "#6B7280")
        _line(fig5a, T, results["lockdown_whole"]["I"],  "Whole-grid",       "#DC2626")
        _line(fig5a, T, results["lockdown_centre"]["I"], "Centre-only",      "#F97316", dash="dash")
        st.plotly_chart(fig5a, width="stretch")
        st.caption(
            "Centre-only lockdown (12% of grid) achieves comparable suppression "
            "to whole-grid lockdown — far more resource efficient."
        )

    with col5b:
        fig5b = _base_fig("Vaccination comparison (200 doses, 5-run mean)")
        _line(fig5b, T, results["vax_none"]["I"],     "No vaccination", "#6B7280")
        _line(fig5b, T, results["vax_uniform"]["I"],  "Uniform",        "#6366F1")
        _line(fig5b, T, results["vax_targeted"]["I"], "Targeted (centre)", "#16A34A", dash="dash")
        st.plotly_chart(fig5b, width="stretch")
        st.caption(
            "200 doses concentrated in the centre zone dramatically outperform "
            "the same doses distributed uniformly across 2,500 cells."
        )

    st.divider()

    # ── Fig 7: Combined strategy ──────────────────────────────────────────────
    st.subheader("Combined strategy")
    fig7 = _base_fig("Combined strategy comparison (5-run mean)")
    _add_lockdown_shade(fig7, LW)
    _line(fig7, T, results["combined_none"]["I"],     "No intervention",             "#6B7280")
    _line(fig7, T, results["combined_vax_only"]["I"], "Targeted vax only",           "#F97316")
    _line(fig7, T, results["combined_blanket"]["I"],  "Uniform vax + whole lockdown","#6366F1")
    _line(fig7, T, results["combined_targeted"]["I"], "Targeted vax + centre lockdown","#16A34A", dash="dash")
    st.plotly_chart(fig7, width="stretch")
    st.caption(
        "The targeted combination (centre vaccination + centre lockdown) achieves "
        "the lowest peak while using fewer resources than the blanket approach."
    )

    st.divider()

    # ── Fig 6: Threshold sweep ────────────────────────────────────────────────
    st.subheader("Epidemic threshold sweep")
    ts = results["threshold_sweep"]
    fig6 = _base_fig("Peak infected vs baseline p_expose (3-run mean)",
                     yaxis="Peak infected", height=340)
    fig6.add_trace(go.Scatter(
        x=ts["baselines"], y=ts["uniform_peak"], name="Uniform grid",
        line=dict(color="#6366F1", width=2),
    ))
    fig6.add_trace(go.Scatter(
        x=ts["baselines"], y=ts["density_peak"], name="Density grid",
        line=dict(color="#DC2626", width=2, dash="dash"),
    ))
    fig6.update_layout(xaxis_title="Baseline p_expose")
    st.plotly_chart(fig6, width="stretch")
    st.caption(
        "The density grid caps worst-case epidemic peaks at higher transmission rates. "
        "At low transmission (p_expose ≤ 0.075) the density grid can produce a slightly "
        "higher peak because the dense centre retains enough transmission to sustain "
        "a small outbreak."
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────

with tab_perf:
    st.subheader("Vectorization speedup")

    if results:
        sp = results["speedup"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Original (loop)", f"{sp['original_s']:.3f}s",
                  help="seiqr.py nested Python loop over all 2,500 cells")
        c2.metric("Vectorized", f"{sp['vectorized_s']:.4f}s",
                  help="model.py NumPy + scipy.signal.convolve2d")
        c3.metric("Speedup", f"{sp['speedup_x']:.0f}×")

    st.markdown(
        """
        **What changed**

        The original `seiqr_advance` function in `seiqr.py` uses a double Python
        `for` loop over every interior cell (48 × 48 = 2,304 iterations per timestep),
        with individual `np.random.rand()` calls inside the loop.

        The vectorized `model.py` replaces this with:

        - **One `scipy.signal.convolve2d` call** over the 50 × 50 infected-state
          array to compute Moore-neighbour counts for all cells simultaneously.
        - **One `rng.random(50, 50)` draw per transition group** (four draws total
          per timestep: S→E, E→I, I→Q/R, Q→R), compared against probability arrays
          via boolean masking.
        - **Zero Python loops** over individual cells.

        At 100 timesteps per run and 5 runs per ensemble, this brings the full
        5-scenario ensemble from ~2.7 s down to ~0.07 s — fast enough to run
        live on every slider interaction in this dashboard.
        """
    )

    with st.expander("Validation details"):
        st.markdown(
            """
            The vectorized implementation was validated against the original on three levels:

            1. **Density map**: pixel-identical output confirmed with `np.allclose`.
            2. **Transition logic**: with one infected seed cell at the grid centre,
               6 of 8 Moore neighbours correctly became Exposed on the first step;
               boundary cells remained permanently Susceptible; the seed cell
               transitioned to I/Q/R without reverting.
            3. **Statistical equivalence**: peak infected and final recovered counts
               across 10 independent seeds are consistent between old and new
               implementations (both draw from the same distribution — ~200–240
               peak I, ~1,950–2,050 final R on a 50×50 grid). They are not
               bit-identical because the two implementations consume random numbers
               in different orders, which is expected and acceptable for a stochastic model.
            """
        )

    # ── Validation against the analytical SEIQR ODE ───────────────────────────
    ov = results.get("ode_validation") if results else None
    if ov:
        st.divider()
        st.subheader("Validation against the analytical SEIQR ODE")

        m_ode = ov["metrics"]["vs_ode"]
        m_rec = ov["metrics"]["vs_discrete"]
        N_int = ov["N_interior"]
        Tv = list(range(len(ov["ode"]["I"])))

        wm_I  = ov["well_mixed_ca"]["I"]
        wm_sd = ov["well_mixed_ca"]["I_std"]
        upper = [m + s for m, s in zip(wm_I, wm_sd)]
        lower = [m - s for m, s in zip(wm_I, wm_sd)]

        figv = go.Figure()
        figv.add_trace(go.Scatter(
            x=Tv + Tv[::-1], y=upper + lower[::-1], fill="toself",
            fillcolor="rgba(37,99,235,0.15)", line=dict(width=0),
            hoverinfo="skip", showlegend=False,
        ))
        figv.add_trace(go.Scatter(
            x=Tv, y=wm_I, name=f"Global-coupling CA ({ov['n_runs']} runs, ±1σ)",
            line=dict(color="#2563EB", width=2)))
        figv.add_trace(go.Scatter(
            x=Tv, y=ov["ode"]["I"], name="Mean-field ODE",
            line=dict(color="#111827", width=2)))
        figv.add_trace(go.Scatter(
            x=Tv, y=ov["local_ca"]["I"], name="Local (spatial) CA",
            line=dict(color="#DC2626", width=2, dash="dash")))
        figv.update_layout(
            title="Infected count: analytical ODE vs well-mixed and spatial CA",
            xaxis_title="Timestep", yaxis_title=f"Infected (of {N_int} interior cells)",
            legend=dict(orientation="h", y=1.15),
            margin=dict(l=0, r=0, t=60, b=0), height=420,
        )
        st.plotly_chart(figv, width="stretch")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R₀ (well-mixed)", f"{ov['rates']['R0']:.1f}")
        c2.metric("Peak I: CA / ODE", f"{m_ode['peak_ca']:.0f} / {m_ode['peak_ref']:.0f}",
                  help="Well-mixed CA ensemble-mean peak vs the ODE peak "
                       f"({m_ode['peak_rel_err_pct']:.1f}% apart)")
        c3.metric("Attack rate: CA / ODE",
                  f"{m_ode['attack_ca']:.3f} / {m_ode['attack_ref']:.3f}")
        c4.metric("RMSE vs exact recursion", f"{m_rec['rmse_pct_of_peak']:.1f}% of peak")

        st.caption(
            "The CA's local rules reduce to the classical well-mixed SEIQR ODE in the "
            "mean-field limit. Driven by the global infected fraction instead of local "
            "neighbours, the CA ensemble (blue, ±1σ) reproduces the exact discrete "
            f"recursion to {m_rec['rmse_pct_of_peak']:.1f}% of peak, and matches the "
            f"continuous ODE's peak height to {m_ode['peak_rel_err_pct']:.1f}% and its final "
            "attack rate almost exactly. The ODE peaks a few steps earlier (a "
            f"continuous-vs-discrete time effect at R₀ ≈ {ov['rates']['R0']:.0f}), which is why "
            "the black and blue curves are offset rather than coincident. The standard local "
            "(spatial) CA (red dashed) departs from the ODE with a lower, later, broader peak. "
            "That gap is the effect of spatial structure and local susceptible depletion, not a "
            f"validation failure. Curves use the participating interior of {N_int} cells; the "
            "grid border is permanently susceptible and excluded."
        )

# =============================================================================
# FOOTER
# =============================================================================

st.divider()
st.markdown(
    "Built by **Ethan Buckley** · "
    "[GitHub](https://github.com/ethanbuckley) · "
    "[LinkedIn](https://www.linkedin.com/in/ethan-buckley/) · "
    "UCL MSci Natural Sciences",
    unsafe_allow_html=False,
)
