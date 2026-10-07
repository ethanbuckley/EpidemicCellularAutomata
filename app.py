"""
app.py: SEIQR Epidemic Cellular Automaton dashboard.

Reads data/results.json for the ensemble scenarios (produced by
run_experiments.py) and runs model.py live for the interactive single-run view.
Default parameter values come from SimConfig so the dashboard, the experiment
suite and the README describe the same model.
"""

import json
import os

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from config import SimConfig
from model import (
    STATE_NAMES,
    initial_grid_single_seed,
    make_density_map,
    run_seiqr,
    vaccinate,
)

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

CFG = SimConfig()
N = CFG.n
NUM_STEPS = CFG.num_steps

_CURVE_COLORS = {
    "S": "#2563EB",
    "E": "#F97316",
    "I": "#DC2626",
    "Q": "#7C3AED",
    "R": "#16A34A",
}

# Discrete colour scale mapping states 0-4 to the S/E/I/Q/R colours; each
# state occupies a 0.2-wide band of [0, 1].
_STATE_COLORSCALE = [
    [k / 5 + edge, _CURVE_COLORS[name]] for k, name in enumerate(STATE_NAMES) for edge in (0.0, 0.2)
]

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "results.json")


# =============================================================================
# DATA LOADING
# =============================================================================


@st.cache_data
def load_results():
    if not os.path.exists(DATA_PATH):
        return None
    with open(DATA_PATH) as f:
        return json.load(f)


results = load_results()


# =============================================================================
# FIGURE HELPERS
# =============================================================================


def _line(fig, t, y, name, color, dash="solid", width=2):
    fig.add_trace(
        go.Scatter(
            x=t,
            y=y,
            name=name,
            mode="lines",
            line=dict(color=color, width=width, dash=dash),
        )
    )


def _base_fig(title, yaxis="Cell count", height=360):
    fig = go.Figure()
    fig.update_layout(
        title=title,
        xaxis_title="Timestep",
        yaxis_title=yaxis,
        legend=dict(orientation="h", y=1.15),
        margin=dict(l=0, r=0, t=60, b=0),
        height=height,
    )
    return fig


def _add_lockdown_shade(fig, window):
    if window:
        fig.add_vrect(
            x0=window[0],
            x1=window[1],
            fillcolor="gray",
            opacity=0.15,
            line_width=0,
            annotation_text="lockdown",
            annotation_position="top left",
        )


def _make_grid_fig(grid: np.ndarray, step: int) -> go.Figure:
    fig = go.Figure(
        go.Heatmap(
            z=grid,
            colorscale=_STATE_COLORSCALE,
            zmin=0,
            zmax=4,
            showscale=False,
            hoverongaps=False,
            hovertemplate="Row %{y}, Col %{x}<br>State %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text=f"Grid at t = {step}", font_size=14),
        xaxis=dict(showticklabels=False, scaleanchor="y"),
        yaxis=dict(showticklabels=False, autorange="reversed"),
        margin=dict(l=0, r=0, t=40, b=0),
        height=380,
    )
    return fig


def _make_curve_fig(curves: dict, highlight_step=None, lockdown_window=None) -> go.Figure:
    """Population curves for one run. ``curves`` maps 'S'..'R' to count lists."""
    t = list(range(len(curves["S"])))
    fig = _base_fig("Population curves", height=380)
    fig.update_layout(legend=dict(orientation="h", y=1.12), margin=dict(t=50))
    _add_lockdown_shade(fig, lockdown_window)
    for name in STATE_NAMES:
        _line(fig, t, curves[name], name, _CURVE_COLORS[name])
    if highlight_step is not None:
        fig.add_vline(x=highlight_step, line_dash="dot", line_color="black", opacity=0.4)
    return fig


# =============================================================================
# HEADER + DISCLAIMER
# =============================================================================

st.title("SEIQR Epidemic Cellular Automaton")
st.caption(
    "Individual extension of a UCL group project. "
    "Original model by Ethan Buckley and team; "
    "see the README for full credits and methodology."
)

st.info(
    f"**Educational project only.** This model is a simplified {N}×{N} grid "
    "simulation for academic exploration of epidemic dynamics. "
    "It does not constitute public-health advice.",
    icon="ℹ️",
)

tab_interactive, tab_report, tab_perf = st.tabs(
    [
        "Interactive Simulation",
        "Report Scenarios",
        "Performance",
    ]
)


# =============================================================================
# TAB 1: INTERACTIVE SINGLE RUN
# =============================================================================

with tab_interactive:
    st.subheader("Configure and run a single simulation")

    col_ctrl, col_vis = st.columns([1, 2], gap="large")

    with col_ctrl:
        st.markdown("**Disease parameters**")
        p_infect = st.slider("p_infect (E→I)", 0.0, 1.0, CFG.p_infect, 0.01)
        p_quarantine = st.slider("p_quarantine (I→Q)", 0.0, 1.0, CFG.p_quarantine, 0.01)
        p_recover_i = st.slider("p_recover_i (I→R)", 0.0, 1.0, CFG.p_recover_i, 0.01)
        p_recover_q = st.slider("p_recover_q (Q→R)", 0.0, 1.0, CFG.p_recover_q, 0.01)
        if p_quarantine + p_recover_i > 1.0:
            st.warning(
                "p_quarantine + p_recover_i exceeds 1. One random draw decides "
                "both exits from I, so the recovery branch is truncated to "
                f"{1.0 - p_quarantine:.2f}."
            )

        st.markdown("**Density map**")
        p_centre = st.slider("p_expose (centre)", 0.0, 1.0, CFG.p_centre, 0.05)
        p_middle = st.slider("p_expose (middle)", 0.0, 1.0, CFG.p_middle, 0.05)
        p_outer = st.slider("p_expose (outer)", 0.0, 1.0, CFG.p_outer, 0.05)

        st.markdown("**Lockdown**")
        lockdown_scope = st.radio("Scope", ["None", "Centre only", "Whole grid"], horizontal=True)
        if lockdown_scope != "None":
            ld_start, ld_end = st.slider(
                "Window (timesteps)",
                0,
                NUM_STEPS,
                (CFG.lockdown_start, CFG.lockdown_end),
                help="Half-open: the lockdown map applies from the start step up to, "
                "but not including, the end step. The window is kept at least "
                "one step wide.",
            )
            if ld_end <= ld_start:
                # Streamlit's range slider allows a zero-width selection, which
                # would be a lockdown that never applies. Widen it to one step.
                ld_end = min(ld_start + 1, NUM_STEPS)
                ld_start = ld_end - 1
                st.caption(f"Zero-width window widened to the single step t = {ld_start}.")
        else:
            ld_start, ld_end = None, None

        st.markdown("**Vaccination**")
        vax_scope = st.radio("Target", ["None", "Uniform", "Centre-first"], horizontal=True)
        vax_doses = st.slider("Doses", 0, 500, CFG.vax_doses, 10, disabled=(vax_scope == "None"))
        vax_eff = st.slider(
            "Efficacy", 0.0, 1.0, CFG.vax_efficacy, 0.05, disabled=(vax_scope == "None")
        )

        seed_val = st.number_input("Random seed", 0, 9999, 42, 1)
        run_btn = st.button("Run simulation", type="primary", width="stretch")

    if run_btn or "sim_curves" not in st.session_state:
        density_map = make_density_map(N, p_centre, p_middle, p_outer)

        if lockdown_scope == "Whole grid":
            lockdown_map = make_density_map(N, CFG.lockdown_p, CFG.lockdown_p, CFG.lockdown_p)
        elif lockdown_scope == "Centre only":
            lockdown_map = make_density_map(N, CFG.lockdown_p, p_middle, p_outer)
        else:
            lockdown_map = None

        rng = np.random.default_rng(seed_val)
        init_grid = initial_grid_single_seed(N)
        if vax_scope != "None":
            init_grid = vaccinate(
                init_grid,
                vax_doses,
                targeted=(vax_scope == "Centre-first"),
                efficacy=vax_eff,
                rng=rng,
            )

        grids, *curves = run_seiqr(
            N,
            density_map,
            p_infect,
            p_quarantine,
            p_recover_i,
            p_recover_q,
            NUM_STEPS,
            lockdown_map=lockdown_map,
            lockdown_start=ld_start,
            lockdown_end=ld_end,
            initial_grid=init_grid,
            rng=rng,
            store_grids=True,
        )
        st.session_state.update(
            dict(
                sim_grids=grids,
                sim_curves=dict(zip(STATE_NAMES, curves, strict=True)),
                sim_lockdown=(ld_start, ld_end) if lockdown_scope != "None" else None,
            )
        )

    with col_vis:
        step = st.slider("Timestep", 0, NUM_STEPS, 0, key="step_slider")

        left, right = st.columns(2)
        with left:
            st.plotly_chart(
                _make_grid_fig(st.session_state["sim_grids"][step], step), width="stretch"
            )
        with right:
            st.plotly_chart(
                _make_curve_fig(
                    st.session_state["sim_curves"],
                    highlight_step=step,
                    lockdown_window=st.session_state.get("sim_lockdown"),
                ),
                width="stretch",
            )

        st.markdown(
            "<small>🔵 Susceptible &nbsp;🟠 Exposed &nbsp;"
            "🔴 Infected &nbsp;🟣 Quarantined &nbsp;🟢 Recovered</small>",
            unsafe_allow_html=True,
        )


# =============================================================================
# TAB 2: REPORT SCENARIOS
# =============================================================================

with tab_report:
    if results is None:
        st.warning(
            "No precomputed results found. "
            "Run `python run_experiments.py` locally to generate `data/results.json`."
        )
        st.stop()

    T = list(range(NUM_STEPS + 1))
    LW = results.get("lockdown_window", [CFG.lockdown_start, CFG.lockdown_end])
    zone_sizes = results.get("zone_sizes", {})
    st.caption(f"Precomputed results, generated at {results['generated_at']} UTC")

    # Fig 4a: uniform vs density grid
    st.subheader("Uniform vs density grid")
    fig4a = _base_fig("Total infected over time (5-run mean)")
    _line(fig4a, T, results["uniform_grid"]["I"], "Uniform grid", "#6366F1")
    _line(fig4a, T, results["density_grid"]["I"], "Density grid", "#DC2626")
    st.plotly_chart(fig4a, width="stretch")

    peak_u = max(results["uniform_grid"]["I"])
    peak_d = max(results["density_grid"]["I"])
    st.caption(
        f"The density grid produces a lower, earlier infection peak (about {peak_d:.0f} cells) "
        f"than the uniform grid (about {peak_u:.0f}). The low-exposure outer zone acts as a "
        "natural brake on transmission."
    )

    # Fig 4b: zone-level wave
    st.subheader("Wave-like propagation by zone")
    fig4b = _base_fig("Fraction of each zone infected (5-run mean)", yaxis="Fraction infected")
    _line(fig4b, T, results["zone_breakdown"]["centre"], "Centre", "#DC2626")
    _line(fig4b, T, results["zone_breakdown"]["middle"], "Middle", "#F97316")
    _line(fig4b, T, results["zone_breakdown"]["outer"], "Outer", "#2563EB")
    st.plotly_chart(fig4b, width="stretch")

    st.caption(
        "Infection spreads outward in a measurable wave: the centre peaks first and "
        "highest, the middle zone follows, and the outer zone peaks last."
    )

    st.divider()

    # Fig 5a + 5b: interventions
    st.subheader("Intervention strategies")
    col5a, col5b = st.columns(2)

    with col5a:
        fig5a = _base_fig("Lockdown comparison (5-run mean)")
        _add_lockdown_shade(fig5a, LW)
        _line(fig5a, T, results["lockdown_none"]["I"], "No lockdown", "#6B7280")
        _line(fig5a, T, results["lockdown_whole"]["I"], "Whole-grid", "#DC2626")
        _line(fig5a, T, results["lockdown_centre"]["I"], "Centre-only", "#F97316", dash="dash")
        st.plotly_chart(fig5a, width="stretch")
        centre_pct = 100 * zone_sizes.get("centre", 289) / (N * N)
        st.caption(
            f"Locking down only the centre zone ({centre_pct:.0f}% of the grid) achieves "
            "comparable suppression to a whole-grid lockdown."
        )

    with col5b:
        fig5b = _base_fig(f"Vaccination comparison ({CFG.vax_doses} doses, 5-run mean)")
        _line(fig5b, T, results["vax_none"]["I"], "No vaccination", "#6B7280")
        _line(fig5b, T, results["vax_uniform"]["I"], "Uniform", "#6366F1")
        _line(fig5b, T, results["vax_targeted"]["I"], "Targeted (centre)", "#16A34A", dash="dash")
        st.plotly_chart(fig5b, width="stretch")
        st.caption(
            f"{CFG.vax_doses} doses concentrated in the centre zone lower the peak far more "
            f"than the same doses spread uniformly across all {N * N:,} cells."
        )

    st.divider()

    # Fig 7: combined strategy
    st.subheader("Combined strategy")
    fig7 = _base_fig("Combined strategy comparison (5-run mean)")
    _add_lockdown_shade(fig7, LW)
    _line(fig7, T, results["combined_none"]["I"], "No intervention", "#6B7280")
    _line(fig7, T, results["combined_vax_only"]["I"], "Targeted vax only", "#F97316")
    _line(fig7, T, results["combined_blanket"]["I"], "Uniform vax + whole lockdown", "#6366F1")
    _line(
        fig7,
        T,
        results["combined_targeted"]["I"],
        "Targeted vax + centre lockdown",
        "#16A34A",
        dash="dash",
    )
    st.plotly_chart(fig7, width="stretch")
    st.caption(
        "The targeted combination (centre vaccination + centre lockdown) achieves "
        "the lowest peak while using fewer resources than the blanket approach."
    )

    st.divider()

    # Fig 6: threshold sweep
    st.subheader("Epidemic threshold sweep")
    ts = results["threshold_sweep"]
    fig6 = _base_fig(
        "Peak infected vs baseline p_expose (3-run mean)", yaxis="Peak infected", height=340
    )
    _line(fig6, ts["baselines"], ts["uniform_peak"], "Uniform grid", "#6366F1")
    _line(fig6, ts["baselines"], ts["density_peak"], "Density grid", "#DC2626", dash="dash")
    fig6.update_layout(xaxis_title="Baseline p_expose")
    st.plotly_chart(fig6, width="stretch")
    st.caption(
        "The density grid caps worst-case epidemic peaks at higher transmission rates. "
        "At low transmission (p_expose ≤ 0.075) the density grid can produce a slightly "
        "higher peak because the dense centre retains enough transmission to sustain "
        "a small outbreak."
    )


# =============================================================================
# TAB 3: PERFORMANCE AND VALIDATION
# =============================================================================

with tab_perf:
    st.subheader("Vectorization speedup")

    sp = results["speedup"] if results else None
    if sp:
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Original (loop)",
            f"{sp['original_s']:.3f}s",
            help=f"seiqr.py nested Python loop over all {N * N:,} cells",
        )
        c2.metric(
            "Vectorized",
            f"{sp['vectorized_s']:.4f}s",
            help="model.py NumPy + scipy.signal.convolve2d",
        )
        c3.metric("Speedup", f"{sp['speedup_x']:.0f}×")

    interior = (N - 2) ** 2
    st.markdown(
        f"""
        **What changed**

        The original `seiqr_advance` function in `seiqr.py` uses a double Python
        `for` loop over every interior cell ({N - 2} × {N - 2} = {interior:,} iterations
        per timestep), with individual `np.random.rand()` calls inside the loop.

        The vectorized `model.py` replaces this with:

        - **One `scipy.signal.convolve2d` call** over the {N} × {N} infected-state
          array to compute Moore-neighbour counts for all cells simultaneously.
        - **One `rng.random(({N}, {N}))` draw per transition group** (four draws total
          per timestep: S→E, E→I, I→Q/R, Q→R), compared against probability arrays
          via boolean masking.
        - **Zero Python loops** over individual cells.
        """
    )
    if sp:
        st.markdown(
            f"At the timings above, a 5-run ensemble takes about "
            f"{5 * sp['original_s']:.1f} s with the original loop and about "
            f"{5 * sp['vectorized_s']:.2f} s vectorised, which is fast enough to run "
            "live in the Interactive Simulation tab."
        )

    with st.expander("Validation details"):
        st.markdown(
            """
            The vectorised implementation is checked against the original by the
            test suite in `tests/test_model.py`, which runs in CI:

            1. **Density map**: the three zones have exactly 289, 800 and 1,411
               cells, with boundaries at the shared radius constants.
            2. **Transition logic**: a susceptible cell is exposed only when it has
               an infected Moore neighbour; quarantined cells do not transmit;
               boundary cells stay susceptible even at maximum exposure; each
               transition fires deterministically at probability 0 and 1.
            3. **Statistical equivalence**: ensemble-mean peak infected and final
               recovered counts over 8 seeds agree between old and new
               implementations. They are not bit-identical because the two
               implementations consume random numbers in different orders, which
               is expected for a stochastic model.
            """
        )

    # Validation against the analytical SEIQR ODE
    ov = results.get("ode_validation") if results else None
    if ov:
        st.divider()
        st.subheader("Validation against the analytical SEIQR ODE")

        m_ode = ov["metrics"]["vs_ode"]
        m_rec = ov["metrics"]["vs_discrete"]
        N_int = ov["N_interior"]
        wm = ov["well_mixed_ca"]
        n_runs = ov["n_runs"]
        n_extinct = wm.get("n_extinct")
        attack_major = wm.get("attack_major")
        Tv = list(range(len(ov["ode"]["I"])))

        wm_I, wm_sd = wm["I"], wm["I_std"]
        upper = [m + s for m, s in zip(wm_I, wm_sd, strict=True)]
        lower = [m - s for m, s in zip(wm_I, wm_sd, strict=True)]

        figv = _base_fig(
            "Infected count: analytical ODE vs well-mixed and spatial CA",
            yaxis=f"Infected (of {N_int} interior cells)",
            height=420,
        )
        figv.add_trace(
            go.Scatter(
                x=Tv + Tv[::-1],
                y=upper + lower[::-1],
                fill="toself",
                fillcolor="rgba(37,99,235,0.15)",
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        _line(figv, Tv, wm_I, f"Global-coupling CA ({n_runs} runs, ±1σ)", "#2563EB")
        _line(figv, Tv, ov["ode"]["I"], "Mean-field ODE", "#111827")
        _line(figv, Tv, ov["local_ca"]["I"], "Local (spatial) CA", "#DC2626", dash="dash")
        st.plotly_chart(figv, width="stretch")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("R₀ (well-mixed)", f"{ov['rates']['R0']:.1f}")
        c2.metric(
            "Peak I: CA / ODE",
            f"{m_ode['peak_ca']:.0f} / {m_ode['peak_ref']:.0f}",
            help="Well-mixed CA ensemble-mean peak vs the ODE peak "
            f"({m_ode['peak_rel_err_pct']:.1f}% apart)",
        )
        c3.metric("RMSE vs exact recursion", f"{m_rec['rmse_pct_of_peak']:.1f}% of peak")
        if n_extinct is not None:
            c4.metric(
                "Runs that died out",
                f"{n_extinct} of {n_runs}",
                help="Runs whose peak infected count never reached "
                f"{100 * ov.get('major_outbreak_threshold', 0.1):.0f}% of the "
                "interior population. A deterministic ODE cannot die out.",
            )
            c5.metric(
                "Attack rate: CA / ODE",
                f"{attack_major:.3f} / {m_ode['attack_ref']:.3f}",
                help="Final recovered fraction over the runs with a major outbreak. "
                f"Over all {n_runs} runs, including those that died out, the CA "
                f"mean is {m_ode['attack_ca']:.3f}.",
            )
        else:
            c4.metric(
                "Attack rate: CA / ODE", f"{m_ode['attack_ca']:.3f} / {m_ode['attack_ref']:.3f}"
            )

        extinction_note = ""
        if n_extinct:
            extinction_note = (
                f" {n_extinct} of the {n_runs} runs died out before a major outbreak, which "
                "the deterministic references cannot do; those runs are included in the "
                "blue mean curve and pull it down by roughly that fraction. Over the runs "
                f"that did take off, the final attack rate is {attack_major:.3f} against the "
                f"ODE's {m_ode['attack_ref']:.3f}."
            )
        st.caption(
            "The CA's local rules reduce to the classical well-mixed SEIQR ODE in the "
            "mean-field limit. Driven by the global infected fraction instead of local "
            "neighbours, the CA ensemble (blue, ±1σ) reproduces the exact discrete "
            f"recursion to {m_rec['rmse_pct_of_peak']:.1f}% of peak (RMSE) and matches the "
            f"continuous ODE's peak height to {m_ode['peak_rel_err_pct']:.1f}%."
            f"{extinction_note} The ODE peaks a few steps earlier (a continuous-vs-discrete "
            f"time effect at R₀ ≈ {ov['rates']['R0']:.0f}), which is why the black and blue "
            "curves are offset rather than coincident. The standard local (spatial) CA "
            "(red dashed) departs from the ODE with a lower, later, broader peak. That gap "
            "is the effect of spatial structure and local susceptible depletion, not a "
            f"validation failure. Curves use the participating interior of {N_int} cells; "
            "the grid border is permanently susceptible and excluded."
        )

# =============================================================================
# FOOTER
# =============================================================================

st.divider()
st.markdown(
    "Built by **Ethan Buckley** · "
    "[GitHub](https://github.com/ethanbuckley) · "
    "[LinkedIn](https://www.linkedin.com/in/ethan-buckley/) · "
    "UCL MSci Natural Sciences"
)
