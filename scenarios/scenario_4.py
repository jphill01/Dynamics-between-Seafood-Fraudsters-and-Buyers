import streamlit as st
import numpy as np
import plotly.graph_objects as go
from System import DynamicalSystem, DEFAULT_PARAMS

from .constants import _C0, _Q0, FULL_INIT
from .plots import plot_4var_ts, plot_bifurcation, plot_return_maps


def _eez_params(beta: float) -> dict:
    return {
        'q1': float(_Q0 + beta * 0.23),
        'c1': float(_C0 + beta * 1.10),
    }


@st.cache_data(show_spinner=False)
def s4_time_series(beta_val: float, sim_time: int) -> dict:
    p = DEFAULT_PARAMS.copy()
    p.update(_eez_params(beta_val))
    state = {k: np.float128(v) for k, v in FULL_INIT.items()}
    sys = DynamicalSystem(p, state, "dimensionalized")
    ts = sys.time_series_plot(time=sim_time)
    return {k: v.astype(np.float64) for k, v in ts.items()}


@st.cache_data(show_spinner=False)
def s4_bifurcation(b_min: float, b_max: float, resolution: int,
                   bif_time: int, burn_frac: float) -> tuple:
    b_sweep = np.linspace(b_min, b_max, resolution)
    burn = int(bif_time * burn_frac)
    bb_b, bb_S, bb_E = [], [], []
    for bv in b_sweep:
        p = DEFAULT_PARAMS.copy()
        p.update(_eez_params(float(bv)))
        state = {k: np.float128(v) for k, v in FULL_INIT.items()}
        sys = DynamicalSystem(p, state, "dimensionalized")
        ts = sys.time_series_plot(time=bif_time)
        s_att = ts['Seafood'][burn:].astype(np.float64)
        e_att = ts['Effort'][burn:].astype(np.float64)
        n = len(s_att)
        bb_b.extend([float(bv)] * n)
        bb_S.extend(s_att.tolist())
        bb_E.extend(e_att.tolist())
    return np.array(bb_b), np.array(bb_S), np.array(bb_E)


@st.cache_data(show_spinner=False)
def s4_stability_heatmap(c1_min: float, c1_max: float,
                         q1_min: float, q1_max: float,
                         resolution: int) -> tuple:
    c1_arr = np.linspace(c1_min, c1_max, resolution)
    q1_arr = np.linspace(q1_min, q1_max, resolution)
    stable_grid = np.full((resolution, resolution), np.nan)
    for i, q1 in enumerate(q1_arr):
        for j, c1 in enumerate(c1_arr):
            p = DEFAULT_PARAMS.copy()
            p.update({'c1': float(c1), 'q1': float(q1)})
            state = {k: np.float128(v) for k, v in FULL_INIT.items()}
            sys = DynamicalSystem(p, state, "dimensionalized")
            result = sys.stability_analysis()
            stable_grid[i, j] = 1.0 if result['stable'] else 0.0
    return c1_arr.astype(np.float64), q1_arr.astype(np.float64), stable_grid


@st.fragment
def scenario_4():
    st.header("Scenario 4 — Non-Enforcement of EEZ")
    st.caption(
        "Fishers access outside-EEZ waters: q₁↑ (more fish), c₁↑ (higher cost). "
        f"pw₁ stays at default ({DEFAULT_PARAMS['pw1']}). "
        "Focus parameter: EEZ violation intensity β ∈ [0, 1]."
    )

    with st.expander("Parameters", expanded=False):
        _c1, _c2, _c3 = st.columns(3)
        with _c1:
            s4_sim = st.slider("Simulation length", 100, 1000, 400, 50, key="s4_sim")
        with _c2:
            s4_res = st.slider("Bifurcation resolution", 50, 500, 200, 50, key="s4_res")
        with _c3:
            s4_rng = st.slider("β sweep range", 0.0, 1.0, (0.0, 1.0), 0.05, key="s4_rng")
        s4_b_vals = st.multiselect(
            "β values for time series & poincare",
            [0.0, 0.10, 0.15, 0.25, 0.40, 0.55, 0.70, 0.85, 1.00],
            default=[0.15, 0.40, 0.70, 1.00],
            key="s4_bv",
        )

    if not s4_b_vals:
        st.warning("Select at least one *β* value.")
        return

    s4_b_vals = sorted(s4_b_vals)
    _burn4 = int(s4_sim * 0.6)

    with st.status("Computing Scenario 4…", expanded=True) as status:
        st.write("Running time-series simulations…")
        ts4 = {b: s4_time_series(float(b), s4_sim) for b in s4_b_vals}
        t4 = np.arange(s4_sim + 1)
        st.write("Computing bifurcation diagram…")
        bb_b, bb_S, bb_E = s4_bifurcation(
            float(s4_rng[0]), float(s4_rng[1]), s4_res, 300, 0.6,
        )
        st.write("Computing stability heatmap…")
        s4_c1_arr, s4_q1_arr, s4_stable = s4_stability_heatmap(
            0.1, 3.0, 0.01, 0.5, 30,
        )
        status.update(label="Scenario 4 ready", state="complete", expanded=False)

    ep_labels = [
        f'β={b}  (q₁={_eez_params(b)["q1"]:.2f}, c₁={_eez_params(b)["c1"]:.2f})'
        for b in s4_b_vals
    ]

    tab_ts, tab_bif, tab_rm, tab_stab = st.tabs(
        ["Time Series", "Bifurcation", "Poincare", "Stability"]
    )

    with tab_ts:
        fig = plot_4var_ts(
            ts4, t4, s4_b_vals, 'β',
            f'EEZ Non-Enforcement — Time Series by Violation Intensity   '
            f'(r={DEFAULT_PARAMS["r"]}  |  q₁↑  c₁↑  |  '
            f'pw₁={DEFAULT_PARAMS["pw1"]} default)',
        )
        for i, lbl in enumerate(ep_labels):
            fig.layout.annotations[i].text = lbl
        st.plotly_chart(fig, width='stretch')

    with tab_bif:
        fig = plot_bifurcation(
            bb_b, bb_S, bb_E,
            xlabel='EEZ Violation Intensity (β)',
            title='Bifurcation over β   '
                  '(β=0 → honest  |  β=1 → q₁=0.30, c₁=2.00)',
        )
        st.plotly_chart(fig, width='stretch')

    with tab_rm:
        fig = plot_return_maps(ts4, s4_b_vals, 'β', _burn4)
        st.plotly_chart(fig, width='stretch')

    with tab_stab:
        s4_stable_clean = np.nan_to_num(s4_stable, nan=0.0)
        colorscale = [[0, '#DC143C'], [1, '#2E8B57']]
        fig = go.Figure(data=go.Heatmap(
            z=s4_stable_clean, x=s4_c1_arr, y=s4_q1_arr,
            colorscale=colorscale, zmin=0, zmax=1,
            showscale=False,
            hovertemplate='c₁=%{x:.2f}<br>q₁=%{y:.3f}<br>%{customdata}<extra></extra>',
            customdata=np.where(s4_stable_clean == 1.0, 'Stable', 'Unstable'),
        ))
        fig.add_trace(go.Scatter(
            x=[_C0], y=[_Q0], mode='markers',
            marker=dict(color='white', size=14, symbol='x',
                        line=dict(color='black', width=2)),
            name='Default (c₀, q₀)',
        ))
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(color='#2E8B57', size=10, symbol='square'),
            name='Stable (ρ < 1)',
        ))
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(color='#DC143C', size=10, symbol='square'),
            name='Unstable (ρ ≥ 1)',
        ))
        fig.update_layout(
            height=600,
            title_text=(
                f'Binary Stability Map — c₁ vs q₁   '
                f'(pw₁={DEFAULT_PARAMS["pw1"]},  r={DEFAULT_PARAMS["r"]})'
            ),
            xaxis_title='Fishing Cost (c₁)',
            yaxis_title='Catchability (q₁)',
            margin=dict(t=60, b=40),
            legend=dict(yanchor='top', y=0.99, xanchor='right', x=0.99),
        )
        st.plotly_chart(fig, width='stretch')
