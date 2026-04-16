import streamlit as st
import numpy as np
import plotly.graph_objects as go
from System import DynamicalSystem, DEFAULT_PARAMS

from .constants import FULL_INIT
from .plots import plot_4var_ts, plot_bifurcation


@st.cache_data(show_spinner=False)
def s6_time_series(esw_val: float, sim_time: int) -> dict:
    p = DEFAULT_PARAMS.copy()
    p['e_sw'] = esw_val
    state = {k: np.float128(v) for k, v in FULL_INIT.items()}
    sys = DynamicalSystem(p, state, "dimensionalized")
    ts = sys.time_series_plot(time=sim_time)
    return {k: v.astype(np.float64) for k, v in ts.items()}


@st.cache_data(show_spinner=False)
def s6_bifurcation(esw_min: float, esw_max: float, resolution: int,
                   bif_time: int, burn_frac: float) -> tuple:
    esw_sweep = np.linspace(esw_min, esw_max, resolution)
    burn = int(bif_time * burn_frac)
    be_x, be_S, be_E = [], [], []
    for esw in esw_sweep:
        p = DEFAULT_PARAMS.copy()
        p['e_sw'] = float(esw)
        state = {k: np.float128(v) for k, v in FULL_INIT.items()}
        sys = DynamicalSystem(p, state, "dimensionalized")
        ts = sys.time_series_plot(time=bif_time)
        s_att = ts['Seafood'][burn:].astype(np.float64)
        e_att = ts['Effort'][burn:].astype(np.float64)
        n = len(s_att)
        be_x.extend([float(esw)] * n)
        be_S.extend(s_att.tolist())
        be_E.extend(e_att.tolist())
    return np.array(be_x), np.array(be_S), np.array(be_E)


@st.cache_data(show_spinner=False)
def s6_spectral_sweep(esw_min: float, esw_max: float, resolution: int) -> tuple:
    esw_vals = np.linspace(esw_min, esw_max, resolution)
    rho_vals = np.empty(resolution)
    for i, esw in enumerate(esw_vals):
        p = DEFAULT_PARAMS.copy()
        p['e_sw'] = float(esw)
        state = {k: np.float128(v) for k, v in FULL_INIT.items()}
        sys = DynamicalSystem(p, state, "dimensionalized")
        result = sys.stability_analysis()
        rho_vals[i] = result['spectral_radius']
    return esw_vals.astype(np.float64), rho_vals.astype(np.float64)


@st.fragment
def scenario_6():
    st.header("Scenario 6 — Wholesale Price Elasticity of Supply")
    st.caption(
        "High ε_sw → wholesale price drops sharply as harvest rises → "
        "dampens effort growth. "
        "Low ε_sw → wholesale price insensitive to supply → effort can spiral. "
        "Focus parameter: ε_sw (supply elasticity)."
    )

    with st.expander("Parameters", expanded=False):
        _c1, _c2, _c3 = st.columns(3)
        with _c1:
            s6_sim = st.slider("Simulation length", 100, 1000, 400, 50, key="s6_sim")
        with _c2:
            s6_res = st.slider("Bifurcation resolution", 50, 500, 200, 50, key="s6_res")
        with _c3:
            s6_rng = st.slider("ε_sw sweep range", 0.0, 5.0, (0.0, 3.0), 0.05, key="s6_rng")
        s6_esw_vals = st.multiselect(
            "ε_sw values for time series",
            [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0],
            default=[0.0, 0.5, 0.9, 2.0],
            key="s6_ev",
        )

    if not s6_esw_vals:
        st.warning("Select at least one *ε_sw* value.")
        return

    s6_esw_vals = sorted(s6_esw_vals)

    with st.status("Computing Scenario 6…", expanded=True) as status:
        st.write("Running time-series simulations…")
        ts6 = {esw: s6_time_series(float(esw), s6_sim) for esw in s6_esw_vals}
        t6 = np.arange(s6_sim + 1)
        st.write("Computing bifurcation diagram…")
        be_x, be_S, be_E = s6_bifurcation(
            float(s6_rng[0]), float(s6_rng[1]), s6_res, 300, 0.6,
        )
        st.write("Computing stability sweep…")
        s6_esw_sweep, s6_rho = s6_spectral_sweep(
            float(s6_rng[0]), float(s6_rng[1]), 100,
        )
        status.update(label="Scenario 6 ready", state="complete", expanded=False)

    s6_tab_ts, s6_tab_bif, s6_tab_stab = st.tabs(
        ["Time Series", "Bifurcation", "Stability"]
    )

    with s6_tab_ts:
        fig = plot_4var_ts(
            ts6, t6, s6_esw_vals, 'ε_sw',
            f'Wholesale Supply Elasticity — Time Series as ε_sw Varies   '
            f'(r={DEFAULT_PARAMS["r"]}  |  Low ε_sw = price insensitive to supply)',
        )
        st.plotly_chart(fig, width='stretch')

    with s6_tab_bif:
        _def_esw = DEFAULT_PARAMS['e_sw']
        fig = plot_bifurcation(
            be_x, be_S, be_E,
            xlabel='Supply Elasticity (ε_sw)',
            title='Bifurcation over ε_sw   '
                  '(Low → price insensitive  |  High → price responsive to supply)',
            vline_x=_def_esw, vline_label=f'Default ε_sw = {_def_esw}',
        )
        st.plotly_chart(fig, width='stretch')

    with s6_tab_stab:
        finite = np.isfinite(s6_rho)
        esw_fin, rho_fin = s6_esw_sweep[finite], s6_rho[finite]
        stable_mask = rho_fin < 1.0
        y_cap = max(float(np.max(rho_fin[rho_fin < 50])) * 1.1, 2.0) if np.any(rho_fin < 50) else 5.0
        rho_plot = np.clip(rho_fin, 0, y_cap)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=esw_fin[stable_mask], y=rho_plot[stable_mask],
            mode='markers', marker=dict(color='#2E8B57', size=6),
            name='Stable (ρ < 1)',
        ))
        fig.add_trace(go.Scatter(
            x=esw_fin[~stable_mask], y=rho_plot[~stable_mask],
            mode='markers', marker=dict(color='#DC143C', size=6),
            name='Unstable (ρ ≥ 1)',
        ))
        fig.add_hline(y=1.0, line_dash='dash', line_color='gray',
                      annotation_text='ρ = 1 (stability boundary)')
        fig.update_layout(
            height=600,
            title_text=(
                f'Spectral Radius vs ε_sw — Fixed-Point Stability   '
                f'(r={DEFAULT_PARAMS["r"]})'
            ),
            xaxis_title='Supply Elasticity (ε_sw)',
            yaxis_title='Spectral Radius  ρ = max|λᵢ|',
            yaxis_range=[0, y_cap],
            margin=dict(t=60, b=40),
            legend=dict(yanchor='top', y=0.99, xanchor='right', x=0.99),
        )
        st.plotly_chart(fig, width='stretch')
