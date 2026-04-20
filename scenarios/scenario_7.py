import streamlit as st
import numpy as np
import plotly.graph_objects as go
from System import DynamicalSystem, DEFAULT_PARAMS

from .constants import FULL_INIT
from .plots import plot_4var_ts, plot_bifurcation


@st.cache_data(show_spinner=False)
def s7_time_series(esm_val: float, sim_time: int) -> dict:
    p = DEFAULT_PARAMS.copy()
    p['e_sm'] = esm_val
    state = {k: np.float128(v) for k, v in FULL_INIT.items()}
    sys = DynamicalSystem(p, state, "dimensionalized")
    ts = sys.time_series_plot(time=sim_time)
    return {k: v.astype(np.float64) for k, v in ts.items()}


@st.cache_data(show_spinner=False)
def s7_bifurcation(esm_min: float, esm_max: float, resolution: int,
                   bif_time: int, burn_frac: float) -> tuple:
    esm_sweep = np.linspace(esm_min, esm_max, resolution)
    burn = int(bif_time * burn_frac)
    be_x, be_S, be_E = [], [], []
    for esm in esm_sweep:
        p = DEFAULT_PARAMS.copy()
        p['e_sm'] = float(esm)
        state = {k: np.float128(v) for k, v in FULL_INIT.items()}
        sys = DynamicalSystem(p, state, "dimensionalized")
        ts = sys.time_series_plot(time=bif_time)
        s_att = ts['Seafood'][burn:].astype(np.float64)
        e_att = ts['Effort'][burn:].astype(np.float64)
        n = len(s_att)
        be_x.extend([float(esm)] * n)
        be_S.extend(s_att.tolist())
        be_E.extend(e_att.tolist())
    return np.array(be_x), np.array(be_S), np.array(be_E)


@st.cache_data(show_spinner=False)
def s7_spectral_sweep(esm_min: float, esm_max: float, resolution: int) -> tuple:
    esm_vals = np.linspace(esm_min, esm_max, resolution)
    rho_vals = np.empty(resolution)
    for i, esm in enumerate(esm_vals):
        p = DEFAULT_PARAMS.copy()
        p['e_sm'] = float(esm)
        state = {k: np.float128(v) for k, v in FULL_INIT.items()}
        sys = DynamicalSystem(p, state, "dimensionalized")
        result = sys.stability_analysis()
        rho_vals[i] = result['spectral_radius']
    return esm_vals.astype(np.float64), rho_vals.astype(np.float64)


@st.fragment
def scenario_7():
    st.header("Scenario 7 — Market Price Elasticity of Supply")
    st.caption(
        "High ε_sm → market price drops sharply as harvest rises → "
        "dampens fraudster incentive. "
        "Low ε_sm → market price insensitive to supply → fraud profit persists. "
        "Focus parameter: ε_sm (market supply elasticity)."
    )

    with st.expander("Parameters", expanded=False):
        _c1, _c2, _c3 = st.columns(3)
        with _c1:
            s7_sim = st.slider("Simulation length", 100, 1000, 400, 50, key="s7_sim")
        with _c2:
            s7_res = st.slider("Bifurcation resolution", 50, 500, 200, 50, key="s7_res")
        with _c3:
            s7_rng = st.slider("ε_sm sweep range", 0.0, 5.0, (0.0, 3.0), 0.05, key="s7_rng")
        s7_esm_vals = st.multiselect(
            "ε_sm values for time series",
            [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0],
            default=[0.0, 0.5, 1.0, 2.0],
            key="s7_ev",
        )

    if not s7_esm_vals:
        st.warning("Select at least one *ε_sm* value.")
        return

    s7_esm_vals = sorted(s7_esm_vals)

    with st.status("Computing Scenario 7…", expanded=True) as status:
        st.write("Running time-series simulations…")
        ts7 = {esm: s7_time_series(float(esm), s7_sim) for esm in s7_esm_vals}
        t7 = np.arange(s7_sim + 1)
        # st.write("Computing bifurcation diagram…")
        # be_x, be_S, be_E = s7_bifurcation(
        #     float(s7_rng[0]), float(s7_rng[1]), s7_res, 300, 0.6,
        # )
        # st.write("Computing stability sweep…")
        # s7_esm_sweep, s7_rho = s7_spectral_sweep(
        #     float(s7_rng[0]), float(s7_rng[1]), 100,
        # )
        status.update(label="Scenario 7 ready", state="complete", expanded=False)

    s7_tab_ts, s7_tab_bif, s7_tab_stab = st.tabs(
        ["Time Series", "Bifurcation", "Stability"]
    )

    with s7_tab_ts:
        fig = plot_4var_ts(
            ts7, t7, s7_esm_vals, 'ε_sm',
            f'Market Supply Elasticity — Time Series as ε_sm Varies   '
            f'(r={DEFAULT_PARAMS["r"]}  |  Low ε_sm = market price insensitive to supply)',
        )
        st.plotly_chart(fig, width='stretch')

    # with s7_tab_bif:
    #     _def_esm = DEFAULT_PARAMS['e_sm']
    #     fig = plot_bifurcation(
    #         be_x, be_S, be_E,
    #         xlabel='Market Supply Elasticity (ε_sm)',
    #         title='Bifurcation over ε_sm   '
    #               '(Low → price insensitive  |  High → price responsive to supply)',
    #         vline_x=_def_esm, vline_label=f'Default ε_sm = {_def_esm}',
    #     )
    #     st.plotly_chart(fig, width='stretch')

    # with s7_tab_stab:
    #     finite = np.isfinite(s7_rho)
    #     esm_fin, rho_fin = s7_esm_sweep[finite], s7_rho[finite]
    #     stable_mask = rho_fin < 1.0
    #     y_cap = max(float(np.max(rho_fin[rho_fin < 50])) * 1.1, 2.0) if np.any(rho_fin < 50) else 5.0
    #     rho_plot = np.clip(rho_fin, 0, y_cap)
    #     fig = go.Figure()
    #     fig.add_trace(go.Scatter(
    #         x=esm_fin[stable_mask], y=rho_plot[stable_mask],
    #         mode='markers', marker=dict(color='#2E8B57', size=6),
    #         name='Stable (ρ < 1)',
    #     ))
    #     fig.add_trace(go.Scatter(
    #         x=esm_fin[~stable_mask], y=rho_plot[~stable_mask],
    #         mode='markers', marker=dict(color='#DC143C', size=6),
    #         name='Unstable (ρ ≥ 1)',
    #     ))
    #     fig.add_hline(y=1.0, line_dash='dash', line_color='gray',
    #                   annotation_text='ρ = 1 (stability boundary)')
    #     fig.update_layout(
    #         height=600,
    #         title_text=(
    #             f'Spectral Radius vs ε_sm — Fixed-Point Stability   '
    #             f'(r={DEFAULT_PARAMS["r"]})'
    #         ),
    #         xaxis_title='Market Supply Elasticity (ε_sm)',
    #         yaxis_title='Spectral Radius  ρ = max|λᵢ|',
    #         yaxis_range=[0, y_cap],
    #         margin=dict(t=60, b=40),
    #         legend=dict(yanchor='top', y=0.99, xanchor='right', x=0.99),
    #     )
    #     st.plotly_chart(fig, width='stretch')
