import streamlit as st
import numpy as np
import plotly.graph_objects as go
from System import DynamicalSystem, DEFAULT_PARAMS

from .constants import FULL_INIT
from .plots import plot_4var_ts, plot_bifurcation


@st.cache_data(show_spinner=False)
def s5_time_series(ed_val: float, sim_time: int) -> dict:
    p = DEFAULT_PARAMS.copy()
    p['e_d'] = ed_val
    state = {k: np.float128(v) for k, v in FULL_INIT.items()}
    sys = DynamicalSystem(p, state, "dimensionalized")
    ts = sys.time_series_plot(time=sim_time)
    return {k: v.astype(np.float64) for k, v in ts.items()}


@st.cache_data(show_spinner=False)
def s5_bifurcation(ed_min: float, ed_max: float, resolution: int,
                   bif_time: int, burn_frac: float) -> tuple:
    ed_sweep = np.linspace(ed_min, ed_max, resolution)
    burn = int(bif_time * burn_frac)
    be_e, be_S, be_E = [], [], []
    for ed in ed_sweep:
        p = DEFAULT_PARAMS.copy()
        p['e_d'] = float(ed)
        state = {k: np.float128(v) for k, v in FULL_INIT.items()}
        sys = DynamicalSystem(p, state, "dimensionalized")
        ts = sys.time_series_plot(time=bif_time)
        s_att = ts['Seafood'][burn:].astype(np.float64)
        e_att = ts['Effort'][burn:].astype(np.float64)
        n = len(s_att)
        be_e.extend([float(ed)] * n)
        be_S.extend(s_att.tolist())
        be_E.extend(e_att.tolist())
    return np.array(be_e), np.array(be_S), np.array(be_E)


@st.cache_data(show_spinner=False)
def s5_spectral_sweep(ed_min: float, ed_max: float, resolution: int) -> tuple:
    ed_vals = np.linspace(ed_min, ed_max, resolution)
    rho_vals = np.empty(resolution)
    for i, ed in enumerate(ed_vals):
        p = DEFAULT_PARAMS.copy()
        p['e_d'] = float(ed)
        state = {k: np.float128(v) for k, v in FULL_INIT.items()}
        sys = DynamicalSystem(p, state, "dimensionalized")
        result = sys.stability_analysis()
        rho_vals[i] = result['spectral_radius']
    return ed_vals.astype(np.float64), rho_vals.astype(np.float64)


@st.fragment
def scenario_5():
    st.header("Scenario 5 — Buyer Dependence on Seafood")
    st.caption(
        "High ε_d → buyers sensitive to fraud perception → self-correcting. "
        "Low ε_d → buyers NEED seafood regardless → fraud persists unchecked. "
        "Focus parameter: ε_d (decreasing = more dependent)."
    )

    with st.expander("Parameters", expanded=False):
        _c1, _c2, _c3 = st.columns(3)
        with _c1:
            s5_sim = st.slider("Simulation length", 100, 1000, 400, 50, key="s5_sim")
        with _c2:
            s5_res = st.slider("Bifurcation resolution", 50, 500, 200, 50, key="s5_res")
        with _c3:
            s5_rng = st.slider("ε_d sweep range", 0.01, 5.0, (0.01, 3.0), 0.05, key="s5_rng")
        s5_ed_vals = st.multiselect(
            "ε_d values for time series",
            [0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0],
            default=[2.0, 1.0, 0.5, 0.1],
            key="s5_ev",
        )

    if not s5_ed_vals:
        st.warning("Select at least one *ε_d* value.")
        return

    s5_ed_vals = sorted(s5_ed_vals)
    _burn5 = int(s5_sim * 0.6)

    with st.status("Computing Scenario 5…", expanded=True) as status:
        st.write("Running time-series simulations…")
        ts5 = {ed: s5_time_series(float(ed), s5_sim) for ed in s5_ed_vals}
        t5 = np.arange(s5_sim + 1)
        st.write("Computing bifurcation diagram…")
        be_e, be_S, be_E = s5_bifurcation(
            float(s5_rng[0]), float(s5_rng[1]), s5_res, 300, 0.6,
        )
        st.write("Computing stability sweep…")
        s5_ed_sweep, s5_rho = s5_spectral_sweep(0.05, 5.0, 100)
        status.update(label="Scenario 5 ready", state="complete", expanded=False)

    s5_tab_ts, s5_tab_bif, s5_tab_stab = st.tabs(
        ["Time Series", "Bifurcation", "Stability"]
    )

    with s5_tab_ts:
        fig = plot_4var_ts(
            ts5, t5, s5_ed_vals, 'ε_d',
            f'Buyer Dependence — Time Series as ε_d Decreases   '
            f'(r={DEFAULT_PARAMS["r"]}  |  Low ε_d = buyers cannot walk away)',
        )
        st.plotly_chart(fig, width='stretch')

    with s5_tab_bif:
        _def_ed = DEFAULT_PARAMS['e_d']
        fig = plot_bifurcation(
            be_e, be_S, be_E,
            xlabel='Demand Elasticity (ε_d)',
            title='Bifurcation over ε_d   '
                  '(Low → buyers dependent  |  High → buyers sensitive to fraud)',
            vline_x=_def_ed, vline_label=f'Default ε_d = {_def_ed}',
        )
        st.plotly_chart(fig, width='stretch')

    with s5_tab_stab:
        finite = np.isfinite(s5_rho)
        ed_fin, rho_fin = s5_ed_sweep[finite], s5_rho[finite]
        stable_mask = rho_fin < 1.0
        y_cap = max(float(np.max(rho_fin[rho_fin < 50])) * 1.1, 2.0) if np.any(rho_fin < 50) else 5.0
        rho_plot = np.clip(rho_fin, 0, y_cap)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ed_fin[stable_mask], y=rho_plot[stable_mask],
            mode='markers', marker=dict(color='#2E8B57', size=6),
            name='Stable (ρ < 1)',
        ))
        fig.add_trace(go.Scatter(
            x=ed_fin[~stable_mask], y=rho_plot[~stable_mask],
            mode='markers', marker=dict(color='#DC143C', size=6),
            name='Unstable (ρ ≥ 1)',
        ))
        fig.add_hline(y=1.0, line_dash='dash', line_color='gray',
                      annotation_text='ρ = 1 (stability boundary)')
        fig.update_layout(
            height=600,
            title_text=(
                f'Spectral Radius vs ε_d — Fixed-Point Stability   '
                f'(r={DEFAULT_PARAMS["r"]})'
            ),
            xaxis_title='Demand Elasticity (ε_d)',
            yaxis_title='Spectral Radius  ρ = max|λᵢ|',
            yaxis_range=[0, y_cap],
            margin=dict(t=60, b=40),
            legend=dict(yanchor='top', y=0.99, xanchor='right', x=0.99),
        )
        st.plotly_chart(fig, width='stretch')
