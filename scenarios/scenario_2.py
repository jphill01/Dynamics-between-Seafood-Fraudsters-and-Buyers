import streamlit as st
import numpy as np
import plotly.graph_objects as go
from System import DynamicalSystem, DEFAULT_PARAMS

from .constants import _PW0, _C0, _Q0, FULL_INIT
from .plots import plot_4var_ts, plot_ts_with_economics, plot_bifurcation, plot_return_maps
from ._status import scenario_header, status_indicator


@st.cache_data(show_spinner=False)
def s2_time_series(pw1_val: float, sim_time: int) -> dict:
    p = DEFAULT_PARAMS.copy()
    p.update({'pw1': pw1_val, 'c1': _C0, 'q1': _Q0})
    state = {k: np.float128(v) for k, v in FULL_INIT.items()}
    sys = DynamicalSystem(p, state, "dimensionalized")
    ts = sys.time_series_plot(time=sim_time)
    return {k: v.astype(np.float64) for k, v in ts.items()}


@st.cache_data(show_spinner=False)
def s2_bifurcation(pw_min: float, pw_max: float, resolution: int,
                   bif_time: int, burn_frac: float) -> tuple:
    pw_sweep = np.linspace(pw_min, pw_max, resolution)
    burn = int(bif_time * burn_frac)
    bp_p, bp_S, bp_E = [], [], []
    for pw in pw_sweep:
        p = DEFAULT_PARAMS.copy()
        p.update({'pw1': float(pw), 'c1': _C0, 'q1': _Q0})
        state = {k: np.float128(v) for k, v in FULL_INIT.items()}
        sys = DynamicalSystem(p, state, "dimensionalized")
        ts = sys.time_series_plot(time=bif_time)
        s_att = ts['Seafood'][burn:].astype(np.float64)
        e_att = ts['Effort'][burn:].astype(np.float64)
        n = len(s_att)
        bp_p.extend([float(pw)] * n)
        bp_S.extend(s_att.tolist())
        bp_E.extend(e_att.tolist())
    return np.array(bp_p), np.array(bp_S), np.array(bp_E)


@st.cache_data(show_spinner=False)
def s2_spectral_sweep(pw_min: float, pw_max: float, resolution: int) -> tuple:
    pw_vals = np.linspace(pw_min, pw_max, resolution)
    rho_vals = np.empty(resolution)
    for i, pw in enumerate(pw_vals):
        p = DEFAULT_PARAMS.copy()
        p.update({'pw1': float(pw), 'c1': _C0, 'q1': _Q0})
        state = {k: np.float128(v) for k, v in FULL_INIT.items()}
        sys = DynamicalSystem(p, state, "dimensionalized")
        result = sys.stability_analysis()
        rho_vals[i] = result['spectral_radius']
    return pw_vals.astype(np.float64), rho_vals.astype(np.float64)


@st.fragment
def scenario_2():
    status_slot = scenario_header("Scenario 2 — Prized / Protected Seafood")
    st.caption(
        f"Fraudsters pay a premium for protected species (pw₁ > pw₀). "
        f"Same gear, same waters: c₁ = c₀ = {_C0}, q₁ = q₀ = {_Q0}. "
        f"Focus parameter: pw₁."
    )

    with st.expander("Parameters", expanded=False):
        _c1, _c2, _c3 = st.columns(3)
        with _c1:
            s2_sim = st.slider("Simulation length", 100, 1000, 400, 50, key="s2_sim")
        with _c2:
            s2_res = st.slider("Bifurcation resolution", 50, 500, 200, 50, key="s2_res")
        with _c3:
            s2_rng = st.slider(
                "pw₁ sweep range", float(_PW0), 8.0, (float(_PW0), 5.0), 0.1,
                key="s2_rng",
            )
        s2_pw_vals = st.multiselect(
            "pw₁ values for time series & poincare",
            [0.25, 0.5, 0.75, 1.0, 1.10, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 3.00, 4.00, 5.00],
            default=[0.5, 0.75, 1.0, 1.50, 2.00],
            key="s2_pw",
        )

    if not s2_pw_vals:
        st.warning("Select at least one *pw₁* value.")
        return

    s2_pw_vals = sorted(s2_pw_vals)
    _burn2 = int(s2_sim * 0.6)

    with status_indicator(status_slot, [
        "Running time-series simulations",
        "Computing bifurcation diagram",
        "Computing stability sweep",
    ]):
        ts2 = {pw: s2_time_series(float(pw), s2_sim) for pw in s2_pw_vals}
        t2 = np.arange(s2_sim + 1)
        bp_p, bp_S, bp_E = s2_bifurcation(
            float(s2_rng[0]), float(s2_rng[1]), s2_res, 300, 0.6,
        )
        s2_pw_sweep, s2_rho = s2_spectral_sweep(0.05, 5.0, 100)

    tab_ts, tab_bif, tab_rm, tab_stab = st.tabs(
        ["Time Series", "Bifurcation", "Poincare", "Stability"]
    )

    with tab_ts:
        fig = plot_ts_with_economics(
            ts2, t2, s2_pw_vals, 'pw₁',
            f'Prized Seafood — Time Series as pw₁ Increases   '
            f'(c₁=c₀={_C0},  q₁=q₀={_Q0},  F_threshold={DEFAULT_PARAMS["F_threshold"]})',
        )
        st.plotly_chart(fig, width='stretch')

    with tab_bif:
        fig = plot_bifurcation(
            bp_p, bp_S, bp_E,
            xlabel='pw₁ (black-market wholesale price)',
            title=f'Bifurcation Diagram over pw₁   (F_threshold={DEFAULT_PARAMS["F_threshold"]})',
            vline_x=float(_PW0), vline_label=f'pw₀ = {_PW0}',
        )
        st.plotly_chart(fig, width='stretch')

    with tab_rm:
        fig = plot_return_maps(ts2, s2_pw_vals, 'pw₁', _burn2)
        st.plotly_chart(fig, width='stretch')

    with tab_stab:
        finite = np.isfinite(s2_rho)
        pw_fin, rho_fin = s2_pw_sweep[finite], s2_rho[finite]
        stable_mask = rho_fin < 1.0
        y_cap = max(float(np.max(rho_fin[rho_fin < 50])) * 1.1, 2.0) if np.any(rho_fin < 50) else 5.0
        rho_plot = np.clip(rho_fin, 0, y_cap)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pw_fin[stable_mask], y=rho_plot[stable_mask],
            mode='markers', marker=dict(color='#2E8B57', size=6),
            name='Stable (ρ < 1)',
        ))
        fig.add_trace(go.Scatter(
            x=pw_fin[~stable_mask], y=rho_plot[~stable_mask],
            mode='markers', marker=dict(color='#DC143C', size=6),
            name='Unstable (ρ ≥ 1)',
        ))
        fig.add_hline(y=1.0, line_dash='dash', line_color='gray',
                      annotation_text='ρ = 1 (stability boundary)')
        fig.update_layout(
            height=600,
            title_text=(
                f'Spectral Radius vs pw₁ — Fixed-Point Stability   '
                f'(c₁=c₀={_C0},  q₁=q₀={_Q0},  F_threshold={DEFAULT_PARAMS["F_threshold"]})'
            ),
            xaxis_title='pw₁ (black-market wholesale price)',
            yaxis_title='Spectral Radius  ρ = max|λᵢ|',
            yaxis_range=[0, y_cap],
            margin=dict(t=60, b=40),
            legend=dict(yanchor='top', y=0.99, xanchor='right', x=0.99),
        )
        st.plotly_chart(fig, width='stretch')
