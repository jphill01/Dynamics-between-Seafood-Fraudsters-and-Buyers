import streamlit as st
import numpy as np
from System import DynamicalSystem, DEFAULT_PARAMS
from text import INTRO
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG (must be first st command)
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    layout="wide",
    page_title="Dynamics of Seafood, Fraudsters, and Buyers",
    page_icon=":fish:",
)

# ════════════════════════════════════════════════════════════════════════════
# SHARED CONSTANTS
# ════════════════════════════════════════════════════════════════════════════
_PW0 = DEFAULT_PARAMS['pw0']
_C0 = DEFAULT_PARAMS['c0']
_Q0 = DEFAULT_PARAMS['q0']

COLORS4 = {'S': '#4682B4', 'E': '#2E8B57', 'F': '#DC143C', 'FP': '#DA70D6'}
S1_COLORS = {'S': '#4682B4', 'E': '#2E8B57'}
S1_NO_FRAUD = {'S': 0.6, 'E': 0.3, 'F': 0.0, 'FP': 0.0}
FULL_INIT = {'S': 0.6, 'E': 0.3, 'F': 0.1, 'FP': 0.1}


# ════════════════════════════════════════════════════════════════════════════
# CACHED COMPUTATION FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

# ── Scenario 1 ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def s1_time_series(r_val: float, sim_time: int) -> dict:
    p = DEFAULT_PARAMS.copy()
    p['r'] = r_val
    state = {k: np.float128(v) for k, v in S1_NO_FRAUD.items()}
    sys = DynamicalSystem(p, state, "dimensionalized")
    ts = sys.time_series_plot(time=sim_time)
    return {k: v.astype(np.float64) for k, v in ts.items()}


@st.cache_data(show_spinner=False)
def s1_bifurcation(r_min: float, r_max: float, resolution: int,
                   bif_time: int, burn_frac: float) -> tuple:
    r_sweep = np.linspace(r_min, r_max, resolution)
    burn = int(bif_time * burn_frac)
    br_r, br_S, br_E = [], [], []
    for rv in r_sweep:
        p = DEFAULT_PARAMS.copy()
        p['r'] = float(rv)
        state = {k: np.float128(v) for k, v in S1_NO_FRAUD.items()}
        sys = DynamicalSystem(p, state, "dimensionalized")
        ts = sys.time_series_plot(time=bif_time)
        s_att = ts['Seafood'][burn:].astype(np.float64)
        e_att = ts['Effort'][burn:].astype(np.float64)
        n = len(s_att)
        br_r.extend([float(rv)] * n)
        br_S.extend(s_att.tolist())
        br_E.extend(e_att.tolist())
    return np.array(br_r), np.array(br_S), np.array(br_E)


# ── Scenario 2 ─────────────────────────────────────────────────────────────

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


# ── Scenario 3 ─────────────────────────────────────────────────────────────

def _blast_params(alpha: float) -> dict:
    return {
        'q1': float(_Q0 + alpha * 0.33),
        'pw1': float(_PW0 - alpha * 0.40),
        'c1': float(_C0 - alpha * 0.80),
    }


@st.cache_data(show_spinner=False)
def s3_time_series(alpha_val: float, sim_time: int) -> dict:
    p = DEFAULT_PARAMS.copy()
    p.update(_blast_params(alpha_val))
    state = {k: np.float128(v) for k, v in FULL_INIT.items()}
    sys = DynamicalSystem(p, state, "dimensionalized")
    ts = sys.time_series_plot(time=sim_time)
    return {k: v.astype(np.float64) for k, v in ts.items()}


@st.cache_data(show_spinner=False)
def s3_bifurcation(a_min: float, a_max: float, resolution: int,
                   bif_time: int, burn_frac: float) -> tuple:
    a_sweep = np.linspace(a_min, a_max, resolution)
    burn = int(bif_time * burn_frac)
    ba_a, ba_S, ba_E = [], [], []
    for av in a_sweep:
        p = DEFAULT_PARAMS.copy()
        p.update(_blast_params(float(av)))
        state = {k: np.float128(v) for k, v in FULL_INIT.items()}
        sys = DynamicalSystem(p, state, "dimensionalized")
        ts = sys.time_series_plot(time=bif_time)
        s_att = ts['Seafood'][burn:].astype(np.float64)
        e_att = ts['Effort'][burn:].astype(np.float64)
        n = len(s_att)
        ba_a.extend([float(av)] * n)
        ba_S.extend(s_att.tolist())
        ba_E.extend(e_att.tolist())
    return np.array(ba_a), np.array(ba_S), np.array(ba_E)


# ── Scenario 4 ─────────────────────────────────────────────────────────────

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


# ── Scenario 5 ─────────────────────────────────────────────────────────────

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
def s5_heatmap(ed_min: float, ed_max: float, ft_min: float, ft_max: float,
               resolution: int, bif_time: int, burn_frac: float) -> tuple:
    ed_hm = np.linspace(ed_min, ed_max, resolution)
    ft_hm = np.linspace(ft_min, ft_max, resolution)
    burn = int(bif_time * burn_frac)
    hm_S = np.full((resolution, resolution), np.nan)
    for i, ft in enumerate(ft_hm):
        for j, ed in enumerate(ed_hm):
            p = DEFAULT_PARAMS.copy()
            p.update({'e_d': float(ed), 'F_threshold': float(ft)})
            state = {k: np.float128(v) for k, v in FULL_INIT.items()}
            sys = DynamicalSystem(p, state, "dimensionalized")
            ts = sys.time_series_plot(time=bif_time)
            hm_S[i, j] = float(np.mean(ts['Seafood'][burn:]))
    return ed_hm.astype(np.float64), ft_hm.astype(np.float64), hm_S.astype(np.float64)


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


# ════════════════════════════════════════════════════════════════════════════
# SCENARIO FRAGMENTS
# ════════════════════════════════════════════════════════════════════════════

@st.fragment
def scenario_1():
    st.header("Scenario 1 — Baseline Bioeconomic Model (No Fraud)")
    st.caption(
        "F = 0, FP = 0 throughout. The system reduces to Seafood (S) vs "
        "Effort (E) only. Focus parameter: intrinsic growth rate *r*."
    )

    with st.expander("Parameters", expanded=False):
        _c1, _c2, _c3 = st.columns(3)
        with _c1:
            s1_sim = st.slider("Simulation length", 100, 1000, 300, 50, key="s1_sim")
        with _c2:
            s1_res = st.slider("Bifurcation resolution", 50, 500, 250, 50, key="s1_res")
        with _c3:
            s1_rng = st.slider("r sweep range", 0.1, 6.0, (0.1, 4.0), 0.1, key="s1_rng")
        s1_r_vals = st.multiselect(
            "r values for time series & poincare",
            [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.75, 4.0, 5.0],
            default=[0.5, 1.5, 2.5, 3.75],
            key="s1_rv",
        )

    if not s1_r_vals:
        st.warning("Select at least one *r* value.")
        return

    s1_r_vals = sorted(s1_r_vals)
    _N = len(s1_r_vals)
    _burn = int(s1_sim * 0.6)

    with st.status("Computing Scenario 1…", expanded=True) as status:
        st.write("Running time-series simulations…")
        ts1 = {rv: s1_time_series(float(rv), s1_sim) for rv in s1_r_vals}
        t1 = np.arange(s1_sim + 1)
        st.write("Computing bifurcation diagram…")
        br_r, br_S, br_E = s1_bifurcation(
            float(s1_rng[0]), float(s1_rng[1]), s1_res, 300, 0.6,
        )
        status.update(label="Scenario 1 ready", state="complete", expanded=False)

    tab_ts, tab_bif, tab_rm = st.tabs(
        ["Time Series", "Bifurcation", "Poincare"]
    )

    with tab_ts:
        fig = make_subplots(
            rows=2, cols=_N,
            subplot_titles=[f'r = {rv}' for rv in s1_r_vals] + [''] * _N,
            shared_xaxes=True, vertical_spacing=0.10, horizontal_spacing=0.05,
        )
        for col, rv in enumerate(s1_r_vals, 1):
            d = ts1[rv]
            fig.add_trace(go.Scatter(
                x=t1, y=d['Seafood'], mode='lines',
                line=dict(color=S1_COLORS['S'], width=1.5),
                name='Seafood (S)', legendgroup='S', showlegend=(col == 1),
            ), row=1, col=col)
            fig.add_trace(go.Scatter(
                x=t1, y=d['Effort'], mode='lines',
                line=dict(color=S1_COLORS['E'], width=1.5),
                name='Effort (E)', legendgroup='E', showlegend=(col == 1),
            ), row=2, col=col)
        fig.update_yaxes(title_text='Seafood (S)', row=1, col=1)
        fig.update_yaxes(title_text='Effort (E)', row=2, col=1)
        fig.update_yaxes(rangemode='tozero')
        fig.update_xaxes(title_text='Time', row=2)
        fig.update_layout(
            height=600,
            title_y = 1.0,
            title_text='Baseline (No Fraud) — Time Series as r Increases',
            legend=dict(orientation='h', yanchor='bottom', y=1.06),
            margin=dict(t=80, b=40),
        )
        st.plotly_chart(fig, width='stretch')

    with tab_bif:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Seafood S*', 'Effort E*'],
            horizontal_spacing=0.08,
        )
        fig.add_trace(go.Scattergl(
            x=br_r, y=br_S, mode='markers',
            marker=dict(color=S1_COLORS['S'], size=2, opacity=0.4),
            showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scattergl(
            x=br_r, y=br_E, mode='markers',
            marker=dict(color=S1_COLORS['E'], size=2, opacity=0.4),
            showlegend=False,
        ), row=1, col=2)
        _def_r = DEFAULT_PARAMS['r']
        fig.add_vline(
            x=_def_r, line_dash='dash', line_color='gray',
            annotation_text=f'Default r = {_def_r}',
            annotation_position='top right', row=1, col=1,
        )
        fig.add_vline(x=_def_r, line_dash='dash', line_color='gray', row=1, col=2)
        fig.update_xaxes(title_text='Intrinsic Growth Rate (r)')
        fig.update_layout(
            height=600,
            title_text='Bifurcation Diagram over r (No Fraud)',
            margin=dict(t=60, b=40),
        )
        st.plotly_chart(fig, width='stretch')

    with tab_rm:
        fig = make_subplots(
            rows=2, cols=_N,
            subplot_titles=[f'r = {rv}' for rv in s1_r_vals] + [''] * _N,
            vertical_spacing=0.12, horizontal_spacing=0.05,
        )
        for col, rv in enumerate(s1_r_vals, 1):
            d = ts1[rv]
            for row, (var, clr) in enumerate([
                ('Seafood', S1_COLORS['S']), ('Effort', S1_COLORS['E']),
            ], 1):
                x = d[var]
                x_t, x_tp1 = x[_burn:-1], x[_burn + 1:]
                fig.add_trace(go.Scattergl(
                    x=x_t, y=x_tp1, mode='markers',
                    marker=dict(color=clr, size=2, opacity=0.6),
                    showlegend=False,
                ), row=row, col=col)
                lo = float(min(x_t.min(), x_tp1.min())) * 0.9
                hi = float(max(x_t.max(), x_tp1.max())) * 1.1
                fig.add_trace(go.Scatter(
                    x=[lo, hi], y=[lo, hi], mode='lines',
                    line=dict(color='black', width=0.8, dash='dash'),
                    showlegend=False,
                ), row=row, col=col)
        fig.update_yaxes(title_text='S(t+1)', row=1, col=1)
        fig.update_yaxes(title_text='E(t+1)', row=2, col=1)
        fig.update_layout(
            height=600,
            title_text='Poincare — x(t) vs x(t+1) (attractor only)',
            margin=dict(t=60, b=40),
        )
        st.plotly_chart(fig, width='stretch')


# ── Helper: 4-variable time-series chart (used by Scenarios 2-5) ──────────

def _plot_4var_ts(ts_dict, t_arr, param_vals, param_label, title, colors=COLORS4):
    _N = len(param_vals)
    fig = make_subplots(
        rows=2, cols=_N,
        subplot_titles=[f'{param_label} = {v}' for v in param_vals] + [''] * _N,
        shared_xaxes=True, vertical_spacing=0.10, horizontal_spacing=0.05,
    )
    for col, v in enumerate(param_vals, 1):
        d = ts_dict[v]
        show = col == 1
        fig.add_trace(go.Scatter(
            x=t_arr, y=d['Seafood'], mode='lines',
            line=dict(color=colors['S'], width=1.5),
            name='Seafood (S)', legendgroup='S', showlegend=show,
        ), row=1, col=col)
        fig.add_trace(go.Scatter(
            x=t_arr, y=d['Effort'], mode='lines',
            line=dict(color=colors['E'], width=1.5),
            name='Effort (E)', legendgroup='E', showlegend=show,
        ), row=1, col=col)
        fig.add_trace(go.Scatter(
            x=t_arr, y=d['Fraudsters'], mode='lines',
            line=dict(color=colors['F'], width=1.5),
            name='Fraudsters (F)', legendgroup='F', showlegend=show,
        ), row=2, col=col)
        fig.add_trace(go.Scatter(
            x=t_arr, y=d['Perception of Fraud'], mode='lines',
            line=dict(color=colors['FP'], width=1.5),
            name='Perception (FP)', legendgroup='FP', showlegend=show,
        ), row=2, col=col)
    fig.update_yaxes(title_text='S / E', row=1, col=1)
    fig.update_yaxes(title_text='F / FP', row=2, col=1)
    fig.update_yaxes(rangemode='tozero')
    fig.update_xaxes(title_text='Time', row=2)
    fig.update_layout(
        height=600, title_text=title,
        title_y = 1.0,
        legend=dict(orientation='h', yanchor='bottom', y=1.06),
        margin=dict(t=100, b=40),
    )
    return fig


def _plot_bifurcation(x_data, s_data, e_data, xlabel, title,
                      vline_x=None, vline_label=None, colors=COLORS4):
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Seafood S*', 'Effort E*'],
        horizontal_spacing=0.08,
    )
    fig.add_trace(go.Scattergl(
        x=x_data, y=s_data, mode='markers',
        marker=dict(color=colors['S'], size=2, opacity=0.4), showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scattergl(
        x=x_data, y=e_data, mode='markers',
        marker=dict(color=colors['E'], size=2, opacity=0.4), showlegend=False,
    ), row=1, col=2)
    if vline_x is not None:
        fig.add_vline(
            x=vline_x, line_dash='dash', line_color='gray',
            annotation_text=vline_label or '',
            annotation_position='top right', row=1, col=1,
        )
        fig.add_vline(x=vline_x, line_dash='dash', line_color='gray', row=1, col=2)
    fig.update_xaxes(title_text=xlabel)
    fig.update_layout(height=600, title_text=title, margin=dict(t=60, b=40))
    return fig


def _plot_return_maps(ts_dict, param_vals, param_label, burn, colors=COLORS4):
    _N = len(param_vals)
    fig = make_subplots(
        rows=2, cols=_N,
        subplot_titles=[f'{param_label} = {v}' for v in param_vals] + [''] * _N,
        vertical_spacing=0.12, horizontal_spacing=0.05,
    )
    for col, v in enumerate(param_vals, 1):
        d = ts_dict[v]
        for row, (var, clr) in enumerate([
            ('Seafood', colors['S']), ('Effort', colors['E']),
        ], 1):
            x = d[var]
            x_t, x_tp1 = x[burn:-1], x[burn + 1:]
            fig.add_trace(go.Scattergl(
                x=x_t, y=x_tp1, mode='markers',
                marker=dict(color=clr, size=2, opacity=0.6), showlegend=False,
            ), row=row, col=col)
            lo = float(min(x_t.min(), x_tp1.min())) * 0.9
            hi = float(max(x_t.max(), x_tp1.max())) * 1.1
            fig.add_trace(go.Scatter(
                x=[lo, hi], y=[lo, hi], mode='lines',
                line=dict(color='black', width=0.8, dash='dash'), showlegend=False,
            ), row=row, col=col)
    fig.update_yaxes(title_text='S(t+1)', row=1, col=1)
    fig.update_yaxes(title_text='E(t+1)', row=2, col=1)
    fig.update_layout(
        height=600,
        title_text='Poincare — x(t) vs x(t+1) (attractor only)',
        margin=dict(t=60, b=40),
    )
    return fig


# ════════════════════════════════════════════════════════════════════════════

@st.fragment
def scenario_2():
    st.header("Scenario 2 — Prized / Protected Seafood")
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
            [1.10, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 3.00, 4.00, 5.00],
            default=[1.10, 1.50, 2.00, 3.00],
            key="s2_pw",
        )

    if not s2_pw_vals:
        st.warning("Select at least one *pw₁* value.")
        return

    s2_pw_vals = sorted(s2_pw_vals)
    _burn2 = int(s2_sim * 0.6)

    with st.status("Computing Scenario 2…", expanded=True) as status:
        st.write("Running time-series simulations…")
        ts2 = {pw: s2_time_series(float(pw), s2_sim) for pw in s2_pw_vals}
        t2 = np.arange(s2_sim + 1)
        st.write("Computing bifurcation diagram…")
        bp_p, bp_S, bp_E = s2_bifurcation(
            float(s2_rng[0]), float(s2_rng[1]), s2_res, 300, 0.6,
        )
        st.write("Computing stability sweep…")
        s2_pw_sweep, s2_rho = s2_spectral_sweep(0.05, 5.0, 100)
        status.update(label="Scenario 2 ready", state="complete", expanded=False)

    tab_ts, tab_bif, tab_rm, tab_stab = st.tabs(
        ["Time Series", "Bifurcation", "Poincare", "Stability"]
    )

    with tab_ts:
        fig = _plot_4var_ts(
            ts2, t2, s2_pw_vals, 'pw₁',
            f'Prized Seafood — Time Series as pw₁ Increases   '
            f'(c₁=c₀={_C0},  q₁=q₀={_Q0},  r={DEFAULT_PARAMS["r"]})',
        )
        st.plotly_chart(fig, width='stretch')

    with tab_bif:
        fig = _plot_bifurcation(
            bp_p, bp_S, bp_E,
            xlabel='pw₁ (black-market wholesale price)',
            title=f'Bifurcation Diagram over pw₁   (r={DEFAULT_PARAMS["r"]})',
            vline_x=float(_PW0), vline_label=f'pw₀ = {_PW0}',
        )
        st.plotly_chart(fig, width='stretch')

    with tab_rm:
        fig = _plot_return_maps(ts2, s2_pw_vals, 'pw₁', _burn2)
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
                f'(c₁=c₀={_C0},  q₁=q₀={_Q0},  r={DEFAULT_PARAMS["r"]})'
            ),
            xaxis_title='pw₁ (black-market wholesale price)',
            yaxis_title='Spectral Radius  ρ = max|λᵢ|',
            yaxis_range=[0, y_cap],
            margin=dict(t=60, b=40),
            legend=dict(yanchor='top', y=0.99, xanchor='right', x=0.99),
        )
        st.plotly_chart(fig, width='stretch')


@st.fragment
def scenario_3():
    st.header("Scenario 3 — Blast / Cyanide Fishing")
    st.caption(
        "Destructive methods: q₁↑  pw₁↓  c₁↓↓ (cost drops much more than price). "
        "A single destruction intensity α ∈ [0, 1] jointly scales all three."
    )

    with st.expander("Parameters", expanded=False):
        _c1, _c2, _c3 = st.columns(3)
        with _c1:
            s3_sim = st.slider("Simulation length", 100, 1000, 400, 50, key="s3_sim")
        with _c2:
            s3_res = st.slider("Bifurcation resolution", 50, 500, 200, 50, key="s3_res")
        with _c3:
            s3_rng = st.slider("α sweep range", 0.0, 1.0, (0.0, 1.0), 0.05, key="s3_rng")
        s3_a_vals = st.multiselect(
            "α values for time series & poincare",
            [0.0, 0.10, 0.15, 0.25, 0.40, 0.55, 0.70, 0.85, 1.00],
            default=[0.15, 0.40, 0.70, 1.00],
            key="s3_av",
        )

    if not s3_a_vals:
        st.warning("Select at least one *α* value.")
        return

    s3_a_vals = sorted(s3_a_vals)
    _burn3 = int(s3_sim * 0.6)

    with st.status("Computing Scenario 3…", expanded=True) as status:
        st.write("Running time-series simulations…")
        ts3 = {a: s3_time_series(float(a), s3_sim) for a in s3_a_vals}
        t3 = np.arange(s3_sim + 1)
        st.write("Computing bifurcation diagram…")
        ba_a, ba_S, ba_E = s3_bifurcation(
            float(s3_rng[0]), float(s3_rng[1]), s3_res, 300, 0.6,
        )
        status.update(label="Scenario 3 ready", state="complete", expanded=False)

    bp_labels = [
        f'α={a}  (q₁={_blast_params(a)["q1"]:.2f}, '
        f'pw₁={_blast_params(a)["pw1"]:.2f}, '
        f'c₁={_blast_params(a)["c1"]:.2f})'
        for a in s3_a_vals
    ]

    tab_ts, tab_bif, tab_rm = st.tabs(
        ["Time Series", "Bifurcation", "Poincare"]
    )

    with tab_ts:
        fig = _plot_4var_ts(
            ts3, t3, s3_a_vals, 'α',
            f'Blast Fishing — Time Series by Destruction Intensity   '
            f'(r={DEFAULT_PARAMS["r"]}  |  q₁↑  pw₁↓  c₁↓↓)',
        )
        for i, lbl in enumerate(bp_labels):
            fig.layout.annotations[i].text = lbl
        st.plotly_chart(fig, width='stretch')

    with tab_bif:
        fig = _plot_bifurcation(
            ba_a, ba_S, ba_E,
            xlabel='Destruction Intensity (α)',
            title='Bifurcation over α   '
                  '(α=0 → honest  |  α=1 → q₁=0.40, pw₁=0.60, c₁=0.10)',
        )
        st.plotly_chart(fig, width='stretch')

    with tab_rm:
        fig = _plot_return_maps(ts3, s3_a_vals, 'α', _burn3)
        st.plotly_chart(fig, width='stretch')


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
        fig = _plot_4var_ts(
            ts4, t4, s4_b_vals, 'β',
            f'EEZ Non-Enforcement — Time Series by Violation Intensity   '
            f'(r={DEFAULT_PARAMS["r"]}  |  q₁↑  c₁↑  |  '
            f'pw₁={DEFAULT_PARAMS["pw1"]} default)',
        )
        for i, lbl in enumerate(ep_labels):
            fig.layout.annotations[i].text = lbl
        st.plotly_chart(fig, width='stretch')

    with tab_bif:
        fig = _plot_bifurcation(
            bb_b, bb_S, bb_E,
            xlabel='EEZ Violation Intensity (β)',
            title='Bifurcation over β   '
                  '(β=0 → honest  |  β=1 → q₁=0.30, c₁=2.00)',
        )
        st.plotly_chart(fig, width='stretch')

    with tab_rm:
        fig = _plot_return_maps(ts4, s4_b_vals, 'β', _burn4)
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
        s5_hm_res = st.slider("Heatmap resolution (N×N grid)", 10, 50, 30, 5, key="s5_hm")

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
        st.write("Computing defence heatmap…")
        ed_hm, ft_hm, hm_S = s5_heatmap(
            0.05, 3.0, 0.05, 0.95, s5_hm_res, 300, 0.6,
        )
        st.write("Computing stability sweep…")
        s5_ed_sweep, s5_rho = s5_spectral_sweep(0.05, 5.0, 100)
        status.update(label="Scenario 5 ready", state="complete", expanded=False)

    s5_tab_ts, s5_tab_bif, s5_tab_hm, s5_tab_stab = st.tabs(
        ["Time Series", "Bifurcation", "Defence Heatmap", "Stability"]
    )

    with s5_tab_ts:
        fig = _plot_4var_ts(
            ts5, t5, s5_ed_vals, 'ε_d',
            f'Buyer Dependence — Time Series as ε_d Decreases   '
            f'(r={DEFAULT_PARAMS["r"]}  |  Low ε_d = buyers cannot walk away)',
        )
        st.plotly_chart(fig, width='stretch')

    with s5_tab_bif:
        _def_ed = DEFAULT_PARAMS['e_d']
        fig = _plot_bifurcation(
            be_e, be_S, be_E,
            xlabel='Demand Elasticity (ε_d)',
            title='Bifurcation over ε_d   '
                  '(Low → buyers dependent  |  High → buyers sensitive to fraud)',
            vline_x=_def_ed, vline_label=f'Default ε_d = {_def_ed}',
        )
        st.plotly_chart(fig, width='stretch')

    with s5_tab_hm:
        fig = go.Figure(data=go.Heatmap(
            z=hm_S, x=ed_hm, y=ft_hm,
            colorscale='YlGnBu',
            colorbar=dict(title='Mean S*'),
            hovertemplate='ε_d=%{x:.2f}<br>F̂=%{y:.2f}<br>Mean S*=%{z:.4f}<extra></extra>',
        ))
        fig.add_trace(go.Scatter(
            x=[DEFAULT_PARAMS['e_d']],
            y=[DEFAULT_PARAMS['F_threshold']],
            mode='markers',
            marker=dict(color='red', size=14, symbol='x'),
            name='Default',
        ))
        fig.update_layout(
            height=600,
            title_text=(
                f'Buyer Defence Landscape — Mean S* across '
                f'(ε_d, F̂) space   (r={DEFAULT_PARAMS["r"]})'
            ),
            xaxis_title='Demand Elasticity (ε_d)',
            yaxis_title='Fraud Detection Threshold (F̂)',
            margin=dict(t=60, b=40),
            legend=dict(yanchor='top', y=0.99, xanchor='right', x=0.99),
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


# ════════════════════════════════════════════════════════════════════════════
# NAVIGATION
# ════════════════════════════════════════════════════════════════════════════

section = st.segmented_control(
    "Navigation",
    ["Introduction", "Scenarios"],
    default="Introduction",
    key="nav_section",
    label_visibility="collapsed",
)

if section == "Introduction":
    st.write(INTRO)

elif section == "Scenarios":
    scenario = st.segmented_control(
        "Scenario",
        ["1: Baseline", "2: Prized Seafood", "3: Blast Fishing",
         "4: EEZ", "5: Buyer Dependence"],
        default="1: Baseline",
        key="nav_scenario",
        label_visibility="collapsed",
    )

    if scenario == "1: Baseline":
        scenario_1()
    elif scenario == "2: Prized Seafood":
        scenario_2()
    elif scenario == "3: Blast Fishing":
        scenario_3()
    elif scenario == "4: EEZ":
        scenario_4()
    elif scenario == "5: Buyer Dependence":
        scenario_5()
