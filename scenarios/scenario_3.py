import streamlit as st
import numpy as np
from System import DynamicalSystem, DEFAULT_PARAMS

from .constants import _PW0, _C0, _Q0, FULL_INIT
from .plots import plot_4var_ts_fp_zoom, plot_bifurcation, plot_return_maps, plot_ts_with_economics
from ._status import scenario_header, status_indicator


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


@st.fragment
def scenario_3():
    status_slot = scenario_header("Scenario 3 — Blast / Cyanide Fishing")
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

    with status_indicator(status_slot, [
        "Running time-series simulations",
        "Computing bifurcation diagram",
    ]):
        ts3 = {a: s3_time_series(float(a), s3_sim) for a in s3_a_vals}
        t3 = np.arange(s3_sim + 1)
        ba_a, ba_S, ba_E = s3_bifurcation(
            float(s3_rng[0]), float(s3_rng[1]), s3_res, 300, 0.6,
        )

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
        fig = plot_4var_ts_fp_zoom(
            ts3, t3, s3_a_vals, 'α',
            f'Blast Fishing — Time Series by Destruction Intensity   '
            f'(F_threshold={DEFAULT_PARAMS["F_threshold"]}  |  q₁↑  pw₁↓  c₁↓↓)',
            fp_ylim=0.05,
        )
        for i, lbl in enumerate(bp_labels):
            fig.layout.annotations[i].text = lbl
        fig = plot_ts_with_economics(
            ts3, t3, s3_a_vals, 'α',
            f'Blast Fishing — Time Series by Destruction Intensity   '
            f'(F_threshold={DEFAULT_PARAMS["F_threshold"]}  |  q₁↑  pw₁↓  c₁↓↓)',
        )
        for i, lbl in enumerate(bp_labels):
            fig.layout.annotations[i].text = lbl
        st.plotly_chart(fig, width='stretch')

    with tab_bif:
        fig = plot_bifurcation(
            ba_a, ba_S, ba_E,
            xlabel='Destruction Intensity (α)',
            title='Bifurcation over α   '
                  f'(F_threshold={DEFAULT_PARAMS["F_threshold"]}  |  α=0 → honest  |  α=1 → q₁=0.40, pw₁=0.60, c₁=0.10)',
        )
        st.plotly_chart(fig, width='stretch')

    with tab_rm:
        fig = plot_return_maps(ts3, s3_a_vals, 'α', _burn3)
        st.plotly_chart(fig, width='stretch')
