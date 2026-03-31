import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from System import DynamicalSystem, DEFAULT_PARAMS

COLORS = {
    'S': 'steelblue', 'E': 'seagreen', 'F': 'crimson', 'FP': 'orchid',
    'Pw': 'darkorange', 'Pm': 'mediumpurple', 'H': 'saddlebrown',
}
INIT_STATE = {
    'S': np.float128(0.6), 'E': np.float128(0.3),
    'F': np.float128(0.1), 'FP': np.float128(0.1),
}


# ════════════════════════════════════════════════════════════════════════════
# SCENARIO 1 — BASELINE BIOECONOMIC MODEL WITHOUT FRAUDSTERS
#
# Gordon-Schaefer / Ricker-style fishery: Seafood (S) and Effort (E) only.
# F = 0, FP = 0 throughout (both are absorbing fixed points).
# With F = 0 the system collapses to:
#   S_{t+1} = S_t · exp( γ_s · r · (1 - S_t/K)  -  q_0 · E_t )
#   E_{t+1} = E_t · exp( γ_e · (q_0 · Pw_t · S_t  -  c_0) )
#
# We explore how the intrinsic growth rate r governs system complexity:
#   low r  → stable bioeconomic equilibrium
#   mid r  → periodic oscillations
#   high r → chaotic boom-bust cycles
# ════════════════════════════════════════════════════════════════════════════
if False:
    NO_FRAUD_STATE = {
        'S': np.float128(0.6), 'E': np.float128(0.3),
        'F': np.float128(0.0), 'FP': np.float128(0.0),
    }

    # ── Fig 1a: Time series at three r values ─────────────────────────────
    r_showcase = [
        {'r': 0.225, 'label': 'r = 0.225  (stable)', 'ls': '-'},
        {'r': 2.50,  'label': 'r = 2.50   (periodic)', 'ls': '-'},
        {'r': 3.75,  'label': 'r = 3.75   (chaotic)', 'ls': '-'},
    ]
    SIM_TIME_1 = 200

    ts1 = {}
    for rc in r_showcase:
        p = DEFAULT_PARAMS.copy()
        p['r'] = rc['r']
        sys1 = DynamicalSystem(
            params=p,
            state={k: v for k, v in NO_FRAUD_STATE.items()},
            type="dimensionalized",
        )
        ts1[rc['r']] = sys1.time_series_plot(time=SIM_TIME_1)

    t_ax1 = np.arange(SIM_TIME_1 + 1)

    fig1a, axes1a = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    fig1a.suptitle(
        'Scenario 1 — Baseline Bioeconomic Dynamics (No Fraud)\n'
        'How intrinsic growth rate $r$ governs complexity',
        fontsize=13,
    )
    for col, rc in enumerate(r_showcase):
        data = ts1[rc['r']]
        ax_s = axes1a[0, col]
        ax_e = axes1a[1, col]

        ax_s.plot(t_ax1, data['Seafood'], color=COLORS['S'], lw=1.8)
        ax_s.plot(t_ax1, data['Harvest'],  color=COLORS['H'], lw=1.2, alpha=0.7)
        ax_s.set_ylim(bottom=0)
        ax_s.grid(True, alpha=0.25)
        ax_s.set_title(rc['label'], fontsize=11)
        if col == 0:
            ax_s.set_ylabel('Seafood / Harvest', fontsize=10)
        ax_s.legend(['S', 'H'], fontsize=8, loc='upper right')

        ax_e.plot(t_ax1, data['Effort'], color=COLORS['E'], lw=1.8)
        ax_e.set_ylim(bottom=0)
        ax_e.grid(True, alpha=0.25)
        ax_e.set_xlabel('Time Step', fontsize=9)
        if col == 0:
            ax_e.set_ylabel('Fishing Effort (E)', fontsize=10)

    plt.tight_layout()
    plt.show()

    # ── Fig 1b: Bifurcation diagram over r ────────────────────────────────
    BIF_R_MIN, BIF_R_MAX = 0.1, 4.0
    BIF_RES   = 200
    BIF_TIME  = 300
    BIF_BURN  = int(BIF_TIME * 0.6)

    r_sweep = np.linspace(BIF_R_MIN, BIF_R_MAX, BIF_RES)
    bif_r, bif_S, bif_E = [], [], []

    for r_val in r_sweep:
        p = DEFAULT_PARAMS.copy()
        p['r'] = float(r_val)
        sys_bif = DynamicalSystem(
            params=p,
            state={k: v for k, v in NO_FRAUD_STATE.items()},
            type="dimensionalized",
        )
        ts_bif = sys_bif.time_series_plot(time=BIF_TIME)
        for s, e in zip(ts_bif['Seafood'][BIF_BURN:], ts_bif['Effort'][BIF_BURN:]):
            bif_r.append(float(r_val))
            bif_S.append(float(s))
            bif_E.append(float(e))

    fig1b, (ax_bs, ax_be) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig1b.suptitle(
        'Scenario 1 — Bifurcation Diagram: Intrinsic Growth Rate ($r$)\n'
        'No fraud  |  Attractor points after burn-in',
        fontsize=12,
    )
    skw = dict(s=0.4, alpha=0.45)
    ax_bs.scatter(bif_r, bif_S, color=COLORS['S'], **skw)
    ax_be.scatter(bif_r, bif_E, color=COLORS['E'], **skw)

    for ax in (ax_bs, ax_be):
        ax.axvline(DEFAULT_PARAMS['r'], color='gray', ls='--', lw=1,
                   label=f'Default r = {DEFAULT_PARAMS["r"]}')
        ax.grid(True, alpha=0.25)

    ax_bs.set_ylabel('Seafood Biomass  $S^*$', fontsize=10)
    ax_be.set_ylabel('Fishing Effort  $E^*$', fontsize=10)
    ax_be.set_xlabel('Intrinsic Growth Rate  ($r$)', fontsize=11)
    ax_bs.legend(fontsize=9, loc='upper left')
    plt.tight_layout()
    plt.show()

    # ── Fig 1c: Phase portrait (S, E) at three r values ──────────────────
    PHASE_TIME = 300
    PHASE_BURN = int(PHASE_TIME * 0.4)
    phase_colors = ['#1B5E20', '#E65100', '#B71C1C']

    fig1c, ax_ph = plt.subplots(figsize=(8, 6))
    for rc, pc in zip(r_showcase, phase_colors):
        p = DEFAULT_PARAMS.copy()
        p['r'] = rc['r']
        sys_ph = DynamicalSystem(
            params=p,
            state={k: v for k, v in NO_FRAUD_STATE.items()},
            type="dimensionalized",
        )
        ts_ph = sys_ph.time_series_plot(time=PHASE_TIME)
        S_ph = ts_ph['Seafood']
        E_ph = ts_ph['Effort']

        ax_ph.plot(S_ph[:PHASE_BURN], E_ph[:PHASE_BURN],
                   color=pc, lw=0.8, alpha=0.2)
        ax_ph.plot(S_ph[PHASE_BURN:], E_ph[PHASE_BURN:],
                   color=pc, lw=1.8, alpha=0.9, label=rc['label'])
        ax_ph.scatter([S_ph[0]], [E_ph[0]],
                      color=pc, s=60, marker='o', zorder=5)
        ax_ph.scatter([S_ph[-1]], [E_ph[-1]],
                      color=pc, s=140, marker='*', zorder=6)

    ax_ph.set_xlabel('Seafood Biomass (S)', fontsize=12)
    ax_ph.set_ylabel('Fishing Effort (E)', fontsize=12)
    ax_ph.set_title(
        'Scenario 1 — Phase Portrait: Seafood vs. Effort\n'
        'Faint = transient  |  Bold = attractor  |  ● start  |  ★ end',
        fontsize=11,
    )
    ax_ph.legend(fontsize=10)
    ax_ph.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ════════════════════════════════════════════════════════════════════════════
# SCENARIO 2 — NON-ENFORCEMENT OF EXCLUSIVE ECONOMIC ZONES (EEZ)
#
# Context: Canada — Newfoundland Northwest Atlantic cod fishery.
# Fraudster wholesalers enable fishers to sell catch harvested outside
# the EEZ. This opens access to less-depleted stocks (higher q₁) but
# at a greater cost per unit effort (higher c₁ — fuel, distance, risk).
#
# In the model:
#   q(F)  = q₀ + (q₁ - q₀)·F   — catchability rises with fraud
#   C(F)  = c₀ + (c₁ - c₀)·F   — cost per unit effort rises with fraud
#   Pw(F) = (pw₀ + (pw₁-pw₀)·F) / (γ_p·H)^ε_sw  — wholesale price
#
# We sweep q₁ (outside-EEZ productivity) and c₁ (outside-EEZ cost) to
# map out where fraud is ecologically sustainable vs. destructive.
#
# Fixed: r = 3.0 (productive species with oscillatory potential),
#        pw1 = DEFAULT (0.81), pw0 = 1.0
# ════════════════════════════════════════════════════════════════════════════
if True:
    EEZ_R = 3.0

    Q1_MIN, Q1_MAX = DEFAULT_PARAMS['q0'], 0.30
    C1_MIN, C1_MAX = DEFAULT_PARAMS['c0'], 2.0
    HMAP_RES  = 35
    HMAP_TIME = 250
    HMAP_BURN = int(HMAP_TIME * 0.6)

    q1_vals = np.linspace(Q1_MIN, Q1_MAX, HMAP_RES)
    c1_vals = np.linspace(C1_MIN, C1_MAX, HMAP_RES)

    mean_S = np.full((HMAP_RES, HMAP_RES), np.nan)
    mean_F = np.full((HMAP_RES, HMAP_RES), np.nan)
    mean_E = np.full((HMAP_RES, HMAP_RES), np.nan)

    for i, c1 in enumerate(c1_vals):
        for j, q1 in enumerate(q1_vals):
            p = DEFAULT_PARAMS.copy()
            p.update({'r': EEZ_R, 'q1': float(q1), 'c1': float(c1)})
            sys_hm = DynamicalSystem(
                params=p,
                state={k: v for k, v in INIT_STATE.items()},
                type="dimensionalized",
            )
            ts_hm = sys_hm.time_series_plot(time=HMAP_TIME)
            mean_S[i, j] = float(np.mean(ts_hm['Seafood'][HMAP_BURN:]))
            mean_F[i, j] = float(np.mean(ts_hm['Fraudsters'][HMAP_BURN:]))
            mean_E[i, j] = float(np.mean(ts_hm['Effort'][HMAP_BURN:]))

    extent = [Q1_MIN, Q1_MAX, C1_MIN, C1_MAX]

    # ── Fig 2a: Heatmaps — mean S* and mean F* ───────────────────────────
    fig2a, (ax_hs, ax_hf) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig2a.suptitle(
        'Scenario 2 — EEZ Non-Enforcement: Parameter Landscape\n'
        f'$r$ = {EEZ_R},  $pw_1$ = {DEFAULT_PARAMS["pw1"]},  $pw_0$ = {DEFAULT_PARAMS["pw0"]}  |  '
        'Mean attractor values after burn-in',
        fontsize=12,
    )

    im_s = ax_hs.imshow(
        mean_S, origin='lower', aspect='auto', extent=extent,
        cmap='YlGnBu', interpolation='bilinear',
    )
    fig2a.colorbar(im_s, ax=ax_hs, label='Mean Seafood Biomass $\\bar{S}^*$')
    ax_hs.set_xlabel('Catchability at full fraud  ($q_1$)', fontsize=10)
    ax_hs.set_ylabel('Cost at full fraud  ($c_1$)', fontsize=10)
    ax_hs.set_title('Seafood Stock Health', fontsize=11)
    ax_hs.scatter([DEFAULT_PARAMS['q1']], [DEFAULT_PARAMS['c1']],
                  color='red', s=80, marker='x', zorder=5, label='Default $(q_1, c_1)$')
    ax_hs.scatter([DEFAULT_PARAMS['q0']], [DEFAULT_PARAMS['c0']],
                  color='white', s=80, marker='o', zorder=5, label='Honest $(q_0, c_0)$')
    ax_hs.legend(fontsize=8, loc='upper left')

    im_f = ax_hf.imshow(
        mean_F, origin='lower', aspect='auto', extent=extent,
        cmap='OrRd', interpolation='bilinear',
    )
    fig2a.colorbar(im_f, ax=ax_hf, label='Mean Fraudster Fraction $\\bar{F}^*$')
    ax_hf.set_xlabel('Catchability at full fraud  ($q_1$)', fontsize=10)
    ax_hf.set_ylabel('Cost at full fraud  ($c_1$)', fontsize=10)
    ax_hf.set_title('Fraud Prevalence', fontsize=11)
    ax_hf.scatter([DEFAULT_PARAMS['q1']], [DEFAULT_PARAMS['c1']],
                  color='red', s=80, marker='x', zorder=5)
    ax_hf.scatter([DEFAULT_PARAMS['q0']], [DEFAULT_PARAMS['c0']],
                  color='white', s=80, marker='o', zorder=5)

    plt.tight_layout()
    plt.show()

    # ── Fig 2b: Time series at four corners of (q1, c1) space ─────────────
    corners = [
        {'q1': 0.10, 'c1': 1.0, 'label': 'Low $q_1$, Low $c_1$',  'color': '#388E3C'},
        {'q1': 0.25, 'c1': 1.0, 'label': 'High $q_1$, Low $c_1$', 'color': '#F57C00'},
        {'q1': 0.10, 'c1': 1.8, 'label': 'Low $q_1$, High $c_1$', 'color': '#1565C0'},
        {'q1': 0.25, 'c1': 1.8, 'label': 'High $q_1$, High $c_1$','color': '#C62828'},
    ]
    SIM_TIME_2 = 300
    t_ax2 = np.arange(SIM_TIME_2 + 1)

    ts2 = {}
    for cn in corners:
        p = DEFAULT_PARAMS.copy()
        p.update({'r': EEZ_R, 'q1': cn['q1'], 'c1': cn['c1']})
        sys_cn = DynamicalSystem(
            params=p,
            state={k: v for k, v in INIT_STATE.items()},
            type="dimensionalized",
        )
        ts2[cn['label']] = sys_cn.time_series_plot(time=SIM_TIME_2)

    state_vars = [
        ('Seafood',             'Seafood (S)',  COLORS['S']),
        ('Effort',              'Effort (E)',   COLORS['E']),
        ('Fraudsters',          'Fraud (F)',    COLORS['F']),
        ('Perception of Fraud', 'Percep. (FP)', COLORS['FP']),
    ]
    fig2b, axes2b = plt.subplots(4, 4, figsize=(16, 12), sharex=True)
    fig2b.suptitle(
        'Scenario 2 — EEZ Dynamics at Four $(q_1, c_1)$ Corners\n'
        f'$r$ = {EEZ_R}  |  columns = corners  |  rows = state variables',
        fontsize=13,
    )
    for col, cn in enumerate(corners):
        data = ts2[cn['label']]
        for row, (var, ylabel, clr) in enumerate(state_vars):
            ax = axes2b[row, col]
            ax.plot(t_ax2, data[var], color=clr, lw=1.5)
            ax.grid(True, alpha=0.25)
            ax.set_ylim(bottom=0)
            if row == 0:
                ax.set_title(
                    f"{cn['label']}\n$q_1$={cn['q1']}, $c_1$={cn['c1']}",
                    fontsize=9, color=cn['color'], fontweight='bold',
                )
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            if row == 3:
                ax.set_xlabel('Time Step', fontsize=9)

    plt.tight_layout()
    plt.show()

    # ── Fig 2c: Bifurcation over q1 at two c1 values ─────────────────────
    c1_bif_vals = [1.0, 1.5]
    c1_bif_colors = ['#1565C0', '#C62828']
    BIF_Q1_RES  = 120
    BIF_Q1_TIME = 200
    BIF_Q1_BURN = int(BIF_Q1_TIME * 0.6)

    q1_sweep = np.linspace(Q1_MIN, Q1_MAX, BIF_Q1_RES)

    fig2c, (ax_bqs, ax_bqf, ax_bqe) = plt.subplots(
        3, 1, figsize=(10, 10), sharex=True,
    )
    fig2c.suptitle(
        'Scenario 2 — Bifurcation: Outside-EEZ Catchability ($q_1$)\n'
        f'$r$ = {EEZ_R}  |  Two cost levels compared  |  '
        f'Default $q_0$ = {DEFAULT_PARAMS["q0"]}',
        fontsize=12,
    )

    for c1_val, c1_clr in zip(c1_bif_vals, c1_bif_colors):
        bq_q, bq_S, bq_F, bq_E = [], [], [], []
        for q1_val in q1_sweep:
            p = DEFAULT_PARAMS.copy()
            p.update({'r': EEZ_R, 'q1': float(q1_val), 'c1': float(c1_val)})
            sys_bq = DynamicalSystem(
                params=p,
                state={k: v for k, v in INIT_STATE.items()},
                type="dimensionalized",
            )
            ts_bq = sys_bq.time_series_plot(time=BIF_Q1_TIME)
            for s, f, e in zip(
                ts_bq['Seafood'][BIF_Q1_BURN:],
                ts_bq['Fraudsters'][BIF_Q1_BURN:],
                ts_bq['Effort'][BIF_Q1_BURN:],
            ):
                bq_q.append(float(q1_val))
                bq_S.append(float(s))
                bq_F.append(float(f))
                bq_E.append(float(e))

        skw = dict(s=0.5, alpha=0.45, label=f'$c_1$ = {c1_val}')
        ax_bqs.scatter(bq_q, bq_S, color=c1_clr, **skw)
        ax_bqf.scatter(bq_q, bq_F, color=c1_clr, **skw)
        ax_bqe.scatter(bq_q, bq_E, color=c1_clr, **skw)

    for ax in (ax_bqs, ax_bqf, ax_bqe):
        ax.axvline(DEFAULT_PARAMS['q0'], color='gray', ls='--', lw=1,
                   label=f'$q_0$ = {DEFAULT_PARAMS["q0"]}  (honest baseline)')
        ax.grid(True, alpha=0.25)

    ax_bqs.set_ylabel('Seafood Biomass  $S^*$', fontsize=10)
    ax_bqf.set_ylabel('Fraudster Fraction  $F^*$', fontsize=10)
    ax_bqe.set_ylabel('Fishing Effort  $E^*$', fontsize=10)
    ax_bqe.set_xlabel('Catchability at Full Fraud  ($q_1$)', fontsize=11)
    ax_bqs.legend(fontsize=9, loc='upper right')
    plt.tight_layout()
    plt.show()
