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
# DEFAULT TIME SERIES — ALL FOUR STATE VARIABLES
# ════════════════════════════════════════════════════════════════════════════
if False:
    sys_default = DynamicalSystem(
        params=DEFAULT_PARAMS.copy(),
        state={k: v for k, v in INIT_STATE.items()},
        type="dimensionalized",
    )
    ts_default = sys_default.time_series_plot(time=1000)
    t_default = np.arange(1001)
    
    print(np.min(ts_default['Seafood']), np.max(ts_default['Seafood']))

    fig_d, ax_d = plt.subplots(figsize=(12, 6))
    ax_d.plot(t_default, ts_default['Seafood'],             color=COLORS['S'],  lw=1.8, label='Seafood (S)')
    ax_d.plot(t_default, ts_default['Effort'],              color=COLORS['E'],  lw=1.8, label='Effort (E)')
    ax_d.plot(t_default, ts_default['Fraudsters'],          color=COLORS['F'],  lw=1.8, label='Fraudsters (F)')
    ax_d.plot(t_default, ts_default['Perception of Fraud'], color=COLORS['FP'], lw=1.8, label='Perception (FP)')
    
    ax2 = ax_d.twinx()
    ax2.plot(t_default, ts_default['Market Price'], color=COLORS['Pm'], lw=1.8, label='Market Price (Pm)')
    ax2.plot(t_default, ts_default['Wholesale Price'], color=COLORS['Pw'], lw=1.8, label='Wholesale Price (Pw)')
    ax2.plot(t_default, ts_default['Harvest'], color=COLORS['H'], lw=1.8, label='Harvest (H)')
    ax2.set_ylabel('Market Price and Harvest', fontsize=11)
    
    ax2.legend(fontsize=10, loc='upper right')
    ax2.set_ylim(bottom=0)
    ax2.grid(True, alpha=0.3)
    ax_d.set_xlabel('Time Step', fontsize=11)
    ax_d.set_ylabel('State Value', fontsize=11)
    ax_d.set_title('Default Parameters — Dimensionalized Time Series', fontsize=13)
    ax_d.legend(fontsize=10, loc='upper right')
    ax_d.set_ylim(bottom=0)
    ax_d.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ════════════════════════════════════════════════════════════════════════════
# BIFURCATION — SEAFOOD VS. γ_m (market price sensitivity)
# ════════════════════════════════════════════════════════════════════════════
if False:
    GM_MIN, GM_MAX = 0.1, 20.0
    GM_RES   = 250
    GM_TIME  = 300
    GM_BURN  = int(GM_TIME * 0.6)

    gm_sweep = np.linspace(GM_MIN, GM_MAX, GM_RES)
    bif_gm, bif_gm_S = [], []

    for gm_val in gm_sweep:
        p = DEFAULT_PARAMS.copy()
        p['gamma_m'] = float(gm_val)
        sys_gm = DynamicalSystem(
            params=p,
            state={k: v for k, v in INIT_STATE.items()},
            type="dimensionalized",
        )
        ts_gm = sys_gm.time_series_plot(time=GM_TIME)
        for s in ts_gm['Seafood'][GM_BURN:]:
            bif_gm.append(float(gm_val))
            bif_gm_S.append(float(s))

    fig_gm, ax_gm = plt.subplots(figsize=(10, 6))
    ax_gm.scatter(bif_gm, bif_gm_S, s=0.4, alpha=0.45, color=COLORS['S'])
    ax_gm.axvline(DEFAULT_PARAMS['gamma_m'], color='gray', ls='--', lw=1.2,
                  label=f'Default $\\gamma_m$ = {DEFAULT_PARAMS["gamma_m"]}')
    ax_gm.set_xlabel('$\\gamma_m$  (market price sensitivity)', fontsize=11)
    ax_gm.set_ylabel('Seafood Biomass  $S^*$  (attractor)', fontsize=11)
    ax_gm.set_title(
        'Bifurcation Diagram: Seafood vs. $\\gamma_m$\n'
        'Default params  |  attractor points after burn-in',
        fontsize=12,
    )
    ax_gm.legend(fontsize=10, loc='upper right')
    ax_gm.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()


# ════════════════════════════════════════════════════════════════════════════
# SHARED CONSTANTS
# ════════════════════════════════════════════════════════════════════════════
PW0 = DEFAULT_PARAMS['pw0']
C0  = DEFAULT_PARAMS['c0']
Q0  = DEFAULT_PARAMS['q0']
HONEST_MARGIN = PW0 - C0


# ════════════════════════════════════════════════════════════════════════════
# SCENARIO 1 — BASELINE BIOECONOMIC MODEL (NO FRAUD)
#
# F = 0, FP = 0 throughout.  System reduces to S vs E only.
# Focus parameter: intrinsic growth rate r.
# ════════════════════════════════════════════════════════════════════════════
if True:
    NO_FRAUD = {
        'S': np.float128(0.6), 'E': np.float128(0.3),
        'F': np.float128(0.0), 'FP': np.float128(0.0),
    }
    r_vals  = [0.5, 1.5, 2.5, 3.75]
    N_R     = len(r_vals)
    SIM1    = 300
    BIF1_T  = 300
    BIF1_B  = int(BIF1_T * 0.6)
    t1      = np.arange(SIM1 + 1)

    ts1 = {}
    for rv in r_vals:
        p = DEFAULT_PARAMS.copy(); p['r'] = rv
        s = DynamicalSystem(p, {k: v for k, v in NO_FRAUD.items()}, "dimensionalized")
        ts1[rv] = s.time_series_plot(time=SIM1)

    # ── 1a: Time Series Grid  (2 rows: S, E) × (N cols: r values) ────────
    fig1a, ax1a = plt.subplots(2, N_R, figsize=(4*N_R, 6), sharex=True)
    fig1a.suptitle('Scenario 1 — Baseline (No Fraud): Time Series as $r$ Increases', fontsize=13)
    for col, rv in enumerate(r_vals):
        d = ts1[rv]
        ax1a[0, col].plot(t1, d['Seafood'], color=COLORS['S'], lw=1.5)
        ax1a[0, col].set_title(f'r = {rv}', fontsize=10)
        ax1a[0, col].set_ylim(bottom=0); ax1a[0, col].grid(True, alpha=0.25)
        ax1a[1, col].plot(t1, d['Effort'],  color=COLORS['E'], lw=1.5)
        ax1a[1, col].set_ylim(bottom=0); ax1a[1, col].grid(True, alpha=0.25)
        ax1a[1, col].set_xlabel('Time', fontsize=9)
        if col == 0:
            ax1a[0, col].set_ylabel('Seafood (S)', fontsize=10)
            ax1a[1, col].set_ylabel('Effort (E)',  fontsize=10)
    plt.tight_layout(); plt.show()

    # ── 1b: Bifurcation over r  (2 panels: S*, E*) ───────────────────────
    r_sweep = np.linspace(0.1, 4.0, 250)
    br_r, br_S, br_E = [], [], []
    for rv in r_sweep:
        p = DEFAULT_PARAMS.copy(); p['r'] = float(rv)
        s = DynamicalSystem(p, {k: v for k, v in NO_FRAUD.items()}, "dimensionalized")
        ts = s.time_series_plot(time=BIF1_T)
        for sv, ev in zip(ts['Seafood'][BIF1_B:], ts['Effort'][BIF1_B:]):
            br_r.append(float(rv)); br_S.append(float(sv)); br_E.append(float(ev))

    fig1b, (a1bs, a1be) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig1b.suptitle('Scenario 1 — Bifurcation: $r$  (No Fraud)', fontsize=12)
    skw = dict(s=0.4, alpha=0.45)
    a1bs.scatter(br_r, br_S, color=COLORS['S'], **skw)
    a1be.scatter(br_r, br_E, color=COLORS['E'], **skw)
    for a in (a1bs, a1be):
        a.axvline(DEFAULT_PARAMS['r'], color='gray', ls='--', lw=1,
                  label=f'Default r = {DEFAULT_PARAMS["r"]}')
        a.grid(True, alpha=0.25)
    a1bs.set_ylabel('$S^*$', fontsize=10); a1be.set_ylabel('$E^*$', fontsize=10)
    a1be.set_xlabel('Intrinsic Growth Rate ($r$)', fontsize=11)
    a1bs.legend(fontsize=9); plt.tight_layout(); plt.show()

    # ── 1c: Return Maps  (2 rows: S, E) × (N cols: r values) ─────────────
    fig1c, ax1c = plt.subplots(2, N_R, figsize=(4*N_R, 7))
    fig1c.suptitle('Scenario 1 — Return Maps: $x_t$ vs $x_{t+1}$  (attractor only)', fontsize=13)
    for col, rv in enumerate(r_vals):
        d = ts1[rv]
        for row, (var, clr, lbl) in enumerate([
            ('Seafood', COLORS['S'], 'S'), ('Effort', COLORS['E'], 'E')
        ]):
            x = d[var]
            x_t, x_tp1 = x[BIF1_B:-1], x[BIF1_B+1:]
            ax = ax1c[row, col]
            ax.scatter(x_t, x_tp1, s=1.5, alpha=0.6, color=clr)
            lo = float(min(x_t.min(), x_tp1.min())) * 0.9
            hi = float(max(x_t.max(), x_tp1.max())) * 1.1
            ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8, alpha=0.4)
            ax.set_aspect('equal', adjustable='datalim')
            ax.grid(True, alpha=0.25)
            if row == 0: ax.set_title(f'r = {rv}', fontsize=10)
            if col == 0: ax.set_ylabel(f'${lbl}_{{t+1}}$', fontsize=10)
            if row == 1: ax.set_xlabel(f'${lbl}_t$', fontsize=9)
    plt.tight_layout(); plt.show()


# ════════════════════════════════════════════════════════════════════════════
# SCENARIO 2 — PRIZED / PROTECTED SEAFOOD
#
# Fraudsters pay a premium for protected species (pw₁ > pw₀).
# Same gear, same waters: c₁ = c₀, q₁ = q₀.
# Focus parameter: pw₁.
# ════════════════════════════════════════════════════════════════════════════
if True:
    pw1_vals = [1.10, 1.50, 2.00, 3.00]
    N_PW     = len(pw1_vals)
    SIM2     = 400
    BIF2_T   = 300
    BIF2_B   = int(BIF2_T * 0.6)
    t2       = np.arange(SIM2 + 1)

    ts2 = {}
    for pw in pw1_vals:
        p = DEFAULT_PARAMS.copy()
        p.update({'pw1': pw, 'c1': C0, 'q1': Q0})
        s = DynamicalSystem(p, {k: v for k, v in INIT_STATE.items()}, "dimensionalized")
        ts2[pw] = s.time_series_plot(time=SIM2)

    # ── 2a: Time Series Grid  (4 rows: S,E,F,FP) × (N cols: pw₁) ────────
    VARS4 = [
        ('Seafood', 'S', COLORS['S']), ('Effort', 'E', COLORS['E']),
        ('Fraudsters', 'F', COLORS['F']), ('Perception of Fraud', 'FP', COLORS['FP']),
    ]
    fig2a, ax2a = plt.subplots(4, N_PW, figsize=(4*N_PW, 10), sharex=True)
    fig2a.suptitle(
        f'Scenario 2 — Prized Seafood: Time Series as $pw_1$ Increases\n'
        f'$c_1=c_0={C0}$,  $q_1=q_0={Q0}$,  $r={DEFAULT_PARAMS["r"]}$', fontsize=13)
    for col, pw in enumerate(pw1_vals):
        d = ts2[pw]
        for row, (var, lbl, clr) in enumerate(VARS4):
            ax = ax2a[row, col]
            ax.plot(t2, d[var], color=clr, lw=1.4)
            ax.set_ylim(bottom=0); ax.grid(True, alpha=0.25)
            if row == 0: ax.set_title(f'$pw_1$ = {pw}', fontsize=10)
            if col == 0: ax.set_ylabel(lbl, fontsize=10)
            if row == 3: ax.set_xlabel('Time', fontsize=9)
    plt.tight_layout(); plt.show()

    # ── 2b: Bifurcation over pw₁  (2 panels: S*, E*) ─────────────────────
    pw_sweep = np.linspace(PW0, 5.0, 200)
    bp_p, bp_S, bp_E = [], [], []
    for pw in pw_sweep:
        p = DEFAULT_PARAMS.copy()
        p.update({'pw1': float(pw), 'c1': C0, 'q1': Q0})
        s = DynamicalSystem(p, {k: v for k, v in INIT_STATE.items()}, "dimensionalized")
        ts = s.time_series_plot(time=BIF2_T)
        for sv, ev in zip(ts['Seafood'][BIF2_B:], ts['Effort'][BIF2_B:]):
            bp_p.append(float(pw)); bp_S.append(float(sv)); bp_E.append(float(ev))

    fig2b, (a2s, a2e) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig2b.suptitle(f'Scenario 2 — Bifurcation: $pw_1$  ($r={DEFAULT_PARAMS["r"]}$)', fontsize=12)
    a2s.scatter(bp_p, bp_S, color=COLORS['S'], **skw)
    a2e.scatter(bp_p, bp_E, color=COLORS['E'], **skw)
    for a in (a2s, a2e):
        a.axvline(PW0, color='gray', ls='--', lw=1, label=f'$pw_0$ = {PW0}')
        a.grid(True, alpha=0.25)
    a2s.set_ylabel('$S^*$', fontsize=10); a2e.set_ylabel('$E^*$', fontsize=10)
    a2e.set_xlabel('$pw_1$  (black-market wholesale price)', fontsize=11)
    a2s.legend(fontsize=9); plt.tight_layout(); plt.show()

    # ── 2c: Return Maps  (2 rows: S, E) × (N cols: pw₁) ─────────────────
    fig2c, ax2c = plt.subplots(2, N_PW, figsize=(4*N_PW, 7))
    fig2c.suptitle('Scenario 2 — Return Maps: $x_t$ vs $x_{t+1}$  (attractor)', fontsize=13)
    for col, pw in enumerate(pw1_vals):
        d = ts2[pw]
        for row, (var, clr, lbl) in enumerate([
            ('Seafood', COLORS['S'], 'S'), ('Effort', COLORS['E'], 'E')
        ]):
            x = d[var]
            x_t, x_tp1 = x[BIF2_B:-1], x[BIF2_B+1:]
            ax = ax2c[row, col]
            ax.scatter(x_t, x_tp1, s=1.5, alpha=0.6, color=clr)
            lo = float(min(x_t.min(), x_tp1.min())) * 0.9
            hi = float(max(x_t.max(), x_tp1.max())) * 1.1
            ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8, alpha=0.4)
            ax.set_aspect('equal', adjustable='datalim')
            ax.grid(True, alpha=0.25)
            if row == 0: ax.set_title(f'$pw_1$ = {pw}', fontsize=10)
            if col == 0: ax.set_ylabel(f'${lbl}_{{t+1}}$', fontsize=10)
            if row == 1: ax.set_xlabel(f'${lbl}_t$', fontsize=9)
    plt.tight_layout(); plt.show()


# ════════════════════════════════════════════════════════════════════════════
# SCENARIO 3 — BLAST / CYANIDE FISHING
#
# Destructive methods: q₁↑, pw₁↓, c₁↓ (c₁ drops much more than pw₁).
# Focus parameter: destruction intensity α ∈ [0, 1] that jointly scales
#   q₁(α) = q₀ + α·0.33    →  [0.07, 0.40]
#   pw₁(α) = pw₀ − α·0.40  →  [1.00, 0.60]
#   c₁(α)  = c₀ − α·0.80   →  [0.90, 0.10]
# ════════════════════════════════════════════════════════════════════════════
if True:
    def blast_params(alpha):
        return {
            'q1': Q0  + alpha * 0.33,
            'pw1': PW0 - alpha * 0.40,
            'c1': C0   - alpha * 0.80,
        }

    alpha_vals = [0.15, 0.40, 0.70, 1.00]
    N_BL  = len(alpha_vals)
    SIM3  = 400
    BIF3_T = 300
    BIF3_B = int(BIF3_T * 0.6)
    t3     = np.arange(SIM3 + 1)

    ts3 = {}
    for a in alpha_vals:
        p = DEFAULT_PARAMS.copy()
        p.update({**blast_params(a)})
        s = DynamicalSystem(p, {k: v for k, v in INIT_STATE.items()}, "dimensionalized")
        ts3[a] = s.time_series_plot(time=SIM3)

    # ── 3a: Time Series Grid  (4 rows) × (N cols: α) ─────────────────────
    fig3a, ax3a = plt.subplots(4, N_BL, figsize=(4*N_BL, 10), sharex=True)
    fig3a.suptitle(
        f'Scenario 3 — Blast Fishing: Time Series by Destruction Intensity\n'
        f'$r={DEFAULT_PARAMS["r"]}$  |  $q_1$↑  $pw_1$↓  $c_1$↓↓', fontsize=13)
    for col, a in enumerate(alpha_vals):
        d = ts3[a]; bp = blast_params(a)
        for row, (var, lbl, clr) in enumerate(VARS4):
            ax = ax3a[row, col]
            ax.plot(t3, d[var], color=clr, lw=1.4)
            ax.set_ylim(bottom=0); ax.grid(True, alpha=0.25)
            if row == 0:
                ax.set_title(
                    f'α={a}\n$q_1$={bp["q1"]:.2f}, $pw_1$={bp["pw1"]:.2f}, $c_1$={bp["c1"]:.2f}',
                    fontsize=9)
            if col == 0: ax.set_ylabel(lbl, fontsize=10)
            if row == 3: ax.set_xlabel('Time', fontsize=9)
    plt.tight_layout(); plt.show()

    # ── 3b: Bifurcation over α  (2 panels: S*, E*) ───────────────────────
    a_sweep = np.linspace(0, 1, 200)
    ba_a, ba_S, ba_E = [], [], []
    for av in a_sweep:
        p = DEFAULT_PARAMS.copy()
        p.update({**blast_params(float(av))})
        s = DynamicalSystem(p, {k: v for k, v in INIT_STATE.items()}, "dimensionalized")
        ts = s.time_series_plot(time=BIF3_T)
        for sv, ev in zip(ts['Seafood'][BIF3_B:], ts['Effort'][BIF3_B:]):
            ba_a.append(float(av)); ba_S.append(float(sv)); ba_E.append(float(ev))

    fig3b, (a3s, a3e) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig3b.suptitle(
        f'Scenario 3 — Bifurcation: Destruction Intensity (α)\n'
        f'α=0 → honest  |  α=1 → $q_1$=0.40, $pw_1$=0.60, $c_1$=0.10', fontsize=12)
    a3s.scatter(ba_a, ba_S, color=COLORS['S'], **skw)
    a3e.scatter(ba_a, ba_E, color=COLORS['E'], **skw)
    for a in (a3s, a3e): a.grid(True, alpha=0.25)
    a3s.set_ylabel('$S^*$', fontsize=10); a3e.set_ylabel('$E^*$', fontsize=10)
    a3e.set_xlabel('Destruction Intensity (α)', fontsize=11)
    plt.tight_layout(); plt.show()

    # ── 3c: Return Maps  (2 rows: S, E) × (N cols: α) ────────────────────
    fig3c, ax3c = plt.subplots(2, N_BL, figsize=(4*N_BL, 7))
    fig3c.suptitle('Scenario 3 — Return Maps: $x_t$ vs $x_{t+1}$  (attractor)', fontsize=13)
    for col, a in enumerate(alpha_vals):
        d = ts3[a]
        for row, (var, clr, lbl) in enumerate([
            ('Seafood', COLORS['S'], 'S'), ('Effort', COLORS['E'], 'E')
        ]):
            x = d[var]
            x_t, x_tp1 = x[BIF3_B:-1], x[BIF3_B+1:]
            ax = ax3c[row, col]
            ax.scatter(x_t, x_tp1, s=1.5, alpha=0.6, color=clr)
            lo = float(min(x_t.min(), x_tp1.min())) * 0.9
            hi = float(max(x_t.max(), x_tp1.max())) * 1.1
            ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8, alpha=0.4)
            ax.set_aspect('equal', adjustable='datalim')
            ax.grid(True, alpha=0.25)
            bp = blast_params(a)
            if row == 0: ax.set_title(f'α = {a}', fontsize=10)
            if col == 0: ax.set_ylabel(f'${lbl}_{{t+1}}$', fontsize=10)
            if row == 1: ax.set_xlabel(f'${lbl}_t$', fontsize=9)
    plt.tight_layout(); plt.show()


# ════════════════════════════════════════════════════════════════════════════
# SCENARIO 4 — NON-ENFORCEMENT OF EEZ
#
# Fishers access outside-EEZ waters: q₁↑ (more fish), c₁↑ (higher cost).
# Focus parameter: EEZ violation intensity β ∈ [0, 1] that jointly scales
#   q₁(β) = q₀ + β·0.23  →  [0.07, 0.30]
#   c₁(β) = c₀ + β·1.10  →  [0.90, 2.00]
# pw₁ stays at default (fraudsters buy illegal catch at discount).
# ════════════════════════════════════════════════════════════════════════════
if True:
    def eez_params(beta):
        return {
            'q1': Q0 + beta * 0.23,
            'c1': C0 + beta * 1.10,
        }

    beta_vals = [0.15, 0.40, 0.70, 1.00]
    N_EZ  = len(beta_vals)
    SIM4  = 400
    BIF4_T = 300
    BIF4_B = int(BIF4_T * 0.6)
    t4     = np.arange(SIM4 + 1)

    ts4 = {}
    for b in beta_vals:
        p = DEFAULT_PARAMS.copy()
        p.update({**eez_params(b)})
        s = DynamicalSystem(p, {k: v for k, v in INIT_STATE.items()}, "dimensionalized")
        ts4[b] = s.time_series_plot(time=SIM4)

    # ── 4a: Time Series Grid  (4 rows) × (N cols: β) ─────────────────────
    fig4a, ax4a = plt.subplots(4, N_EZ, figsize=(4*N_EZ, 10), sharex=True)
    fig4a.suptitle(
        f'Scenario 4 — EEZ Non-Enforcement: Time Series by Violation Intensity\n'
        f'$r={DEFAULT_PARAMS["r"]}$  |  $q_1$↑  $c_1$↑  |  $pw_1={DEFAULT_PARAMS["pw1"]}$ (default)', fontsize=13)
    for col, b in enumerate(beta_vals):
        d = ts4[b]; ep = eez_params(b)
        for row, (var, lbl, clr) in enumerate(VARS4):
            ax = ax4a[row, col]
            ax.plot(t4, d[var], color=clr, lw=1.4)
            ax.set_ylim(bottom=0); ax.grid(True, alpha=0.25)
            if row == 0:
                ax.set_title(
                    f'β={b}\n$q_1$={ep["q1"]:.2f}, $c_1$={ep["c1"]:.2f}',
                    fontsize=9)
            if col == 0: ax.set_ylabel(lbl, fontsize=10)
            if row == 3: ax.set_xlabel('Time', fontsize=9)
    plt.tight_layout(); plt.show()

    # ── 4b: Bifurcation over β  (2 panels: S*, E*) ───────────────────────
    b_sweep = np.linspace(0, 1, 200)
    bb_b, bb_S, bb_E = [], [], []
    for bv in b_sweep:
        p = DEFAULT_PARAMS.copy()
        p.update({**eez_params(float(bv))})
        s = DynamicalSystem(p, {k: v for k, v in INIT_STATE.items()}, "dimensionalized")
        ts = s.time_series_plot(time=BIF4_T)
        for sv, ev in zip(ts['Seafood'][BIF4_B:], ts['Effort'][BIF4_B:]):
            bb_b.append(float(bv)); bb_S.append(float(sv)); bb_E.append(float(ev))

    fig4b, (a4s, a4e) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig4b.suptitle(
        f'Scenario 4 — Bifurcation: EEZ Violation Intensity (β)\n'
        f'β=0 → honest  |  β=1 → $q_1$=0.30, $c_1$=2.00', fontsize=12)
    a4s.scatter(bb_b, bb_S, color=COLORS['S'], **skw)
    a4e.scatter(bb_b, bb_E, color=COLORS['E'], **skw)
    for a in (a4s, a4e): a.grid(True, alpha=0.25)
    a4s.set_ylabel('$S^*$', fontsize=10); a4e.set_ylabel('$E^*$', fontsize=10)
    a4e.set_xlabel('EEZ Violation Intensity (β)', fontsize=11)
    plt.tight_layout(); plt.show()

    # ── 4c: Return Maps  (2 rows: S, E) × (N cols: β) ────────────────────
    fig4c, ax4c = plt.subplots(2, N_EZ, figsize=(4*N_EZ, 7))
    fig4c.suptitle('Scenario 4 — Return Maps: $x_t$ vs $x_{t+1}$  (attractor)', fontsize=13)
    for col, b in enumerate(beta_vals):
        d = ts4[b]
        for row, (var, clr, lbl) in enumerate([
            ('Seafood', COLORS['S'], 'S'), ('Effort', COLORS['E'], 'E')
        ]):
            x = d[var]
            x_t, x_tp1 = x[BIF4_B:-1], x[BIF4_B+1:]
            ax = ax4c[row, col]
            ax.scatter(x_t, x_tp1, s=1.5, alpha=0.6, color=clr)
            lo = float(min(x_t.min(), x_tp1.min())) * 0.9
            hi = float(max(x_t.max(), x_tp1.max())) * 1.1
            ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8, alpha=0.4)
            ax.set_aspect('equal', adjustable='datalim')
            ax.grid(True, alpha=0.25)
            if row == 0: ax.set_title(f'β = {b}', fontsize=10)
            if col == 0: ax.set_ylabel(f'${lbl}_{{t+1}}$', fontsize=10)
            if row == 1: ax.set_xlabel(f'${lbl}_t$', fontsize=9)
    plt.tight_layout(); plt.show()


# ════════════════════════════════════════════════════════════════════════════
# SCENARIO 5 — BUYER DEPENDENCE ON SEAFOOD (LOW ε_d)
#
# When ε_d is high, buyers are sensitive to fraud perception — demand
# crashes, market price falls, and fraudsters lose their margin.
# When ε_d is low, buyers NEED seafood regardless — demand holds even
# when fraud is rampant. This removes the self-correcting mechanism and
# lets fraud persist unchecked.
#
# Focus parameter: ε_d (decreasing = more dependent).
# ════════════════════════════════════════════════════════════════════════════
if True:
    ed_vals = [2.0, 1.0, 0.5, 0.1]
    N_ED    = len(ed_vals)
    SIM5    = 400
    BIF5_T  = 300
    BIF5_B  = int(BIF5_T * 0.6)
    t5      = np.arange(SIM5 + 1)

    ts5 = {}
    for ed in ed_vals:
        p = DEFAULT_PARAMS.copy()
        p.update({'e_d': ed})
        s = DynamicalSystem(p, {k: v for k, v in INIT_STATE.items()}, "dimensionalized")
        ts5[ed] = s.time_series_plot(time=SIM5)

    # ── 5a: Time Series Grid  (4 rows) × (N cols: ε_d decreasing) ────────
    fig5a, ax5a = plt.subplots(4, N_ED, figsize=(4*N_ED, 10), sharex=True)
    fig5a.suptitle(
        f'Scenario 5 — Buyer Dependence: Time Series as $\\epsilon_d$ Decreases\n'
        f'$r={DEFAULT_PARAMS["r"]}$  |  Low $\\epsilon_d$ = buyers cannot walk away from seafood',
        fontsize=13)
    for col, ed in enumerate(ed_vals):
        d = ts5[ed]
        for row, (var, lbl, clr) in enumerate(VARS4):
            ax = ax5a[row, col]
            ax.plot(t5, d[var], color=clr, lw=1.4)
            ax.set_ylim(bottom=0); ax.grid(True, alpha=0.25)
            if row == 0: ax.set_title(f'$\\epsilon_d$ = {ed}', fontsize=10)
            if col == 0: ax.set_ylabel(lbl, fontsize=10)
            if row == 3: ax.set_xlabel('Time', fontsize=9)
    plt.tight_layout(); plt.show()

    # ── 5b: Bifurcation over ε_d  (2 panels: S*, E*) ─────────────────────
    ed_sweep = np.linspace(0.01, 3.0, 200)
    be_e, be_S, be_E = [], [], []
    for ed in ed_sweep:
        p = DEFAULT_PARAMS.copy()
        p.update({'e_d': float(ed)})
        s = DynamicalSystem(p, {k: v for k, v in INIT_STATE.items()}, "dimensionalized")
        ts = s.time_series_plot(time=BIF5_T)
        for sv, ev in zip(ts['Seafood'][BIF5_B:], ts['Effort'][BIF5_B:]):
            be_e.append(float(ed)); be_S.append(float(sv)); be_E.append(float(ev))

    fig5b, (a5s, a5e) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig5b.suptitle(
        f'Scenario 5 — Bifurcation: Demand Elasticity ($\\epsilon_d$)\n'
        f'Low $\\epsilon_d$ → buyers dependent  |  '
        f'High $\\epsilon_d$ → buyers sensitive to fraud', fontsize=12)
    a5s.scatter(be_e, be_S, color=COLORS['S'], **skw)
    a5e.scatter(be_e, be_E, color=COLORS['E'], **skw)
    for a in (a5s, a5e):
        a.axvline(DEFAULT_PARAMS['e_d'], color='gray', ls='--', lw=1,
                  label=f'Default $\\epsilon_d$ = {DEFAULT_PARAMS["e_d"]}')
        a.grid(True, alpha=0.25)
    a5s.set_ylabel('$S^*$', fontsize=10); a5e.set_ylabel('$E^*$', fontsize=10)
    a5e.set_xlabel('Demand Elasticity ($\\epsilon_d$)', fontsize=11)
    a5s.legend(fontsize=9); plt.tight_layout(); plt.show()

    # ── 5c: Heatmap — (ε_d, F_threshold) → mean S* ──────────────────────
    #
    # Two levers of buyer defence: how sensitive they are (ε_d) and how
    # quickly they notice fraud (F_threshold). This map shows which
    # combinations protect the stock and which allow collapse.
    # ─────────────────────────────────────────────────────────────────────
    HM_RES = 30
    ed_hm  = np.linspace(0.05, 3.0, HM_RES)
    ft_hm  = np.linspace(0.05, 0.95, HM_RES)
    hm_S   = np.full((HM_RES, HM_RES), np.nan)

    for i, ft in enumerate(ft_hm):
        for j, ed in enumerate(ed_hm):
            p = DEFAULT_PARAMS.copy()
            p.update({'e_d': float(ed), 'F_threshold': float(ft)})
            s = DynamicalSystem(p, {k: v for k, v in INIT_STATE.items()}, "dimensionalized")
            ts = s.time_series_plot(time=BIF5_T)
            hm_S[i, j] = float(np.mean(ts['Seafood'][BIF5_B:]))

    fig5c, ax5c = plt.subplots(figsize=(9, 6))
    im = ax5c.imshow(
        hm_S, origin='lower', aspect='auto',
        extent=[ed_hm[0], ed_hm[-1], ft_hm[0], ft_hm[-1]],
        cmap='YlGnBu', interpolation='bilinear',
    )
    fig5c.colorbar(im, ax=ax5c, label='Mean Seafood Biomass $\\bar{S}^*$')
    ax5c.scatter([DEFAULT_PARAMS['e_d']], [DEFAULT_PARAMS['F_threshold']],
                 color='red', s=100, marker='x', zorder=5, label='Default')
    ax5c.set_xlabel('Demand Elasticity ($\\epsilon_d$)', fontsize=11)
    ax5c.set_ylabel('Fraud Detection Threshold ($\\hat{F}$)', fontsize=11)
    ax5c.set_title(
        'Scenario 5 — Buyer Defence Landscape\n'
        f'Mean $S^*$ across ($\\epsilon_d$, $\\hat{{F}}$) space  |  $r={DEFAULT_PARAMS["r"]}$',
        fontsize=12)
    ax5c.legend(fontsize=9, loc='upper right')
    plt.tight_layout(); plt.show()
