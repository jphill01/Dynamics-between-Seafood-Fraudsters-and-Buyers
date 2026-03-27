import numpy as np
import matplotlib.pyplot as plt

from System import DynamicalSystem


default_params = {
    'gamma_m': 5.0,
    'gamma_f': 1.0,
    'gamma_fp': 1.0,
    'gamma_s': 1.0,
    'gamma_e': 0.225,
    'gamma_p': 1.0,
    'e_d': 1.0,
    'e_sw': 1.0,
    'e_sm': 1.0,
    'K': 1.0,
    
    'F_threshold': 0.5,
    'q': 0.07,
    'r': 1.0,
    'pw0': 1.0,
    'c0': 0.9, # Chosen to be illustative
    
    'pw1': 0.81,
    'c1': 0.153
}
init_state = {
    'S': np.float128(0.6),
    'E': np.float128(0.3),
    'F': np.float128(0.1),
    'FP': np.float128(0.1)
}

'''
Bioeconomic Model - What if Fraudsters didn't exist?
Gordon-Schaeffer Model, Yodzis PhD thesis, Fryxell 2017

Seafood equation follows closely to a Ricker logistic model, with the addition of qE (fishing mortality) in the exponent.
    Adding qE allows for the addition of instantaneous consideration of fishing mortality to the reproduction efforts of the seafood.
    It also allows for better numerical stability, where seafood can never reach negative values.

Effort equatino follows a logistic growth model, where effort is driven by profit-per-unit-effort (qSP_w - C). 
    This highlights the idea that fraud drives individual incentives to increase effort, even if it's not sustainable in the long run.
    Because we care about revenue and costs per unit effort, this allows fraud to enable bad actors in the fishing industry to cheat the system,
    and even encourage others to do so.
    
Analysing the effects of the bioeconomic model without fraudsters is important because it allows us to understand the basic dynamics of the system.
'''
if True:
    params = default_params.copy()
    # params['r'] = 3.75
    # params['pw1'] = params['pw0']
    # params['F_threshold'] = 0.01
    params['q'] = 0.01
    system = DynamicalSystem(
        params=params,
        state={
            'S': np.float128(0.6),
            'E': np.float128(0.3),
            'F': np.float128(0.1),
            'FP': np.float128(0.1)
        },
        type="dimensionalized"
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ts_data = system.time_series_plot(time=500)

    # Plot State Variables on the primary Y-axis
    l1 = ax.plot(ts_data['Seafood'], label='Seafood (S)', color='blue')
    l2 = ax.plot(ts_data['Effort'], label='Effort (E)', color='green') 
    l3 = ax.plot(ts_data['Fraudsters'], label='Fraudsters (F)', color='red')
    l4 = ax.plot(ts_data['Perception of Fraud'], label='Perception (FP)', color='pink')
    ax.set_ylim(0, 5)
    # l5 = ax.plot(ts_data['Harvest'], label='Harvest', color='brown')
    ax.set_ylabel('State [0, 1]')
    
    # Create a secondary Y-axis for the Prices
    # ax2 = ax.twinx()
    # l6 = ax2.plot(ts_data['Market Price'], label='Market Price', color='orange', linestyle='--')
    # l7 = ax2.plot(ts_data['Wholesale Price'], label='Wholesale Price', color='purple', linestyle='--')
    # ax2.set_ylabel('Price ($)')
    
    # Combine legends from both axes    
    lines = l1 + l2 + l3 + l4
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper right', fontsize='small')
    
    ax.set_title('nuh')
    ax.set_xlabel('Time')
    ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.show()
# Bioeconomic Bifurcation Diagram
if False:
    params = default_params.copy()
    system = DynamicalSystem(
        params=params,
        state={
            'S': np.float128(0.6),
            'E': np.float128(0.3),
            'F': np.float128(0.0),
            'FP': np.float128(0.0)
        },
        type="dimensionalized"
    )

    bif_data = system.bioeconomic_bifucation_plot(
        r_range=(0, 4),
        resolution=500,
        time=500,
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(bif_data['r'], bif_data['S'], s=0.3, color='steelblue', alpha=0.5)
    ax.set_xlabel('r  (intrinsic growth rate)')
    ax.set_ylabel('Seafood biomass S  (attractor)')
    ax.set_title('Bioeconomic Bifurcation Diagram  –  r  vs  Seafood Transient')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
        
'''
EFFECTS OF VARYING PW1
-> When pw1 is 0.1 (below c1), we see that there's no incentive for fishers to fish.
This is because they actually earn less money fishing because the price that wholesalers
set to purchase their catch is lower than the price that they would set for themselves.
This could be due to a variety of reasons
- desync between fishers and wholesalers, where the black market price to sell their catch is...
- wholesalers are weary of buying fish from fishers because they don't know if the fish quality is maintained
- ...

-> When pw1 is 10.0 (much above pw0)
- It's still sorta like normal oscillations, but there's much more "hookie" at play when it comes to the relationship
between wholesalers and fishers. Effort is still willing to grow because it's still profitable to do so, but when 
it becomes profitable to cheat, they follow suit. However, the wholesale price will grow a lot more compared to market price, 
fraudsters quickly alter from entering to leaving the market. This causes these spikes while effort still grows overall

CASE STUDY: 
Destructive Fishing and Fisheries Enforcement in Eastern Indonesia (Bailey and Sumaila 2015) -> This could be something with q0 and q1 where catchability fluctuates.

Fraud enables bad actors in the fishing industry to cheat the system, and even encourage others to do so.
This opens the door to IUU fishing, where fishers are entered into a realm where no rules apply.
This opportunity leads to a certain scenarios:
- Fishers simply are incentivized to fish more effectively using unethical fishing methods.
  An example of this is through blast fishing, where fishers use explosives to kill fish.
  

'''
if False:
    pass


# ════════════════════════════════════════════════════════════════════════════
# EEZ FRAUD SCENARIO
#
# Story: Fraudster wholesalers enable fishers to sell catch harvested outside
# the Exclusive Economic Zone (EEZ). Fishers bear higher costs (c1 > c0) —
# longer trips, fuel, legal risk — but receive a fraud premium (pw1 > pw0)
# because fraudsters need the volume and absorb the legal risk of purchase.
#
# Parameter regime enforced throughout:
#   pw1 > pw0       : fraudsters offer a price premium for outside-EEZ catch
#   c1  > c0        : fishing outside EEZ is more expensive
#   pw1 > c1        : fishing remains profitable even under full fraud
#   pw1-c1 > pw0-c0 : fraud always yields a HIGHER net profit margin
#
# We vary the "fraud premium gap" Δ = pw1 - c1 across three scenarios
# (marginal, moderate, high) and trace the ecological consequences.
# ════════════════════════════════════════════════════════════════════════════
if False:
    PW0, C0       = default_params['pw0'], default_params['c0']   # 1.0, 0.9
    HONEST_MARGIN = PW0 - C0                                       # 0.1

    # Each scenario: pw1 > pw0=1.0, c1 > c0=0.9, pw1 > c1, pw1-c1 > 0.1
    eez_scenarios = [
        {'pw1': 1.12, 'c1': 1.00, 'color': '#388E3C', 'label': 'Marginal (Δ=0.12)'},
        {'pw1': 1.40, 'c1': 1.10, 'color': '#F57C00', 'label': 'Moderate (Δ=0.30)'},
        {'pw1': 1.80, 'c1': 1.20, 'color': '#C62828', 'label': 'High (Δ=0.60)'},
    ]

    SIM_TIME   = 300
    INIT_STATE = {
        'S': np.float128(0.6), 'E': np.float128(0.3),
        'F': np.float128(0.1), 'FP': np.float128(0.1),
    }

    ts_results = {}
    for sc in eez_scenarios:
        p = default_params.copy()
        p.update({'pw1': sc['pw1'], 'c1': sc['c1']})
        sys_sc = DynamicalSystem(
            params=p,
            state={k: v for k, v in INIT_STATE.items()},
            type="dimensionalized",
        )
        ts_results[sc['label']] = sys_sc.time_series_plot(time=SIM_TIME)

    t_axis = np.arange(SIM_TIME + 1)

    # ── Figure 1: Fisher Profit Incentive Landscape ───────────────────────
    #
    # Normalized profit margin at representative effort (γ_p·H = 1):
    #   π(F) = (pw0 - c0)  +  F · [(pw1 - pw0) - (c1 - c0)]
    #
    # All scenarios share the same honest intercept; the slope encodes how
    # much more profitable each additional fraudster makes fishing.
    # ─────────────────────────────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    F_range = np.linspace(0, 1, 300)

    for sc in eez_scenarios:
        slope  = (sc['pw1'] - PW0) - (sc['c1'] - C0)
        margin = HONEST_MARGIN + F_range * slope
        ax1.plot(F_range, margin, color=sc['color'], lw=2.5, label=sc['label'])
        ax1.annotate(
            f"  π={sc['pw1'] - sc['c1']:.2f}",
            xy=(1.0, sc['pw1'] - sc['c1']),
            color=sc['color'], fontsize=10, va='center',
        )

    ax1.axhline(HONEST_MARGIN, color='gray', ls='--', lw=1.5,
                label=f'Honest baseline  (π = {HONEST_MARGIN})')
    ax1.axhline(0, color='black', lw=0.8, ls=':')
    ax1.set_xlim(0, 1.15)
    ax1.set_ylim(-0.05, 0.85)
    ax1.set_xlabel('Fraction of Fraudster Wholesalers (F)', fontsize=12)
    ax1.set_ylabel('Normalized Profit Margin per Unit Effort', fontsize=12)
    ax1.set_title(
        'EEZ Fraud: Fisher Profit Incentive as Fraud Prevalence Grows\n'
        r'$pw_1 > pw_0$: fraudster premium  |  $c_1 > c_0$: outside-EEZ cost',
        fontsize=11,
    )
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ── Figure 2: Time Series Comparison (4 state vars × 3 scenarios) ────
    state_panels = [
        ('Seafood',             'Seafood Biomass (S)'),
        ('Effort',              'Fishing Effort (E)'),
        ('Fraudsters',          'Fraudster Fraction (F)'),
        ('Perception of Fraud', 'Buyer Fraud Perception (FP)'),
    ]
    fig2, axes2 = plt.subplots(4, 3, figsize=(14, 12), sharex=True)
    fig2.suptitle(
        'EEZ Fraud Dynamics: System Response Across Fraud Premium Levels\n'
        r'Higher $\Delta = pw_1 - c_1$  →  stronger incentive to harvest outside EEZ',
        fontsize=13,
    )

    for col, sc in enumerate(eez_scenarios):
        data = ts_results[sc['label']]
        for row, (var, ylabel) in enumerate(state_panels):
            ax = axes2[row, col]
            ax.plot(t_axis, data[var], color=sc['color'], lw=1.5)
            ax.grid(True, alpha=0.25)
            ax.set_ylim(bottom=0)
            if row == 0:
                ax.set_title(
                    f"{sc['label']}\n$pw_1$={sc['pw1']},  $c_1$={sc['c1']}",
                    fontsize=10,
                )
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            if row == 3:
                ax.set_xlabel('Time Step', fontsize=9)

    plt.tight_layout()
    plt.show()

    # ── Figure 3: Phase Portrait — Seafood vs. Fraudsters ────────────────
    #
    # Trajectory in (S, F) phase space. Faint = transient, bold = attractor.
    # Circles mark the shared starting point; stars mark the long-run end.
    # ─────────────────────────────────────────────────────────────────────
    BURN_TS = int(SIM_TIME * 0.5)
    fig3, ax3 = plt.subplots(figsize=(8, 6))

    for sc in eez_scenarios:
        S_traj = ts_results[sc['label']]['Seafood']
        F_traj = ts_results[sc['label']]['Fraudsters']
        ax3.plot(S_traj[:BURN_TS], F_traj[:BURN_TS],
                 color=sc['color'], lw=1.0, alpha=0.25)
        ax3.plot(S_traj[BURN_TS:], F_traj[BURN_TS:],
                 color=sc['color'], lw=2.0, alpha=0.9, label=sc['label'])
        ax3.scatter([S_traj[0]],  [F_traj[0]],
                    color=sc['color'], s=80,  marker='o', zorder=5)
        ax3.scatter([S_traj[-1]], [F_traj[-1]],
                    color=sc['color'], s=160, marker='*', zorder=6)

    ax3.set_xlabel('Seafood Biomass (S)', fontsize=12)
    ax3.set_ylabel('Fraudster Fraction (F)', fontsize=12)
    ax3.set_title(
        'Phase Portrait: Seafood Biomass vs. Fraudster Fraction\n'
        'Faint = transient  |  Bold = attractor  |  Circles = start  |  Stars = end',
        fontsize=11,
    )
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ── Figure 4: Bifurcation — sweep pw1 at fixed c1 ────────────────────
    #
    # With c1 fixed, we sweep pw1 from the honest baseline (pw0) upward.
    # Two vertical markers delineate three qualitative regimes:
    #   pw1 < c1         : fraud is unprofitable (fishers lose money via fraud)
    #   c1 ≤ pw1 < c1+Δ₀: fraud profitable but no better than honest fishing
    #   pw1 ≥ c1+Δ₀     : fraud strictly dominates — EEZ violations pay off
    # ─────────────────────────────────────────────────────────────────────
    C1_BIF        = 1.05
    EEZ_THRESHOLD = C1_BIF + HONEST_MARGIN   # pw1 where fraud beats honest = 1.15
    BIF_TIME      = 200
    BIF_RES       = 120
    BIF_BURN      = int(BIF_TIME * 0.6)

    pw1_sweep = np.linspace(PW0, 2.5, BIF_RES)
    bif_pw1, bif_S, bif_F, bif_E = [], [], [], []

    for pw1_val in pw1_sweep:
        p = default_params.copy()
        p.update({'r': 3.75, 'pw1': float(pw1_val), 'c1': C1_BIF})
        sys_b = DynamicalSystem(
            params=p,
            state={k: v for k, v in INIT_STATE.items()},
            type="dimensionalized",
        )
        ts_b = sys_b.time_series_plot(time=BIF_TIME)
        for s, f, e in zip(
            ts_b['Seafood'][BIF_BURN:],
            ts_b['Fraudsters'][BIF_BURN:],
            ts_b['Effort'][BIF_BURN:],
        ):
            bif_pw1.append(float(pw1_val))
            bif_S.append(float(s))
            bif_F.append(float(f))
            bif_E.append(float(e))

    fig4, (ax4a, ax4b, ax4c) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fig4.suptitle(
        f'Bifurcation: Effect of the Fraudster Price Premium ($pw_1$)\n'
        f'Fixed $c_1$={C1_BIF},  $r$=3.75  |  Honest profit margin = {HONEST_MARGIN}',
        fontsize=12,
    )
    skw = dict(s=0.5, alpha=0.5)
    ax4a.scatter(bif_pw1, bif_S, color='steelblue', **skw)
    ax4b.scatter(bif_pw1, bif_F, color='crimson',   **skw)
    ax4c.scatter(bif_pw1, bif_E, color='seagreen',  **skw)

    vline_styles = [
        (C1_BIF,        'red',    '--', f'$pw_1 = c_1$ = {C1_BIF}  (fraud breakeven)'),
        (EEZ_THRESHOLD, 'orange', '-.', f'$pw_1$ = {EEZ_THRESHOLD:.2f}  (fraud > honest fishing)'),
    ]
    for ax in (ax4a, ax4b, ax4c):
        for xv, col, ls, lbl in vline_styles:
            ax.axvline(xv, color=col, ls=ls, lw=1.3, label=lbl)
        ax.grid(True, alpha=0.25)

    ax4a.set_ylabel('Seafood Biomass  $S^*$', fontsize=10)
    ax4b.set_ylabel('Fraudster Fraction  $F^*$', fontsize=10)
    ax4c.set_ylabel('Fishing Effort  $E^*$', fontsize=10)
    ax4c.set_xlabel('Wholesale Price at Full Fraud  ($pw_1$)', fontsize=11)
    ax4a.legend(fontsize=8, loc='upper right')

    plt.tight_layout()
    plt.show()


# ════════════════════════════════════════════════════════════════════════════
# EEZ SCENARIO — CATCHABILITY SWEEP
#
# Story: Fraudsters unlock access to outside-EEZ fishing grounds. Fishers
# pay higher costs (c1 > c0) for the privilege, but gain access to less-
# depleted stocks. We proxy this abundance via catchability q: as q rises,
# each unit of effort yields more fish.
#
# The question: at what level of outside-EEZ productivity does the
# combination of fraud and high catchability drive the stock toward collapse?
#
# Fixed: pw1=1.30 (fraudster price premium), c1=1.10 (EEZ cost > c0=0.9)
# Swept: q across a range representing increasing outside-EEZ productivity
# ════════════════════════════════════════════════════════════════════════════
if True:
    EEZ_PW1    = 1.30
    EEZ_C1     = 1.10
    Q_MIN      = 0.01
    Q_MAX      = 0.25
    N_SHOWCASE = 5       # number of q values for time series / phase portrait
    BIF_RES_Q  = 100
    BIF_TIME_Q = 200
    BIF_BURN_Q = int(BIF_TIME_Q * 0.6)
    SIM_TIME_Q = 300

    INIT_STATE_Q = {
        'S': np.float128(0.6), 'E': np.float128(0.3),
        'F': np.float128(0.1), 'FP': np.float128(0.1),
    }

    q_showcase = np.linspace(Q_MIN, Q_MAX, N_SHOWCASE)
    cmap_q     = plt.cm.plasma
    colors_q   = cmap_q(np.linspace(0.1, 0.85, N_SHOWCASE))
    sm_q       = plt.cm.ScalarMappable(
        cmap=cmap_q, norm=plt.Normalize(Q_MIN, Q_MAX)
    )
    sm_q.set_array([])

    # ── Run showcase simulations ──────────────────────────────────────────
    ts_q = {}
    for q_val in q_showcase:
        p = default_params.copy()
        p.update({'pw1': EEZ_PW1, 'c1': EEZ_C1, 'q': float(q_val)})
        sys_q = DynamicalSystem(
            params=p,
            state={k: v for k, v in INIT_STATE_Q.items()},
            type="dimensionalized",
        )
        ts_q[float(q_val)] = sys_q.time_series_plot(time=SIM_TIME_Q)

    t_axis_q = np.arange(SIM_TIME_Q + 1)

    # ── Figure 1: Time Series Overlay (2×2 grid, lines coloured by q) ────
    state_panels_q = [
        ('Seafood',             'Seafood Biomass (S)'),
        ('Effort',              'Fishing Effort (E)'),
        ('Fraudsters',          'Fraudster Fraction (F)'),
        ('Perception of Fraud', 'Buyer Fraud Perception (FP)'),
    ]
    fig_q1, axes_q1 = plt.subplots(2, 2, figsize=(13, 8))
    fig_q1.suptitle(
        'EEZ Catchability Sweep: System Dynamics as Outside-EEZ Productivity Grows\n'
        f'Fixed: $pw_1$={EEZ_PW1},  $c_1$={EEZ_C1} (> $c_0$={default_params["c0"]}),  '
        r'$r$=3.75   |   colour = catchability $q$',
        fontsize=12,
    )

    for idx, (var, ylabel) in enumerate(state_panels_q):
        ax = axes_q1.flatten()[idx]
        for q_val, col in zip(q_showcase, colors_q):
            ax.plot(t_axis_q, ts_q[float(q_val)][var],
                    color=col, lw=1.6, alpha=0.9,
                    label=f'q = {q_val:.3f}')
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlabel('Time Step', fontsize=9)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.25)

    axes_q1.flatten()[1].legend(fontsize=9, loc='upper right', title='Catchability q')
    fig_q1.subplots_adjust(right=0.88)
    cax_q1 = fig_q1.add_axes([0.91, 0.15, 0.02, 0.68])
    fig_q1.colorbar(sm_q, cax=cax_q1).set_label('Catchability (q)', fontsize=10)
    plt.show()

    # ── Figure 2: Phase Portrait in (S, F) coloured by q ─────────────────
    BURN_Q = int(SIM_TIME_Q * 0.5)
    fig_q2, ax_q2 = plt.subplots(figsize=(8, 6))

    for q_val, col in zip(q_showcase, colors_q):
        S_traj = ts_q[float(q_val)]['Seafood']
        F_traj = ts_q[float(q_val)]['Fraudsters']
        ax_q2.plot(S_traj[:BURN_Q], F_traj[:BURN_Q],
                   color=col, lw=0.9, alpha=0.2)
        ax_q2.plot(S_traj[BURN_Q:], F_traj[BURN_Q:],
                   color=col, lw=2.0, alpha=0.95)
        ax_q2.scatter([S_traj[0]],  [F_traj[0]],
                      color=col, s=70,  marker='o', zorder=5)
        ax_q2.scatter([S_traj[-1]], [F_traj[-1]],
                      color=col, s=150, marker='*', zorder=6)

    fig_q2.colorbar(sm_q, ax=ax_q2).set_label('Catchability (q)', fontsize=10)
    ax_q2.set_xlabel('Seafood Biomass (S)', fontsize=12)
    ax_q2.set_ylabel('Fraudster Fraction (F)', fontsize=12)
    ax_q2.set_title(
        'Phase Portrait: Seafood vs. Fraudsters Across Catchability Levels\n'
        'Faint = transient  |  Bold = attractor  |  Circles = start  |  Stars = end',
        fontsize=11,
    )
    ax_q2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ── Figure 3: Bifurcation over q ─────────────────────────────────────
    #
    # Fine sweep of q; vertical reference line at the default value (0.07).
    # Three qualitative stories visible in the attractor cloud:
    #   low q  : high-effort, low-catch regime — fraud present but stocks hold
    #   mid q  : oscillatory / period-doubling transition
    #   high q : high-catch, stock-collapse regime driven by fraud
    # ─────────────────────────────────────────────────────────────────────
    q_sweep = np.linspace(Q_MIN, Q_MAX, BIF_RES_Q)
    bif_q, bif_qS, bif_qF, bif_qE = [], [], [], []

    for q_val in q_sweep:
        p = default_params.copy()
        p.update({'pw1': EEZ_PW1, 'c1': EEZ_C1, 'q': float(q_val)})
        sys_bq = DynamicalSystem(
            params=p,
            state={k: v for k, v in INIT_STATE_Q.items()},
            type="dimensionalized",
        )
        ts_bq = sys_bq.time_series_plot(time=BIF_TIME_Q)
        for s, f, e in zip(
            ts_bq['Seafood'][BIF_BURN_Q:],
            ts_bq['Fraudsters'][BIF_BURN_Q:],
            ts_bq['Effort'][BIF_BURN_Q:],
        ):
            bif_q.append(float(q_val))
            bif_qS.append(float(s))
            bif_qF.append(float(f))
            bif_qE.append(float(e))

    fig_q3, (ax_q3a, ax_q3b, ax_q3c) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fig_q3.suptitle(
        'Bifurcation: Outside-EEZ Catchability ($q$) as the Driver\n'
        f'Fixed: $pw_1$={EEZ_PW1},  $c_1$={EEZ_C1},  $r$=3.75  |  '
        f'Default $q$={default_params["q"]} shown for reference',
        fontsize=12,
    )
    skw_q = dict(s=0.5, alpha=0.5)
    ax_q3a.scatter(bif_q, bif_qS, color='steelblue', **skw_q)
    ax_q3b.scatter(bif_q, bif_qF, color='crimson',   **skw_q)
    ax_q3c.scatter(bif_q, bif_qE, color='seagreen',  **skw_q)

    for ax in (ax_q3a, ax_q3b, ax_q3c):
        ax.axvline(default_params['q'], color='gray', ls='--', lw=1.3,
                   label=f'Default $q$ = {default_params["q"]}')
        ax.grid(True, alpha=0.25)

    ax_q3a.set_ylabel('Seafood Biomass  $S^*$', fontsize=10)
    ax_q3b.set_ylabel('Fraudster Fraction  $F^*$', fontsize=10)
    ax_q3c.set_ylabel('Fishing Effort  $E^*$', fontsize=10)
    ax_q3c.set_xlabel('Catchability  ($q$)', fontsize=11)
    ax_q3a.legend(fontsize=9, loc='upper right')

    plt.tight_layout()
    plt.show()