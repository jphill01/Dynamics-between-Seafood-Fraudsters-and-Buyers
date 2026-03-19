import numpy as np
import matplotlib.pyplot as plt

from System import DynamicalSystem

system = DynamicalSystem(
    params={
        'gamma_m': 10.0,
        'gamma_f': 1.0,
        'gamma_s': 1.0,
        'gamma_e': 0.225,
        'gamma_p': 1.0,
        'gamma_fp': 1.0,
        'e_d': 1.0,
        'e_sw': 1.0,
        'e_sm': 1.0,
        'K': 1.0,
        
        'F_threshold': 0.5,
        'q': 0.07,
        'r': 0.225,
        'pw0': 1.0,
        'c0': 0.9,
        
        'pw1': 0.81,
        'c1': 0.153
    },
    state={
        'S': np.float128(0.6),
        'E': np.float128(0.3),
        'F': np.float128(0.1),
        'FP': np.float128(0.1)
    }
)

''' STORY ABOUT PW1 V. PW0'''
vals = {'lower': 0.1, 'little_lower': 0.81, 'little_higher': 5.0, 'higher': 10.0}
nut = system.effects_of_pw1(pw1=vals)
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Effects of varying pw1')

plots_info = [
    ('lower', f'Lower pw1 ({vals["lower"]})'),
    ('little_lower', f'Little Lower pw1 ({vals["little_lower"]})'),
    ('little_higher', f'Little Higher pw1 ({vals["little_higher"]})'),
    ('higher', f'Higher pw1 ({vals["higher"]})')
]

for ax, (key, title) in zip(axs.flatten(), plots_info):
    ax.plot(nut[key]['Seafood'], label='Seafood (S)')
    ax.plot(nut[key]['Effort'], label='Effort (E)')
    ax.plot(nut[key]['Fraudsters'], label='Fraudsters (F)')
    ax.plot(nut[key]['Perception of Fraud'], label='Perception (FP)')
    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel('State')
    ax.grid(True)
    ax.legend()

# fig, ax = plt.subplots(figsize=(10, 6))
# ts_data = system.time_series_plot(time=500, title="System Dynamics over Time", x_label="Time", y_label="State", ax=ax)
# ax.legend()

# fig, axs = plt.subplots(2, 2, figsize=(15, 10))
# fig.suptitle('Bifurcation Diagrams for e_sm (Range: 0.01 to 5.0)')

# state_vars = [
#     ('seafood', 'Seafood (S)'),
#     ('effort', 'Effort (E)'),
#     ('fraudsters', 'Fraudsters (F)'),
#     ('p_fraudsters', 'Perception of Fraud (FP)')
# ]

# init_state = {'S': 0.6, 'E': 0.3, 'F': 0.1, 'FP': 0.1}

# for ax, (var, title) in zip(axs.flatten(), state_vars):
#     system.bifurcation_plot(
#         ax=ax,
#         init_vals=init_state,
#         param_name='g',
#         param_range=(0.01, 5.0),
#         resolution=300,
#         time=300,
#         y_state_var=var
#     )
#     # Overwrite title to be cleaner
#     ax.set_title(f'Impact on {title}')

plt.tight_layout()
plt.show()