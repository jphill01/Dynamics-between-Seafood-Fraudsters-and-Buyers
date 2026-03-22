import numpy as np
import matplotlib.pyplot as plt

from System import DynamicalSystem


default_params = {
    'gamma_m': 5.0,
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
When Fraud has no effect
'''
if True:
    params = default_params.copy()
    params['c1'] = params['c0']
    params['pw1'] = params['pw0']
    params['F_threshold'] = 0.01
    system = DynamicalSystem(
        params=params,
        state=init_state
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ts_data = system.time_series_plot(time=500, title="System Dynamics over Time", x_label="Time", y_label="State", ax=ax)
    
    # Plot State Variables on the primary Y-axis
    l1 = ax.plot(ts_data['Seafood'], label='Seafood (S)', color='blue')
    l2 = ax.plot(ts_data['Effort'], label='Effort (E)', color='green')
    l3 = ax.plot(ts_data['Fraudsters'], label='Fraudsters (F)', color='red')
    l4 = ax.plot(ts_data['Perception of Fraud'], label='Perception (FP)', color='pink')
    l5 = ax.plot(ts_data['Harvest'], label='Harvest', color='brown')
    ax.set_ylabel('State [0, 1]')
    
    # Create a secondary Y-axis for the Prices
    ax2 = ax.twinx()
    # l6 = ax2.plot(ts_data['Market Price'], label='Market Price', color='orange', linestyle='--')
    # l7 = ax2.plot(ts_data['Wholesale Price'], label='Wholesale Price', color='purple', linestyle='--')
    l8 = ax2.plot(ts_data['Nondim Market Price'], label='Nondim Market Price', color='orange', linestyle='--')
    l9 = ax2.plot(ts_data['Nondim Wholesale Price'], label='Nondim Wholesale Price', color='purple', linestyle='--')
    ax2.set_ylabel('Price ($)')
    
    # Combine legends from both axes    
    lines = l1 + l2 + l3 + l4 + l5 + l8 + l9
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper right', fontsize='small')
    
    ax.set_title('nuh')
    ax.set_xlabel('Time')
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
'''
if False:
    system = DynamicalSystem(
        params=default_params,
        state=init_state
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
        # Plot State Variables on the primary Y-axis
        l1 = ax.plot(nut[key]['Seafood'], label='Seafood (S)', color='blue')
        l2 = ax.plot(nut[key]['Effort'], label='Effort (E)', color='green')
        l3 = ax.plot(nut[key]['Fraudsters'], label='Fraudsters (F)', color='red')
        l4 = ax.plot(nut[key]['Perception of Fraud'], label='Perception (FP)', color='pink')
        l5 = ax.plot(nut[key]['Harvest'], label='Harvest', color='brown')
        ax.set_ylabel('State [0, 1]')
        
        # Create a secondary Y-axis for the Prices
        ax2 = ax.twinx()
        l6 = ax2.plot(nut[key]['Market Price'], label='Market Price', color='orange', linestyle='--')
        l7 = ax2.plot(nut[key]['Wholesale Price'], label='Wholesale Price', color='purple', linestyle='--')
        l8 = ax2.plot(nut[key]['Nondim Market Price'], label='Nondim Market Price', color='orange', linestyle='--')
        l9 = ax2.plot(nut[key]['Nondim Wholesale Price'], label='Nondim Wholesale Price', color='purple', linestyle='--')
        ax2.set_ylabel('Price ($)')
        
        # Combine legends from both axes    
        lines = l1 + l2 + l3 + l4 + l5 + l6 + l7 + l8 + l9
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper right', fontsize='small')
        
        ax.set_title(title)
        ax.set_xlabel('Time')
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.show()
'''
BIFURCATION DIAGRAMS if e_sm from 0.01 to 5.0
'''
if False:
    system = DynamicalSystem(
        params=default_params,
        state=init_state
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ts_data = system.time_series_plot(time=500, title="System Dynamics over Time", x_label="Time", y_label="State", ax=ax)
    ax.legend()

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Bifurcation Diagrams for e_sm (Range: 0.01 to 5.0)')

    state_vars = [
        ('seafood', 'Seafood (S)'),
        ('effort', 'Effort (E)'),
        ('fraudsters', 'Fraudsters (F)'),
        ('p_fraudsters', 'Perception of Fraud (FP)')
    ]

    init_state = {'S': 0.6, 'E': 0.3, 'F': 0.1, 'FP': 0.1}

    for ax, (var, title) in zip(axs.flatten(), state_vars):
        system.bifurcation_plot(
            ax=ax,
            init_vals=init_state,
            param_name='g',
            param_range=(0.01, 5.0),
            resolution=300,
            time=300,
            y_state_var=var
        )
        # Overwrite title to be cleaner
        ax.set_title(f'Impact on {title}')

    plt.tight_layout()
    plt.show()