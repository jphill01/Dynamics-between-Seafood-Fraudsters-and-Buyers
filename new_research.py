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
    params['r'] = 3.75
    # params['pw1'] = params['pw0']
    # params['F_threshold'] = 0.01
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