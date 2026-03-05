import warnings
import numbers
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from numpy.linalg import eigvals
from matplotlib.colors import ListedColormap

from Plots import Plots
from System import DynamicalSystem

BASE_PARAMS = {
    'gamma_m': 20.0,
    'gamma_f': 1.0,
    'gamma_s': 1.0,
    'gamma_e': 1.0,
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
}


def ed_fp_demand(low_harvest, high_harvest, e_ms=1, time=2000):
    '''Create plots'''
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(2, 2, 1, projection='3d')

    '''Set up the x and y axis (Fp and e_d respectively)'''
    x_vals = np.linspace(0, 0.999, 100)
    y_vals = np.linspace(0, 5, 100)
    X, Y = np.meshgrid(x_vals, y_vals)

    '''Initialize the revenue and cost lists'''
    demand = np.zeros((100, 100))
    demand_2 = np.zeros((100, 100))

    '''Calculate the revenue and costs at every pair of seafood and fraudster levels'''
    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            demand[j, i] = DynamicalSystem.demand(x, low_harvest, y, e_ms)
            demand_2[j, i] = DynamicalSystem.demand(x, high_harvest, y, e_ms)

    Plots.surface_plot(
        [X, X],
        [Y, Y],
        [demand, demand_2],
        ax,
        title="Demand against CES & Perceived Fraud (2.1)",
        y_label="Demand Elasticity",
        x_label="Perceived Fraudsters",
        z_label="Demand",
        surface_label=["Demand (Low Harvest)", "Demand (High Harvest)"],
        surface_color=['red', 'blue'],
        view=[10, 20, 0]
    )
def ed_fp():
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(2, 2, 1)
    
    init_vals = [0.5, 0.5, 0.5, 0.5]
    Plots.bifurcation_plot(
        ax=ax,
        init_vals=init_vals,
        params=BASE_PARAMS,
        param_name='e_d',
        param_linspace=[0.01, 5.0, 500],
        time=500,
        y_state_var='p_fraudsters'
    )

# ed_fp_demand(low_harvest=0.01, high_harvest=0.05)
ed_fp()
plt.show()
