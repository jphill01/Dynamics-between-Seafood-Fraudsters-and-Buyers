# %pip install SALib

import numbers
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.optimize import least_squares
from numpy.linalg import eigvals


# import warnings
# warnings.filterwarnings("ignore")
bump = 1e-9
bump = 0
legacy_gamma_fp = 1.0
seafood_state = lambda S, E, gamma_s: S * np.exp(gamma_s * (1 - S - E))
def effort_state(E, S, F, q, pw, c, e_sw, gamma_p, gamma_e): 
    term_1 = (F * (pw - 1) + 1)
    denom_e = (gamma_p * E * S + bump) ** e_sw
    term_2 = (q * S * (term_1 / denom_e))
    term_3 = (F * (c - 1)) + 1
    
    return E * np.exp(gamma_e * (term_2 - term_3))
def fraudster_state(S, E, F, Fp, gamma_f, gamma_m, gamma_p, pw, e_sw, e_sm, e_d):
    denom_market = (E * S)**(e_sm/2) + bump
    price_market = gamma_m * ((1 - Fp)**(e_d/2) / denom_market)
    
    denom_wholesale = (gamma_p * E * S)**(e_sw) + bump
    price_wholesale = (F * (pw - 1) + 1) / denom_wholesale
    
    delta = gamma_f * (price_market - price_wholesale)
    if bump != 0:
        delta = np.clip(delta, -20, 20) # Prevent overflow
    return (F * np.exp(delta)) / (1 + F * (np.exp(delta) - 1))
def p_fraudster_state(F, Fp, F_threshold): 
    delta_fp = legacy_gamma_fp * (F - F_threshold)
    exp_delta_fp = np.exp(delta_fp)
    
    return (Fp * exp_delta_fp) / (1 + Fp * (exp_delta_fp - 1))

def line_graph(x_series, y_series, ax, **kwargs):
    """
    Creates a line graph in 2-D or 3-D (optional)

    Args:
        x_series (list[list[Number]]): Values along the x-axis.
        y_series (list[list[Number]]): Values along the y-axis. Same structure as `x_series`.
        ax (plt.Axes): Matplotlib Axes to build graph
        **kwargs: Additional keyword arguments containing graph metadata
            title (str): Title of the graph
            x_label (str): Label along x-axis
            y_label (str): Label along y-axis
            line_label (list[str]): Label(s) for the line(s)
            line_color (list[str]): Color(s) for the line(s)
            x_lim (tuple): The range of the x-axis
            y_lim (tuple): The range of the y-axis

    Returns:
        None
    """
    if not hasattr(x_series, '__iter__') or not hasattr(y_series, '__iter__'):
        raise Exception("x_series and y_series must be iterable")

    if len(x_series) != len(y_series) or len(x_series) == 0:
        raise Exception('x_series and y_series must not be empty')


    # Check if each element in the axis_series is an iterable
    if all(hasattr(a, '__iter__') for a in x_series) and all(hasattr(a, '__iter__') for a in y_series):
        # For each list within the axis_series, check if each element is a number
        if all(isinstance(a, numbers.Number) for sublist in x_series for a in sublist) and all(isinstance(a, numbers.Number) for sublist in y_series for a in sublist):
            # For each list within the axis_series, check if each the list matches the size of the list in the other axis_series
            if all(len(a) == len(b) for a, b in zip(x_series, y_series)):
                line_label_exists = 'line_label' in kwargs and len(kwargs['line_label']) == len(x_series) and all(isinstance(a, str) for a in kwargs['line_label'])
                line_color_exists = 'line_color' in kwargs and len(kwargs['line_color']) == len(x_series) and all(isinstance(a, str) for a in kwargs['line_color'])
                for i in range(len(x_series)):
                    ax.plot(
                    x_series[i],
                    y_series[i],
                    label=kwargs['line_label'][i]
                        if line_label_exists else None,
                    color=kwargs['line_color'][i]
                        if line_color_exists else None,
                    )
                    if line_label_exists:
                        ax.legend()
            else:
                raise Exception("x_series shape does not match y_series")
        else:
            raise Exception("Series elements aren't series of numbers")

    else:
        raise Exception("Series elements aren't iterable")

    if 'title' in kwargs:
      ax.set_title(kwargs['title'])
    if 'x_label' in kwargs:
      ax.set_xlabel(kwargs['x_label'])
    if 'y_label' in kwargs:
      ax.set_ylabel(kwargs['y_label'])
    if 'x_lim' in kwargs:
      ax.set_xlim(kwargs['x_lim'])
    if 'y_lim' in kwargs:
      ax.set_ylim(kwargs['y_lim'])

    ax.grid(True)



    # Old parameters that are still relevant in the nondimensionalized system

def system_map(state, params):
    S, E, F, FP = state
    # Unpack parameters dictionary for clarity
    gamma_m = params['gamma_m']
    gamma_f = params['gamma_f']
    gamma_s = params['gamma_s']
    gamma_e = params['gamma_e']
    gamma_p = params['gamma_p']
    e_sm = params['e_sm']
    e_sw = params['e_sw']
    e_d = params['e_d']
    q = params['q']
    pw = params['pw']
    c = params['c']
    F_threshold = params['F_threshold']
    
    # Calculate next steps
    S_new = seafood_state(S, E, gamma_s)
    E_new = effort_state(E, S, F, q, pw, c, e_sw, gamma_p, gamma_e)
    F_new = fraudster_state(S, E, F, FP, gamma_f, gamma_m, gamma_p, pw, e_sw, e_sm, e_d)
    Fp_new = p_fraudster_state(F, FP, F_threshold)
    return np.array([S_new, E_new, F_new, Fp_new])
def time_series(ax, params, initial_vals, time):
    seafood, effort, fraudsters, p_fraudsters = np.array([initial_vals[0]], dtype=np.longdouble), np.array([initial_vals[1]], dtype=np.longdouble), np.array([initial_vals[2]], dtype=np.longdouble), np.array([initial_vals[3]], dtype=np.longdouble)
    time_period = []
    for i in range(time):
        time_period.append(i)
        seafood = np.append(seafood, seafood_state(seafood[i], effort[i], params['gamma_s']))
        effort = np.append(effort, effort_state(effort[i], seafood[i], fraudsters[i], params['q'], params['pw'], params['c'], params['e_sw'], params['gamma_p'], params['gamma_e']))
        fraudsters = np.append(fraudsters, fraudster_state(seafood[i], effort[i], fraudsters[i], p_fraudsters[i], params['gamma_f'], params['gamma_m'], params['gamma_p'], params['pw'], params['e_sw'], params['e_sm'], params['e_d']))
        p_fraudsters = np.append(p_fraudsters, p_fraudster_state(fraudsters[i], p_fraudsters[i], params['F_threshold']))
    time_period.append(time)
            
    line_graph([time_period, time_period, time_period, time_period], [seafood, effort, fraudsters, p_fraudsters], ax, title=f"Time Series", y_label="Levels", x_label="Time (t)", line_label=["Seafood", "Effort", "Fraudsters", "Perceived Fraudsters"], line_color=["Blue", "green", "red", "pink"], y_lim=y_lim)
def bifurcation(ax, params, param_name, param_linspace, time, y_state_var):
    bifurcation_transient = int(time - (0.25 * time))
    
    def run_system(params, steps=time, transient=bifurcation_transient):
        # Initial starting points
        S, E, F, FP = init_vals
        
        trajectory = []
        
        for t in range(steps):
            # Update
            S_new = seafood_state(S, E, params['gamma_s'])
            E_new = effort_state(E, S, F, params['q'], params['pw'], params['c'], 
                                params['e_sw'], params['gamma_p'], params['gamma_e'])
            F_new = fraudster_state(S, E, F, FP, params['gamma_f'], params['gamma_m'], 
                                    params['gamma_p'], params['pw'], params['e_sw'], 
                                    params['e_sm'], params['e_d'])
            FP_new = p_fraudster_state(F, FP, params['F_threshold'])
            
            # Safety Clip
            S, E, F, FP = np.maximum([S_new, E_new, F_new, FP_new], 1e-6)
            
            # Store only after transient period to see steady state behavior
            if t > transient:
                if y_state_var == "fraudsters":
                    trajectory.append(F)
                elif y_state_var == "p_fraudsters":
                    trajectory.append(FP)
                elif y_state_var == "seafood":
                    trajectory.append(S)
                elif y_state_var == "effort":
                    trajectory.append(E)
                
        return trajectory
    param_values = np.linspace(param_linspace[0], param_linspace[1], param_linspace[2])

    x_vals = []
    y_vals = []

    print(f"Generating Bifurcation Diagram for {param_name}...")

    for val in param_values:
        # Update param
        current_params = params.copy()
        current_params[param_name] = val
        
        # Run
        points = run_system(current_params)
        
        # Append to lists
        for p in points:
            x_vals.append(val)
            y_vals.append(p)

    # Plot
    ax.scatter(x_vals, y_vals, s=0.5, c='black', alpha=0.5)
    ax.set_title(f'Bifurcation Diagram: Impact of {param_name} on {y_state_var}')
    ax.set_xlabel(param_name)
    ax.set_ylabel(y_state_var)
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3)
def stability_analysis(params):
    def equations_to_zero(x):
        """
        For fixed points, we want X_{t+1} - X_{t} = 0.
        """
        # Enforce non-negativity with absolute value for physical feasibility 
        # during solver steps (optional but helps stability)
        x_safe = np.abs(x) 
        return system_map(x_safe, params) - x_safe

    def compute_jacobian(func, x, epsilon=1e-6):
        """
        Computes the Jacobian matrix numerically using central differences.
        J_ij = d(f_i)/d(x_j)
        """
        n = len(x)
        J = np.zeros((n, n))
        
        for j in range(n):
            perturb = np.zeros(n)
            perturb[j] = epsilon
            
            # Central difference: (f(x+h) - f(x-h)) / 2h
            f_plus = func(x + perturb, params)
            f_minus = func(x - perturb, params)
            
            J[:, j] = (f_plus - f_minus) / (2 * epsilon)
            
        return J

def poincare(ax, params, initial_vals, time):
    seafood, effort, fraudsters, p_fraudsters = np.array([initial_vals[0]], dtype=np.longdouble), np.array([initial_vals[1]], dtype=np.longdouble), np.array([initial_vals[2]], dtype=np.longdouble), np.array([initial_vals[3]], dtype=np.longdouble)
    time_period = []
    for i in range(time):
        time_period.append(i)
        seafood = np.append(seafood, seafood_state(seafood[i], effort[i], params['gamma_s']))
        effort = np.append(effort, effort_state(effort[i], seafood[i], fraudsters[i], params['q'], params['pw'], params['c'], params['e_sw'], params['gamma_p'], params['gamma_e']))
        fraudsters = np.append(fraudsters, fraudster_state(seafood[i], effort[i], fraudsters[i], p_fraudsters[i], params['gamma_f'], params['gamma_m'], params['gamma_p'], params['pw'], params['e_sw'], params['e_sm'], params['e_d']))
        p_fraudsters = np.append(p_fraudsters, p_fraudster_state(fraudsters[i], p_fraudsters[i], params['F_threshold']))
    time_period.append(time)
         
    # For seafood
    line_graph([seafood[:-1]], [seafood[1:]], ax, title=f"Poincare", y_label="Seafood + 1", x_label="Seafood", line_label=["Seafood"], line_color=["Blue"], y_lim=y_lim)
def contour_plot(ax, params, initial_vals, time):
    def root_func(x, p):
        """The function we want to find the root for: G(x) - x = 0"""
        return system_map(x, p) - x
    def get_numerical_jacobian(func, x, p, epsilon=1e-8):
        """Approximates the Jacobian matrix using finite differences."""
        n = len(x)
        J = np.zeros((n, n))
        f0 = func(x, p)
        for i in range(n):
            x_eps = np.copy(x)
            x_eps[i] += epsilon
            f_eps = func(x_eps, p)
            J[:, i] = (f_eps - f0) / epsilon
        return J

    resolution = 2
    pw1_vals = np.linspace(0.01, 1, resolution)
    c1_vals = np.linspace(0.01, 0.17, resolution)
    
    stability_map = np.zeros((resolution, resolution))
    
    current_guess = np.array([0.5, 26, 0.5, 0.5])
    
    p = params
    simulated_x = np.array([0.5, 26, 0.5, 0.5])
    for _ in range(100):
        simulated_x = system_map(simulated_x, p)
    print(simulated_x)
    res = least_squares(
        root_func, 
        simulated_x, 
        args=(p,), 
        bounds=(0, np.inf), 
        method='trf'
    )
    print(res.x, res.message, res.success)
    
# Initial starting values 
init_vals = [0.5, 0.5, 0.2, 0.5]
total_time = 50

fig, axs = plt.subplots(3, 2, figsize=(15, 15))
y_lim = [0, 1.1]
y_lim = None

def exec(params, init_vals, total_time, **kwargs):
    if not kwargs['fire']:
        return
    # Converts old params into non-dimensionalized params
    if not kwargs['legacy']: 
        new_params = {
            'gamma_m': params['gamma_m']  / (params['pw0'] * (params['r'] * params['K']) ** (params['e_sm'] / 2.0)),  # gamma_m / (Pw0 * (r * K)^(e_sm / 2)
            'gamma_f': params['gamma_f'] * params['pw0'],                   # gamma_f * Pw0
            'gamma_s': params['gamma_s'] * params['r'],                     # gamma_s * r
            'gamma_e': params['gamma_e'] * params['c0'],                    # gamma_e * C0
            'gamma_p': params['gamma_p'] * params['r'] * params['K'],       # gamma_p * r * K
            'e_sm': params['e_sm'],
            'e_sw': params['e_sw'],
            'e_d': params['e_d'],
            'F_threshold': params['F_threshold'],
            'q': (params['q'] * params['pw0'] * params['K']) / params['c0'],    # (q * Pw0 * K) / C0
            'pw': params['pw1'] / params['pw0'],                                # Pw1/Pw0
            'c': params['c1'] / params['c0'],                                   # C1/C0
        }
        params = new_params
    
    
    time_series(axs[0][0], params, init_vals, total_time)
    poincare(axs[0][1], params, init_vals, total_time)
    
    if ('param_bifurcation' in kwargs and 'param_range' in kwargs):
        bifurcation(axs[1][0], params, kwargs['param_bifurcation'], kwargs['param_range'], total_time, 'fraudsters')
        bifurcation(axs[1][1], params, kwargs['param_bifurcation'], kwargs['param_range'], total_time, 'p_fraudsters')
        bifurcation(axs[2][0], params, kwargs['param_bifurcation'], kwargs['param_range'], total_time, 'effort')
        bifurcation(axs[2][1], params, kwargs['param_bifurcation'], kwargs['param_range'], total_time, 'seafood')
    
    
    # plt.show()
    stability_analysis(params)

 
exec(
    params={
        'gamma_m': 1.0,
        'gamma_f': 1.0,
        'gamma_s': 1.0,
        'gamma_e': 1.0,
        'gamma_p': 1.0,
        'gamma_fp': 1.0,
        'e_d': 1.0,
        'e_sw': 1.0,
        'e_sm': 1.0,
        'F_threshold': 0.5,
        'q': 0.5,
        'pw': 1.0,
        'c': 0.5
    },
    init_vals=[0.5, 0.5, 0.5, 0.5], # [S, E, F, FP]
    param_bifurcation='c',
    param_range=[0.1, 1.0, 300],
    total_time=300,
    fire=False,
    legacy=True,
    comments='''
        PRE:
            This is kinda a based off running some stuff with setting all params to 1 and seeing what I could do there. 
            Seems like (obviously) pw and c are the drivers for this entire system, with q being an "enabler" of sorts and F_threshold being the threshold of
            the perception of fraudsters advocating against fraud. 
            
        POST:
            k so maybe I need the gammas cause there's literally nothing happening and it's just always stable. 
            q lineraly correlated to where Effort will be, and pw + c doesn't do anything
    '''
)

exec(
    params={
        'gamma_m': 1.0,
        'gamma_f': 1.0,
        'gamma_s': 1.0,
        'gamma_e': 1.0,
        'gamma_p': 10.0,
        'gamma_fp': 1.0,
        'e_d': 1.0,
        'e_sw': 1.0,
        'e_sm': 1.0,
        'F_threshold': 0.5,
        'q': 0.5,
        'pw': 1.0,
        'c': 0.5
    },
    init_vals=[0.5, 0.5, 0.5, 0.5], # [S, E, F, FP]
    param_bifurcation='c',
    param_range=[0.1, 1.0, 300],
    total_time=300,
    fire=False,
    legacy=True,
    comments='''
        COOL!
        param_p is a driver for ossicilations? There's osscillations that's for sure. It's like weird square like shapes. Cool to share.
    '''
)

exec(
    params={
        'gamma_m': 1.0,
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
        'q': 1.0,
        'r': 0.225,
        'pw0': 1.0,
        'c0': 0.17,
        
        'pw1': 0.01,
        'c1': 0.001
    },
    init_vals=[0.5, 0.5, 0.5, 0.5], # [S, E, F, FP]
    # param_bifurcation='pw1',
    # param_range=[0.01, 1.0, 300],
    total_time=500,
    fire=True,
    legacy=False,
    comments='''
        PRE: I have a baseline parameter set (essentially) from Yodzis' PhD thesis.
        The only thing that wasn't covered was Pw1 and C1. 
        This is assuming that all elasticities and gammas are one, and F_threshold is some arbitrary number like 0.5.
        
        The question now is what should Pw1 and C1 be?
    '''
)



