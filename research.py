import warnings
import numbers
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from numpy.linalg import eigvals
from matplotlib.colors import ListedColormap

lowest_point = 1e-12
legacy_gamma_fp = 1.0
def seafood_state(S, E, gamma_s): 
    return np.max([S * np.exp(gamma_s * (1 - S - E)), -1*np.inf])
def effort_state(E, S, F, q, pw, c, e_sw, gamma_p, gamma_e): 
    term_1 = (F * (pw - 1) + 1)
    denom_e = (gamma_p * E * S ) ** e_sw
    term_2 = (q * S * (term_1 / denom_e))
    term_3 = (F * (c - 1)) + 1
    
    return np.max([E * np.exp(gamma_e * (term_2 - term_3)), -1*np.inf])
def fraudster_state(S, E, F, Fp, gamma_f, gamma_m, gamma_p, pw, e_sw, e_sm, e_d):
    with warnings.catch_warnings(record=True) as recorded_warnings:
        # Ensure all warnings are captured
        # if Fp > 1:
        #     Fp = 1 - 1e-9
        if F > 1:
            # print(f"F WENT OVER{F}")
            F = 1 - 1e-12
        warnings.simplefilter("always")
        denom_market = (E * S)**(e_sm/2)
        price_market = gamma_m * ((1 - Fp)**(e_d/2) / denom_market)
        
        denom_wholesale = (gamma_p * E * S)**(e_sw)
        price_wholesale = (F * (pw - 1) + 1) / denom_wholesale
        
        delta = gamma_f * (price_market - price_wholesale)
    if recorded_warnings:
        print(f"Captured {len(recorded_warnings)} warning(s):")
        print(f"S: {S}")
        print(f"E: {E}")
        print(f"F: {F}")
        print(f"Fp: {Fp}")
        print(f"market price: {price_market}")
        for w in recorded_warnings:
            print(f"- Message: {w.message}, Category: {w.category.__name__}")
    if gamma_m < 1.3 and gamma_m > 1.0:
        print(f"S: {S}")
        print(f"E: {E}")
        print(f"F: {F}")
        print(f"Fp: {Fp}")
        print(f"market price: {price_market}")
        print(gamma_m)
    return np.max([(F * np.exp(delta)) / (1 + F * (np.exp(delta) - 1)), lowest_point])
def p_fraudster_state(F, Fp, F_threshold): 
    delta_fp = legacy_gamma_fp * (F - F_threshold)
    exp_delta_fp = np.exp(delta_fp)

    return np.max([(Fp * exp_delta_fp) / (1 + Fp * (exp_delta_fp - 1)), lowest_point])
Fprice_m = lambda S, E, FP, q, e_sm, e_d, gamma_m: gamma_m * np.sqrt((1-FP)**e_d/(q*E*S)**e_sm)
Fprice_w = lambda S, E, F, q, pw0, pw1, e_sw, gamma_p: ((pw1 - pw0)*F + pw0)/(gamma_p*q*E*S)**e_sw

def market_and_wholesale_price(S, E, F, Fp, e_sm, e_sw, e_d, gamma_m, gamma_p, pw0, pw1, q):
    # denom_market = (E * S)**(e_sm/2)
    # price_market = gamma_m * ((1 - Fp)**(e_d/2) / denom_market)
    
    # denom_wholesale = (gamma_p * E * S)**(e_sw)
    # price_wholesale = (F * (pw - 1) + 1) / denom_wholesale
    
    return [Fprice_m(S, E, Fp, q, e_sm, e_d, gamma_m), Fprice_w(S, E, F, q, pw0, pw1, e_sw, gamma_p), pw0 / (gamma_p * q * E * S) ** e_sw, pw1 / (gamma_p * q * E * S) ** e_sw]
    return [price_market, price_wholesale]
def non_dim_params(params):
    return {
        'gamma_m': params['gamma_m'] / (params['pw0'] * (params['r'] * params['K']) ** (params['e_sm'] / 2.0)),
        'gamma_f': params['gamma_f'] * params['pw0'],
        'gamma_s': params['gamma_s'] * params['r'],
        'gamma_e': params['gamma_e'] * params['c0'],
        'gamma_p': params['gamma_p'] * params['r'] * params['K'],
        'e_sm': params['e_sm'], 'e_sw': params['e_sw'], 'e_d': params['e_d'],
        'F_threshold': params['F_threshold'], 'legacy_gamma_fp': params['gamma_fp'],
        'q': (params['q'] * params['pw0'] * params['K']) / params['c0'],
        'pw': params['pw1'] / params['pw0'],
        'c': params['c1'] / params['c0'],
    }
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

def time_series(ax, params, initial_vals, time, include_market=False): 
    nondim_p = {
        'gamma_m': 2.0,
        'gamma_f': 1.0,
        'gamma_s': 1.0,
        'gamma_e': 1.0,
        'gamma_p': 1.0,
        'gamma_fp': 1.0,
        'e_d': 1.0,
        'e_sw': 1.0,
        'e_sm': 2.0,
        'K': 1.0,
        
        'F_threshold': 0.5,
        'q': 0.07,
        'r': 0.225,
        'pw0': 1.0,
        'c0': 0.9,
        
        'pw1': 0.81,
        'c1': 0.153
    }
    
    seafood, effort, fraudsters, p_fraudsters = np.array([initial_vals[0]], dtype=np.longdouble), np.array([initial_vals[1]], dtype=np.longdouble), np.array([initial_vals[2]], dtype=np.longdouble), np.array([initial_vals[3]], dtype=np.longdouble)
    buf = market_and_wholesale_price(initial_vals[0], initial_vals[1], initial_vals[2], initial_vals[3], params['e_sm'], params['e_sw'], params['e_d'], params['gamma_m'], params['gamma_p'], nondim_p['pw0'], nondim_p['pw1'], nondim_p['q'])
    market_price = np.array([buf[0]], dtype=np.float128)
    wholesale_price = np.array([buf[1]], dtype=np.float128)
    wholesale_pw0_h = np.array([buf[2]], dtype=np.float128)
    wholesale_pw1_h = np.array([buf[3]], dtype=np.float128)
    
    time_period = []
    for i in range(time):
        time_period.append(i)
        seafood = np.append(seafood, seafood_state(seafood[i], effort[i], params['gamma_s']))
        effort = np.append(effort, effort_state(effort[i], seafood[i], fraudsters[i], params['q'], params['pw'], params['c'], params['e_sw'], params['gamma_p'], params['gamma_e']))
        fraudsters = np.append(fraudsters, fraudster_state(seafood[i], effort[i], fraudsters[i], p_fraudsters[i], params['gamma_f'], params['gamma_m'], params['gamma_p'], params['pw'], params['e_sw'], params['e_sm'], params['e_d']))
        p_fraudsters = np.append(p_fraudsters, p_fraudster_state(fraudsters[i], p_fraudsters[i], params['F_threshold']))
        buf = market_and_wholesale_price(seafood[i], effort[i], fraudsters[i], p_fraudsters[i], params['e_sm'], params['e_sw'], params['e_d'], params['gamma_m'], params['gamma_p'], nondim_p['pw0'], nondim_p['pw1'], nondim_p['q'])
        market_price = np.append(market_price, buf[0])
        wholesale_price = np.append(wholesale_price, buf[1])
        wholesale_pw0_h = np.append(wholesale_pw0_h, buf[2])
        wholesale_pw1_h = np.append(wholesale_pw1_h, buf[3])
    time_period.append(time)
    
    if include_market:
        line_graph([time_period, time_period, time_period, time_period], [wholesale_pw0_h, wholesale_pw1_h, market_price, wholesale_price], ax, title=f"Time Series", y_label="Levels", x_label="Time (t)", line_label=["pw0_h", "pw1_h", "Market Price", "Wholesale Price"], line_color=["Blue", "green","orange", "yellow"], y_lim=y_lim)
    else:
        line_graph([time_period, time_period, time_period, time_period], [seafood, effort, fraudsters, p_fraudsters], ax, title=f"Time Series demand elasticity: {params["e_d"]}", y_label="Levels", x_label="Time (t)", line_label=["Seafood", "Effort", "Fraudsters", "Perceived Fraudsters"], line_color=["Blue", "green", "red", "pink"], y_lim=y_lim)

    return [seafood, effort, fraudsters, p_fraudsters]

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
        points = run_system(non_dim_params(current_params))
        
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
def stability_analysis(params, initial_vals):
    def residuals(x, p):
        """
        For fixed points, we want X_{t+1} - X_{t} = 0.
        """
        return system_map(x, p) - x   
    def get_numerical_jacobian(func, x, p, epsilon=1e-8):
        """Approximates the Jacobian matrix using finite differences."""
        n = len(x)
        J = np.zeros((n, n), dtype=np.float128)
        f0 = func(x, p)
        for i in range(n):
            x_eps = np.copy(x)
            x_eps[i] += epsilon
            f_eps = func(x_eps, p)
            J[:, i] = (f_eps - f0) / epsilon
        return J
        
    res = least_squares(
        residuals, 
        initial_vals, 
        args=(params,), 
        bounds=(0, np.inf), 
        method='trf',
        max_nfev=10000,
        x_scale='jac',
    )
    fixed_point = np.array(res.x, dtype=np.float128)
    
    if not res.success:
        return {
            "fixed_point": None,
            "jacobian": None,
            "max_eigenvalue_mag": None,
            "success": False
        }
        
    J = np.array(get_numerical_jacobian(func=system_map, x=fixed_point, p=params), dtype=np.float64)
    eigenvalues = eigvals(J)
    print(f"Eigenvalues: {eigenvalues}")
    max_eigenvalue_mag = np.max(np.abs(eigenvalues))
    
    return {
        "fixed_point": fixed_point,
        "jacobian": J,
        "max_eigenvalue_mag": max_eigenvalue_mag,
        "success": res.success
    }
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
def contour_plot(ax, params, initial_vals, x_param, y_param, x_range, y_range, resolution, time):
    non_dim_p = non_dim_params(params)
    x_vals = np.linspace(x_range[0], x_range[1], resolution)
    y_vals = np.linspace(y_range[0], y_range[1], resolution)

    stability_map = np.zeros((resolution, resolution))

    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            # Update dynamic parameters
            p = non_dim_p.copy()
            
            '''THIS IS HARD CODED!!!!!!!!'''
            p['pw'] = x / params['pw0']
            p['c'] = y / params['c0']
            try:
                ret_val = stability_analysis(p, [0.5, 0.5, 0.5, 0.5])
                if ret_val['max_eigenvalue_mag'] < 1.0:
                    stability_map[i, j] = 1  # Stable
                else:
                    stability_map[i, j] = 2  # Unstable (Oscillatory/Chaotic)
            except:
                stability_map[i, j] = 3  # Something went wrong
        print(i)

    # Same colormap: Extinct (Black), Stable (Blue), Unstable (Orange), Divergent/No Root (Red)
    cmap = ListedColormap(['blue', 'red', 'black'])

    x_grid, y_grid = np.meshgrid(x_vals, y_vals)
    ax.pcolormesh(x_grid, y_grid, stability_map, cmap=cmap, vmin=0.5, vmax=3.5, shading='auto')

    # ax.colorbar(ticks=[1.0, 2.0, 3.0], 
    #             format=plt.FuncFormatter(lambda val, loc: ['Stable', 'Unstable', "Unknown"][int(val-1)]))

    ax.set_title('Jacobian Eigenvalue Stability Map', fontsize=14)
    ax.set_xlabel(x_param, fontsize=12)
    ax.set_ylabel(y_param, fontsize=12)
    
# Initial starting values 
init_vals = [0.5, 0.5, 0.2, 0.5]
total_time = 50

fig, axs = plt.subplots(3, 2, figsize=(15, 15))
y_lim = [0, 1.1]
y_lim = None

def exec(params, init_vals, total_time, **kwargs):
    if not kwargs['fire'] or kwargs['legacy']:
        return
    
    state_var_vectors = time_series(axs[0][0], non_dim_params(params), init_vals, total_time)
    state_var_vectors = time_series(axs[0][1], non_dim_params(params), init_vals, total_time, True)
    if ("stability" in kwargs.keys() and kwargs['stability'] == True):
        res = stability_analysis(non_dim_params(params), np.array([state_var_vectors[0][-1], state_var_vectors[1][-1], state_var_vectors[2][-1], state_var_vectors[3][-1]], dtype=np.float64))
        if res['success']:    
            print(f"FIXED POINT: {res['fixed_point']}")
            print(f"Max Eigenvalue Magnitude: {res['max_eigenvalue_mag']}")
    
    if ('param_bifurcation' in kwargs and 'param_range' in kwargs):
        bifurcation(axs[1][0], params, kwargs['param_bifurcation'], kwargs['param_range'], total_time, 'fraudsters')
        bifurcation(axs[1][1], params, kwargs['param_bifurcation'], kwargs['param_range'], total_time, 'p_fraudsters')
        bifurcation(axs[2][0], params, kwargs['param_bifurcation'], kwargs['param_range'], total_time, 'effort')
        bifurcation(axs[2][1], params, kwargs['param_bifurcation'], kwargs['param_range'], total_time, 'seafood')
    
    plt.show()

exec(
    params={
        'gamma_m': 2.0,
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
    },
    init_vals=[0.6, 0.3, 0.1, 0.1], # [S, E, F, FP]
    param_bifurcation='gamma_m',
    param_range=[0.01, 10, 100],
    stability=False,
    total_time=100,
    fire=False,
    legacy=False,
    comments='''
        PRE: e_d at 0??
    '''
)

exec(
    params={
        'gamma_m': 23.0,
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
    },
    init_vals=[0.6, 0.3, 0.1, 0.1], # [S, E, F, FP]
    # param_bifurcation='e_d',
    # param_range=[0.01, 10, 300],
    stability=False,
    total_time=100,
    fire=False,
    legacy=False,
    comments='''
        PRE: Let's see how e_d being close to inelastic effects the plot
        
        POST: e_d mainly affects Fp and its oscillations. The amplitude
        of Fp's oscillations get smaller and approach 1 as e_d approaches 0.
        
        I think a good baseline set of parameters is the following:
        {
            'gamma_m': 20.0,  -> Anything above ~4.5 will drive oscillations
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
        
        A good 
    '''
)

exec(
    params={
        'gamma_m': 23,
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
    },
    init_vals=[0.5, 0.5, 0.5, 0.1], # [S, E, F, FP]
    # param_bifurcation='gamma_m',
    # param_range=[0.01, 6.0, 600],
    total_time=500,
    fire=False,
    legacy=False,
    comments='''
        PRE: See how hard we could blast gamma_m
        
        POST: WOAHHHHHHH. gamma_m at 23 seems to be quite numerically stable compared to values such as 20 and 25.
        This is baseline for now.  
    '''
)

exec(
    params={
        'gamma_m': 2.0,
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
    },
    init_vals=[0.5, 0.5, 0.5, 0.1], # [S, E, F, FP]
    param_bifurcation='gamma_m',
    param_range=[0.01, 6.0, 600],
    total_time=100,
    fire=False,
    legacy=False,
    comments='''
        PRE: Seems like google thinks that I should set pw0=1 and pw1=0.81, and c0=0.9 and c1=0.153
        
        POST: Cool bifurcation diagrams but that's really it
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
        'q': 0.07,
        'r': 0.225,
        'pw0': 1.0,
        'c0': 0.9,
        
        'pw1': 1.0,
        'c1': 0.153
    },
    init_vals=[0.5, 0.5, 0.5, 0.5], # [S, E, F, FP]
    param_bifurcation='gamma_m',
    param_range=[3.01, 6.0, 300],
    total_time=500,
    fire=False,
    legacy=False,
    comments='''
        PRE: I'll raise gamma_m and see what's up. 
         
        POST: ok so it really does nothing until around 4.5, where I think it has some numerical instability. 
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
        'q': 0.07,
        'r': 0.225,
        'pw0': 1.0,
        'c0': 0.9,
        
        'pw1': 1.0,
        'c1': 0.153
    },
    init_vals=[0.5, 0.5, 0.5, 0.5], # [S, E, F, FP]
    # param_bifurcation='c1',
    # param_range=[0.01, 0.9, 300],
    total_time=500,
    fire=False,
    legacy=False,
    comments='''
        PRE: When I thought that pw0 = 1 and c0 = 0.17, the graphs seemed to be only stable.
        When q = 0.07, effort was between 0 and 1, and when q = 1, effort was around 26. 
        I've determined that q should be low (so 0.07 for now), but decided that perhaps the ratio 
        between pw1:pw0 and c1:c0 would be 1 and 0.17 respectively. 
        I've set c0 to be 0.9 just to ensure that fishing is still profitable even w/o fraudsters.
        By setting pw0 and pw1 as 1 (so fraudsters having no impact on the fishing), c1 because the 
        bifurcation parameter and we get to see how greatly c1 impacts the system. 
        
        POST: It's still not right. Market price simply isn't high enough. We can perhaps lower Pw0
        towards 0 to grow nondim(gamma_m), or simply raise gamma_m.
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
    fire=False,
    legacy=False,
    comments='''
        PRE: I have a baseline parameter set (essentially) from Yodzis' PhD thesis.
        The only thing that wasn't covered was Pw1 and C1. 
        This is assuming that all elasticities and gammas are one, and F_threshold is some arbitrary number like 0.5.
        
        The question now is what should Pw1 and C1 be?
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