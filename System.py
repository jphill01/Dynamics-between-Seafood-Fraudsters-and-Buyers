import numpy as np
import numbers
import warnings
import matplotlib.pyplot as plt

CLOSE_TO_ZERO = np.finfo(np.float64).eps
CLOSE_TO_ONE = 1 - np.finfo(np.float64).epsneg
POSITIVE_INF = np.inf
class DynamicalSystem():
    # STATE VARIABLES (nondimensionalized)
    def seafood_state(self):
        S = self.state['S']
        E = self.state['E']
        gamma_s = self.nondim_params['gamma_s']
        
        '''
            Artificially create a floor near 0+.
            Reduces risk of numerical imprecisions 
            (and values reaching areas they shouldn't reach).
        '''
        S_next = np.clip(
            [S * np.exp(gamma_s * (1 - S - E))],
            CLOSE_TO_ZERO,
            POSITIVE_INF
        )[0]
        
        return S_next
    def effort_state(self): 
        S = self.state['S']
        E = self.state['E']
        F = self.state['F']
        q = self.nondim_params['q']
        pw = self.nondim_params['pw']
        c = self.nondim_params['c']
        e_sw = self.nondim_params['e_sw']
        gamma_p = self.nondim_params['gamma_p']
        gamma_e = self.nondim_params['gamma_e']

        term_1 = (F * (pw - 1) + 1)
        denom_e = (gamma_p * E * S ) ** e_sw
        term_2 = (q * S * (term_1 / denom_e))
        term_3 = (F * (c - 1)) + 1
        
        '''
            Artificially create a floor near 0+.
            Reduces risk of numerical imprecisions 
            (and values reaching areas they shouldn't reach).
        '''
        E_next = np.clip(
            [E * np.exp(gamma_e * (term_2 - term_3))],
            CLOSE_TO_ZERO,
            POSITIVE_INF
        )[0]
        
        return E_next
    def fraudster_state(self):
        S = self.state['S']
        E = self.state['E']
        F = self.state['F']
        FP = self.state['FP']
        pw = self.nondim_params['pw']
        e_sw = self.nondim_params['e_sw']
        e_sm = self.nondim_params['e_sm']
        e_d = self.nondim_params['e_d']
        gamma_f = self.nondim_params['gamma_f']
        gamma_m = self.nondim_params['gamma_m']
        gamma_p = self.nondim_params['gamma_p']
        
        '''
            1.0 and 0.0 are known fixed points.
            Better to simply return what we know instead of
            calculating the result and risking numerical imprecisions. 
        '''
        if F == 1.0:
            return 1.0
        if F == 0.0:
            return 0.0
        
        with warnings.catch_warnings(record=True) as recorded_warnings:
            denom_market = (E * S)**(e_sm/2)
            price_market = gamma_m * ((1 - FP)**(e_d/2) / denom_market)
            
            denom_wholesale = (gamma_p * E * S)**(e_sw)
            price_wholesale = (F * (pw - 1) + 1) / denom_wholesale
            
            delta = gamma_f * (price_market - price_wholesale)
            
            '''
            Artifically clip between 0 and 1 (noninclusive).
            Reduces risk of numerical imprecisions 
            (and values reaching areas they shouldn't reach).
            '''
            F_next = np.clip(
                [(F * np.exp(delta)) / (1 + F * (np.exp(delta) - 1))],
                CLOSE_TO_ZERO,
                CLOSE_TO_ONE
            )[0]
        if recorded_warnings:
            print(f"Captured {len(recorded_warnings)} warning(s):")
            print(f"S: {S}")
            print(f"E: {E}")
            print(f"F: {F}")
            print(f"Fp: {FP}")
            print(f"market price: {price_market}")
            print(f"wholesale price: {price_wholesale}")
            print(self.params)
            for w in recorded_warnings:
                print(f"- Message: {w.message}, Category: {w.category.__name__}")
                
        return F_next
    def p_fraudster_state(self): 
        F = self.state['F']
        FP = self.state['FP']
        F_threshold = self.nondim_params['F_threshold']
        gamma_fp = self.nondim_params['gamma_fp']
        
        exp_delta_fp = np.exp(gamma_fp * (F - F_threshold))

        '''
            Artifically clip between 0 and 1 (noninclusive).
            Reduces risk of numerical imprecisions 
            (and values reaching areas they shouldn't reach).
        '''
        FP_next = np.clip(
            [(FP * exp_delta_fp) / (1 + FP * (exp_delta_fp - 1))],
            CLOSE_TO_ZERO,
            CLOSE_TO_ONE
        )[0]
        
        return FP_next
        
    # VARIABLES (dimensionful)
    def harvest(self):
        S = self.state['S']
        E = self.state['E']
        q = self.params['q']
        
        '''
            Artificially create a floor near 0+.
            Reduces risk of numerical imprecisions 
            (and values reaching areas they shouldn't reach).
        '''
        return np.clip([q * E * S], np.finfo(type(S)).eps, np.inf)[0]
    def demand(self):
        FP = self.state['FP']
        e_sm = self.params['e_sm']
        e_d = self.params['e_d']
        
        H = self.harvest()
        
        '''
            Artificially create a floor near 0+.
            Reduces risk of numerical imprecisions 
            (and values reaching areas they shouldn't reach).
        '''
        return np.clip([np.sqrt((FP)**e_d * H**e_sm)], np.finfo(type(FP)).eps, np.inf)[0]
    def market_price(self):
        FP = self.state['FP']
        e_d = self.params['e_d']
        e_sm = self.params['e_sm']
        H = self.harvest()
        
        '''
            Artificially create a floor near 0+.
            Reduces risk of numerical imprecisions 
            (and values reaching areas they shouldn't reach).
        '''
        return np.clip([np.sqrt((1-FP)**e_d / H**e_sm)], np.finfo(type(FP)).eps, np.inf)[0]
    def wholesale_price(self):
        F = self.state['F']
        pw0 = self.params['pw0']
        pw1 = self.params['pw1']
        e_sw = self.params['e_sw']
        H = self.harvest()
        
        '''
            Artificially create a floor near 0+.
            Reduces risk of numerical imprecisions 
            (and values reaching areas they shouldn't reach).
        '''
        return np.clip([(F*(pw1 - pw0) + pw0) / H**e_sw], np.finfo(type(F)).eps, np.inf)[0]
    
    def __init__(self, params, state):
        '''
        Args:
            params (dict): A dictionary of system parameters containing:
                Expected keys:
                * gamma_m (float128): [Add description for gamma_m].
                * gamma_p (float128): [Add description for gamma_p].
                * gamma_f (float128): [Add description for gamma_f].
                * gamma_s (float128): [Add description for gamma_s].
                * gamma_e (float128): [Add description for gamma_e].
                * gamma_fp (float128): [Add description for gamma_fp].
                * e_d (float128): [Add description for e_d].
                * e_sw (float128): [Add description for e_sw].
                * e_sm (float128): [Add description for e_sm].
                * K (float128): Carrying capacity of the seafood population.
                * F_threshold (float128): Threshold limit for fraud.
                * q (float128): Catchability coefficient.
                * r (float128): Intrinsic growth rate.
                * pw0 (float128): [Add description for pw0].
                * c0 (float128): [Add description for c0].
                * pw1 (float128): [Add description for pw1].
                * c1 (float128): [Add description for c1].
            state: A dictionary of the initial system state containing:
                Expected keys:
                * S (float128): Seafood biomass.
                * E (float128): Fishing effort.
                * F (float128): Current level of fraud.
                * FP (float128): Public perception of fraud.
        '''
        self._params = {}
        self._state = {}
        
        self.params = params
        self.state = state
    
    def system_map(self) -> dict:        
        '''
        Get system's values for the next time step
        
        Returns: 
            dict: 
        '''
        # It's important to get the variables before calculating the state variables
        market_price = self.market_price()
        wholesale_price = self.wholesale_price()
        harvest = self.harvest()
        demand = self.demand()
        
        S_next = self.seafood_state()
        E_next = self.effort_state()
        F_next = self.fraudster_state()
        FP_next = self.p_fraudster_state()
        
        return {
            'S': S_next,
            'E': E_next,
            'F': F_next,
            'FP': FP_next,
            'market_price': market_price,
            'wholesale_price': wholesale_price,
            'harvest': harvest,
            'demand': demand
        }
    
    def time_series_plot(self, ax, time, title, x_label, y_label):
        # Initializing the state variable vectors
        seafood = np.array(self.state['S'], dtype=np.float128)
        effort = np.array(self.state['E'], dtype=np.float128)
        fraudsters = np.array(self.state['F'], dtype=np.float128)
        p_fraudsters = np.array(self.state['FP'], dtype=np.float128)
        
        # Filling the state variable vectors
        time_steps = []
        for i in range(time):
            time_steps.append(i)
            result = self.system_map()
            
            # Update state
            self.state = {'S': result['S'], 'E': result['E'], 'F': result['F'], 'FP': result['FP']}
            seafood = np.append(seafood, result['S'])
            effort = np.append(effort, result['E'])
            fraudsters = np.append(fraudsters, result['F'])
            p_fraudsters = np.append(p_fraudsters, result['FP'])
        time_steps.append(time)
        
        # Set plot labels
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        
        # Plot vectors
        ax.plot(time_steps, seafood, label='Seafood (S)', color='blue', linewidth=2)
        ax.plot(time_steps, effort, label='Effort (E)', color='green', linewidth=2)
        ax.plot(time_steps, fraudsters, label='Fraud (F)', color='red', linewidth=2)
        ax.plot(time_steps, p_fraudsters, label='Perception (FP)', color='pink', linewidth=2)
        
        ax.grid(True)     
    def surface_plot(x_series, y_series, z_series, ax, **kwargs):
        """
        Creates a line graph in 2-D or 3-D (optional)

        Args:
            x_series (list[list[list[Number]]]): Values along the x-axis.
            y_series (list[list[list[Number]]]): Values along the y-axis. Same structure as `x_series`.
            z_series (list[list[list[Number]]]): Values along the z-axis. Same structure as `x_series`.
            ax (plt.Axes): Matplotlib Axes to build graph. Should have 3D projection.
            **kwargs: Additional keyword arguments containing graph metadata
                title (str): Title of the graph
                x-label (str): Label along x-axis
                y_label (str): Label along y-axis
                z_label (str): Label along z-axis
                surface_label (list[str]): Label(s) for the surface(s)
                surface_color (list[str]): Color(s) for the surface(s)
                x_lim (tuple): The range of the x-axis
                y_lim (tuple): The range of the y-axis
                z_lim (tuple): The range of the z-axis
                view (tuple): The view of the plot
        Returns:
            None
        """
        if isinstance(ax, plt.Axes) == False:
            raise Exception("ax must be a Matplotlib Axes object")

        if not hasattr(x_series, '__iter__') or not hasattr(y_series, '__iter__') or not hasattr(z_series, '__iter__'):
            raise Exception("x_series, y_series, and z_series must be iterable")

        if not (len(x_series) == len(y_series) == len(z_series) and len(z_series) != 0):
            raise Exception('x_series and y_series must not be empty')

        def is_3d_list_of_numbers(x):
            for sub1 in x:
                if not hasattr(sub1, '__iter__'):
                    return False
                for sub2 in sub1:
                    if not hasattr(sub2, '__iter__'):
                        return False
                    if not all(isinstance(item, numbers.Number) for item in sub2):
                        return False
            return True

        # Check if each element in the axis_series is an iterable
        if all(hasattr(a, '__iter__') for a in x_series) and all(hasattr(a, '__iter__') for a in y_series) and all(hasattr(a, '__iter__') for a in z_series):
            # For each list within the axis_series, check if each element is a number
            if is_3d_list_of_numbers(x_series) and is_3d_list_of_numbers(y_series) and is_3d_list_of_numbers(z_series):
                # For each list within the axis_series, check if each the list matches the size of the list in the other axis_series
                if np.shape(x_series) == np.shape(y_series) == np.shape(z_series):
                    surface_label_exists = 'surface_label' in kwargs and len(kwargs['surface_label']) == len(x_series) and all(isinstance(a, str) for a in kwargs['surface_label'])
                    for i in range(len(x_series)):
                        ax.plot_surface(
                            x_series[i],
                            y_series[i],
                            z_series[i],
                            alpha=0.5,
                            edgecolor='none',
                            linewidth=0.5,
                            label=kwargs['surface_label'][i]
                                if surface_label_exists else None,
                            color=kwargs['surface_color'][i]
                                if 'surface_color' in kwargs and len(kwargs['surface_color']) == len(x_series) and all(isinstance(a, str) for a in kwargs['surface_color']) else None,
                        )
                        # if surface_label_exists:
                            # ax.legend()
                else:
                    raise Exception("The series' shapes don't match one another")
            else:
                raise Exception("Surfaces aren't series of numbers")
        else:
            raise Exception("Series elements aren't iterable")

        if 'title' in kwargs:
            ax.set_title(kwargs['title'])
        if 'x_label' in kwargs:
            ax.set_xlabel(kwargs['x_label'])
        if 'y_label' in kwargs:
            ax.set_ylabel(kwargs['y_label'])
        if 'z_label' in kwargs:
            ax.set_zlabel(kwargs['z_label'])
        if 'x_lim' in kwargs:
            ax.set_xlim(kwargs['x_lim'])
        if 'y_lim' in kwargs:
            ax.set_ylim(kwargs['y_lim'])
        if 'z_lim' in kwargs:
            ax.set_zlim(kwargs['z_lim'])
        if 'view' in kwargs and len(np.shape(kwargs['view'])) > 0 and np.shape(kwargs['view'])[0] == 3:
            ax.view_init(kwargs['view'][0], kwargs['view'][1], kwargs['view'][2])

        ax.grid(True)
    def bifurcation_plot(self, ax, init_vals, param_name, param_range, resolution, time, y_state_var):
        transient = int(time - (0.25 * time))
        
        def run_system():
            trajectory = []
            self.state = init_vals
            
            for t in range(time):
                # Update
                next = self.system_map()
                S, E, F, FP = next['S'], next['E'], next['F'], next['FP']
                self.state = {'S': S, 'E': E, 'F': F, 'FP': FP}
                
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
        
        param_values = np.linspace(param_range[0], param_range[1], resolution)

        x_vals = []
        y_vals = []

        print(f"Generating Bifurcation Diagram for {param_name}...")

        for val in param_values:
            # Update param
            current_params = self.params
            current_params[param_name] = val
            self.params = current_params
                                
            # Run
            points = run_system()
            
            # Append to lists
            for p in points:
                x_vals.append(val)
                y_vals.append(p)
            print("val param_value done")

        # Plot
        ax.scatter(x_vals, y_vals, s=0.5, c='black', alpha=0.5)
        ax.set_title(f'Bifurcation Diagram: Impact of {param_name} on {y_state_var}')
        ax.set_xlabel(param_name)
        ax.set_ylabel(y_state_var)
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3)
        
    @property
    def state(self):
        return self._state
    @state.setter
    def state(self, value):
        self._state = value
        
    @property
    def params(self):
        return self._params
    @params.setter
    def params(self, value):
        self._params = value
    
    @property
    def nondim_params(self):
        params = self.params
        return {
            'gamma_m': params['gamma_m'] / (params['pw0'] * (params['r'] * params['K']) ** (params['e_sm'] / 2.0)),
            'gamma_p': params['gamma_p'] * params['r'] * params['K'],
            'gamma_f': params['gamma_f'] * params['pw0'],
            'gamma_s': params['gamma_s'] * params['r'],
            'gamma_e': params['gamma_e'] * params['c0'],
            'gamma_fp': params['gamma_fp'],
            'e_sm': params['e_sm'], 'e_sw': params['e_sw'], 'e_d': params['e_d'],
            'F_threshold': params['F_threshold'],
            'q': (params['q'] * params['pw0'] * params['K']) / params['c0'],
            'pw': params['pw1'] / params['pw0'],
            'c': params['c1'] / params['c0'],
        }







sys = DynamicalSystem(
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
    state={
        'S': 0.6,
        'E': 0.3,
        'F': 0.1,
        'FP': 0.1
    }
)
fig, ax = plt.subplots(figsize=(6,4))
# sys.time_series_plot(ax, 500, "idk", "idk as well", "mane what")
sys.bifurcation_plot(
    ax=ax,
    init_vals = {
        'S': 0.6,
        'E': 0.3,
        'F': 0.1,
        'FP': 0.1
    },
    param_name='gamma_m',
    param_range=[0.01, 10, 100],
    resolution=100,
    time=100,
    y_state_var='p_fraudsters'
)
plt.show()