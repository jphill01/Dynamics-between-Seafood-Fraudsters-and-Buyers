import numpy as np
import warnings

CLOSE_TO_ZERO = np.finfo(np.float128).eps
CLOSE_TO_ONE = 1 - np.finfo(np.float128).epsneg
POSITIVE_INF = np.inf
class DynamicalSystem():
    # CONSTRUCTOR
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
            np.finfo(np.float128).eps,
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
    def demand(self, **kwargs):
        FP = self.state['FP'] if 'FP' not in kwargs else kwargs['FP']
        e_sm = self.params['e_sm'] if 'e_sm' not in kwargs else kwargs['e_sm']
        e_d = self.params['e_d'] if 'e_d' not in kwargs else kwargs['e_d']
        
        H = self.harvest() if 'H' not in kwargs else kwargs['H']
        
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
        return np.clip([np.sqrt((1-FP)**e_d / H**e_sm)], CLOSE_TO_ZERO, POSITIVE_INF)[0]
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
    
    # TESTING
    def nondim_market_price(self):
        FP = self.state['FP']
        e_d = self.nondim_params['e_d']
        e_sm = self.nondim_params['e_sm']
        E = self.state['E']
        S = self.state['S']
        gamma_m = self.nondim_params['gamma_m']
        
        denom_market = (E * S)**(e_sm/2)
        price_market = gamma_m * ((1 - FP)**(e_d/2) / denom_market)
        
        return np.clip([price_market], CLOSE_TO_ZERO, POSITIVE_INF)[0]
    def nondim_wholesale_price(self):
        F = self.state['F']
        E = self.state['E']
        S = self.state['S']
        pw = self.nondim_params['pw']
        e_sw = self.nondim_params['e_sw']
        gamma_p = self.nondim_params['gamma_p']
        
        denom_wholesale = (gamma_p * E * S)**(e_sw)
        price_wholesale = (F * (pw - 1) + 1) / denom_wholesale
        
        return np.clip([price_wholesale], CLOSE_TO_ZERO, POSITIVE_INF)[0]
    
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
        nondim_market_price = self.nondim_market_price()
        nondim_wholesale_price = self.nondim_wholesale_price()
        
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
            'demand': demand,
            'nondim_market_price': nondim_market_price,
            'nondim_wholesale_price': nondim_wholesale_price
        }
    def time_series_plot(self, time, title="", x_label="", y_label="", ax=None) -> dict:
        # Initializing the state variable vectors
        seafood = np.array(self.state['S'], dtype=np.float128)
        effort = np.array(self.state['E'], dtype=np.float128)
        fraudsters = np.array(self.state['F'], dtype=np.float128)
        p_fraudsters = np.array(self.state['FP'], dtype=np.float128)
        
        # Calculate initial prices for time = 0
        harvest_arr = np.array(self.harvest(), dtype=np.float128)
        market_price_arr = np.array(self.market_price(), dtype=np.float128)
        wholesale_price_arr = np.array(self.wholesale_price(), dtype=np.float128)
        
        nondim_market_price = np.array(self.nondim_market_price(), dtype=np.float128)
        nondim_wholesale_price = np.array(self.nondim_wholesale_price(), dtype=np.float128)
        
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
            
            # Append new prices
            market_price_arr = np.append(market_price_arr, result['market_price'])
            wholesale_price_arr = np.append(wholesale_price_arr, result['wholesale_price'])
            harvest_arr = np.append(harvest_arr, result['harvest'])
            
            nondim_market_price = np.append(nondim_market_price, result['nondim_market_price'])
            nondim_wholesale_price = np.append(nondim_wholesale_price, result['nondim_wholesale_price'])
            
        time_steps.append(time)
        
        if ax is not None:
            # Set plot labels
            ax.set_title(title)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            
            # Plot vectors
            ax.plot(time_steps, seafood, label='Seafood (S)', color='blue', linewidth=2)
            ax.plot(time_steps, effort, label='Effort (E)', color='green', linewidth=2)
            ax.plot(time_steps, fraudsters, label='Fraud (F)', color='red', linewidth=2)
            ax.plot(time_steps, p_fraudsters, label='Perception (FP)', color='pink', linewidth=2)
            
            # # Plot prices as dashed lines
            # ax.plot(time_steps, market_price_arr, label='Market Price', color='orange', linewidth=2, linestyle='--')
            # ax.plot(time_steps, wholesale_price_arr, label='Wholesale Price', color='purple', linewidth=2, linestyle='--')
            
            ax.grid(True)
            
        return {
            'Seafood': seafood,
            'Effort': effort,
            'Fraudsters': fraudsters,
            'Perception of Fraud': p_fraudsters,
            'Market Price': market_price_arr,
            'Wholesale Price': wholesale_price_arr,
            'Harvest': harvest_arr,
            'Nondim Market Price': nondim_market_price,
            'Nondim Wholesale Price': nondim_wholesale_price
        }
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

        # Plot
        ax.scatter(x_vals, y_vals, s=0.5, c='black', alpha=0.5)
        ax.set_title(f'Bifurcation Diagram: Impact of {param_name} on {y_state_var}')
        ax.set_xlabel(param_name)
        ax.set_ylabel(y_state_var)
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3)
    
    # SCENARIOS
    def ed_fp_demand(self, low_harvest, high_harvest, e_ms=1, resolution=100):
        '''Set up the x and y axis (Fp and e_d respectively)'''
        FP_vals = np.linspace(0, 0.999, resolution)
        e_d_vals = np.linspace(0, 5, resolution)
        # X, Y = np.meshgrid(FP_vals, e_d_vals)

        '''Initialize the revenue and cost lists'''
        demand = np.zeros((resolution, resolution))
        demand_2 = np.zeros((resolution, resolution))

        '''Calculate the revenue and costs at every pair of seafood and fraudster levels'''
        for i, FP in enumerate(FP_vals):
            for j, e_d in enumerate(e_d_vals):
                demand[j, i] = self.demand(FP=FP, H=low_harvest, e_d=e_d, e_ms=e_ms)
                demand_2[j, i] = self.demand(FP=FP, H=high_harvest, e_d=e_d, e_ms=e_ms)

        return {
            "Low Harvest": demand,
            "High Harvest": demand_2
        }
    def effects_of_pw1(self, pw1, time=40):
        '''
        Args: 
            pw1: {'lower': int, 'little_lower': int, 'little_higher': int, 'higher': int}
            time: int
        Returns:
            dict: {'lower': time_series, 'little_lower': time_series, 'little_higher': time_series, 'higher': time_series}
        '''
        ret_val = {}
        start_state = self.state
        p = self.params
        p['pw1'] = pw1['lower']
        self.params = p
        ret_val['lower'] = self.time_series_plot(time=time, title="Lower pw1")
        
        p = self.params
        p['pw1'] = pw1['little_lower']
        self.params = p
        self.state = start_state
        ret_val['little_lower'] = self.time_series_plot(time=time, title="Little Lower pw1")
        
        p = self.params
        p['pw1'] = pw1['little_higher']
        self.params = p
        self.state = start_state
        ret_val['little_higher'] = self.time_series_plot(time=time, title="Little Higher pw1")
        
        p = self.params
        p['pw1'] = pw1['higher']
        self.params = p
        self.state = start_state
        ret_val['higher'] = self.time_series_plot(time=time, title="Higher pw1")
        
        self.state = start_state
        return ret_val
    
    # FUNCTION PROPERTIES
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
        