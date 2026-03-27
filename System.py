import numpy as np
import warnings

CLOSE_TO_ZERO = np.finfo(np.float128).eps
CLOSE_TO_ONE = 1 - np.finfo(np.float128).epsneg
POSITIVE_INF = np.inf
DEFAULT_PARAMS = {
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
class DynamicalSystem():
    # CONSTRUCTOR
    def __init__(self, params, state, type="nondimensionalized"):
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
            type: A string indicating the type of system to be initialized.
                Expected values:
                * "nondimensionalized": Initializes the system in nondimensionalized form.
                * "dimensionalized": Initializes the system in dimensionalized form.
        '''
        self._params = {}
        self._state = {}
        self._type = type
        
        self.params = params if params is not None else DEFAULT_PARAMS
        self.state = state
    
    # STATE VARIABLES (nondimensionalized)
    def seafood_state_nondim(self):
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
    def effort_state_nondim(self): 
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
    def fraudster_state_nondim(self):
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
    def p_fraudster_state_nondim(self): 
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
    
    # STATE VARIABLES (dimensionful)
    def seafood_state_dimful(self):
        S = self.state['S']
        E = self.state['E']
        r = self.params['r']
        K = self.params['K']
        q = self.params['q']
        gamma_s = self.params['gamma_s']
        
        '''
            Artificially create a floor near 0+.
            Reduces risk of numerical imprecisions 
            (and values reaching areas they shouldn't reach).
        '''
        S_next = np.clip(
            [S * np.exp(gamma_s * r * (1 - S / K) - q * E)],
            np.finfo(np.float128).eps,
            POSITIVE_INF
        )[0]

        return S_next
    def effort_state_dimful(self):
        S = self.state['S']
        E = self.state['E']
        q = self.params['q']
        gamma_e = self.params['gamma_e']
                
        Pw = self.wholesale_price()
        C = self.cost()
                
        E_next = np.clip(
            [E * np.exp(gamma_e * (q * Pw * S - C))],
            CLOSE_TO_ZERO,
            POSITIVE_INF
        )[0]
        
        return E_next
    def fraudster_state_dimful(self):
        F = self.state['F']
        gamma_f = self.params['gamma_f']
        
        pm = self.market_price()
        pw = self.wholesale_price()
        delta = gamma_f * (pm - pw)
        
        return np.clip([F * np.exp(delta) / (1 + F * (np.exp(delta) - 1))], CLOSE_TO_ZERO, CLOSE_TO_ONE)[0]
    def p_fraudster_state_dimful(self):
        F = self.state['F']
        FP = self.state['FP']
        F_threshold = self.params['F_threshold']
        gamma_fp = self.params['gamma_fp']
        exp_delta_fp = np.exp(gamma_fp * (F - F_threshold))
        
        return np.clip([FP * exp_delta_fp / (1 + FP * (exp_delta_fp - 1))], CLOSE_TO_ZERO, CLOSE_TO_ONE)[0]
    
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
        gamma_p = self.params['gamma_p']
        H = self.harvest()
        
        '''
            Artificially create a floor near 0+.
            Reduces risk of numerical imprecisions 
            (and values reaching areas they shouldn't reach).
        '''
        return np.clip([(F*(pw1 - pw0) + pw0) / ((gamma_p * H)**e_sw)], np.finfo(type(F)).eps, np.inf)[0]
    def cost(self):
        F = self.state['F']
        c0 = self.params['c0']
        c1 = self.params['c1']
        
        return F * (c1 - c0) + c0
    
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
        
        S_next_nondim = self.seafood_state_nondim()
        E_next_nondim = self.effort_state_nondim()
        F_next_nondim = self.fraudster_state_nondim()
        FP_next_nondim = self.p_fraudster_state_nondim()
        
        S_next_dimful = self.seafood_state_dimful()
        E_next_dimful = self.effort_state_dimful()
        F_next_dimful = self.fraudster_state_dimful()
        FP_next_dimful = self.p_fraudster_state_dimful()
        
        return {
            'S_nondim': S_next_nondim,
            'E_nondim': E_next_nondim,
            'F_nondim': F_next_nondim,
            'FP_nondim': FP_next_nondim,
            'market_price': market_price,
            'wholesale_price': wholesale_price,
            'harvest': harvest,
            'demand': demand,
            'nondim_market_price': nondim_market_price,
            'nondim_wholesale_price': nondim_wholesale_price,
            'S_dimful': S_next_dimful,
            'E_dimful': E_next_dimful,
            'F_dimful': F_next_dimful,
            'FP_dimful': FP_next_dimful
        }
    def time_series_plot(self, time, title="", x_label="", y_label="", ax=None) -> dict:
        
        if self.type == "nondimensionalized":
            # Initializing the nondim state variable vectors
            seafood_nondim = np.array(self.state['S'], dtype=np.float128)
            effort_nondim = np.array(self.state['E'], dtype=np.float128)
            fraudsters_nondim = np.array(self.state['F'], dtype=np.float128)
            p_fraudsters_nondim = np.array(self.state['FP'], dtype=np.float128)
            
            nondim_market_price = np.array(self.nondim_market_price(), dtype=np.float128)
            nondim_wholesale_price = np.array(self.nondim_wholesale_price(), dtype=np.float128)
        elif self.type == "dimensionalized":
            # Initializing the dimful state variable vectors
            seafood_dimful = np.array(self.state['S'], dtype=np.float128)
            effort_dimful = np.array(self.state['E'], dtype=np.float128)
            fraudsters_dimful = np.array(self.state['F'], dtype=np.float128)
            p_fraudsters_dimful = np.array(self.state['FP'], dtype=np.float128)
        
            harvest_arr = np.array(self.harvest(), dtype=np.float128)
            market_price_arr = np.array(self.market_price(), dtype=np.float128)
            wholesale_price_arr = np.array(self.wholesale_price(), dtype=np.float128)
        
        # Filling the state variable vectors
        time_steps = []
        for i in range(time):
            time_steps.append(i)
            result = self.system_map()
            
            # Update state
            self.state = {'S': result['S_nondim'], 'E': result['E_nondim'], 'F': result['F_nondim'], 'FP': result['FP_nondim']}
            if self.type == "nondimensionalized":
                # Update nondimensionalized state variables
                seafood_nondim = np.append(seafood_nondim, result['S_nondim'])
                effort_nondim = np.append(effort_nondim, result['E_nondim'])
                fraudsters_nondim = np.append(fraudsters_nondim, result['F_nondim'])
                p_fraudsters_nondim = np.append(p_fraudsters_nondim, result['FP_nondim'])
                
                nondim_market_price = np.append(nondim_market_price, result['nondim_market_price'])
                nondim_wholesale_price = np.append(nondim_wholesale_price, result['nondim_wholesale_price'])
            elif self.type == "dimensionalized":
                # Update dimensionalized state variables
                seafood_dimful = np.append(seafood_dimful, result['S_dimful'])
                effort_dimful = np.append(effort_dimful, result['E_dimful'])
                fraudsters_dimful = np.append(fraudsters_dimful, result['F_dimful'])
                p_fraudsters_dimful = np.append(p_fraudsters_dimful, result['FP_dimful'])            
                
                market_price_arr = np.append(market_price_arr, result['market_price'])
                wholesale_price_arr = np.append(wholesale_price_arr, result['wholesale_price'])
                harvest_arr = np.append(harvest_arr, result['harvest'])
        time_steps.append(time)
        
        if self.type == "nondimensionalized":
            return {
                'Seafood': seafood_nondim,
                'Effort': effort_nondim,
                'Fraudsters': fraudsters_nondim,
                'Perception of Fraud': p_fraudsters_nondim,
                'Nondim Market Price': nondim_market_price,
                'Nondim Wholesale Price': nondim_wholesale_price,
            }
        elif self.type == "dimensionalized":
            return {
                'Seafood': seafood_dimful,
                'Effort': effort_dimful,
                'Fraudsters': fraudsters_dimful,
                'Perception of Fraud': p_fraudsters_dimful,
                'Market Price': market_price_arr,
                'Wholesale Price': wholesale_price_arr,
                'Harvest': harvest_arr,
            }
        else:
            raise ValueError(f"Invalid system type: {self.type}")
        
    # SCENARIOS
    def bioeconomic_bifucation_plot(self, r_range=(0, 4), resolution=500, time=500, burn_in=None):
        '''
        Compute a bifurcation diagram by sweeping the intrinsic growth rate r.

        For each r value the system is run from the current initial state for
        `time` steps.  The first `burn_in` steps are discarded as transient;
        the remaining attractor points are collected.  The method restores the
        original params/state before returning.

        Args:
            r_range  (tuple): (min_r, max_r) inclusive range for the sweep.
            resolution (int): Number of r values to sample across r_range.
            time       (int): Number of time steps to simulate per r value.
            burn_in    (int): Steps to discard as transient (default: time // 2).

        Returns:
            dict:
                'r' (np.ndarray): Bifurcation parameter value for every
                                  attractor point (shape: resolution * attractor_len).
                'S' (np.ndarray): Seafood biomass attractor points matching 'r'.
        '''
        if burn_in is None:
            burn_in = time // 2

        r_values = np.linspace(r_range[0], r_range[1], resolution)
        saved_state = {k: v for k, v in self.state.items()}
        saved_params = self.params.copy()

        r_bif = []
        S_bif = []

        for r in r_values:
            params = saved_params.copy()
            params['r'] = r
            self.params = params
            self.state = {k: v for k, v in saved_state.items()}

            S_trajectory = [self.state['S']]
            for _ in range(time):
                result = self.system_map()
                if self._type == "dimensionalized":
                    self.state = {
                        'S': result['S_dimful'],
                        'E': result['E_dimful'],
                        'F': result['F_dimful'],
                        'FP': result['FP_dimful'],
                    }
                    S_trajectory.append(result['S_dimful'])
                else:
                    self.state = {
                        'S': result['S_nondim'],
                        'E': result['E_nondim'],
                        'F': result['F_nondim'],
                        'FP': result['FP_nondim'],
                    }
                    S_trajectory.append(result['S_nondim'])

            attractor = S_trajectory[burn_in:]
            r_bif.extend([r] * len(attractor))
            S_bif.extend(attractor)

        self.params = saved_params
        self.state = saved_state

        return {
            'r': np.array(r_bif, dtype=np.float64),
            'S': np.array(S_bif, dtype=np.float64),
        }
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
    def type(self):
        return self._type
        
    @property
    def nondim_params(self):
        params = self.params.copy()
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
        