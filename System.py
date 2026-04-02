import numpy as np
import warnings

CLOSE_TO_ZERO = np.finfo(np.float128).eps
CLOSE_TO_ONE = 1 - np.finfo(np.float128).epsneg
POSITIVE_INF = np.inf
DEFAULT_PARAMS = {
    'gamma_m': 10.0,
    'gamma_p': 1.0,
    'gamma_s': 1.0,
    'gamma_e': 0.225,
    'gamma_f': 1.0,
    'gamma_fp': 1.0,
    
    'e_d': 1.0,
    'e_sw': 1.0,
    'e_sm': 1.0,
    
    'K': 1.0,
    'F_threshold': 0.5,
    'r': 0.225,
    
    'q0': 0.07,
    'q1': 0.15,
    'pw0': 1.0,
    'pw1': 0.81,
    'c0': 0.9,
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
                * q0 (float128): Catchability coefficient when no fraudsters are present.
                * q1 (float128): Catchability coefficient when fraudsters are present.
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
        gamma_s = self.params['gamma_s']
        
        q = self.catchability()
        
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
        gamma_e = self.params['gamma_e']
                
        revenue = self.revenue_per_unit_effort()
        cost = self.cost_per_unit_effort()
                
        E_next = np.clip(
            [E * np.exp(gamma_e * (revenue - cost))],
            CLOSE_TO_ZERO,
            POSITIVE_INF
        )[0]
        
        return E_next
    def fraudster_state_dimful(self):
        F = self.state['F']
        
        if F == 1.0:
            return 1.0
        if F == 0.0:
            return 0.0
        
        gamma_f = self.params['gamma_f']
        
        pm = self.market_price()
        pw = self.wholesale_price()
        delta = gamma_f * (pm - pw)
        
        return np.clip([F * np.exp(delta) / (1 + F * (np.exp(delta) - 1))], CLOSE_TO_ZERO, CLOSE_TO_ONE)[0]
    def p_fraudster_state_dimful(self):
        F = self.state['F']
        FP = self.state['FP']
        
        if FP == 1.0:
            return 1.0
        if FP == 0.0:
            return 0.0
        
        F_threshold = self.params['F_threshold']
        gamma_fp = self.params['gamma_fp']
        exp_delta_fp = np.exp(gamma_fp * (F - F_threshold))
        
        return np.clip([FP * exp_delta_fp / (1 + FP * (exp_delta_fp - 1))], CLOSE_TO_ZERO, CLOSE_TO_ONE)[0]
    
    # VARIABLES (dimensionful)
    def catchability(self):
        q0 = self.params['q0']
        q1 = self.params['q1']
        F = self.state['F']
        return q0 + (q1 - q0) * F
    def harvest(self):
        S = self.state['S']
        E = self.state['E']
        q = self.catchability()
        
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
        gamma_m = self.params['gamma_m']
        e_d = self.params['e_d']
        e_sm = self.params['e_sm']
        H = self.harvest()
        
        '''
            Artificially create a floor near 0+.
            Reduces risk of numerical imprecisions 
            (and values reaching areas they shouldn't reach).
        '''
        return np.clip([np.sqrt((1-FP)**e_d / H**e_sm) * gamma_m], CLOSE_TO_ZERO, POSITIVE_INF)[0]
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
    def revenue_per_unit_effort(self):
        q = self.catchability()
        Pw = self.wholesale_price()
        S = self.state['S']
        E = self.state['E']
        return q * Pw * S
    def cost_per_unit_effort(self):
        F = self.state['F']
        c0 = self.params['c0']
        c1 = self.params['c1']   
        return F * (c1 - c0) + c0
      
    def system_map(self) -> dict:        
        '''
        Get system's values for the next time step.
        Only computes the state update matching self.type to avoid
        calling nondim/dimful functions with incompatible params.
        '''
        market_price = self.market_price()
        wholesale_price = self.wholesale_price()
        catchability = self.catchability()
        revenue = self.revenue_per_unit_effort()
        cost = self.cost_per_unit_effort()
        harvest = self.harvest()
        demand = self.demand()
        
        if self.type == "dimensionalized":
            S_next = self.seafood_state_dimful()
            E_next = self.effort_state_dimful()
            F_next = self.fraudster_state_dimful()
            FP_next = self.p_fraudster_state_dimful()
        else:
            S_next = self.seafood_state_nondim()
            E_next = self.effort_state_nondim()
            F_next = self.fraudster_state_nondim()
            FP_next = self.p_fraudster_state_nondim()
        
        return {
            'S': S_next,
            'E': E_next,
            'F': F_next,
            'FP': FP_next,
            'market_price': market_price,
            'wholesale_price': wholesale_price,
            'catchability': catchability,
            'revenue_per_unit_effort': revenue,
            'cost_per_unit_effort': cost,
            'harvest': harvest,
            'demand': demand,
        }
    def time_series_plot(self, time, title="", x_label="", y_label="", ax=None) -> dict:
        seafood = np.array(self.state['S'], dtype=np.float128)
        effort = np.array(self.state['E'], dtype=np.float128)
        fraudsters = np.array(self.state['F'], dtype=np.float128)
        p_fraudsters = np.array(self.state['FP'], dtype=np.float128)
        harvest_arr = np.array(self.harvest(), dtype=np.float128)
        market_price_arr = np.array(self.market_price(), dtype=np.float128)
        wholesale_price_arr = np.array(self.wholesale_price(), dtype=np.float128)
        
        for i in range(time):
            result = self.system_map()
            self.state = {
                'S': result['S'], 'E': result['E'],
                'F': result['F'], 'FP': result['FP'],
            }
            
            seafood = np.append(seafood, result['S'])
            effort = np.append(effort, result['E'])
            fraudsters = np.append(fraudsters, result['F'])
            p_fraudsters = np.append(p_fraudsters, result['FP'])
            market_price_arr = np.append(market_price_arr, result['market_price'])
            wholesale_price_arr = np.append(wholesale_price_arr, result['wholesale_price'])
            harvest_arr = np.append(harvest_arr, result['harvest'])
        
        return {
            'Seafood': seafood,
            'Effort': effort,
            'Fraudsters': fraudsters,
            'Perception of Fraud': p_fraudsters,
            'Market Price': market_price_arr,
            'Wholesale Price': wholesale_price_arr,
            'Harvest': harvest_arr,
        }
        
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
        