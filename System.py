import numpy as np
import warnings

class DynamicalSystem():
    # STATE VARIABLES (nondimensionalized)
    def _seafood_state(self):
        S = self.state['S']
        E = self.state['E']
        gamma_s = self.nondim_params['gamma_s']
        
        '''
            Artificially create a floor near 0+.
            Reduces risk of numerical imprecisions 
            (and values reaching areas they shouldn't reach).
        '''
        S_next = np.clip([S * np.exp(gamma_s * (1 - S - E))], np.finfo(type(S)).eps, np.inf)[0]
        
        # Update state and return
        self.state['S'] = S_next
        return S_next
    def _effort_state(self): 
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
        E_next = np.clip([E * np.exp(gamma_e * (term_2 - term_3))], np.finfo(type(S)).eps, np.inf)[0]
        
        # Update state and return
        self.state['E'] = E_next
        return E_next
    def _fraudster_state(self):
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
        F_next = np.clip([(F * np.exp(delta)) / (1 + F * (np.exp(delta) - 1))], np.finfo(type(S)).eps, 1.0 - np.finfo(type(S)).epsneg)[0]
        
        # Update state and return
        self.state['F'] = F_next
        return F_next
    def _p_fraudster_state(self): 
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
        FP_next = np.clip([(FP * exp_delta_fp) / (1 + FP * (exp_delta_fp - 1))], np.finfo(type(F)).eps, 1 - np.finfo(type(F)).epsneg)[0]
        
        # Update state and return
        self.state['FP'] = FP_next
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
        e_ms = self.params['e_ms']
        e_d = self.params['e_d']
        
        H = self.harvest()
        
        '''
            Artificially create a floor near 0+.
            Reduces risk of numerical imprecisions 
            (and values reaching areas they shouldn't reach).
        '''
        return np.clip([np.sqrt((FP)**e_d * H**e_ms)], np.finfo(type(FP)).eps, np.inf)[0]
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
        H = self.harvest
        
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
        self.params = params
        self.nondim_params = {
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
 