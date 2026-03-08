import numpy as np
import warnings

LOWEST_POINT = 1e-12
legacy_gamma_fp = 1.0
class DynamicalSystem():
    # STATE VARIABLES
    @staticmethod
    def seafood_state(S, E, gamma_s): 
        return np.max([S * np.exp(gamma_s * (1 - S - E)), -1*np.inf])
    @staticmethod
    def effort_state(S, E, F, q, pw, c, e_sw, gamma_p, gamma_e): 
        term_1 = (F * (pw - 1) + 1)
        denom_e = (gamma_p * E * S ) ** e_sw
        term_2 = (q * S * (term_1 / denom_e))
        term_3 = (F * (c - 1)) + 1
        
        return np.max([E * np.exp(gamma_e * (term_2 - term_3)), -1*np.inf])
    @staticmethod
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
        
        return np.max([(F * np.exp(delta)) / (1 + F * (np.exp(delta) - 1)), LOWEST_POINT])
    @staticmethod
    def p_fraudster_state(F, Fp, F_threshold): 
        delta_fp = legacy_gamma_fp * (F - F_threshold)
        exp_delta_fp = np.exp(delta_fp)

        return np.max([(Fp * exp_delta_fp) / (1 + Fp * (exp_delta_fp - 1)), LOWEST_POINT])
    
    # VARIABLES
    @staticmethod
    def demand(Fp, H, e_d, e_ms):
        return np.sqrt((1-Fp)**e_d*H**e_ms)
    @staticmethod
    def demand(Fp, H, e_d, e_ms):
        return np.sqrt((1-Fp)**e_d*H**e_ms)
    
    
    def __init__(self, params, state):
        # Nondim params
        self.params = {
            'gamma_m': params['gamma_m'] / (params['pw0'] * (params['r'] * params['K']) ** (params['e_sm'] / 2.0)),
            'gamma_f': params['gamma_f'] * params['pw0'],
            'gamma_s': params['gamma_s'] * params['r'],
            'gamma_e': params['gamma_e'] * params['c0'],
            'gamma_p': params['gamma_p'] * params['r'] * params['K'],
            'e_sm': params['e_sm'], 'e_sw': params['e_sw'], 'e_d': params['e_d'],
            'F_threshold': params['F_threshold'],
            'q': (params['q'] * params['pw0'] * params['K']) / params['c0'],
            'pw': params['pw1'] / params['pw0'],
            'c': params['c1'] / params['c0'],
        }
        self.state = state
    
    def system_map(self, state):
        S, E, F, FP = state
        
        # Calculate next steps
        S_new = DynamicalSystem.seafood_state(S, E, self.params["gamma_s"])
        E_new = DynamicalSystem.effort_state(S, E, F, self.params['q'], self.params['pw'], self.params['c'], self.params['e_sw'], self.params['gamma_p'], self.params['gamma_e'])
        F_new = DynamicalSystem.fraudster_state(S, E, F, FP, self.params['gamma_f'], self.params['gamma_m'], self.params['gamma_p'], self.params['pw'], self.params['e_sw'], self.params['e_sm'], self.params['e_d'])
        Fp_new = DynamicalSystem.p_fraudster_state(F, FP, self.params['F_threshold'])
        return np.array([S_new, E_new, F_new, Fp_new])
 