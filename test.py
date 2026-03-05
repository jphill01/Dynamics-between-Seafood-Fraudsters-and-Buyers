import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from matplotlib.colors import ListedColormap

# --- 1. System Definition ---
bump = 1e-8

def system_step(x, p):
    """Calculates x_{t+1} given x_t and parameters p."""
    S, E, F, Fp = x
    
    # Seafood state
    S_next = S * np.exp(p['gamma_s'] * (1 - S - E))
    
    # Effort state
    term_1 = (F * (p['pw'] - 1) + 1)
    denom_e = (p['gamma_p'] * E * S + bump) ** p['e_sw']
    term_2 = (p['q'] * S * (term_1 / denom_e))
    term_3 = (F * (p['c'] - 1)) + 1
    E_next = E * np.exp(p['gamma_e'] * (term_2 - term_3))
    
    # Fraudster state
    denom_market = (E * S)**(p['e_sm']/2) + bump
    price_market = p['gamma_m'] * ((1 - Fp)**(p['e_d']/2) / denom_market)
    denom_wholesale = (p['gamma_p'] * E * S)**(p['e_sw']) + bump
    price_wholesale = (F * (p['pw'] - 1) + 1) / denom_wholesale
    delta = p['gamma_f'] * (price_market - price_wholesale)
    delta = np.clip(delta, -20, 20)
    F_next = (F * np.exp(delta)) / (1 + F * (np.exp(delta) - 1))
    
    # P_Fraudster state
    delta_fp = p['legacy_gamma_fp'] * (F - p['F_threshold'])
    exp_delta_fp = np.exp(delta_fp)
    Fp_next = (Fp * exp_delta_fp) / (1 + Fp * (exp_delta_fp - 1))
    
    return np.array([S_next, E_next, F_next, Fp_next])

def root_func(x, p):
    """The function we want to find the root for: G(x) - x = 0"""
    return system_step(x, p) - x

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

# --- 2. Parameter Setup ---
params = {
    'gamma_m': 1.0, 'gamma_f': 1.0, 'gamma_s': 1.0, 'gamma_e': 1.0,
    'gamma_p': 1.0, 'gamma_fp': 1.0, 'e_d': 1.0, 'e_sw': 1.0,
    'e_sm': 1.0, 'K': 1.0, 'F_threshold': 0.5, 'q': 1.0,
    'r': 0.225, 'pw0': 1.0, 'c0': 0.17
}

resolution = 2
pw1_vals = np.linspace(0.01, 1, resolution)
c1_vals = np.linspace(0.01, 0.17, resolution)

# Calculate base nondimensionalized params that don't change
base_p = {
    'gamma_m': params['gamma_m'] / (params['pw0'] * (params['r'] * params['K']) ** (params['e_sm'] / 2.0)),
    'gamma_f': params['gamma_f'] * params['pw0'],
    'gamma_s': params['gamma_s'] * params['r'],
    'gamma_e': params['gamma_e'] * params['c0'],
    'gamma_p': params['gamma_p'] * params['r'] * params['K'],
    'e_sm': params['e_sm'], 'e_sw': params['e_sw'], 'e_d': params['e_d'],
    'F_threshold': params['F_threshold'], 'legacy_gamma_fp': params['gamma_fp'],
    'q': (params['q'] * params['pw0'] * params['K']) / params['c0']
}

# --- 3. Grid Evaluation ---
stability_map = np.zeros((resolution, resolution))

# Initial guess for the first root-finding attempt
current_guess = np.array([0.5, 26, 0.5, 0.5])


p = base_p.copy()
p['pw'] = 0.01 / params['pw0']
p['c'] = 0.001 / params['c0']
simulated_x = np.array([0.5, 26, 0.5, 0.5])
for _ in range(100):
    simulated_x = system_step(simulated_x, p)
print(simulated_x)
res = least_squares(
    root_func, 
    simulated_x, 
    args=(p,), 
    bounds=(0, np.inf), 
    method='trf'
)
print(res.x, res.message, res.success)

# for i, c1 in enumerate(c1_vals):
#     for j, pw1 in enumerate(pw1_vals):
#         # Update dynamic parameters
#         p = base_p.copy()
#         p['pw'] = pw1 / params['pw0']
#         p['c'] = c1 / params['c0']
        
#         # 1. Find the fixed point
#         # Using the guess from the previous loop iteration speeds this up immensely
#         x_star, infodict, ier, mesg = fsolve(root_func, current_guess, args=(p,), full_output=True)
        
#         # If fsolve didn't converge, mark as divergent/unknown
#         if ier != 1:
#             print(mesg)
#             stability_map[i, j] = 3 
#             continue
            
#         # Update our guess for the next iteration to be this fixed point
#         current_guess = x_star
        
#         # Check if the fixed point represents biological extinction (S <= 0)
#         if x_star[0] <= 1e-4:
#             stability_map[i, j] = 0
#             continue
            
#         # 2. Compute Jacobian at the fixed point using the transition function G(x)
#         J = get_numerical_jacobian(system_step, x_star, p)
        
#         # 3. Calculate eigenvalues
#         eigenvalues = np.linalg.eigvals(J)
#         max_eigenvalue_mag = np.max(np.abs(eigenvalues))
        
#         # 4. Classify based on spectral radius
#         if max_eigenvalue_mag < 1.0:
#             stability_map[i, j] = 1  # Stable
#         else:
#             stability_map[i, j] = 2  # Unstable (Oscillatory/Chaotic)

# --- 4. Plotting ---
plt.figure(figsize=(10, 7))

# Same colormap: Extinct (Black), Stable (Blue), Unstable (Orange), Divergent/No Root (Red)
cmap = ListedColormap(['black', 'dodgerblue', 'darkorange', 'crimson'])

pw1_grid, c1_grid = np.meshgrid(pw1_vals, c1_vals)
plt.pcolormesh(pw1_grid, c1_grid, stability_map, cmap=cmap, shading='auto')

plt.colorbar(ticks=[0.375, 1.125, 1.875, 2.625], 
             format=plt.FuncFormatter(lambda val, loc: ['Extinct', 'Stable', 'Unstable', 'Divergent'][int(val)]))

plt.title('Jacobian Eigenvalue Stability Map', fontsize=14)
plt.xlabel('pw1 (Wholesale Price Parameter)', fontsize=12)
plt.ylabel('c1 (Cost Parameter)', fontsize=12)
plt.tight_layout()
# plt.show()