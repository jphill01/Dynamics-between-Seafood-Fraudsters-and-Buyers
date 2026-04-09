# Dynamics between Seafood Fraudsters and Buyers

A discrete-time dynamical systems model exploring how seafood fraud propagates through supply chains and how buyer awareness, market forces, and fishing practices create feedback loops that either stabilize or collapse fish stocks.

## Motivation

Seafood is one of the most heavily traded food commodities globally, supplying roughly a quarter of the world's animal protein intake. The complexity of global supply chains makes it difficult to ensure the authenticity and traceability of seafood products, creating opportunities for fraud — mislabelling, substitution, adulteration, and unethical harvesting. This project models the interactions between key players in the supply chain to understand when fraud self-corrects and when it spirals out of control.

## The Model

The system tracks four state variables over discrete time steps:

| Variable | Symbol | Range | Description |
|----------|--------|-------|-------------|
| **Seafood biomass** | $S$ | $(0, K]$ | Fish stock level, normalized to carrying capacity $K = 1$ |
| **Fishing effort** | $E$ | $(0, \infty)$ | Magnitude of harvesting effort by fishers |
| **Fraudster share** | $F$ | $[0, 1]$ | Proportion of wholesalers engaged in fraud |
| **Fraud perception** | $F^p$ | $[0, 1]$ | Proportion of buyers aware of fraudulent activity |

These interact through four coupled mechanisms:

1. **Seafood & Effort** — Effort grows when fishing is profitable (revenue > cost); higher effort depletes fish stocks, which recover via logistic growth at intrinsic rate $r$.
2. **Fishers & Wholesalers** — Fraudulent wholesalers enable illegal fishing methods, lowering costs and wholesale prices. Fraud grows when the gap between market price and wholesale price widens.
3. **Wholesalers & Buyers** — Buyers detect fraud when $F$ exceeds a threshold $\hat{F}$. Rising perception of fraud reduces demand and depresses market prices, squeezing fraudster margins.
4. **Feedback loop** — Lower market prices reduce the incentive for fraud, allowing perception to decay, demand to recover, and the cycle to potentially repeat.

The full equations and derivations are presented in the Streamlit app (sourced from `text.py`).

## Repository Structure

| File | Purpose |
|------|---------|
| `System.py` | Core `DynamicalSystem` class — implements both **nondimensionalized** and **dimensionalized** forms of the model, with methods for single-step mapping and time series generation |
| `app.py` | Streamlit web app — renders the model introduction and visualizations |
| `new_research.py` | Research script — runs 5 scenario experiments with time series, bifurcation diagrams, and return maps via matplotlib |
| `text.py` | Markdown narrative and LaTeX equations displayed in the Streamlit app |
| `requirements.txt` | Pinned Python dependencies |

## Research Scenarios

`new_research.py` investigates five scenarios using the dimensionalized model, each with time series grids, bifurcation diagrams, and return maps:

| # | Scenario | Focus Parameter | What It Tests |
|---|----------|-----------------|---------------|
| 1 | **Baseline (no fraud)** | Intrinsic growth rate $r$ | Pure bioeconomic S–E dynamics; onset of oscillations and chaos as $r$ increases |
| 2 | **Prized / protected seafood** | Fraudster wholesale price $pw_1$ | What happens when illegal catch commands a premium over legal catch |
| 3 | **Blast / cyanide fishing** | Destruction intensity $\alpha$ | Joint effect of higher catchability, lower costs, and lower wholesale prices from destructive methods |
| 4 | **EEZ non-enforcement** | Violation intensity $\beta$ | Fishers access outside-EEZ waters — higher catchability but higher costs |
| 5 | **Buyer dependence** | Demand elasticity $\varepsilon_d$ | When buyers need seafood regardless of fraud, the self-correcting mechanism breaks down; includes a heatmap over $(\varepsilon_d, \hat{F})$ space |

## Parameters

Key parameters and their roles:

| Parameter | Description |
|-----------|-------------|
| $\gamma_s, \gamma_e, \gamma_f, \gamma_{fp}$ | Response speeds for seafood, effort, fraudsters, and fraud perception |
| $\gamma_m, \gamma_p$ | Market price scaling and wholesale price scaling |
| $r$ | Intrinsic growth rate of seafood biomass |
| $K$ | Carrying capacity (normalized to 1) |
| $q_0, q_1$ | Catchability when fraud is 0% vs 100% (dimensionalized). $q_0, q_1$ not considered in nondimensionalized system; combined as just $q$ (todo) |
| $pw_0, pw_1$ | Wholesale price at 0% vs 100% fraud |
| $c_0, c_1$ | Fishing cost at 0% vs 100% fraud |
| $\varepsilon_d$ | Demand elasticity — buyer sensitivity to perceived fraud |
| $\varepsilon_{s,w}, \varepsilon_{s,m}$ | Supply elasticities for wholesale and market prices |
| $\hat{F}$ | Fraud detection threshold — minimum fraud level buyers can perceive |

Default parameters (dimensionalized, from `System.py`):

```python
{
    'gamma_m': 10.0, 'gamma_p': 1.0, 'gamma_s': 1.0,
    'gamma_e': 0.225, 'gamma_f': 1.0, 'gamma_fp': 1.0,
    'e_d': 1.0, 'e_sw': 1.0, 'e_sm': 1.0,
    'K': 1.0, 'F_threshold': 0.5, 'r': 0.225,
    'q0': 0.07, 'q1': 0.15,
    'pw0': 1.0, 'pw1': 0.81,
    'c0': 0.9, 'c1': 0.153,
}
```

## Stability Analysis

The `DynamicalSystem` class includes numerical tools for local stability analysis of the 4D discrete map. Three methods work together:

### `find_fixed_point(initial_guess=None, warmup_steps=500, tol=1e-10)`

Finds a fixed point $x^*$ satisfying $G(x^*) = x^*$ by solving the root-finding problem $G(x) - x = 0$.

**Algorithm:** `scipy.optimize.fsolve` — MINPACK's `hybrd` routine, a modified Powell hybrid method that switches between Newton steps and scaled gradient-descent steps. By default, the solver is warm-started by simulating the map forward 500 steps from the current state to get close to the attractor before refinement.

### `jacobian(state=None, h=None)`

Computes the $4 \times 4$ Jacobian matrix $J$ of the map at a given state using **central finite differences**:

$$J_{ji} = \frac{G_j(x + h \, e_i) - G_j(x - h \, e_i)}{2h}$$

The default perturbation $h = \varepsilon^{1/3} \cdot \max(1, |x_i|)$ where $\varepsilon \approx 2.2 \times 10^{-16}$ (float64 machine epsilon) gives $O(h^2)$ accuracy — the optimal trade-off between truncation and round-off error.

### `stability_analysis(initial_guess=None, warmup_steps=500, tol=1e-10)`

Orchestrates the full workflow: finds the fixed point, computes the Jacobian, and extracts eigenvalues via `numpy.linalg.eig` (LAPACK's `dgeev` — implicit QR iteration).

For a discrete-time map, the fixed point is **stable** when the spectral radius $\rho = \max_i |\lambda_i| < 1$ (all eigenvalues inside the unit circle) and **unstable** when $\rho > 1$. The transition at $\rho = 1$ through a complex conjugate pair is a **Neimark-Sacker bifurcation** (the discrete-time analog of a Hopf bifurcation).

**Example:**

```python
from System import DynamicalSystem, DEFAULT_PARAMS
import numpy as np

params = DEFAULT_PARAMS.copy()
params.update({'pw1': 1.5, 'c1': 0.9, 'q1': 0.07})
state = {'S': np.float128(0.6), 'E': np.float128(0.3),
         'F': np.float128(0.1), 'FP': np.float128(0.1)}
sys = DynamicalSystem(params, state, "dimensionalized")

result = sys.stability_analysis()
print(f"Spectral radius: {result['spectral_radius']:.4f}")
print(f"Stable: {result['stable']}")
print(f"Classification: {result['classification']}")
```

## Getting Started

**Requirements:** Python 3.10+

```bash
pip install -r requirements.txt
```

**Run the Streamlit app:**

```bash
streamlit run app.py
```

**Run the research scenarios** (generates matplotlib figures interactively):

```bash
python new_research.py
```

Toggle individual scenarios on/off by changing the `if True` / `if False` guards at the top of each section in `new_research.py`.

## References

To be finished
