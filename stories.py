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
    
    return np.max([(F * np.exp(delta)) / (1 + F * (np.exp(delta) - 1)), lowest_point])
def p_fraudster_state(F, Fp, F_threshold): 
    delta_fp = legacy_gamma_fp * (F - F_threshold)
    exp_delta_fp = np.exp(delta_fp)

    return np.max([(Fp * exp_delta_fp) / (1 + Fp * (exp_delta_fp - 1)), lowest_point])


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
                    if surface_label_exists:
                        ax.legend()
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