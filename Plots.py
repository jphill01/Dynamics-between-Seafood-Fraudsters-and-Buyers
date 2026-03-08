import numbers
import numpy as np
import matplotlib.pyplot as plt

from System import DynamicalSystem

class Plots():
    @staticmethod
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
    @staticmethod
    def bifurcation_plot(ax, init_vals, params, param_name, param_linspace, time, y_state_var):
        transient = int(time - (0.25 * time))
        
        def run_system(system, steps=time):
            trajectory = []
            next = init_vals
            
            for t in range(steps):
                # Update
                next = system.system_map(next)
                S, E, F, FP = next
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
            
            system = DynamicalSystem(current_params)
                    
            # Run
            points = run_system(system)
            
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
        