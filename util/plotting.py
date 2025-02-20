
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_times2(test_names, var, mean_times_data, std_times_data, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
   
    palette = sns.color_palette("Set1", 6)  
    plot_styles = {
        'naive': (palette[0], 'o', 'Contract-Then-Compress', '--'),   # First color, circle marker, dashed line
        'random': (palette[1], 's', 'Randomized', '-'),  # Second color, square marker, solid line
        'zipup': (palette[2], 'D', 'Zip-up', '-'),       # Third color, diamond marker, solid line
        'density': (palette[3], '^', 'Density Matrix', '-'), # Fourth color, triangle up marker, solid line
        'fit': (palette[4], 'v', 'Fitting', '-'),        # Fifth color, triangle down marker, solid line
        'rand_then_orth': ('g', '^', 'Randomized C-T-C', '--'),      
        'nyst': ('b', 'v', 'Nystrom Contraction', '--'),
        'random+oversample': ('c', 'v', 'random+oversample', '--'),

    }

    for name in test_names:
        if name in plot_styles and name in mean_times_data:
            color, marker, label, linestyle = plot_styles[name]
            mean_times = np.array(mean_times_data[name])
            std_times = np.array(std_times_data[name])
            ax.plot(var, mean_times, color=color, marker=marker, label=label, markersize=6, linewidth=1.5, linestyle=linestyle)
            ax.fill_between(var, mean_times - std_times, mean_times + std_times, color=color, alpha=0.3)
            
    ax.set_yscale('log')
    ax.grid()
    ax.set_xlabel(r'Fixed Bond Dimension $\overline{\chi}$', fontsize=16)
    ax.set_ylabel('Runtime', fontsize=16)
    #ax.legend(fontsize=11, ncol=2)
    ax.tick_params(axis='both', which='major', labelsize=16)
    return ax
    
def plot_accuracy2(test_names, var, mean_acc_data, std_acc_data, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
        
    palette = sns.color_palette("Set1", 6)  
    plot_styles = {
        'naive': (palette[0], 'o', 'Contract-Then-Compress', '-'),   # First color, circle marker, dashed line
        'random': (palette[1], 's', 'Randomized', '--'),  # Second color, square marker, solid line
        'zipup': (palette[2], 'D', 'Zip-up', '-'),       # Third color, diamond marker, solid line
        'density': (palette[3], '^', 'Density Matrix', '-'), # Fourth color, triangle up marker, solid line
        'fit': (palette[4], 'v', 'Fitting', '-'),        # Fifth color, triangle down marker, solid line
        'rand_then_orth': ('g', '^', 'Randomized C-T-C', '--'),      
        'nyst': ('b', 'v', 'Nystrom Contraction', '--'),
        'random+oversample': ('c', 'v', 'rrandom+oversample', '--'),

    }

    for name in test_names:
        if name in plot_styles and name in mean_acc_data:
            color, marker, label, linestyle = plot_styles[name]
            mean_acc = np.array(mean_acc_data[name])
            std_acc = np.array(std_acc_data[name])
            ax.plot(var, mean_acc, color=color, marker=marker, label=label, markersize=6, linewidth=1.5, linestyle=linestyle)
            ax.fill_between(var, mean_acc - std_acc, mean_acc + std_acc, color=color, alpha=0.3)
      
    ax.set_yscale('log')
    ax.grid()
    ax.set_xlabel(r'Fixed Bond Dimension $\overline{\chi}$', fontsize=16)
    ax.set_ylabel('Relative Error', fontsize=16)
    # ax.legend(fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=16)
    return ax
    
def plot_generic(ylabel, title, data_arrays, labels=None, log_scale=False,show_legend=True):
    """
    Plots multiple data arrays against iterations.
    
    Args:
    - ylabel (str): Label for the y-axis.
    - title (str): Title for the plot.
    - data_arrays (list of arrays): List containing data arrays.
    - labels (list of str, optional): Labels for each data array.
    - log_scale (bool, optional): Whether to use a logarithmic y-scale.
    
    """
    # Ensure that all arrays are of the same length
    assert all(len(data_array) == len(data_arrays[0]) for data_array in data_arrays), "All data arrays must be of the same length"
    
    iterations = np.arange(1, len(data_arrays[0]) + 1)
    
    plt.figure( dpi=150)
    # Line styles for the three algorithms
    line_styles = ['-', ':', '--']
    
    for idx, (data_array, label) in enumerate(zip(data_arrays, labels)):
        # Choose line style based on the current index (0, 1, 2)
        style = line_styles[idx % 3]
        plt.plot(iterations, data_array, label=label, marker='o', linestyle=style)
        
    plt.xlabel('Iteration')
    plt.ylabel(ylabel)
    if log_scale:
        title += " (log scale)"
        plt.yscale('log')
    plt.title(title)
    plt.legend()
    #if show_legend:
        #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.show()

def plot_runtimes(*times, labels=None,log_scale=False,show_legend=True):
    """
    Plots runtimes against iterations.
    
    Args:
    - *times (tuple of arrays): Multiple arrays containing runtimes.
    - labels (list of str, optional): Labels for each runtime array.
    
    """
    plot_generic('Time (s)', 'Runtime Comparison for different methods', *times, labels=labels,log_scale=log_scale,show_legend=show_legend)

def plot_bond_dimensions(*bond_dims, labels=None,show_legend=True):
    """
    Plots bond dimensions against iterations.
    
    Args:
    - *bond_dims (tuple of arrays): Multiple arrays containing bond dimensions.
    - labels (list of str, optional): Labels for each bond dimension array.
    
    """

    plot_generic( 'Max Bond Dimension', 'Max Bond Dimension for different contraction methods', *bond_dims, labels=labels,show_legend=show_legend)

def plot_norms(*norms, labels=None,show_legend=True):
    """
    Plots norms against iterations.
    
    Args:
    - *norms (tuple of arrays): Multiple arrays containing norms.
    - labels (list of str, optional): Labels for each norm array.

    """

    plot_generic('Norm Difference', 'Norm Difference between contraction methods', *norms, labels=labels,show_legend=show_legend)

def plot_all(times,bonds,norms,labels,show_legend=True):
    #plot_runtimes(times, labels=labels,show_legend=show_legend)
    plot_runtimes(times, labels=labels,log_scale=True,show_legend=show_legend)
    plot_bond_dimensions(bonds, labels=labels,show_legend=show_legend)
    plot_norms(norms, labels=labels,show_legend=show_legend)
    
def compare_algorithms(times_list, bonds_list, norms_list, labels_list,show_legend=True):
    """
    Plots data sets for multiple algorithms on the same plots using plot_all.
    
    Args:
    - times_list (list of list of arrays): Each list contains time data arrays for a particular algorithm.
    - bonds_list (list of list of arrays): Each list contains bond dimension data arrays for a particular algorithm.
    - norms_list (list of list of arrays): Each list contains norms data arrays for a particular algorithm.
    - labels_list (list of list of str): Each list contains labels for a particular algorithm.
    """

    # Combine data from each algorithm
    all_times = [time for sublist in times_list for time in sublist]
    all_bonds = [bond for sublist in bonds_list for bond in sublist]
    all_norms = [norm for sublist in norms_list for norm in sublist]
    all_labels = [label for sublist in labels_list for label in sublist]

    # Plot all data on the same plots
    plot_all(all_times, all_bonds, all_norms, all_labels,show_legend=show_legend)

def plot_time_evolution(time_steps, magnetization_x_values, rmagnetization_x_values,
                        magnetization_y_values, rmagnetization_y_values,
                        magnetization_z_values, rmagnetization_z_values,
                        baseline_times, random_times,
                        baseline_bonds, random_bonds,
                        random_accs):
    
    # Create subplots
    fig, axs = plt.subplots(2, 3, figsize=(17, 7))
    
    # Title with number of steps
    fig.text(0.05, 1.05, f'Number of steps: {len(time_steps)}', fontsize=12,
             verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', edgecolor='0.3'))
    
    # Plot X Magnetization
    axs[0,0].plot(time_steps, magnetization_x_values, color='red', label='mps-mpo')
    axs[0,0].plot(time_steps, rmagnetization_x_values, color='orange', label='Random', linestyle='dashed')
    axs[0,0].set_title('X Magnetization at Last Site')
    axs[0,0].set_xlabel('Time')
    axs[0,0].set_ylabel(r'$\langle \sigma_{x}^{(N-1)} \rangle$')
    axs[0,0].legend()

    # Plot Y Magnetization
    axs[0,1].plot(time_steps, magnetization_y_values, color='green', label='mps-mpo')
    axs[0,1].plot(time_steps, rmagnetization_y_values, color='lightgreen', label='Random', linestyle='dashed')
    axs[0,1].set_title('Y Magnetization at Last Site')
    axs[0,1].set_xlabel('Time')
    axs[0,1].set_ylabel(r'$\langle \sigma_{y}^{(N-1)} \rangle$')
    axs[0,1].legend()

    # Plot Z Magnetization
    axs[0,2].plot(time_steps, magnetization_z_values, color='blue', label='mps-mpo')
    axs[0,2].plot(time_steps, rmagnetization_z_values, color='lightblue', label='Random', linestyle='dashed')
    axs[0,2].set_title('Z Magnetization at Last Site')
    axs[0,2].set_xlabel('Time')
    axs[0,2].set_ylabel(r'$\langle \sigma_{z}^{(N-1)} \rangle$')
    axs[0,2].legend()

    # Runtime Comparison
    axs[1, 0].plot(time_steps, baseline_times, color='purple', label='Baseline Time')
    axs[1, 0].plot(time_steps, random_times, color='violet', label='Random Times', linestyle='dashed')
    axs[1, 0].set_title('Runtime Comparison per Iteration')
    axs[1, 0].set_xlabel('Time')
    axs[1, 0].set_ylabel('Runtime')
    axs[1, 0].legend()

    # Bond Dimension Comparison
    axs[1, 1].plot(time_steps, baseline_bonds, color='brown', label='Baseline Bonds')
    axs[1, 1].plot(time_steps, random_bonds, color='tan', label='Random Bonds', linestyle='dashed')
    axs[1, 1].set_title('Bond Dimension Comparison')
    axs[1, 1].set_xlabel('Time')
    axs[1, 1].set_ylabel('Bond Dimension')
    axs[1, 1].legend()

    # Time Step Accuracy
    axs[1, 2].plot(time_steps, random_accs, color='cyan', label='Random Accuracy', linestyle='dashed')
    axs[1, 2].set_title('Accuracy per timestep')
    axs[1, 2].set_xlabel('Time')
    axs[1, 2].set_ylabel(r'$||\psi(t)_{naive}-\psi(t)_{rand}||_F/||\psi(t)_{naive}||_F$')
    axs[1, 2].legend()

    # Adjust layout and display
    plt.tight_layout()
    plt.show()


# ===========================================================================
# Plotting
# ===========================================================================

def plot_times(test_names, var, times_data, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
   
    palette = sns.color_palette("Set1", 6)  
    plot_styles = {
        'naive': (palette[0], 'o', 'Contract-Then-Compress', '--'),   # First color, circle marker, dashed line
        'random': (palette[1], 's', 'Randomized Contraction', '-'),  # Second color, square marker, solid line
        'zipup': (palette[2], 'D', 'Zip-up Contraction', '-'),       # Third color, diamond marker, solid line
        'density': (palette[3], '^', 'Density Matrix Contraction', '-'), # Fourth color, triangle up marker, solid line
        'fit': (palette[4], 'v', 'Fitting Contraction', '-'),        # Fifth color, triangle down marker, solid line
        'fit2': (palette[4], 'v', 'Fitting Contraction', '-'),        # Fifth color, triangle down marker, solid line

        'rand_then_orth': ('g', '^', 'Rand_then_orth Contraction', '--'),      
        'nyst': ('b', 'v', 'Nystrom Contraction', '--'),
    }

    for name in test_names:
        if name in plot_styles and name in times_data:
            color, marker, label, linestyle = plot_styles[name]
            ax.plot(var, times_data[name], color=color, marker=marker, label=label, markersize=6, linewidth=1.5, linestyle=linestyle)
            
    ax.set_title ("Runtime Comparison")
    ax.set_yscale('log')
    ax.grid()
    ax.set_xlabel('Bond Dimension')
    ax.set_ylabel('Time (s)')
    ax.legend()
    return ax
    
def plot_accuracy(test_names, var, accuracy_data, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
        
    palette = sns.color_palette("Set1", 6)  
    plot_styles = {
        'naive': (palette[0], 'o', 'Contract-Then-Compress', '--'),   # First color, circle marker, dashed line
        'random': (palette[1], 's', 'Randomized Contraction', '-'),  # Second color, square marker, solid line
        'zipup': (palette[2], 'D', 'Zip-up Contraction', '-'),       # Third color, diamond marker, solid line
        'density': (palette[3], '^', 'Density Matrix Contraction', '-'), # Fourth color, triangle up marker, solid line
        'fit': (palette[4], 'v', 'Fitting Contraction', '-'),        # Fifth color, triangle down marker, solid line
        'rand_then_orth': ('g', '^', 'Rand_then_orth Contraction', '--'),
        'nyst': ('b', 'v', 'Nystrom Contraction', '--'),

    }

    for name in test_names:
        if name in plot_styles and name in accuracy_data:
            color, marker, label, linestyle = plot_styles[name]
            ax.plot(var, accuracy_data[name], color=color, marker=marker, label=label, markersize=6, linewidth=1.5, linestyle=linestyle)
      
    ax.set_title("Accuracy Comparison (relative error)")
    ax.set_yscale('log')
    ax.grid()
    ax.set_xlabel('Bond Dimension')
    ax.set_ylabel('Relative Error')
    ax.legend()
    return ax

def plot_times_reversed(test_names,cutoffs, times_data, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
        palette = sns.color_palette("Set1", 5)  
    plot_styles = {
        'naive': (palette[0], 'o', 'Contract-Then-Compress', '--'),   # First color, circle marker, dashed line
        'random': (palette[1], 's', 'Randomized Contraction', '-'),  # Second color, square marker, solid line
        'zipup': (palette[2], 'D', 'Zip-up Contraction', '-'),       # Third color, diamond marker, solid line
        'density': (palette[3], '^', 'Density Matrix Contraction', '-'), # Fourth color, triangle up marker, solid line
        'fit': (palette[4], 'v', 'Fitting Contraction', '-'),        # Fifth color, triangle down marker, solid line
    }

    for name in test_names:
        if name in plot_styles and name in times_data:
            color, marker, label, linestyle = plot_styles[name]
            ax.plot(cutoffs[::-1], times_data[name][::-1], color=color, marker=marker, label=label, markersize=6, linewidth=1.5, linestyle=linestyle)
    
  
    ax.set_title("Runtime Comparison")
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.invert_xaxis()  # Reverse x-axis
    ax.set_xlabel('Cutoff Value')
    ax.set_ylabel('Time (s)')
    ax.legend()
    return ax


def plot_accuracy_reversed(test_names,cutoffs, accuracy_data, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
        palette = sns.color_palette("Set1", 5)  
    plot_styles = {
        'naive': (palette[0], 'o', 'Contract-Then-Compress', '--'),   # First color, circle marker, dashed line
        'random': (palette[1], 's', 'Randomized Contraction', '-'),  # Second color, square marker, solid line
        'zipup': (palette[2], 'D', 'Zip-up Contraction', '-'),       # Third color, diamond marker, solid line
        'density': (palette[3], '^', 'Density Matrix Contraction', '-'), # Fourth color, triangle up marker, solid line
        'fit': (palette[4], 'v', 'Fitting Contraction', '-'),        # Fifth color, triangle down marker, solid line
    }

    for name in test_names:
        # if name =='naive':
        #     continue
        if name in plot_styles and name in accuracy_data:
            color, marker, label, linestyle = plot_styles[name]
            ax.plot(cutoffs[::-1], accuracy_data[name][::-1], color=color, marker=marker, label=label, markersize=6, linewidth=1.5, linestyle=linestyle)
    
      # Adding a dashed red diagonal line
    # min_cutoff = min(cutoffs)
    # max_cutoff = max(cutoffs)
    # ax.plot([min_cutoff, max_cutoff], [min_cutoff, max_cutoff], linestyle='--', color='grey', )    
    ax.set_title("Accuracy Comparison (Relative Error)")
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.invert_xaxis()  # Reverse x-axis
    ax.set_xlabel('Cutoff Value')
    ax.set_ylabel('Relative Error')
    ax.legend()
    return ax


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker

import seaborn as sns 
import os 


def plot_results(results, N, dt, steps, contraction_types):
    fig, axs = plt.subplots(2, 5, figsize=(20, 10))  # Adjusted for 5 columns
    palette = sns.color_palette("Set1", 7)
    colors = {
        'naive': palette[0],   # First color from the palette
        'random': palette[1],  # Second color from the palette
        'zipup': palette[2],   # Third color from the palette
        'density': palette[3], # Fourth color from the palette
        'fit': palette[4],     # Fifth color from the palette
        'baseline': palette[6], 
        'rand_then_orth': 'g', # Green for 'rand_then_orth'
        'nyst': 'b',           # Blue for 'nyst'
    }
    # First Row: Magnetization Differences Contour Plots
    for i, ct in enumerate(contraction_types + ['baseline']):
        ax = axs[1, i]
        levels = np.linspace(0, np.abs(results['mags_diff'][ct]).max(), 300)
        contour = ax.contourf(np.abs(results['mags_diff'][ct].T), levels=levels, cmap='plasma', vmin=0)
        cbar = fig.colorbar(contour, ax=ax)
        cbar.set_label('Magnitude')
        ax.set_title(f'Diff {ct.capitalize()}')
        ax.set_xlabel('Particle Index')
        ax.set_ylabel('Timestep')
    
    # Second Row
    # Magnetization at Midpoint
    for idx, ct in enumerate(contraction_types + ['baseline']):
        axs[0, 0].plot(np.linspace(0, dt * steps, steps + 1), results['mags_modified'][ct][N // 2], 
                       label=ct.capitalize(), color=colors[ct], linestyle='--', linewidth=7-idx)
    axs[0, 0].set_xlabel('Time')
    axs[0, 0].set_ylabel('Magnetization at Midpoint')
    axs[0, 0].legend()
    axs[0, 0].set_title('Magnetization at Midpoint')

    # Bond Dimension Over Time
    for ct in contraction_types + ['baseline']:
        axs[1, 1].plot(results['bond_dim'][ct], label=ct.capitalize(), color=colors[ct])
    axs[0, 1].set_xlabel('Timestep')
    axs[0, 1].set_ylabel('Bond Dimension')
    axs[0, 1].legend()
    axs[0, 1].set_title('Bond Dimension Over Time')

    # MPS Krylov Runtimes
    for ct in contraction_types:
        axs[0, 2].plot(results['krylov_times'][ct], label=f'Krylov {ct.capitalize()}', color=colors[ct])
    axs[0, 2].set_xlabel('Timestep')
    axs[0, 2].set_ylabel('Time (s)')
    axs[0, 2].set_yscale('log')
    axs[0, 2].legend()
    axs[0, 2].set_title('MPS Krylov Runtime')

    # Expand Runtimes
    for ct in contraction_types:
        axs[0, 3].plot(results['expand_times'][ct], label=f'Expand {ct.capitalize()}', color=colors[ct])
    axs[0, 3].set_xlabel('Timestep')
    axs[0, 3].set_ylabel('Time (s)')
    axs[0, 3].set_yscale('log')
    axs[0, 3].legend()
    axs[0, 3].set_title('Expand Runtime')

    # TDVP Runtimes
    for ct in contraction_types + ['baseline']:
        axs[0, 4].plot(results['tdvp_times'][ct], label=f'TDVP {ct.capitalize()}', color=colors[ct])
    axs[0, 4].set_xlabel('Timestep')
    axs[0, 4].set_ylabel('Time (s)')
    axs[0, 4].set_yscale('log')
    axs[0, 4].legend()
    axs[0, 4].set_title('TDVP Runtime')

    plt.tight_layout()
    plt.show()

def write_results_to_csv(results, t, contraction_type, filename):
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)

    try:
        # Ensure t-1 is within valid range
        index = t - 1 if t > 0 else 0

        mags_modified = results['mags_modified'][contraction_type][:, index+1] if index+1 < results['mags_modified'][contraction_type].shape[1] else None
        mags_diff = results['mags_diff'][contraction_type][:, index+1] if index+1 < results['mags_diff'][contraction_type].shape[1] else None
        
        # Adjust index for timing data (t-1 because timing data is recorded for previous step)
        krylov_time = results['krylov_times'][contraction_type][index] if index < len(results['krylov_times'][contraction_type]) else float('nan')
        expand_time = results['expand_times'][contraction_type][index] if index < len(results['expand_times'][contraction_type]) else float('nan')
        tdvp_time = results['tdvp_times'][contraction_type][index] if index < len(results['tdvp_times'][contraction_type]) else float('nan')
        bond_dim = results['bond_dim'][contraction_type][index] if index < len(results['bond_dim'][contraction_type]) else float('nan')
        
        # Prepare the data dictionary
        data = {
            'Step': [t],
            'Contraction_Type': [contraction_type],
            'Mags_Modified': [','.join(map(str, mags_modified)) if mags_modified is not None else ''],
            'Mags_Diff': [','.join(map(str, mags_diff)) if mags_diff is not None else ''],
            'Krylov_Time': [krylov_time],
            'Expand_Time': [expand_time],
            'TDVP_Time': [tdvp_time],
            'Bond_Dim': [bond_dim]
        }
        
        # Check and format evolution errors (only for 'zipup' and 'random')
        evolution_errors = None
        if contraction_type in results['evolution_errors']:
            evolution_errors = results['evolution_errors'][contraction_type][:, index] if index < results['evolution_errors'][contraction_type].shape[1] else None
            
            # Add evolution errors to the data dictionary with proper column names
            if evolution_errors is not None:
                for i, error in enumerate(evolution_errors):
                    data[f'Evolution_Error_KryVec_{i+1}'] = [error]
        
        # Convert data to DataFrame
        df = pd.DataFrame(data)

        # Write the data to CSV, ensuring headers are included only for the first write
        df.to_csv(filename, mode='a', header=not os.path.isfile(filename), index=False)

    except Exception as e:
        print(f"Error writing to CSV: {e}") 
        
def final_plots(results, N, dt):
    def plot_magnetization(matrix, xlabel, ylabel, title, text_annotation, dt):
        rows, cols = matrix.shape
        fig, ax = plt.subplots(figsize=(8, 8))  # Adjust aspect ratio based on matrix shape

        levels = np.linspace(0, np.max(matrix), 300)

        # Adjust x-axis to range from 1 to 101
        contour = ax.contourf(np.linspace(1, 101, cols), np.linspace(0, (rows-1)*dt, rows), matrix, levels=levels, cmap='jet', vmin=0)
        
        # Adjust fraction to match color bar length with the plot height
        cbar = fig.colorbar(contour, ax=ax, location='right', pad=0.1, fraction=0.0435)
        
        # Set the colorbar to display full values up to two decimal places
        cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        
        cbar.locator = ticker.MaxNLocator(nbins=5)
        cbar.update_ticks()

        cbar.ax.yaxis.offsetText.set(size=18)
        cbar.ax.tick_params(labelsize=18)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.set_xlabel(xlabel, fontsize=18)
        ax.set_ylabel(ylabel, fontsize=18)

        # Set custom x-axis tick marks at intervals of 20 starting from 1
        ax.set_xticks([1, 21, 41, 61, 81, 101])

        if text_annotation:
            ax.text(0.95, 0.05, text_annotation, color='white', fontsize=20,
                    horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)

        ax.set_aspect(cols / ((rows-1) * dt))

        if title:
            plt.title(title, fontsize=18)
        
        plt.show()

    random_magnetizations = np.abs(results['mags_diff']['random'].T)
    plot_magnetization(random_magnetizations, 
                    xlabel='Lattice Position $i$', 
                    ylabel='Time $tJ$', 
                    title=None, 
                    text_annotation=None,
                    dt=dt)
    
    # random_magnetizations = results['mags_modified']['random']
    # naive_magnetizations = results['mags_modified']['naive']
    # # diff = np.abs(random_magnetizations - naive_magnetizations)
    # relative_error = np.abs(random_magnetizations.T - naive_magnetizations.T)+1e-16
    # # naive_magnetizations #TODO: try abs here 

    # plot_magnetization(relative_error, 
    #                 xlabel='Lattice Position $r$', 
    #                 title=None,
    #                 ylabel='Time $tJ$', 
    #                 text_annotation=None, 
    #                 dt=dt)
    
    # zipup_magnetizations = results['mags_modified']['zipup']
    # naive_magnetizations = results['mags_modified']['naive']
    # # diff = np.abs(random_magnetizations - naive_magnetizations)
    # relative_error = np.abs(zipup_magnetizations.T - naive_magnetizations.T)+1e-16
    # # naive_magnetizations #TODO: try abs here 

    # plot_magnetization(relative_error, 
    #                 xlabel='Lattice Position $r$', 
    #                 title=None,
    #                 ylabel='Time $tJ$', 
    #                 text_annotation=None, 
    #                 dt=dt)

def bond_dimensions(results):
    bond_dims = {
        'random': [],
        'density': [],
        'naive': [],
        'zipup': [],
        'baseline': []
    }

    contraction_types = list(bond_dims.keys())

    for ct in contraction_types:
        bond_dim_for_ct = []

        for t in range(1,len(results['bond_dim'][ct])):
            bond_dim = results['bond_dim'][ct][t] if ct != 'baseline' else 0
            bond_dim_for_ct.append(bond_dim)

        bond_dims[ct] = bond_dim_for_ct

    return bond_dims

def plot_bond_dimensions(bond_dims):
    # Define the plot styles for each method
    palette = sns.color_palette("Set1", 6)
    plot_styles = {
        'naive': (palette[0], 'o', 'Contract-Then-Compress', '--'),   # First color, circle marker, dashed line
        'random': (palette[1], 's', 'Randomized', '-'),  # Second color, square marker, solid line
        'zipup': (palette[2], 'D', 'Zip-Up', '-'),       # Third color, diamond marker, solid line
        'density': (palette[3], '^', 'Density Matrix', '-'), # Fourth color, triangle up marker, solid line
        'fit': (palette[4], 'v', 'Fitting', '-'),        # Fifth color, triangle down marker, solid line
        'rand_then_orth': ('g', '^', 'Randomized C-T-C', '--'),      
        'nyst': ('b', 'v', 'Nystrom', '--'),
    }
    
    fig, ax = plt.subplots(figsize=(8, 8))  

    for ct, dims in bond_dims.items():
        style = plot_styles.get(ct, ('k', 'o', f'{ct.capitalize()}', '-'))  # Default style if not found
        color, marker, label, linestyle = style

        if ct == "baseline":
            pass  # Skip baseline
        else:
            plt.plot(dims, label=label, color=color, linestyle=linestyle)
            plt.scatter(range(len(dims)), dims, color=color, marker=marker, s=50)  # Adding markers to each point
    
    plt.tick_params(axis='both', which='major', labelsize=18)
    ax.yaxis.set_tick_params(labelsize=18)  # Uniform y-axis tick size
    plt.xlabel('Timestep', fontsize=20)
    plt.ylabel('Bond Dimension', fontsize=20)
    plt.yscale("log")
    plt.legend(fontsize=15)
    plt.grid()
    plt.show()

def expand_times_runtime(results):
    expand_times = {
        'random': [],
        'density': [],
        'naive': [],
        'zipup': [],
        'baseline': []
    }

    contraction_types = list(expand_times.keys())

    for ct in contraction_types:
        expand_time_for_ct = []

        for t in range(1,len(results['expand_times'][ct])):
            expand_time = results['expand_times'][ct][t] if ct != 'baseline' else 0
            expand_time_for_ct.append(expand_time)

        expand_times[ct] = expand_time_for_ct

    return expand_times

def plot_expand_times(expand_times):
    # Define the plot styles for each method
    palette = sns.color_palette("Set1", 6)
    plot_styles = {
        'naive': (palette[0], 'o', 'Contract-Then-Compress', '--'),   # First color, circle marker, dashed line
        'random': (palette[1], 's', 'Randomized', '-'),  # Second color, square marker, solid line
        'zipup': (palette[2], 'D', 'Zip-Up', '-'),       # Third color, diamond marker, solid line
        'density': (palette[3], '^', 'Density Matrix', '-'), # Fourth color, triangle up marker, solid line
        'fit': (palette[4], 'v', 'Fitting', '-'),        # Fifth color, triangle down marker, solid line
        'rand_then_orth': ('g', '^', 'Randomized C-T-C', '--'),      
        'nyst': ('b', 'v', 'Nystrom', '--'),
    }
    
    fig, ax = plt.subplots(figsize=(8, 8))  

    for ct, times in expand_times.items():
        style = plot_styles.get(ct, ('k', 'o', f'{ct.capitalize()}', '-'))  # Default style if not found
        color, marker, label, linestyle = style

        if ct == "baseline":
            pass  # Skip baseline
        else:
            plt.plot(times, label=label, color=color, linestyle=linestyle)
            plt.scatter(range(len(times)), times, color=color, marker=marker, s=50)  # Adding markers to each point
    
    plt.tick_params(axis='both', which='major', labelsize=18)
    ax.yaxis.set_tick_params(labelsize=18)  # Uniform y-axis tick size
    plt.xlabel('Timestep', fontsize=20)
    plt.ylabel('Expand Time (sec)', fontsize=20)
    plt.legend(fontsize=15)
    plt.grid()
    plt.yscale('log')  # Log scale for y-axis if needed
    plt.show()
    
def total_runtime(results):
    total_times = {
        'naive': [],
        'density':[],
        'zipup': [],
        'random': [],
        # 'fit': [],
       
        #'zipup': [],
        #'baseline': []
    }

    contraction_types = list(total_times.keys())

    for ct in contraction_types:
        total_runtime_for_ct = []
        
        for t in range(len(results['tdvp_times'][ct])):
            krylov_time = results['krylov_times'][ct][t] if ct != 'baseline' else 0
            expand_time = results['expand_times'][ct][t] if ct != 'baseline' else 0
            tdvp_time = results['tdvp_times'][ct][t]
            
            total_time = krylov_time + expand_time + tdvp_time
            
            total_runtime_for_ct.append(total_time)
        
        total_times[ct] = total_runtime_for_ct
    
    return total_times

def krylov_times_runtime(results):
    krylov_times = {
        'random': [],
        'density': [],
        'naive': [],
        'zipup': [],
        #'baseline': []
    }

    contraction_types = list(krylov_times.keys())

    for ct in contraction_types:
        krylov_time_for_ct = []

        for t in range(len(results['krylov_times'][ct])):
            krylov_time = results['krylov_times'][ct][t] if ct != 'baseline' else 0
            krylov_time_for_ct.append(krylov_time)

        krylov_times[ct] = krylov_time_for_ct

    return krylov_times

def plot_krylov_times(krylov_times):
    # Define the new plot styles for each method
    palette = sns.color_palette("Set1", 6)
    plot_styles = {
        'naive': (palette[0], 'o', 'Contract-Then-Compress', '--'),   # First color, circle marker, dashed line
        'random': (palette[1], 's', 'Randomized', '-'),  # Second color, square marker, solid line
        'zipup': (palette[2], 'D', 'Zip-up', '-'),       # Third color, diamond marker, solid line
        'density': (palette[3], '^', 'Density Matrix', '-'), # Fourth color, triangle up marker, solid line
        'fit': (palette[4], 'v', 'Fitting', '-'),        # Fifth color, triangle down marker, solid line
        'rand_then_orth': ('g', '^', 'Randomized C-T-C', '--'),      
        'nyst': ('b', 'v', 'Nystrom', '--'),
    }
    
    fig, ax = plt.subplots(figsize=(8, 16))  

    for ct, times in krylov_times.items():
        style = plot_styles.get(ct, ('k', 'o', f'{ct.capitalize()}', '-'))  # Default style if not found
        color, marker, label, linestyle = style

        if ct == "baseline":
            pass  # Skip baseline
        else:
            plt.plot(np.array(range(len(times)))+1, times, label=label, color=color, linestyle=linestyle)
            plt.scatter(np.array(range(len(times)))+1, times, color=color, marker=marker, s=70,linewidths=5)  # Adding markers to each point
    
    plt.xticks([0,5,10,15,20,25]) 
    #3e-1 to 1e3
    plt.tick_params(axis='both', which='major', labelsize=36)
    ax.yaxis.set_tick_params(labelsize=36)  
    plt.xlabel('Timestep', fontsize=34)
    plt.ylabel('Krylov Runtime per Timestep (sec)', fontsize=34)
    plt.grid()
    plt.ylim(3e-1, 1e3)

    #plt.legend(fontsize=11)
    plt.yscale('log')  # Log scale for y-axis
    plt.show()


def plot_total_times(total_times):
    # Define the new plot styles for each method
    palette = sns.color_palette("Set1", 6)
    plot_styles = {
        'naive': (palette[0], 'o', 'C-T-C', '-'),   # First color, circle marker, dashed line
        'random': (palette[1], 's', 'Randomized', '-'),  # Second color, square marker, solid line
        'zipup': (palette[2], 'D', 'Zip-Up', '-'),       # Third color, diamond marker, solid line
        'density': (palette[3], '^', 'Density Matrix', '-'), # Fourth color, triangle up marker, solid line
        'fit': (palette[4], 'v', 'Fitting', '-'),        # Fifth color, triangle down marker, solid line
        'rand_then_orth': ('g', '^', 'Randomized C-T-C', '--'),      
        'nyst': ('b', 'v', 'Nystrom', '--'),
    }
    
    fig, ax = plt.subplots(figsize=(8, 16))  

    for ct, times in total_times.items():
        style = plot_styles.get(ct, ('k', 'o', f'{ct.capitalize()}', '-'))  # Default style if not found
        color, marker, label, linestyle = style

        if ct == "baseline":
            pass  # Skip baseline
        else:
            plt.plot(np.array(range(len(times)))+1, times, label=label, color=color, linestyle=linestyle)
            plt.scatter(np.array(range(len(times)))+1, times, color=color, marker=marker, s=70,linewidths=5)  # Adding markers to each point
    
    plt.xticks([0,5,10,15,20,25]) 
    plt.tick_params(axis='both', which='major', labelsize=36)
    ax.yaxis.set_tick_params(labelsize=36)  
    plt.xlabel('Timestep', fontsize=34)
    plt.ylabel('Total Runtime per Timestep (sec)', fontsize=34)
    plt.grid()
    plt.ylim(3e-1, 1e3)

    plt.legend(fontsize=34)
    plt.yscale('log')  # Log scale for y-axis
    plt.show()


def plot_relative_errors_line_plot_with_gradient(results, contraction_types, steps, basis_size):
    # Define time steps correctly using the steps variable
    time_steps = np.arange(1, steps + 1)  # Corresponding to time steps from 1 to 25 (inclusive)

    # Filter out 'naive' and 'baseline' from the contraction types
    filtered_contraction_types = [ct for ct in contraction_types if ct not in ['naive', 'baseline']]

    # Define a color palette for each contraction type
    palette = sns.color_palette("Set1", 6)  # Set1 is a good choice for distinct colors
    plot_colors = {
        'naive': palette[0],
        'random': palette[1],
        'zipup': palette[2],
        'density': palette[3],
        'fit': palette[4],
    }
    legend_names = {
        'random': 'Random',
        'zipup': 'Zip-Up',
        'density': 'Density',
        'fit': 'Fit',
    }
    
    # Get the evolution errors for each contraction type
    evolution_errors_by_method = {ct: results['evolution_errors'][ct] for ct in filtered_contraction_types if ct in results['evolution_errors']}

    plt.figure(figsize=(8, 8))  

    for ct in filtered_contraction_types:
        for j in range(basis_size - 1):
            if ct == "density":
                continue

            # Extract the errors for the time steps from 0 to (steps - 1) (corresponding to time steps 1 to 25 on the plot)
            errors = evolution_errors_by_method[ct][j, :steps]  # Use steps to slice up to the correct number of steps (25)
            non_zero_indices = errors != 0  # Find indices where errors are non-zero
            
            # Only plot the non-zero errors
            non_zero_errors = errors[non_zero_indices]
            non_zero_time_steps = time_steps[non_zero_indices]  # Corresponding time steps (1 to steps)

            if len(non_zero_errors) == 0:  # If no non-zero values, skip this Krylov vector
                continue

            color = plot_colors[ct]
            rgba_color = (*color, 1 - (0.99 * (j / basis_size)))  # Apply gradient to color
            
            # Plot only the non-zero values
            plt.plot(non_zero_time_steps, non_zero_errors, label=f'{legend_names[ct]} KryVec {j+1}', color=rgba_color, marker='o')

    # Plot settings
    plt.xlabel('Time Step', fontsize=18)
    plt.ylabel('Relative Error (Log Scale)', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.xticks(ticks=np.arange(1, steps + 1, 5), labels=np.arange(1, steps + 1, 5))  # Time steps from 1 to steps (25)

    plt.yscale('log')
    plt.legend(fontsize=15)
    plt.grid(True)
    plt.tight_layout()

    # Show plot
    plt.show()

def load_specific_experiment_results(file_path):
    # Check if file exists
    try:
        results_df = pd.read_csv(file_path)
        print(f"Data loaded from {file_path}")
        return results_df
    except FileNotFoundError:
        print(f"File {file_path} does not exist.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return None

def extract_experiment_data(df, names):
    # Initialize the dictionaries
    times = {name: [] for name in names}
    std_times = {name: [] for name in names}
    accs = {name: [] for name in names}
    std_accs = {name: [] for name in names}
    
    # Loop through each name and populate the dictionaries
    for name in names:
        times_col = f'{name} Mean Time'
        std_times_col = f'{name} Time Std'
        accs_col = f'{name} Mean Accuracy'
        std_accs_col = f'{name} Accuracy Std'
        
        # Populate the dictionaries by extracting data from the dataframe
        times[name] = df[times_col].tolist()
        std_times[name] = df[std_times_col].tolist()
        accs[name] = df[accs_col].tolist()
        std_accs[name] = df[std_accs_col].tolist()
    
    return times, std_times, accs, std_accs
    

def plot_runtime_vs_accuracy(test_names, times_data, std_times_data, accuracy_data, std_acc_data, var, mean_times_data, std_times_data_var):
    # Create a figure with two horizontal subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), constrained_layout=False)  # 18x8 figure
    global_font_size = 18
    global_marker_size = 8
    global_line_width = 2
    plt.subplots_adjust(wspace=0.3, top=0.85)  # Increase the width space between the subplots and space for legend

    # Plot runtime vs accuracy on the first subplot (ax1)
    palette = sns.color_palette("Set1", 6)
    plot_styles = {
        'naive': (palette[0], 'o', 'C-T-C', '-'),
        'random': (palette[1], '^', 'Randomized', '-'),
        'zipup': (palette[2], 'D', 'Zip-Up', '-'),
        'density': (palette[3], 'd', 'Density Matrix', '-'),
        'fit': (palette[4], 's', 'Fitting', '-'),
        'rand_then_orth': ('g', '*', 'Randomized C-T-C', '-'),
    }

    handles = []
    labels = []

    for name in test_names:
        if name in plot_styles and name in times_data and name in accuracy_data:
            color, marker, label, linestyle = plot_styles[name]
            times = np.array(times_data[name])
            std_times = np.array(std_times_data[name])
            accuracy = np.array(accuracy_data[name])
            std_acc = np.array(std_acc_data[name])
            if name == 'naive':
                line, = ax1.plot(times, accuracy, color=color, marker=marker, label=label, markersize=8, 
                            linewidth=global_line_width, linestyle=linestyle, markerfacecolor='none', markeredgewidth=1, alpha=1)
            # Plot the line
            else:
                line, = ax1.plot(times, accuracy, color=color, marker=marker, label=label,
                            markersize=global_marker_size, linewidth=global_line_width, linestyle=linestyle)

            # Add standard deviation for times and accuracy
            ax1.fill_between(times, accuracy - std_acc, accuracy + std_acc, color=color, alpha=0.2)
            ax1.fill_betweenx(accuracy, times - std_times, times + std_times, color=color, alpha=0.2)

            handles.append(line)
            labels.append(label)

    # Set log scales and labels
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, which='both', ls='--', lw=0.5)
    ax1.set_xlabel('Runtime (sec)', fontsize=global_font_size)
    ax1.set_ylabel('Relative Error', fontsize=global_font_size)
    ax1.tick_params(axis='both', which='major', labelsize=global_font_size)

    # Remove manual aspect ratio setting for ax1
    # ax1.set_aspect(x_range / y_range)  # Removed

    # Legend for the first plot above the figure
    handles, labels = ax1.get_legend_handles_labels()  # Assuming both plots have the same labels
    fig.legend(handles, labels, loc='lower center', fontsize=20, ncol=4, bbox_to_anchor=(0.5, .95))

    # Now plot requested bond dimension vs runtime on the second subplot (ax2)
    def plot_times(test_names, var, mean_times_data, std_times_data, ax):
        plot_styles = {
            'naive': (palette[0], 'o', 'C-T-C', '-'),
            'random': (palette[1], '^', 'Randomized', '-'),
            'zipup': (palette[2], 'D', 'Zip-Up', '-'),
            'density': (palette[3], 'd', 'Density Matrix', '-'),
            'fit': (palette[4], 's', 'Fitting', '-'),
            'rand_then_orth': ('g', '*', 'Randomized C-T-C', '-'),
        }

        for name in test_names:
            if name in plot_styles and name in mean_times_data:
                color, marker, label, linestyle = plot_styles[name]
                mean_times = np.array(mean_times_data[name])
                std_times = np.array(std_times_data[name])

                if name == 'naive':
                    ax.plot(var, mean_times, color=color, marker=marker, label=label,
                            markersize=global_marker_size, linewidth=global_line_width,
                            linestyle=linestyle, markerfacecolor='none', markeredgewidth=1, alpha=1)
                else:
                    ax.plot(var, mean_times, color=color, marker=marker, label=label,
                            markersize=global_marker_size, linewidth=global_line_width, linestyle=linestyle)
                    ax.fill_between(var, mean_times - std_times, mean_times + std_times, color=color, alpha=0.3)

        ax.set_yscale('log')
        ax.grid()
        ax.set_xlabel(r'Requested Output Bond Dimension $\overline{\chi}$', fontsize=global_font_size)
        ax.set_ylabel('Runtime', fontsize=global_font_size)
        ax.tick_params(axis='both', which='major', labelsize=global_font_size)

    # Call plot_times for the second subplot (ax2)
    plot_times(test_names, var, mean_times_data, std_times_data_var, ax=ax2)

    # Adjust layout to prevent clipping
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to leave space for legend
    plt.show()