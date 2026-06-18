"""
Reservoir state analysis is conducted at the single-cell level and aims at understanding
the neural dynamics during the bot's navigation. In this script, the reservoir states
beforehand recorded during the bot's navigation are loaded to process to the population analysis.

 The script allows several analytical processes:

- SI Index Computation: Calculates the Selectivity Index (SI) of neurons corresponding to different bot trajectories.

- Place Cells, Head-Direction Cells, and Splitter Cells Activity Plotting: Visualizes the activity patterns of place cells, head-direction cells, and splitter cells, providing insights into spatial and directional encoding in the reservoir.

- Mean Firing Rate Plotting: Plots the mean firing rate of individual neurons during both correct and error trials,
 highlighting different neural activity dynamics in different behavioral contexts.

"""
import re
import pandas as pd
import sys
import os
sys.path.append(r'/Users/nchaix/Documents/PhD/code/splitter_cells_in_reservoir/')
from maze import Maze
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from matplotlib.gridspec import GridSpec
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Trajectory ANCOVA p-value cutoff for splitter-cell labelling (pipeline + plots).
SC_P_THRESHOLD = 1e-4

plt.rc('font', size=14)
def darken_color(color, factor=0.7):
    import matplotlib.colors as mcolors
    rgb = mcolors.to_rgb(color)  # Convert color to RGB
    return tuple([factor * x for x in rgb])
COLORS = {'RL': 'red', 'LR': darken_color('blueviolet', factor=0.7), 'LL': 'olivedrab', 'RR': 'darkorange'}


def load_positions(path):
    """Load position data from a specified path.

        Args:
        path (str): The path to the directory containing the position data file.

        Returns:
        numpy.ndarray: An array containing position data.
        """
    return np.load(path + 'positions.npy')


def load_reservoir_states(path):
    """Load reservoir state data from a specified path.

        Args:
        path (str): The path to the directory containing the reservoir state data file.

        Returns:
        numpy.ndarray: An array containing reservoir state data.
        """
    return np.load(path + 'reservoir_states.npy')


def load_orientations(path):
    """Load orientation data from a specified path.

        Args:
        path (str): The path to the directory containing the orientation data file.

        Returns:
        numpy.ndarray: An array containing orientation data.
        """
    return np.load(path + 'output.npy')


def find_location_indexes_xy(x_positions, y_positions):
    """
        This function identifies location indexes ('m' for middle, 'r' for right, 'l' for left)
        based on the provided y positions. It iterates through the positions and determines
        the location based on specific thresholds.

        Args:
        y_positions (numpy.ndarray): An array containing y positions.

        Returns:
        tuple: A tuple containing two lists:
            - locations: A list of location identifiers ('m', 'r', 'l').
            - locations_indexes: A list of corresponding indexes for each identified location.
        """
    flag_middle = False
    flag_left = False
    flag_right = False
    locations = []
    locations_indexes = []
    for i, pos in enumerate(y_positions):
        if 140 < pos < 160 and not flag_middle:
            if 100 < x_positions[i] < 200:
                locations.append('m')
                locations_indexes.append(i)
                flag_middle = True
                flag_left = False
                flag_right = False
        elif pos < 140 and not flag_right:
            locations.append('r')
            locations_indexes.append(i)
            flag_middle = False
            flag_right = True
            flag_left = False
        elif pos > 160 and not flag_left:
            locations.append('l')
            locations_indexes.append(i)
            flag_middle = False
            flag_right = False
            flag_left = True
    return locations, locations_indexes


def get_average_activity(range, reservoir_states):
    """This function calculates the average activity of reservoir states within the specified range.

        Args:
        range: A array containing the start and end indices of the range.
        reservoir_states (numpy.ndarray): An array containing reservoir state data.

        Returns:
        float: The average activity of reservoir states within the specified range.
        """
    selected_states = reservoir_states[range[0]:range[1]]
    return np.mean(selected_states, axis=0)


def find_trajectory_indexes(positions, min_y=140, max_y=160):
    """
    Generate legend based on y_positions and x_positions.
    'l': left loop (y > 175)
    'r': right loop (y < 125)
    'rl': right to left in central corridor (125 <= y <= 175, 50 <= x <= 250)
    'lr': left to right in central corridor (125 <= y <= 175, 50 <= x <= 250)
    'rr': right to right in central corridor
    'll': left to left in central corridor
    """
    legend = []
    last_side_1 = 'l'
    last_side_2 = 'r'
    # Track last known side before entering the central corridor
    in_corridor = False  # Track if the bot is in the corridor
    y_positions = positions[:, 1]
    x_positions = positions[:, 0]

    for i in range(len(y_positions)):
        y = y_positions[i]
        x = x_positions[i]
        if y > max_y:
            legend.append('l')
            in_corridor = False
            last_side_2 = 'l'
        elif y < min_y:
            legend.append('r')
            last_side_2 = 'r'
            in_corridor = False
        elif min_y <= y <= max_y:
            if 100 <= x <= 200:
                if not in_corridor:
                    in_corridor = True
                    if last_side_1 + last_side_2 == 'll':
                        corridor_pos = 'rr'
                        last_side_1, last_side_2 = 'l', 'r'
                    elif last_side_1 + last_side_2 == 'rr':
                        corridor_pos = 'll'
                        last_side_1, last_side_2 = 'r', 'l'
                    elif last_side_1 + last_side_2 == 'lr':
                        corridor_pos = 'rl'
                        last_side_1, last_side_2 = 'r', 'r'
                    elif last_side_1 + last_side_2 == 'rl':
                        corridor_pos = 'lr'
                        last_side_1, last_side_2 = 'l', 'l'
                    else:
                        print('Issue in guessing the next loop')
                legend.append(corridor_pos)
            else:
                legend.append(last_side_2)
                in_corridor = False
        else:
            print('Issue 2')
    legend = np.array(legend)
    # Find indices where the value changes
    change_indices = np.where(legend[:-1] != legend[1:])[0] + 1
    split_indices = np.split(np.arange(len(legend)), change_indices)
    split_values = np.split(legend, change_indices)
    all_idx = {}
    for val, idx in zip(split_values, split_indices):
        key = val[0]  # Category name
        if key not in all_idx:
            all_idx[key] = []
        all_idx[key].append(idx.tolist())

    return all_idx


def find_activity_ranges(locations, locations_indexes):
    """
    This function determines the ranges of activity indexes for different pathways in the central corridor,
    including:
    - Right to left (R-L)
    - Left to right (L-R)
    - Left to left (L-L)
    - Right to right (R-R)

    Args:
    locations (list): A list of location identifiers ('m', 'r', 'l').
    locations_indexes (list): A list of corresponding indexes for each identified location.

    Returns:
    dict: A dictionary containing activity ranges for each pathway:
        - 'RL': Activity ranges for right to left pathway.
        - 'LR': Activity ranges for left to right pathway.
        - 'RR': Activity ranges for right to right pathway.
        - 'LL': Activity ranges for left to left pathway.
        - 'r_loop': Activity ranges for the right loop.
        - 'l_loop': Activity ranges for the left loop.
    """

    activity_ranges = {'RL': [], 'LR': [], 'RR': [], 'LL': [], 'r_loop':[], 'l_loop':[]}
    for i in range(1, len(locations) - 1):
        if locations[i] == 'm':
            if locations[i-1] == 'r':
                activity_ranges['r_loop'].append([locations_indexes[i-1], locations_indexes[i]])
                if locations[i+1] == 'r':
                    activity_ranges['RR'].append([locations_indexes[i], locations_indexes[i]+100])
                elif locations[i+1] == 'l':
                    activity_ranges['RL'].append([locations_indexes[i], locations_indexes[i]+100])
            elif locations[i-1] == 'l':
                activity_ranges['l_loop'].append([locations_indexes[i - 1], locations_indexes[i]])
                if locations[i+1] == 'r':
                    activity_ranges['LR'].append([locations_indexes[i], locations_indexes[i]+100])
                elif locations[i+1] == 'l':
                    activity_ranges['LL'].append([locations_indexes[i], locations_indexes[i]+100])
    return activity_ranges


def plot_splitter_activity_evolution_corridor_grid_comparison(*paths, indexes=[259, 709, 166, 219, 259, 335, 453, 761]):
    fig, axes = plt.subplots(nrows=len(indexes), ncols=len(paths), figsize=(10, 10))  # , sharey=True)
    #fig.suptitle('Comparison of neuronal firing activity across 3 models ')

    for k, path in enumerate(paths):
        positions = load_positions(path)
        reservoir_activities = load_reservoir_states(path)
        traj_idx = find_trajectory_indexes(positions=positions)

        for j, cell in enumerate(indexes):
            ax = axes[j, k]  # Use the loop index to reference the correct subplot
            all_reservoir_traj = {}
            min_lengths = {}
            for key in ['rl', 'lr', 'rr', 'll']:
                if key in traj_idx and traj_idx[key]:
                    all_reservoir_traj[key] = [reservoir_activities.T[cell][traversal] for traversal in traj_idx[key]]
                    min_lengths[key] = min(len(traversal) for traversal in all_reservoir_traj[key])

            # Trim all traversals to the smallest length
            for key in all_reservoir_traj:
                all_reservoir_traj[key] = [traversal[:min_lengths[key]] for traversal in all_reservoir_traj[key]]

            for key, val in all_reservoir_traj.items():
                mean = np.mean(val, axis=0)
                std = np.std(val, axis=0)
                x = np.arange(len(mean))
                # Plot mean and std shading for 'From right'
                ax.plot(x, mean, color=COLORS[key.upper()], alpha=0.7, label=key)
                ax.fill_between(x, mean - std, mean + std,
                                         color=COLORS[key.upper()], alpha=0.2)


            if j == 0:
                ax.text(0.2, 0.05, f'SC1: Cell #{cell}', transform=ax.transAxes, ha='center', fontsize=10)
            else:
                ax.text(0.2, 0.05, f'SC2: Cell #{cell}', transform=ax.transAxes, ha='center', fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            # Get the current y-tick values
            yticks = ax.get_yticks()
            #if i % 3 == 0:
            if k == 0:
                ax.set_ylabel('Firing activity')
    plt.tight_layout()
    plt.show()


def dynamics_reorganisation_SC(file_init_model="../data/R-L/no_cues/reservoir_states/",
                               file_lesioned_model="../data/R-L/no_cues/reservoir_states_killed_n/",
                               save=False):

    split_1 = find_splitter_cells_ANCOVA(file_init_model, save=save)
    split_2 = find_splitter_cells_ANCOVA(file_lesioned_model, save=save)

    print('N splitter cell init:', len(split_1))
    print('N splitter cell after:', len(split_2))

    count = 0
    new_SC = []
    for cell in split_2:
        if cell not in split_1:
            count += 1
            new_SC.append(cell)
    print('New splitter cells:', count)
    print('New SC:', new_SC)

    count = 0
    old_SC = []
    for cell in split_1:
        if cell not in split_2:
            count += 1
            old_SC.append(cell)
    print('Not splitter cells:', count)
    print('Old SC:', old_SC)


def _trajectory_names_for_cell_type(cell_type='SC'):
    """Trajectory pair labels used in the per-neuron ANCOVA (splitter pipeline)."""
    if cell_type == 'SC':
        return ('LR', 'RL')
    if cell_type == 'pro':
        return ('LR', 'LL')
    if cell_type == 'retro':
        return ('LR', 'RR')
    raise ValueError(f"Unknown cell_type {cell_type!r}; use 'SC', 'pro', or 'retro'.")


def build_crossing_level_ancova_table(path, cell_type='SC', max_crossings=16):
    """
    Build crossing-level design matrix for the splitter ANCOVA.

    Each row is one corridor crossing: mean reservoir activity over the
    crossing window, with trajectory label (LR / RL), lateral position, and
    orientation covariates.

    Returns
    -------
    trajectories : ndarray, shape (n_crossings,)
    firing_activities : ndarray, shape (n_neurons, n_crossings)
    lateral_positions : ndarray, shape (n_crossings,)
    orientations_values : ndarray, shape (n_crossings,)
    n_crossings_by_traj : dict[str, int]
    """
    res_activity = load_reservoir_states(path)
    orientations = load_orientations(path)
    positions = load_positions(path)

    y_positions = positions[:, 1]
    x_positions = positions[:, 0]
    locations, locations_indexes = find_location_indexes_xy(x_positions, y_positions)
    activity_ranges = find_activity_ranges(locations, locations_indexes)

    traj_names = _trajectory_names_for_cell_type(cell_type)
    trajectories = []
    firing_activities = []
    lateral_positions = []
    orientations_values = []
    n_crossings_by_traj = {}

    for trajectory in traj_names:
        n_avail = len(activity_ranges[trajectory])
        n_use = min(max_crossings, n_avail)
        n_crossings_by_traj[trajectory] = n_use
        print(f'For {trajectory}, {n_avail} crossings ({n_use} used)')
        for idx in range(n_use):
            crossing_range = activity_ranges[trajectory][idx]
            trajectories.append(trajectory)
            firing_activities.append(
                get_average_activity(crossing_range, res_activity))
            orientations_values.append(
                get_average_activity(crossing_range, orientations))
            lateral_positions.append(
                get_average_activity(crossing_range, y_positions))

    trajectories = np.asarray(trajectories)
    firing_activities = np.asarray(firing_activities).T
    lateral_positions = np.asarray(lateral_positions)
    orientations_values = np.asarray(orientations_values)
    return (trajectories, firing_activities, lateral_positions,
            orientations_values, n_crossings_by_traj)


def compute_trajectory_ancova_per_neuron(path, cell_type='SC', max_crossings=16,
                                         jitter=1e-6):
    """
    Per-neuron ANCOVA p-values for trajectory (LR vs RL) in the splitter pipeline.

    For each reservoir unit *i*, fits the same model as
    :func:`find_splitter_cells_ANCOVA` on corridor-crossing averages::

        firing_i ~ C(trajectory) + lateral_position + orientation

    and records one p-value per neuron for the trajectory factor (Type-II ANOVA),
    i.e. how strongly mean firing differs between LR and RL after covariate adjustment.

    Parameters
    ----------
    path : str
        Directory with ``reservoir_states.npy``, ``positions.npy``, ``output.npy``.
    cell_type : {'SC', 'pro', 'retro'}
        Which trajectory pair to compare (default ``'SC'`` → LR and RL).
    max_crossings : int
        Maximum crossings per trajectory type (default 16).
    jitter : float
        Tiny Gaussian noise added to firing per crossing (numerical stability).

    Returns
    -------
    results : dict
        ``p_trajectory`` : ndarray, shape (n_neurons,) — trajectory factor p-values
        ``f_trajectory`` : ndarray, shape (n_neurons,) — trajectory F statistics
        ``p_lateral`` : ndarray, shape (n_neurons,)
        ``p_orientation`` : ndarray, shape (n_neurons,)
        ``trajectories``, ``firing_activities``, ``lateral_positions``,
        ``orientations_values``, ``n_crossings_by_traj`` — design data
    """
    (trajectories, firing_activities, lateral_positions, orientations_values,
     n_crossings_by_traj) = build_crossing_level_ancova_table(
         path, cell_type=cell_type, max_crossings=max_crossings)

    n_neurons = firing_activities.shape[0]
    p_trajectory = np.full(n_neurons, np.nan)
    f_trajectory = np.full(n_neurons, np.nan)
    p_lateral = np.full(n_neurons, np.nan)
    p_orientation = np.full(n_neurons, np.nan)

    for i in range(n_neurons):
        data = pd.DataFrame({
            'trajectory': trajectories,
            'lateral_positions': lateral_positions,
            'value': firing_activities[i],
            'orientations': orientations_values,
        })
        if jitter:
            data['value'] = data['value'] + np.random.normal(
                0, jitter, size=len(data))
        model = ols(
            'value ~ C(trajectory) + lateral_positions + orientations',
            data=data).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        p_trajectory[i] = anova_table['PR(>F)']['C(trajectory)']
        f_trajectory[i] = anova_table['F']['C(trajectory)']
        p_lateral[i] = model.pvalues['lateral_positions']
        p_orientation[i] = model.pvalues['orientations']

    return {
        'p_trajectory': p_trajectory,
        'f_trajectory': f_trajectory,
        'p_lateral': p_lateral,
        'p_orientation': p_orientation,
        'trajectories': trajectories,
        'firing_activities': firing_activities,
        'lateral_positions': lateral_positions,
        'orientations_values': orientations_values,
        'n_crossings_by_traj': n_crossings_by_traj,
        'cell_type': cell_type,
    }


def get_trajectory_ancova_pvalues(path, **kwargs):
    """
    Return the length-``n_neurons`` vector of trajectory ANOVA p-values.

    Convenience wrapper around :func:`compute_trajectory_ancova_per_neuron`.
    """
    return compute_trajectory_ancova_per_neuron(path, **kwargs)['p_trajectory']


def plot_trajectory_ancova_pvalue_histogram(path, cell_type='SC', max_crossings=16,
                                          p_threshold=SC_P_THRESHOLD, bins=20,
                                          ancova_results=None, save_path=None,
                                          show=True, ax=None, ylim=1, title=None):
    """
    Histogram of per-neuron trajectory ANCOVA p-values (LR vs RL) for the reservoir.

    Parameters
    path : str
    cell_type, max_crossings
        Passed to :func:`compute_trajectory_ancova_per_neuron` if ``ancova_results``
        is not provided.
    p_threshold : float
        Vertical line marking the splitter-cell cutoff (default 1e-4).
    bins : int
        Number of histogram bins on [0, 1].
    ancova_results : dict or None
        Precomputed output from :func:`compute_trajectory_ancova_per_neuron`.
    save_path : str or None
        If set, save the figure to this path.
    show : bool
        Call ``plt.show()`` when True.
    ax : matplotlib.axes.Axes or None
        Axis to draw on; if None, a new figure is created.
    ylim : float
        Right limit of the x-axis (p-value zoom; default 1).
    title : str or None
        Panel title; default shows n and count below ``p_threshold``.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes (``fig`` is None if ``ax`` was supplied)
    ancova_results : dict
    """
    if ancova_results is None:
        ancova_results = compute_trajectory_ancova_per_neuron(
            path, cell_type=cell_type, max_crossings=max_crossings)

    p = np.asarray(ancova_results['p_trajectory'], dtype=float)
    p = p[np.isfinite(p)]
    n_sig = int(np.sum(p < p_threshold))

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    else:
        fig = ax.figure

    ax.hist(p, bins=bins, range=(0, 1), color=COLORS['LR'], edgecolor='white',
            alpha=0.85, linewidth=0.6)
    ax.axvline(p_threshold, color=COLORS['RL'], ls='--', lw=1.5,
               label=f'Splitter threshold ({p_threshold:g})')
    ax.set_xlabel('Trajectory ANCOVA p-value (LR vs RL)')
    ax.set_ylabel('Number of neurons')
    sc_line = f'{n_sig} splitter cell{"s" if n_sig != 1 else ""} detected (p < {p_threshold:g})'
    if title is None:
        title = f'Distribution of trajectory p-values\n({sc_line})'
    else:
        title = f'{title}\n{sc_line}'
    ax.set_title(title)
    ax.legend(frameon=False, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, ylim)

    if own_fig:
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        if show:
            plt.show()

    return fig, ax, ancova_results


def plot_trajectory_ancova_pvalue_histogram_panels(path_kwargs_list, save_path=None,
                                                   show=True, figsize=(14, 4.5)):
    """
    Side-by-side trajectory ANCOVA p-value histograms.

    Parameters
    ----------
    path_kwargs_list : list of (path, kwargs)
        Each entry is a reservoir folder path and a dict of keyword arguments
        forwarded to :func:`plot_trajectory_ancova_pvalue_histogram` (e.g.
        ``bins``, ``ylim``, ``title``, ``max_crossings``).
    save_path, show, figsize
        Applied to the combined figure.

    Returns
    -------
    fig, axes, results_list
    """
    n = len(path_kwargs_list)
    fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)
    axes = axes.ravel()
    results_list = []
    for ax, (path, kw) in zip(axes, path_kwargs_list):
        kw = dict(kw)
        kw.setdefault('p_threshold', SC_P_THRESHOLD)
        _, _, res = plot_trajectory_ancova_pvalue_histogram(
            path, ax=ax, show=False, **kw)
        results_list.append(res)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    if show:
        plt.show()
    return fig, axes, results_list


def find_splitter_cells_ANCOVA(path, save=False, type='SC',
                               p_threshold=SC_P_THRESHOLD,
                               max_crossings=16):
    """
    Identify splitter cells: neurons with strong LR/RL trajectory effect (ANCOVA).

    Uses :func:`compute_trajectory_ancova_per_neuron`; cells with
    ``p_trajectory < p_threshold`` (default 1e-4) are labelled splitter cells,
    ranked by descending trajectory F statistic.
    """
    ancova = compute_trajectory_ancova_per_neuron(
        path, cell_type=type, max_crossings=max_crossings)

    splitter_cells = []
    f_values = []
    for i, p_traj in enumerate(ancova['p_trajectory']):
        if np.isfinite(p_traj) and p_traj < p_threshold:
            splitter_cells.append(i)
            f_values.append(ancova['f_trajectory'][i])

    ranked_splitter_cells = [
        cell for _, cell in sorted(zip(f_values, splitter_cells), reverse=True)]

    print('Number of splitter cells:', len(splitter_cells))
    if save:
        np.save(
            arr=np.asarray(splitter_cells, dtype=np.intp),
            file=path + 'splitter_cells_index.npy')

    return ranked_splitter_cells


def plot_raster(ax, activities, colors, dt=0.1):
    """
    Plots a raster plot of spike times for given neural activity.

    Parameters:
    - ax: Matplotlib axis to plot on
    - activities: List of neural activity arrays (one per neuron)
    - colors: List of colors for each neuron
    - dt: Time step in seconds
    """
    for j, (activity, color) in enumerate(zip(activities, colors)):
        if np.all(activity == 0):  # Skip completely silent neurons
            continue

        # Normalize activity within its range
        min_act, max_act = np.min(activity), np.max(activity)
        if max_act - min_act == 0:
            normalized_activity = np.zeros_like(activity)
        else:
            normalized_activity = (activity - min_act) / (max_act - min_act)

        # Set firing rates dynamically based on activity range
        baseline_rate = max(1, np.percentile(activity, 5))  # 5th percentile as baseline (at least 1 Hz)
        max_rate_change = max(5, np.percentile(activity, 95) - baseline_rate)  # Adaptive dynamic range

        # Compute firing rates
        firing_rates = baseline_rate + normalized_activity * max_rate_change
        firing_rates = np.clip(firing_rates, 0, None)  # Ensure non-negative rates

        # Generate spike times using Poisson process
        spike_times = np.where(np.random.poisson(firing_rates * dt) > 0)[0]

        # Plot spikes
        ax.vlines(spike_times, j + 0.7, j + 1.3, color=color)

    ax.set_ylim(0.5, len(activities) + 0.5)
    ax.set_xticks([])
    ax.grid(False)


def plot_splitter_activity_evolution_and_raster(path, indexes):
    positions = load_positions(path)
    reservoir_activities = load_reservoir_states(path)
    traj_idx = find_trajectory_indexes(positions=positions)

    num_cells = len(indexes)
    ncols = 3
    nrows = (num_cells + ncols - 1) // ncols  # Calculate rows needed for 3 columns

    # Create the figure with custom GridSpec for subplots
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle('Splitter cell activities in the central corridor')

    for i, cell in enumerate(indexes):
        row = i // ncols
        col = i % ncols
        gs = GridSpec(2, 1, height_ratios=[4, 1], hspace=0.05)  # 4:1 height ratio
        outer_grid = GridSpec(nrows, ncols, figure=fig)[row, col].subgridspec(2, 1, height_ratios=[4, 1])

        ax_activity = fig.add_subplot(outer_grid[0])  # Top for activity
        ax_raster = fig.add_subplot(outer_grid[1])  # Bottom for raster

        all_reservoir_traj = {}
        min_lengths = {}

        for key in ['rl', 'lr', 'rr', 'll']:
            if key in traj_idx and traj_idx[key]:
                all_reservoir_traj[key] = [reservoir_activities.T[cell][traversal] for traversal in traj_idx[key]]
                min_lengths[key] = min(len(traversal) for traversal in all_reservoir_traj[key])

        # Trim all traversals to the smallest length
        for key in all_reservoir_traj:
            all_reservoir_traj[key] = [traversal[:min_lengths[key]] for traversal in all_reservoir_traj[key]]

        for key, val in all_reservoir_traj.items():
            mean = np.mean(val, axis=0)
            std = np.std(val, axis=0)
            x = np.arange(len(mean))
            # Plot mean and std shading for 'From right'
            ax_activity.plot(x, mean, color=COLORS[key.upper()], alpha=0.7, label=key)
            ax_activity.fill_between(x, mean - std, mean + std,
                                     color=COLORS[key.upper()], alpha=0.2)

        ax_activity.text(0.1, 0.95, f'Cell #{cell}', transform=ax_activity.transAxes, ha='center', fontsize=10)

        ax_activity.set_xticks([])
        ax_activity.set_yticks([])
        ax_activity.grid(True, linestyle='--', alpha=0.5)
        ax_activity.spines['top'].set_visible(False)
        ax_activity.spines['right'].set_visible(False)

        plot_raster(ax_raster, (all_reservoir_traj['rl'][0], all_reservoir_traj['lr'][0]), (COLORS['RL'], COLORS['LR']))  # Right traversal raster

        ax_raster.set_yticks([])
        ax_raster.spines['top'].set_visible(False)
        ax_raster.spines['right'].set_visible(False)
        ax_raster.spines['bottom'].set_visible(False)
        ax_raster.spines['left'].set_visible(False)

        if i % ncols == 0:
            ax_raster.set_ylabel('Raster')
            ax_activity.set_ylabel('Firing activity')

        if i > ncols - 1:
            ax_raster.set_xlabel('Time')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_splitter_activity_retro_pro(path,
                                     indexes=[259, 709, 166, 219, 259, 335, 453, 761]):
    num_cells = len(indexes)
    num_cols = int(np.ceil(np.sqrt(num_cells)))
    num_rows = int(np.ceil((num_cells) / num_cols))
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, 10), squeeze=False)
    axes = axes.flatten()
    plot_idx = 0
    positions = load_positions(path)
    reservoir_activities = load_reservoir_states(path)
    traj_idx = find_trajectory_indexes(positions=positions, min_y=140, max_y=160)

    handles, labels = [], []

    for j, cell in enumerate(indexes):
        if plot_idx >= len(axes):
            break
        ax = axes[plot_idx]
        plot_idx += 1

        all_reservoir_traj = {}
        min_lengths = {}
        for key in ['rl', 'lr', 'rr', 'll']:
            if key in traj_idx and traj_idx[key]:
                all_reservoir_traj[key] = [reservoir_activities.T[cell][traversal] for traversal in traj_idx[key]]
                min_lengths[key] = min(len(traversal) for traversal in all_reservoir_traj[key])

        for key in all_reservoir_traj:
            all_reservoir_traj[key] = [traversal[:min_lengths[key]] for traversal in all_reservoir_traj[key]]

        for key, val in all_reservoir_traj.items():
            mean = np.mean(val, axis=0)
            std = np.std(val, axis=0)
            x = np.arange(len(mean))
            line, = ax.plot(x, mean, color=COLORS[key.upper()], alpha=0.7, label=key.upper())
            ax.fill_between(x, mean - std, mean + std, color=COLORS[key.upper()], alpha=0.2)

            if j == 0:
                handles.append(line)
                labels.append(key.upper())

        ax.text(0.2, 0.05, f'Cell #{cell}', transform=ax.transAxes, ha='center', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if plot_idx % num_cols == 1:
            ax.set_ylabel('Firing activity')

    fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.02, 1, 1])
    plt.show()


def plot_SC_count_evolution(evol_param='connectivity'):
    leak_rate_vals = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    connectivity_vals = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    colors = ('C1', 'C4', 'C3')
    if evol_param == 'lr':
        for i,connectivity in enumerate(connectivity_vals):
            means, stds = [], []

            for lr in leak_rate_vals:
                path = f"../data/R-L/no_cues/forced_mode/connectivity_{connectivity}/lr_{lr}/"
                all_sc = []

                for seed in range(1, 11):
                    seeded_path = path + f'seed_{seed}/'
                    sc = np.load(seeded_path + 'splitter_cells_index.npy')
                    all_sc.append(len(sc))

                mean_val = np.mean(all_sc)
                std_val = np.std(all_sc)

                means.append(mean_val)
                stds.append(std_val)

            means = np.array(means)
            stds = np.array(stds)

            plt.plot(leak_rate_vals, means, 'x-', label=f'Connectivity={connectivity}', c=colors[i])
            plt.fill_between(leak_rate_vals, means - stds, means + stds, alpha=0.2, color=colors[i])

        plt.xlabel("Leak rate")
        plt.title("Splitter cell analysis with fixed connectivity")
        plt.legend()
    elif evol_param == 'connectivity':
        for i, lr in enumerate((0.1, 1)):
            means, stds = [], []
            for connectivity in connectivity_vals:
                path = f"../data/R-L/no_cues/forced_mode/lr_{lr}/connectivity_{connectivity}/"
                all_sc = []

                for seed in range(1, 11):
                    seeded_path = path + f'seed_{seed}/'
                    sc = np.load(seeded_path + 'splitter_cells_index.npy')
                    all_sc.append(len(sc))

                mean_val = np.mean(all_sc)
                std_val = np.std(all_sc)

                means.append(mean_val)
                stds.append(std_val)

            means = np.array(means)
            stds = np.array(stds)

            plt.plot(connectivity_vals, means, 'x-', label=f'Leak rate={lr}', color= colors[i])
            plt.fill_between(connectivity_vals, np.maximum(means - stds, 0), means + stds, alpha=0.2, color=colors[i])

        plt.xlabel("Connectivity")
        plt.title("Splitter cell analysis with fixed leak rate")
        plt.legend()
    elif evol_param == 'sr':
        means, stds = [], []
        sr_vals = [0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1, 1.1]

        for sr in sr_vals:
            path = f"../data/R-L/no_cues/forced_mode/sr_evol/sr_{sr}/"
            all_sc = []

            for seed in range(1, 11):
                seeded_path = path + f'seed_{seed}/'
                sc = np.load(seeded_path + 'splitter_cells_index.npy')
                all_sc.append(len(sc))

            mean_val = np.mean(all_sc)
            std_val = np.std(all_sc)

            means.append(mean_val)
            stds.append(std_val)

        means = np.array(means)
        stds = np.array(stds)

        plt.plot(sr_vals, means, 'x-')
        plt.fill_between(sr_vals, means - stds, means + stds, alpha=0.2)

        plt.xlabel("Spectral radius")
        plt.title("Splitter cell analysis with fixed leak rate and connectivity")

    else:
        Exception, "Fixed parameter name not recognized"

    plt.ylabel("Splitter Cell Count")

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(False)
    plt.show()


def count_all_SC(evol_param='connectivity'):
    if evol_param == 'lr':
        leak_rate_vals = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
        for connectivity in (0.1, 0.5,1):
            print(f'Connectivity={connectivity}')
            for lr in leak_rate_vals:
                print(f'Leak rate={lr}')
                for seed in np.arange(1, 11):
                    path = f"../data/R-L/no_cues/forced_mode/lr_evol/lr_{lr}/seed_{seed}/"
                    splitter_cells = find_splitter_cells_ANCOVA(path=path, save=True, type='SC')
    elif evol_param == 'connectivity':
        print('Fixed leak rate')
        connectivity_vals = [0, 0.1, 0.2, 0.5, 0.7, 0.8, 1]
        for lr in (0.1,  1):
            for connectivity in connectivity_vals:
                for seed in np.arange(1, 11):
                    path = f"../data/R-L/no_cues/forced_mode/co_evol/connectivity_{connectivity}/seed_{seed}/"
                    splitter_cells = find_splitter_cells_ANCOVA(path=path, save=True, type='SC')
    elif evol_param == 'sr':
        sr_vals = [0.8, 0.9, 0.95, 0.99]
        sr_vals = [0.6, 0.7, 1, 1.1]

        for sr in sr_vals:
            for seed in np.arange(1, 11):
                path = f"../data/R-L/no_cues/forced_mode/sr_evol/sr_{sr}/seed_{seed}/"
                splitter_cells = find_splitter_cells_ANCOVA(path=path, save=True, type='SC')


def plot_decoder_accuracy_evolution(evol_param='sr'):
    sr_vals = [0.6, 0.7, 0.8, 0.9, 0.95, 1, 1.1]
    lr_vals = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    co_vals = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    connectivity = 1
    lr = 1
    sr = 0.99
    #all_vals = lr_vals
    all_vals = co_vals

    means_accuracy_pos = []
    means_accuracy_dec = []
    means_accuracy_or = []

    stds_accuracy_pos = []
    stds_accuracy_dec = []
    stds_accuracy_or = []
    for val in all_vals:
        accuracy_pos = []
        accuracy_dec = []
        accuracy_or = []
        for seed in np.arange(1, 11):
            #path = f'../data/R-L/no_cues/forced_mode/sr_evol/sr_{val}/seed_{seed}/'
            #path = f'../data/R-L/no_cues/forced_mode/lr_evol/connectivity_{connectivity}/lr_{val}/seed_{seed}/'
            path = f'../data/R-L/no_cues/forced_mode/co_evol/lr_{lr}/connectivity_{val}/seed_{seed}/'
            accuracy = np.load(path + 'decoder_accuracy.npy', allow_pickle=True).item()
            accuracy_pos.append(accuracy['position'])
            accuracy_dec.append(accuracy['decision'])
            accuracy_or.append(accuracy['orientation'])
        means_accuracy_pos.append(np.mean(accuracy_pos))
        means_accuracy_dec.append(np.mean(accuracy_dec))
        means_accuracy_or.append(np.mean(accuracy_or))
        stds_accuracy_pos.append(np.std(accuracy_pos))
        stds_accuracy_dec.append(np.std(accuracy_dec))
        stds_accuracy_or.append(np.std(accuracy_or))

    # Plot all three accuracies in the same figure
    plt.figure(figsize=(8, 6))

    plt.plot(all_vals, means_accuracy_pos, 'x-', label="Position")
    plt.fill_between(all_vals, np.array(means_accuracy_pos) - np.array(stds_accuracy_pos),
                     np.array(means_accuracy_pos) + np.array(stds_accuracy_pos), alpha=0.2)

    plt.plot(all_vals, means_accuracy_dec, 'o-', label="Decision")
    plt.fill_between(all_vals, np.array(means_accuracy_dec) - np.array(stds_accuracy_dec),
                     np.array(means_accuracy_dec) + np.array(stds_accuracy_dec), alpha=0.2)

    plt.plot(all_vals, means_accuracy_or, 's-', label="Orientation")
    plt.fill_between(all_vals, np.array(means_accuracy_or) - np.array(stds_accuracy_or),
                     np.array(means_accuracy_or) + np.array(stds_accuracy_or), alpha=0.2)

    plt.ylim(0.7, 1)
    plt.ylabel("Decoder accuracy")
    #plt.xlabel("Spectral radius")
    #plt.xlabel("Leak rate")
    plt.xlabel("Connectivity")
    #plt.title("Decoder Accuracy vs. Spectral radius")
    #plt.title("Decoder Accuracy vs. Leak rate")
    plt.title("Decoder Accuracy vs. Connectivity")
    plt.legend()
    plt.grid()
    plt.show()


def count_all_SC():
    all_sc = []
    for i in (2, 5, 10, 21, 22, 27, 30, 37, 38, 39):
        file = '/Users/nchaix/Documents/PhD/code/splitter_cells_in_reservoir_march_2025_tests/data/R-L/no_cues/'
        file += f'seed_{i}/reservoir_states_1_killed_bis/'
        sc_list = np.load(file + 'splitter_cells_index_bis.npy')
        n_sc = len(sc_list)
        print(sc_list)
        all_sc.append(n_sc)
    print(np.mean(all_sc))
    print(np.std(all_sc))


# ---------------------------------------------------------------------------
# Lesion-series analysis helpers
# ---------------------------------------------------------------------------
import os, re
from scipy.interpolate import interp1d


def _count_crossings(path):
    """Count LR and RL corridor crossings for one reservoir-states folder.

    Returns
    -------
    (n_lr, n_rl) : int, int
    """
    try:
        pos = load_positions(path)
        loc, loc_idx = find_location_indexes_xy(pos[:, 0], pos[:, 1])
        ar = find_activity_ranges(loc, loc_idx)
        return len(ar.get('LR', [])), len(ar.get('RL', []))
    except Exception:
        return 0, 0


def _infer_n_total_from_root(root):
    """Read reservoir width (number of neurons) from the first available seed."""
    for seed_dir in sorted(os.listdir(root)):
        if not re.match(r'seed_\d+$', seed_dir):
            continue
        f = os.path.join(root, seed_dir, 'reservoir_states', 'reservoir_states.npy')
        if os.path.isfile(f):
            return int(np.load(f, mmap_mode='r').shape[1])
    return None


def _load_lesion_data(root, n_total=None, exclude_seeds=None, min_crossings=10):
    """Load SC counts and compute n_active across lesion iterations for all seeds.

    An iteration is included only when both LR and RL have at least
    *min_crossings* corridor crossings.  Neurons killed at a skipped
    iteration are still accumulated in the cumulative-kill counter so that
    n_active remains correct for subsequent iterations.

    Parameters
    ----------
    n_total : int or None
        Reservoir size. If None, inferred from ``reservoir_states.npy`` shape.

    Returns
    -------
    dict  seed -> {'iters': list[int], 'n_sc': list[int],
                   'n_active': list[int], 'prop': list[float]}
    """
    if exclude_seeds is None:
        exclude_seeds = set()
    if n_total is None:
        n_total = _infer_n_total_from_root(root)
        if n_total is None:
            raise ValueError(f'Could not infer n_total from reservoir_states in {root}')
    data = {}
    for seed_dir in sorted(os.listdir(root)):
        m = re.match(r'seed_(\d+)$', seed_dir)
        if not m:
            continue
        seed = int(m.group(1))
        if seed in exclude_seeds:
            continue
        seed_path = os.path.join(root, seed_dir)
        raw = {}          # it -> n_sc
        subdir_map = {}   # it -> subfolder name  (for crossing check)
        for sub in os.listdir(seed_path):
            f = os.path.join(seed_path, sub, 'splitter_cells_index.npy')
            if not os.path.exists(f):
                continue
            if sub == 'reservoir_states':
                it = 0
            else:
                m2 = re.match(r'reservoir_states_1_killed_(\d+)$', sub)
                if not m2:
                    continue
                it = int(m2.group(1))
            raw[it] = len(np.load(f, allow_pickle=True))
            subdir_map[it] = sub

        iters_out, n_sc, n_active, prop = [], [], [], []
        n_act = n_total
        cumkilled = 0
        for it in sorted(raw):
            sc = raw[it]
            path = os.path.join(seed_path, subdir_map[it]) + os.sep
            n_lr, n_rl = _count_crossings(path)
            if min(n_lr, n_rl) >= min_crossings:
                iters_out.append(it)
                n_sc.append(sc)
                n_active.append(n_act)
                prop.append(sc / n_act if n_act > 0 else np.nan)
            # always advance the kill counter — those neurons are gone regardless
            cumkilled += sc
            n_act = n_total - cumkilled

        data[seed] = dict(iters=iters_out, n_sc=n_sc, n_active=n_active, prop=prop)
    return data


def plot_lesion_series(root='../data/R-L/no_cues/5000_units',
                       n_total=None,
                       exclude_seeds=None,
                       save=False,
                       min_crossings=10):
    """Figure 1 – absolute lesion iteration on x-axis.

    Three panels in one row: (A) number of splitter cells, (B) total active cells,
    (C) N valid seeds per iteration. Individual seed traces in light grey;
    mean ± std as coloured band. N seeds per point annotated below x-axis.
    """
    if exclude_seeds is None:
        exclude_seeds = {2}          # default: exclude collapsed seed

    data = _load_lesion_data(root, n_total=n_total, exclude_seeds=exclude_seeds, min_crossings=min_crossings)

    # ---- collect per-iteration arrays ----
    max_iter = max(it for d in data.values() for it in d['iters'])
    all_iters = np.arange(0, max_iter + 1)

    def gather(key):
        """Return (matrix NaN-padded, N per iter) where rows = seeds."""
        rows = []
        for d in data.values():
            row = np.full(max_iter + 1, np.nan)
            for it, v in zip(d['iters'], d[key]):
                row[it] = v
            rows.append(row)
        mat = np.array(rows)
        n_per = np.sum(~np.isnan(mat), axis=0)
        return mat, n_per

    sc_mat, n_per = gather('n_sc')
    act_mat, _ = gather('n_active')

    mean_sc = np.nanmean(sc_mat, axis=0)
    std_sc = np.nanstd(sc_mat, axis=0)
    mean_act = np.nanmean(act_mat, axis=0)
    std_act = np.nanstd(act_mat, axis=0)

    BLUE = '#2E86AB'
    GREEN = '#3BB273'
    GREY = '#B0B0B0'

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.subplots_adjust(wspace=0.35, top=0.78, bottom=0.22)

    def _panel_label(ax, letter):
        ax.text(0.02, 1.06, letter, transform=ax.transAxes,
                fontsize=13, fontweight='bold', va='bottom', ha='left',
                clip_on=False)

    def _plot_mean_trace(ax, mat, mean, std, ylabel, color, ylim_pad=None):
        for row in mat:
            valid = ~np.isnan(row)
            ax.plot(all_iters[valid], row[valid],
                    color=GREY, lw=0.8, alpha=0.55, zorder=1)
        valid = ~np.isnan(mean)
        ax.fill_between(all_iters[valid],
                        (mean - std)[valid], (mean + std)[valid],
                        color=color, alpha=0.20, zorder=2)
        ax.plot(all_iters[valid], mean[valid],
                color=color, lw=2.2, label='Mean ± std', zorder=3)
        ax.axhline(mean[0], color=color, lw=1, ls='--', alpha=0.5)
        ax.set_ylabel(ylabel)
        ax.set_xticks(all_iters)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(frameon=False, fontsize=10)
        if ylim_pad is not None:
            lo, hi = ylim_pad
            ax.set_ylim(min(mean[valid]) - lo, max(mean[valid]) + hi)
        for it in all_iters:
            ax.annotate(f'N={int(n_per[it])}',
                        xy=(it, -0.12), xycoords=('data', 'axes fraction'),
                        ha='center', va='top', fontsize=7.5, color='#555555',
                        clip_on=False)
        ax.set_xlabel('Lesion iteration', labelpad=18)

    _plot_mean_trace(axes[0], sc_mat, mean_sc, std_sc,
                     'Number of splitter cells', BLUE, ylim_pad=(50, 250))
    _panel_label(axes[0], 'A')
    _plot_mean_trace(axes[1], act_mat, mean_act, std_act,
                     'Total active cells', GREEN)
    _panel_label(axes[1], 'B')

    ax = axes[2]
    ax.bar(all_iters, n_per, color=BLUE, alpha=0.75, width=0.6)
    ax.set_ylabel('N seeds')
    ax.set_xticks(all_iters)
    ax.set_yticks(np.arange(0, int(n_per.max()) + 2, 2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('Valid seeds per iteration', fontsize=10, pad=12)
    ax.set_xlabel('Lesion iteration', labelpad=8)
    _panel_label(ax, 'C')

    fig.suptitle('Splitter cell dynamics across lesion iterations', y=0.92)
    if save:
        fig.savefig(os.path.join(root, 'lesion_series_absolute.pdf'),
                    bbox_inches='tight')
    plt.show()


def plot_lesion_series_normalised(root='../data/R-L/no_cues/5000_units',
                                  n_total=None,
                                  exclude_seeds=None,
                                  ylim = 0.05,
                                  n_interp=21,
                                  save=False,
                                  min_crossings=10):
    """Figure 2 – x-axis normalised to [0, 1] per seed (fraction of lesion series).

    Two panels: (A) number of splitter cells, (B) proportion SC / active cells.
    Individual seed traces shown in light grey; mean ± std as coloured band.
    Interpolation uses linear interpolation onto a common *n_interp*-point grid.
    """
    if exclude_seeds is None:
        exclude_seeds = {2}

    data = _load_lesion_data(root, n_total=n_total, exclude_seeds=exclude_seeds, min_crossings=min_crossings)
    x_common = np.linspace(0, 1, n_interp)

    def interpolate_seeds(key):
        rows = []
        for d in data.values():
            iters = np.array(d['iters'], dtype=float)
            vals  = np.array(d[key],    dtype=float)
            if len(iters) < 2:
                # single point: flat line
                rows.append(np.full(n_interp, vals[0]))
                continue
            x_norm = (iters - iters[0]) / (iters[-1] - iters[0])
            f = interp1d(x_norm, vals, kind='linear', bounds_error=False,
                         fill_value=(vals[0], vals[-1]))
            rows.append(f(x_common))
        return np.array(rows)

    sc_mat   = interpolate_seeds('n_sc')
    prop_mat = interpolate_seeds('prop')
    act_mat  = interpolate_seeds('n_active')

    mean_sc   = np.mean(sc_mat,   axis=0)
    std_sc    = np.std(sc_mat,    axis=0)
    mean_prop = np.mean(prop_mat, axis=0)
    std_prop  = np.std(prop_mat,  axis=0)
    mean_act  = np.mean(act_mat,  axis=0)
    std_act   = np.std(act_mat,   axis=0)

    BLUE   = '#2E86AB'
    ORANGE = '#E07A5F'
    GREEN  = '#3BB273'
    GREY   = '#B0B0B0'
    XLABEL = 'Normalised lesion progression (0 = baseline, 1 = last lesion)'

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.subplots_adjust(wspace=0.35, hspace=0.5)

    for ax, mat, mean, std, ylabel, color in [
        (axes[0, 0], sc_mat,   mean_sc,   std_sc,
         'Number of splitter cells',      BLUE),
        (axes[0, 1], prop_mat, mean_prop, std_prop,
         'Proportion SC / active cells',  ORANGE),
        (axes[1, 0], act_mat,  mean_act,  std_act,
         'Total active cells',            GREEN),
    ]:
        for row in mat:
            ax.plot(x_common, row, color=GREY, lw=0.8, alpha=0.55, zorder=1)

        ax.fill_between(x_common, mean - std, mean + std,
                        color=color, alpha=0.20, zorder=2)
        ax.plot(x_common, mean, color=color, lw=2.2,
                label='Mean ± std', zorder=3)
        ax.axhline(mean[0], color=color, lw=1, ls='--', alpha=0.5)

        ax.set_xlabel(XLABEL)
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, ylim)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(frameon=False, fontsize=10)

    # bottom-right: N seeds per interpolation is meaningless (all seeds contribute
    # to every x_common point after interpolation), so show the raw N-per-iter bar
    # chart as a reference panel instead.
    max_iter = max(it for d in data.values() for it in d['iters'])
    all_iters_raw = np.arange(0, max_iter + 1)
    n_per_raw = np.array([
        sum(1 for d in data.values() if it in d['iters'])
        for it in all_iters_raw
    ])
    ax = axes[1, 1]
    ax.bar(all_iters_raw, n_per_raw, color=BLUE, alpha=0.75, width=0.6)
    ax.set_xlabel('Lesion iteration (raw)')
    ax.set_ylabel('N seeds')
    ax.set_xticks(all_iters_raw)
    ax.set_yticks(np.arange(0, int(n_per_raw.max()) + 2, 2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('Valid seeds per iteration', fontsize=10)

    fig.suptitle('Splitter cell dynamics - normalised lesion series', y=1.01)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(root, 'lesion_series_normalised.pdf'),
                    bbox_inches='tight')
    plt.show()



if __name__ == '__main__':

    #plot_decoder_accuracy_evolution(evol_param='sr')

    # 1- Find splitter cells
    #path = f"../data/R-L/no_cues/5000_units/seed_15/reservoir_states_1_killed_4/"
    #splitter_cells = find_splitter_cells_ANCOVA(path=path, save=False, type='SC')

    # 2- Plot splitter cells
    # plot_splitter_activity_evolution_and_raster(path=
    #   "../data/R-L/no_cues/seed_1/reservoir_states/", indexes=[243, 51, 358, 432, 620, 675])

    # 3- After lesioning splitter cells through the simulation, compare the activities of the cells
    # Observe the reorganisation of the dynamics: some splitter cells became not splitter cells anymore in the new model,
    # and new splitter cells appear

    # dynamics_reorganisation_SC(file_init_model="../data/R-L/cues/seed_1234/reservoir_states/",
    #                           file_lesioned_model="../data/R-L/cues/seed_1234/reservoir_states_1_killed/",
    #                          save=False)

    # 4- Plot to have a visualisation of this reorganization
    # plot_splitter_activity_evolution_corridor_grid_comparison(
    # "../data/R-L/cues/seed_1234/reservoir_states/",
    # "../data/R-L/cues/seed_1234/reservoir_states_1_killed/", indexes=[149, 719, 663, 302, 186])

    # 5 - Plot the cells during the RR-LL task to differentiate retrospective and prospective cells.
    #path = "../data/RR-LL/cues/seed_5/reservoir_states/"
    #plot_splitter_activity_retro_pro(path, indexes=np.concatenate((pro, retro, rest)))

    #plot_SC_count_evolution(fixed_param='lr')

    # Investigate the evolution of the number of splitter cels when varying one parameter of the resevoir
    #count_all_SC(evol_param='sr')
    #plot_SC_count_evolution(evol_param='sr')

    #for seed in (5,6,7,8,9,10,11,12,13,14,15,1234):
    #    print('seed',seed)
    #    path = f"../data/RR-LL/cues/seed_{seed}/reservoir_states/"
    #    splitter_cells = find_splitter_cells_ANCOVA(path=path, save=False, type='SC')
    #    print(len(splitter_cells))

    #idx_kill = 0
    #for seed in np.arange(0, 10):
    #    print('Seed:', seed)
    #    path = f'../data/R-L/no_cues/forced_mode/kill_multiple_times/seed_{seed}/kill_{idx_kill}/'
    #    splitter_cells = find_splitter_cells_ANCOVA(path=path, save=False, type='SC')
    #    print(len(splitter_cells))

    #for seed in np.arange(0, 10):
    #    path = f'../data/R-L/no_cues/forced_mode/kill_multiple_times/seed_{seed}/kill_0/'
    #    splitter_cells = find_splitter_cells_ANCOVA(path=path, save=True, type='SC')


    # Folder 5000_units: multiple lesion iterations
    root = '../data/R-L/no_cues/3000_units_lr_04'
    # Figure 1 – absolute x-axis + N-seeds companion panel
    #plot_lesion_series(root=root, save=True, min_crossings=10)
    # Figure 2 – normalised x-axis [0→1]
    #plot_lesion_series_normalised(root=root, save=True, min_crossings=15)

    # Count splitter cells in:
    """stds = []
    means = []
    ratios_means = []
    ratios_stds = []
    init_total = {i:1000 for i in range(10)}
    means_decoder = {'position': [],
               'orientation': [],
               'decision': []}
    stds_decoder = {'position': [],
                     'orientation': [],
                     'decision': []}
    for j in range(10):
        all_n = []
        all_ratio = []
        all_decoder = {'position': [],
               'orientation': [],
               'decision': []}
        for i in range(10):
            file = f'/Users/nchaix/Documents/PhD/code/' \
                   f'splitter_cells_in_reservoir_march_2025_tests/data/R-L/' \
                   f'no_cues/forced_mode/kill_multiple_times/seed_{i}/kill_{j}/'

            if os.path.exists(file):
                sc = np.load(file + 'splitter_cells_index_bis.npy')
                decoder = np.load(file + 'input_decoder_decoder_accuracy.npy', allow_pickle=True)
                decoder = decoder.item()
                for key, item in decoder.items():
                    all_decoder[key].append(decoder[key])
                n_sc = len(sc)
                ratio = 100*n_sc/init_total[i]
                #init_total[i] -= n_sc
                all_n.append(n_sc)
                all_ratio.append(ratio)
            else:
                pass
        for key, item in decoder.items():
            means_decoder[key].append(np.mean(all_decoder[key]))
            stds_decoder[key].append(np.std(all_decoder[key]))
        ratios_means.append(np.mean(all_ratio))
        ratios_stds.append(np.std(all_ratio))
        means.append(np.mean(all_n))
        stds.append(np.std(all_n))"""

    """means = np.array(means)
    stds = np.array(stds)
    x = np.arange(len(means))
    plt.ylabel('Number of splitter cells')
    plt.xlabel('Lesion iteration (cumulative splitter cell removals)')
    plt.title('Number of new splitter cells after cumulative lesion')
    plt.plot(x, means)
    plt.fill_between(x, means-stds, means+stds, alpha=0.3)
    plt.show()"""

    """ratios_means = np.array(ratios_means)
    ratios_stds = np.array(ratios_stds)
    x = np.arange(len(ratios_means))
    plt.plot(x, ratios_means)
    plt.fill_between(x, ratios_means - ratios_stds, ratios_means + ratios_stds, alpha=0.3)
    plt.show()"""
    """markers = {'position':'x-', 'decision':'o-', 'orientation': 's-'}
    for key, item in decoder.items():
        means_decoder[key] = np.array(means_decoder[key])
        stds_decoder[key] = np.array(stds_decoder[key])
        print(means_decoder[key] )
        x = np.arange(len(means_decoder[key]))
        plt.plot(x, means_decoder[key], markers[key], label=key)
        plt.fill_between(x, means_decoder[key] - stds_decoder[key], means_decoder[key] + stds_decoder[key],
                         alpha=0.3)
    plt.ylim(0.7, 1)
    plt.title('Decoder accuracy after cumulative lesion')
    plt.xlabel('Lesion iteration (cumulative splitter cell removals)')
    plt.ylabel('Decoder accuracy')
    plt.legend()
    plt.show()"""

    path = '../data/R-L/no_cues/3000_units_lr_04/seed_1/reservoir_states/'
    #fig, ax, results = plot_trajectory_ancova_pvalue_histogram(
    #path,
    #    p_threshold=SC_P_THRESHOLD,
    #    bins=3000,
    #    save_path='ancova_pvalue_histogram.pdf',  # optional
    #)

    #path = '../data/R-L/no_cues/3000_units_lr_04/seed_6/reservoir_states_1_killed_3/'
    #plot_trajectory_ancova_pvalue_histogram(
    #    path,
    #    bins=1500,
    #    ylim=0.01,
    #    save_path='../data/R-L/no_cues/3000_units_lr_04/fig_ancova_pvalue_hist_seed3_15cross.pdf',)


    root = '../data/R-L/no_cues/3000_units_lr_04'
    plot_trajectory_ancova_pvalue_histogram_panels(
        [
            (f'{root}/seed_1/reservoir_states/', dict(
                bins=1500,
                ylim=0.01,
                title='seed_1,  baseline (iter 0)',
            )),
            (f'{root}/seed_6/reservoir_states_1_killed_3/', dict(
                bins=1500,
                ylim=0.01,
                title='seed_6,  lesion iter 3',
            )),
        ],
        save_path=f'{root}/fig_ancova_pvalue_hist_seed1_seed6.pdf',
    )













































