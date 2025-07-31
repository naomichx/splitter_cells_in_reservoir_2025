"""
Reservoir state analysis is conducted at the population level. In this script, the reservoir states
beforehand recorded during the bot's navigation are loaded to process to the population analysis.


 The script allows several analytical processes:

- 3D PCA Analysis: principal component analysis (PCA) on the reservoir states.
                   Additional information, such as the Euclidean distance between
                  points is incorporated into the analysis.

- SVM Classification:  support vector machine (SVM) classification to categorize
                       the reservoir states based on which direction the bot will take at the next decision point

- UMAP Analysis - Central Corridor: Applies Uniform Manifold Approximation and Projection (UMAP)
                                    on the reservoir states specifically when the bot enters the central corridor.
                                    This analysis helps in distinguishing various trajectories
                                     in a higher-dimensional space.

- UMAP Analysis - Error Case: Implements UMAP on the reservoir states during error cases,
                              providing insights into the internal dynamics of the reservoir when errors occur.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score
from scipy.spatial import distance
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm, to_rgb
from matplotlib.lines import Line2D
from scipy.spatial import KDTree
import sys
sys.path.append(r'/Users/nchaix/Documents/PhD/code/splitter_cells_in_reservoir/')
from maze import Maze
plt.rc('font', size=12)


def darken_color(color, factor=0.7):
    import matplotlib.colors as mcolors
    rgb = mcolors.to_rgb(color)  # Convert color to RGB
    return tuple([factor * x for x in rgb])

# Useful functions
def split_train_test(input, output, nb_train, max_len_test=100000):
    """ Splits the input and output lists in two lists: one for the training phase, one for the testing phase.
    Inputs:
    - input: input list
    - output: output list
    - nb_train:number of element in the training lists. nb_test = len(input)-nb_train
    - max_len_test : max length of the testing list (so it is not too big).
    Outputs:
    - X_train, X_test: two lists from input that were split at the nb_train index. len(X_test) <= max_len_test
    - Y_train, Y_test: two lists from output that were split at the nb_train index. len(Y_test) <= max_len_test"""

    X_train, Y_train, X_test, Y_test = input[:nb_train], output[:nb_train], input[nb_train:], output[nb_train:]
    if len(X_test) > max_len_test:
        X_test, Y_test = X_test[:max_len_test], Y_test[:max_len_test]

    return X_train, Y_train, X_test, Y_test


def generate_legend_RR_LL(positions, min_y=145, max_y=155):
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
    # Verify on the simulation what were the two last loops done by the bot
    last_side_1 = 'r'
    last_side_2 = 'l' # Track last known side before entering the central corridor
    in_corridor = False  # Track if the bot is in the corridor
    y_positions = positions[:,1]

    for i in range(len(y_positions)):
        y = y_positions[i]
        if y > max_y:
            legend.append('l')
            in_corridor = False
            last_side_2 = 'l'
        elif y < min_y:
            legend.append('r')
            last_side_2 = 'r'
            in_corridor = False
        elif min_y <= y <= max_y:
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
            print('Issue 2')

    return np.array(legend)


def generate_legend_RL(positions, min_y=145, max_y= 155):
    """
    Generate legend based on y_positions.
    'm': middle corridor (now adjusted to take the previous corridor value)
    'r': right loop
    'l': left loop
    """
    legend = []
    previous_legend = 'r'  #  Verify on the simulation what were the last loop done by the bot
    y_positions = positions[:, 1]

    for pos in y_positions:
        if min_y <= pos <= max_y:
            #legend.append(previous_legend)  # Use previous corridor value
            if previous_legend == 'l':
                legend.append('lr')
            elif previous_legend == 'r':
                legend.append('rl')
        elif pos < min_y:
            legend.append('r')
            previous_legend = 'r'
        elif pos > max_y:
            legend.append('l')
            previous_legend = 'l'
        else:
            print('Error in labelling')

    return np.array(legend)


def plot_PCA_3D(path=None, task='R-L', plot_distance=False,
                central_corridor_only=False,
                sequence=None):
    """
    This function loads reservoir states data and positions, performs PCA to reduce dimensionality to 3D,
    and visualizes the data in a 3D line plot.

    Parameters:
    - cues (bool): Whether to include cues in the data path. Defaults to False.

    Returns:
    None
    """

    start, end = sequence[0], sequence[1]

    # Load reservoir states and positions
    res_activity = np.load(path + 'reservoir_states.npy')
    positions = np.load(path + 'positions.npy')


    if task == 'R-L':
        legend = generate_legend_RL(positions, min_y=140, max_y=160)
    elif task == 'RR-LL':
        legend = generate_legend_RR_LL(positions, min_y=130, max_y=170)
    else:
        print('Error in task settings')

    res_activity = res_activity[start:end]
    y_positions = positions[start:end,1]
    x_positions = positions[start:end,0]
    legend = legend[start:end]

    # Perform PCA
    pca = PCA(n_components=3)
    x = StandardScaler().fit_transform(res_activity)
    principalComponents = pca.fit_transform(x)

    Xax = principalComponents[:, 0]
    Yax = principalComponents[:, 1]
    Zax = principalComponents[:, 2]

    cdict = {"r": 'red', "rl": 'red', "rr": 'red', 'll': darken_color('blueviolet', factor=0.7),
             'lr': darken_color('blueviolet', factor=0.7),"l": darken_color('blueviolet', factor=0.7)}
    labl = {"r": '', "rl": 'RL', "lr": 'LR', 'll': 'LL', 'rr': 'RR', 'l': ''}
    if central_corridor_only:
        alpha_values = {"l": 0, "r": 0, "rl": 1, "lr": 1, "ll": 1, "rr": 1}
    else:
        alpha_values = {"l": 1, "r": 1, "rl": 1, "lr": 1, "ll": 1, "rr": 1}
    ls = {"l": '-', "r": '-', "rl": '-', "lr": '-', "ll": '-', "rr": '-'}

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')


    for l in np.unique(legend):
        ix = np.where(legend == l)[0]  # Get indices where legend matches label
        # Find contiguous segments
        segments = np.split(ix, np.where(np.diff(ix) > 1)[0] + 1)

        for i, segment in enumerate(segments):
            if len(segment) > 1:  # Avoid single-point segments
                ax.plot(Xax[segment], Yax[segment], Zax[segment], c=cdict[l] ,
                        alpha=alpha_values.get(l, 1.0),  label= None,
                        linestyle= ls[l], linewidth=1.5)


    if plot_distance:
        indices_rl = np.where(legend == 'rl')[0]
        indices_lr = np.where(legend == 'lr')[0]
        x_positions_rl = x_positions[indices_rl]
        x_positions_lr = x_positions[indices_lr]


        linestyles= ['--', '-.', ':']
        colors = ['C7', 'C8', 'C9']
        #for x_target, key, c in zip([200, 150, 100], ['C1', 'C2', 'C3'], colors):
        for x_target, key, c in zip([200, 100, 80], ['C1', 'C2', 'C3'], colors):
            if len(x_positions_rl) > 0 and len(x_positions_lr) > 0:
                idx_rl = indices_rl[np.argmin(np.abs(x_positions_rl - x_target))]
                idx_lr = indices_lr[np.argmin(np.abs(x_positions_lr - x_target))]

                dist = distance.euclidean((Xax[idx_rl], Yax[idx_rl], Zax[idx_rl]),
                                          (Xax[idx_lr], Yax[idx_lr], Zax[idx_lr]))

                ax.plot([Xax[idx_rl], Xax[idx_lr]], [Yax[idx_rl], Yax[idx_lr]],
                        [Zax[idx_rl], Zax[idx_lr]],
                        color=c, linestyle='--', linewidth=1, label=None)

                ax.scatter3D(Xax[idx_rl], Yax[idx_rl], Zax[idx_rl], s=30, marker='o',
                             color=c, alpha=1,label=f"{key}: {round(dist, 2)}")
                ax.scatter3D(Xax[idx_lr], Yax[idx_lr], Zax[idx_lr], s=30, marker='o',
                             color=c, alpha=1,label=None)

    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_zlabel("PCA 3")
    # ax.set_title('PCA analysis of the reservoir states')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    ax.legend(bbox_to_anchor=(1.01, 1.02))
    plt.show()


def compute_pca_distances(path=None, task='R-L', sequence=None, x_targets= [100, 110, 120,130,140, 150, 160, 170, 180], tol=3, save= False):
    """
    Compute mean PCA distance at C2 points (x ≈ 150):
    - Intra-distance within 'rl' segments.
    - Inter-distance between 'rl' and 'lr' segments.

    Parameters:
    - Xax, Yax, Zax: PCA-transformed coordinates.
    - x_positions: Original x positions.
    - legend: Trajectory labels.
    - x_target: The x-coordinate for C2 (default=150).
    - tol: Tolerance range to select C2 points (default=5).

    Returns:
    - intra_rl_dist: Mean PCA distance within 'rl' at C2.
    - inter_dist: Mean PCA distance between 'rl' and 'lr' at C2.
    """
    start, end = sequence[0], sequence[1]

    # Load reservoir states and positions
    res_activity = np.load(path + 'reservoir_states.npy')
    positions = np.load(path + 'positions.npy')

    if task == 'R-L':
        legend = generate_legend_RL(positions, min_y=149, max_y=151)
    elif task == 'RR-LL':
        legend = generate_legend_RR_LL(positions, min_y=149, max_y=151)
    else:
        print('Error in task settings')

    res_activity = res_activity[start:end]
    y_positions = positions[start:end, 1]
    x_positions = positions[start:end, 0]
    legend = legend[start:end]

    # Perform PCA
    pca = PCA(n_components=3)
    x = StandardScaler().fit_transform(res_activity)
    principalComponents = pca.fit_transform(x)

    Xax = principalComponents[:, 0]
    Yax = principalComponents[:, 1]
    Zax = principalComponents[:, 2]

    distances = {}

    for x_target in x_targets:
        distances[x_target]= {}
        # Select indices near C2 (x ≈ 150 ± tol)
        indices_rl = np.where((legend == 'rl') & (np.abs(x_positions - x_target) <= tol))[0]
        indices_lr = np.where((legend == 'lr') & (np.abs(x_positions - x_target) <= tol))[0]

        # Extract PCA points at C2
        c2_rl = np.column_stack((Xax[indices_rl], Yax[indices_rl], Zax[indices_rl]))
        c2_lr = np.column_stack((Xax[indices_lr], Yax[indices_lr], Zax[indices_lr]))

        # Compute mean PCA distance within 'rl' at C2
        if len(c2_rl) > 1:
            intra_rl_dist = np.mean([distance.euclidean(c2_rl[i], c2_rl[j])
                                     for i in range(len(c2_rl))
                                     for j in range(i + 1, len(c2_rl))])
        else:
            intra_rl_dist = np.nan  # Not enough points to compute

        # Compute mean PCA distance within 'lr' at C2
        if len(c2_rl) > 1:
            intra_lr_dist = np.mean([distance.euclidean(c2_lr[i], c2_lr[j])
                                     for i in range(len(c2_lr))
                                     for j in range(i + 1, len(c2_lr))])
        else:
            intra_lr_dist = np.nan  # Not enough points to compute

        # Compute mean PCA distance between 'rl' and 'lr' at C2
        if len(c2_rl) > 0 and len(c2_lr) > 0:
            inter_dist = np.mean([distance.euclidean(c2_rl[i], c2_lr[j])
                                  for i in range(len(c2_rl))
                                  for j in range(len(c2_lr))])
        else:
            inter_dist = np.nan  # Not enough points to compute

        print(f"Mean PCA distance within 'rl' at C2: {intra_rl_dist:.2f}")
        print(f"Mean PCA distance within 'lr' at C2: {intra_lr_dist:.2f}")
        print(f"Mean PCA distance between 'rl' and 'lr' at C2: {inter_dist:.2f}")

        distances[x_target]['intra_rl'] = intra_rl_dist
        distances[x_target]['intra_lr'] = intra_lr_dist
        distances[x_target]['inter'] = inter_dist

    if save:
        np.save(arr=distances, file=path + f'PCA_distances_corridor.npy', allow_pickle=True)


    return intra_rl_dist,intra_lr_dist, inter_dist


def plot_PCA_distance_evolution(path):
    all_intra = []
    all_inter = []
    for seed in seeds:
        print(seed)
        intra = []
        inter = []
        path += 'seed_{seed}/reservoir_states/'
        distances = np.load(path + f'PCA_distances_corridor.npy', allow_pickle=True).item()
        for x_target in x_targets:
            intra.append(np.mean((distances[x_target]['intra_lr'], distances[x_target]['intra_rl'])))
            inter.append(distances[x_target]['inter'])
        all_intra.append(intra)
        all_inter.append(inter)

    means_intra = np.mean(all_intra, axis=0)
    stds_intra = np.std(all_intra, axis=0)

    means_inter = np.mean(all_inter, axis=0)
    stds_inter = np.std(all_inter, axis=0)

    plt.scatter(x_targets, means_intra, color='grey', marker='x')
    plt.plot(x_targets, means_intra, color='grey', label='$d_{intra}$', ls='--')
    plt.fill_between(x_targets, means_intra - stds_intra, means_intra + stds_intra, color='grey', alpha=0.5)

    plt.scatter(x_targets, means_inter, color='black', marker='x')
    plt.plot(x_targets, means_inter, color='black', label='$d_{inter}$', ls='--')
    plt.fill_between(x_targets, means_inter - stds_inter, means_inter + stds_inter, color='black', alpha=0.5)
    plt.legend(loc='upper right')
    plt.xlabel('Position x in the central corridor')
    plt.ylabel('PCA distance')
    plt.ylim(0, 22)
    # plt.title('PCA distance between intra and inter trajectories (R-L task no cues)')
    plt.grid(True)
    plt.show()


def draw_maze_and_markers():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    maze = Maze()
    maze.draw_static(ax)
    ax.axis('off')  # Hide the empty axis
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    # To plot all the PCA plots:

    # R-L, no-cues:
    path = '../data/R-L/no_cues/seed_1/reservoir_states/'
    #plot_PCA_3D(path=path, task='R-L', plot_distance=False, central_corridor_only=True, sequence=[1780, 8020])#sequence=[1780, 2320])
    #compute_pca_distances(path=path, task='R-L', sequence=[0, 10000], x_target=150, tol=1)


    # R-L, cues:
    path = '../data/R-L/cues/seed_1234/reservoir_states/'
    #plot_PCA_3D(path=path, task='R-L', plot_distance=False, central_corridor_only=True, sequence=[1780, 8050])#sequence=[1780, 2350])

    #plot_PCA_3D(path=path, task='R-L', plot_distance=True, central_corridor_only=True, sequence=[1780, 4000])

    # RR-LL, no-cues:
    #path = '../data/RR-LL/no_cues/seed_1234/reservoir_states/'
    #plot_PCA_3D(path=path, task='RR-LL', plot_distance=False, central_corridor_only=False, sequence=[0, 2000])

    # RR-LL, cues:
    #path = '../data/RR-LL/cues/seed_1234/reservoir_states/'
    #plot_PCA_3D(path=path, task='RR-LL', plot_distance=False, central_corridor_only=False, sequence=[2300,6000])

    ## R-L, no cues,  kill 100% SC
    #path = '../data/R-L/no_cues/seed_1/reservoir_states_1_killed/'
    #plot_PCA_3D(path=path, task='R-L', plot_distance=True, central_corridor_only=True, sequence=[1780, 2320])

    ## R-L,  cues,  kill 100% SC
    #path = '../data/R-L/cues/seed_1234/reservoir_states_1_killed/'
    #plot_PCA_3D(path=path, task='R-L', plot_distance=True, central_corridor_only=True, sequence=[1780, 2320])


    ## Calculate the mean PCA distance between intra and inter trajectories
    #seeds = (1, 2, 5, 10, 21, 22, 27, 30, 37, 38, 39)
    #seeds = (5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 1234)
    #x_targets = [100, 110, 120, 130, 140, 150, 160, 170, 180]
    #for seed in seeds:
   #     print('Seed:', seed)
    #    path = f'../data/R-L/cues/seed_{seed}/reservoir_states/'
     #   compute_pca_distances(path=path, task='R-L', sequence=[0, 10000], x_targets=x_targets, tol=1, save=True)
    #plot_PCA_distance_evolution()



















