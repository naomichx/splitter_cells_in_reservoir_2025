import numpy as np
import math
import matplotlib.pyplot as plt


def darken_color(color, factor=0.7):
    import matplotlib.colors as mcolors
    rgb = mcolors.to_rgb(color)  # Convert color to RGB
    return tuple([factor * x for x in rgb])


def build_head_direction_zones(path):
    orientations = np.load(path + 'output.npy') + 2*math.pi
    input = np.load(path + 'input.npy')
    print(orientations)
    outputs = []   #  [0, pi/4, pi/2, 3pi/4, pi, -3pi/4, -pi/2, -pi/4]

    for theta in orientations:
        if 0 <= theta%(2*math.pi) < math.pi/4:
            outputs.append([1, 0, 0, 0, 0, 0, 0, 0])
        elif math.pi/4 <= theta%(2*math.pi) < math.pi/2:
            outputs.append([0, 1, 0, 0, 0, 0, 0, 0])
        elif math.pi/2 <= theta%(2*math.pi) < 3*math.pi/4:
            outputs.append([0, 0, 1, 0, 0, 0, 0, 0])
        elif 3*math.pi/4 <= theta%(2*math.pi) < math.pi:
            outputs.append([0, 0, 0, 1, 0, 0, 0, 0])
        elif math.pi <= theta%(2*math.pi) < 5*math.pi/4:
            outputs.append([0, 0, 0, 0, 1, 0, 0, 0])
        elif  5*math.pi/4  <= theta%(2*math.pi) < 6*math.pi/4 :
            outputs.append([0, 0, 0, 0, 0, 1, 0, 0])
        elif 6*math.pi/4  <= theta%(2*math.pi) < 7*math.pi/4 :
            outputs.append([0, 0, 0, 0, 0, 0, 1, 0])
        elif 7*math.pi/4 < theta%(2*math.pi) < 8*math.pi/4:
            outputs.append([0, 0, 0, 0, 0, 0, 0, 1])
        else:
            outputs.append([0, 0, 0, 0, 0, 0, 0, 0])
        #print(outputs[-1])

    np.save(path + 'head_direction_zones.npy', arr=outputs)


def plot_cells_in_maze(positions, reservoir_states, idx):
    norm_states = (reservoir_states - np.min(reservoir_states)) / (np.max(reservoir_states) - np.min(reservoir_states))
    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.title(f'Cell n°{idx}')
    sc = plt.scatter(positions[:, 0], positions[:, 1], c=norm_states[:, idx], cmap='seismic', s=10, alpha=0.5)
    plt.colorbar(sc, label="Firing Activity")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.show()


def plot_orientation_pred(path):
    plt.rcParams.update({'font.size': 12})
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    titles = ['Decoding from reservoir', 'Decoding from input']
    file_suffix = ['', 'input_decoder_']
    orientation_true = np.load(path + 'output.npy')
    orientation_true = np.unwrap(orientation_true)

    idx = 0
    for ax, suffix, title in zip(axes, file_suffix, titles):
        orientation_pred = np.load(path + f'{suffix}predicted_orientations.npy')
        print(orientation_pred)
        orientations = {0: math.pi / 4, 1: math.pi / 2, 2: 3 * math.pi / 4, 3: math.pi,
                        4: 5 * math.pi / 4, 5: 6 * math.pi / 4, 6: 7 * math.pi / 4, 7: 8 * math.pi / 4}
        n = 2500
        # Plot the true continuous orientation with a dashed line
        ax.plot(orientation_true[:n], label='True Orientation', linestyle='--', color='black', alpha=0.8, lw=1)

        # Add horizontal gridlines for discrete levels
        discrete_levels = list(set(orientations.values()))
        ax.hlines(discrete_levels[:n], xmin=0, xmax=len(orientation_pred[:n]), color='gray', linestyles='dashed',
                  alpha=0.3)

        # Fill between the predicted orientation intervals (orientations[orient] is the upper limit,
        # orientations[orient-1] is the lower limit)

        for i in range(1, n):
            upper_limit = orientations[orientation_pred[i]]

            if orientation_pred[i] > 0:
                lower_limit = orientations[orientation_pred[i] - 1]
            else:
                lower_limit = 0

            # Fill the area between lower and upper limits
            ax.fill_between([i - 1, i], lower_limit, upper_limit, color='C3', alpha=0.1)
            if i == 1:
                ax.plot([], [], color='C3', label='Predicted orientation', alpha=0.5)
        # Customize y-ticks for clarity
        plt.yticks(
            [0, math.pi / 4, math.pi / 2, 3 * math.pi / 4, math.pi, 5 * math.pi / 4, 3 * math.pi / 2, 7 * math.pi / 4,
             2 * math.pi],
            ['0', 'π/4', 'π/2', '3π/4', 'π', '5π/4', '6π/4', '7π/4', '2π'])

        # Add labels and legend
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Orientation')
        if idx == 1:
            ax.legend(loc='upper right')
        idx += 1
    plt.tight_layout()
    plt.show()


def plot_orientation_pred(path):
    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots(figsize=(5, 5))
    title = 'Decoding from reservoir'
    orientation_true = np.load(path + 'output.npy')
    orientation_true = np.unwrap(orientation_true)
    orientation_pred = np.load(path + f'input_decoder_predicted_orientations.npy')

    orientations = {
        0: math.pi / 4, 1: math.pi / 2, 2: 3 * math.pi / 4, 3: math.pi,
        4: 5 * math.pi / 4, 5: 6 * math.pi / 4, 6: 7 * math.pi / 4, 7: 2 * math.pi
    }

    n = 2500
    ax.plot(orientation_true[:n], label='True Orientation', linestyle='--', color='black', alpha=0.8, lw=1)

    discrete_levels = list(set(orientations.values()))
    ax.hlines(discrete_levels, xmin=0, xmax=n, color='gray', linestyles='dashed', alpha=0.3)

    for i in range(1, n):
        upper_limit = orientations[orientation_pred[i]]
        lower_limit = orientations[orientation_pred[i] - 1] if orientation_pred[i] > 0 else 0
        ax.fill_between([i - 1, i], lower_limit, upper_limit, color='C3', alpha=0.1)
        if i == 1:
            ax.plot([], [], color='C3', label='Predicted orientation', alpha=0.5)

    ax.set_yticks([0, math.pi / 4, math.pi / 2, 3 * math.pi / 4, math.pi,
                   5 * math.pi / 4, 3 * math.pi / 2, 7 * math.pi / 4, 2 * math.pi])
    ax.set_yticklabels(['0', 'π/4', 'π/2', '3π/4', 'π', '5π/4', '6π/4', '7π/4', '2π'])
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Orientation')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

def plot_decision_pred(path):
    plt.rcParams.update({'font.size': 12})
    fig, axes = plt.subplots(1, 2, figsize=(8, 7), sharex=True, sharey=True)
    titles = ['Decoding from reservoir', 'Decoding from input']
    file_suffix = ['', 'input_decoder_']

    idx = 0
    for ax, suffix, title in zip(axes, file_suffix, titles):
        decisions = np.load(path + f'{suffix}predicted_decisions.npy')
        positions = np.load(path + 'positions.npy')
        dict_colors = {1: 'red', 0: darken_color('blueviolet', factor=0.7)}

        for i in range(500, 4000):
            color = dict_colors[decisions[i]]

            if i == 500:
                ax.plot([], [], color='red', label='Left predicted', alpha=1)
                ax.plot([], [], color=darken_color('blueviolet', factor=0.7), label='Right predicted', alpha=0.5)

            ax.plot([positions[i][0], positions[i + 1][0]],
                    [positions[i][1], positions[i + 1][1]],
                    color=color,
                    linewidth=1,
                    alpha=0.2)

        ax.set_title(title, fontsize=12)
        ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.set_xlabel('X Position', fontsize=12)
        ax.set_ylabel('Y Position', fontsize=12)
        if idx == 0:
            ax.legend(fontsize=12, loc='lower left')#, frameon=True)
        idx += 1

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    path = '../data/R-L/no_cues/seed_1/reservoir_states/'
    #plot_decision_pred(path=path)#, reservoir_in_decoder=False)
    plot_orientation_pred(path)
