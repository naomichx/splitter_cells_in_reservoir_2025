""" Main script to run.
Before running the script, certain configurations are required:

- task :
        1) 'R-L' (alternation task)
        2) 'RR-LL' (half-alternation task)

- simulation_mode:
                1) 'walls': the bot navigates and takes direction automatically using Braitenberg algorithms.
                            Walls are added to guide the bot in the right direction.
                   Some walls are added so as to force the bot taking the right direction
                2) 'data': the bot is data-driven and navigates based on the provided position file.
                3) 'esn': the bot moves based on ESN predictions, trained using supervised learning.

- save_reservoir_states: set to True if the reservoir states and the bot's positions and orientation need to be recorded
- save_bot_states: set to True if the bot's positions and orientation need to be recorded
- path_to_save: folder to save

"""

import matplotlib.animation as animation
from experiment import Experiment
import matplotlib.pyplot as plt
import numpy as np
import os
from analysis.single_cell_analysis import find_splitter_cells_ANCOVA, SC_P_THRESHOLD

task = 'R-L' #'RR-LL', 'R-L'
seed = 1
simulation_mode = "esn"  # data, walls, esn, data_esn (controlled by data, but esn is still running)
cues = False
percentage_killed_neurons = 1 # set to 0 if no SC are killed, to 1 if all SC are killed
save_reservoir_states = True
save_bot_states = True
save_decoding = False
decoder = False  # place_cells, decision_cells, head_direction, None
reservoir_in_decoder = False
idx_kill = 0
save_splitter_cells = True
n_les = 1

if __name__ == '__main__':

    if cues:
        path_to_save = 'data/' + task + '/cues/'
    else:
        initial_path = 'data/' + task + '/no_cues/3000_units_lr_04/'

    path_to_save = initial_path + f'seed_{str(seed)}/'

    #if not os.path.exists(initial_path+f'seed_{seed}/reservoir_states_1_killed_{n_les-1}/splitter_cells_index.npy'):
    #    print(f'skipping seed {seed} and n_les {n_les} because the file does not exist')
   #     continue

    # TO stop the process when there are no splitter cells left between two lesions
    if n_les >= 2:
        prev_sc_path = initial_path + f'seed_{seed}/reservoir_states_1_killed_{n_les - 1}/splitter_cells_index.npy'
        if n_les == 2:
            prev_prev_sc_path = initial_path + f'seed_{seed}/reservoir_states/splitter_cells_index.npy'
        else:
            prev_prev_sc_path = initial_path + f'seed_{seed}/reservoir_states_1_killed_{n_les - 2}/splitter_cells_index.npy'
        n_sc_prev = len(np.load(prev_sc_path))
        n_sc_prev_prev = len(np.load(prev_prev_sc_path))
        if n_sc_prev == n_sc_prev_prev:
            print(f'skipping seed {seed}: n_splitter_cells unchanged ({n_sc_prev}) '
                    f'between lesions {n_les - 2} and {n_les - 1} — no splitter cells left')
            #break

    if percentage_killed_neurons != 0:
        path_to_save += f'reservoir_states_{percentage_killed_neurons}_killed_{n_les}/'
    else:
        path_to_save += f'reservoir_states/'

    if simulation_mode == 'data_esn':
        data_folder = 'data/R-L/no_cues/'
        #data_folder = 'data/R-L/no_cues/seed_1/reservoir_states/'
        model_file = "model_settings/model_RL_no_cues_forced_mode.json"
        path_to_save = f'data/R-L/no_cues/forced_mode/kill_multiple_times/seed_{seed}/kill_{idx_kill}/'
        neurons_to_kill_file = []
        for idx in range(idx_kill):
            neurons_to_kill_file.append(f'data/R-L/no_cues/forced_mode/kill_multiple_times/seed_{seed}/kill_{idx}' \
                                    f'/splitter_cells_index_bis.npy')
    else:
        neurons_to_kill_file = [initial_path+f'seed_{seed}/reservoir_states/splitter_cells_index.npy']
                                #initial_path+f'seed_{seed}/reservoir_states_1_killed/splitter_cells_index.npy',
                                #initial_path+f'seed_{seed}/reservoir_states_1_killed_2/splitter_cells_index.npy',
                                #initial_path+f'seed_{seed}/reservoir_states_1_killed_3/splitter_cells_index.npy',
                                #initial_path+f'seed_{seed}/reservoir_states_1_killed_4/splitter_cells_index.npy',
                                #initial_path+f'seed_{seed}/reservoir_states_1_killed_5/splitter_cells_index.npy']
        if n_les >= 2:
            for n in np.arange(1, n_les):
                neurons_to_kill_file.append(initial_path+f'seed_{seed}/reservoir_states_1_killed_{n}/splitter_cells_index.npy')
        print('neurons_to_kill_file:', neurons_to_kill_file)

        if task == 'R-L':
            print('Run the alternation task (R-L) ...')
            if cues:
                data_folder = "data/R-L/cues/"
                model_file = "model_settings/model_RL_cues.json"

            else:
                data_folder = 'data/R-L/no_cues/'
                model_file = "model_settings/model_RL_no_cues_3000_units.json"
        elif task == 'RR-LL':
            print('Run the half-alternation task (RR-LL) ...')
            if cues:
                model_file = "model_settings/model_RR-LL_cues.json"
                data_folder = "data/RR-LL/cues/"
            else:
                model_file = "model_settings/model_RR-LL_no_cues.json"
                data_folder = "data/RR-LL/no_cues/"

        else:
            raise ValueError("Task name {}".format(task) + " is not recognized.")

    # Set up the experiment
    exp = Experiment(seed, model_file, data_folder, simulation_mode=simulation_mode,
                        task=task, cues=cues,
                        save_reservoir_states=save_reservoir_states,
                        save_bot_states=save_bot_states, percentage_killed_neurons=percentage_killed_neurons,
                        neurons_to_kill_file=neurons_to_kill_file,
                        decoder=decoder, reservoir_in_decoder=reservoir_in_decoder, connectivity=None,
                        leak_rate=None, spectral_radius=None)

    for i in range(10000):
        exp.run(i)

    # Set up the animation
    anim = animation.FuncAnimation(exp.simulation_visualizer.fig, exp.run,  frames=10000, interval=1, repeat=False)
    plt.tight_layout()
    plt.show()

    # Save data after animation completes
    print('Saving path:', path_to_save)
    if save_bot_states:
        os.makedirs(path_to_save, exist_ok=True)
        np.save(path_to_save + 'positions.npy', exp.bot.all_positions)
        np.save(path_to_save + 'output.npy', exp.bot.all_orientations)
        np.save(path_to_save + 'd_output.npy', exp.bot.all_d_orientations)
        if cues:
            input_data = np.concatenate((exp.bot.all_sensors_vals, exp.bot.all_cues), axis=1)
            np.save(path_to_save + 'input.npy', input_data)
        else:
            np.save(path_to_save + 'input.npy', exp.bot.all_sensors_vals)
        print('Bot information saved!')

    if save_reservoir_states:
        np.save(path_to_save + 'reservoir_states.npy', exp.model.reservoir_states)
        print('Reservoir state saved!')

    if save_decoding:
            assert decoder is True, "Decoder must be True when save_decoding is enabled."
            add_title = '' if reservoir_in_decoder else 'input_decoder_'
            np.save(path_to_save + add_title+ 'decoder_accuracy.npy', exp.model.decoder_accuracy)
            np.save(path_to_save + add_title + 'predicted_decisions.npy', exp.bot.all_predicted_dec)
            np.save(path_to_save + add_title + 'predicted_orientations.npy', exp.bot.all_predicted_or)
            np.save(path_to_save + add_title + 'predicted_positions.npy', exp.bot.all_predicted_pos)
            print('Decoding information state saved!')

    try:
        splitter_cells = find_splitter_cells_ANCOVA(
            path=path_to_save, save=save_splitter_cells, type='SC',
            p_threshold=SC_P_THRESHOLD)
    except Exception as e:
        print(f'Error finding splitter cells: {e}')
        #continue












