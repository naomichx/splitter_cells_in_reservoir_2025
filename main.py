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
from analysis.single_cell_analysis import find_splitter_cells_ANCOVA

task = 'R-L' #'RR-LL', 'R-L'
seed = 1
simulation_mode = "data_esn"  # data, walls, esn, data_esn (controlled by data, but esn is still running)
training_mode = 'offline' # offline, online
cues = False
percentage_killed_neurons = 1 # set to 0 if no SC are killed, to 1 if all SC are killed
save_reservoir_states = True
save_bot_states = True
save_decoding = True
decoder = True  # place_cells, decision_cells, head_direction, None
reservoir_in_decoder = False
idx_kill = 0

if __name__ == '__main__':

    for seed in np.arange(10):
        if cues:
            path_to_save = 'data/' + task + '/cues/'
        else:
            path_to_save = 'data/' + task + '/no_cues/'

        path_to_save += f'seed_{str(seed)}/'

        if percentage_killed_neurons != 0:
            path_to_save += f'reservoir_states_{percentage_killed_neurons}_killed_bis/'
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
            neurons_to_kill_file = ''
            if task == 'R-L':
                print('Run the alternation task (R-L) ...')
                if cues:
                    data_folder = "data/R-L/cues/"
                    if training_mode == 'online':
                        model_file = "model_settings/model_RL_cues_RLS.json"
                    elif training_mode == 'offline':
                        model_file = "model_settings/model_RL_cues.json"
                    else:
                        raise ValueError("Training mode {}".format(training_mode) + " is not recognized.")
                else:
                    data_folder = 'data/R-L/no_cues/'
                    if training_mode == 'online':
                        model_file = "model_settings/model_RL_no_cues_RLS.json"
                    elif training_mode == 'offline':
                        model_file = "model_settings/model_RL_no_cues.json"
                    else:
                        raise ValueError('Error in defining training mode')

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
                         training_mode=training_mode, task=task, cues=cues,
                         save_reservoir_states=save_reservoir_states,
                         save_bot_states=save_bot_states, percentage_killed_neurons=percentage_killed_neurons,
                         neurons_to_kill_file=neurons_to_kill_file,
                         decoder=decoder, reservoir_in_decoder=reservoir_in_decoder, connectivity=None,
                         leak_rate=None, spectral_radius=None)

        for i in range(10000):
            exp.run(i)

        # Set up the animation
        anim = animation.FuncAnimation(exp.simulation_visualizer.fig, exp.run,  frames=20000, interval=1, repeat=False)
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

        splitter_cells = find_splitter_cells_ANCOVA(path=path_to_save, save=True, type='SC')












