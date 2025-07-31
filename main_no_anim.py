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



if __name__ == '__main__':

    seed_vals = [i for i in range(1, 11)]
    connectivity_vals = [0.3, 0.4, 0.6, 0.9]
    leak_rate_vals = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    #sr_vals = [0.6, 0.7, 0.8, 0.9, 0.95,1, 1.1]
    lr = 1
    spectral_radius = None
    connectivity = 1
   #connectivity = None
    #for spectral_radius in sr_vals:
    #for connectivity in connectivity_vals:
        #print('Sr:', spectral_radius)
    for lr in leak_rate_vals:
        for seed in seed_vals:
            print('Seed:', seed)
            task = 'R-L'  # 'RR-LL', 'R-L'
            simulation_mode = "data_esn"  # data, walls, esn, data_esn (controlled by data, but esn is still running)
            training_mode = 'online'  # offline, online
            cues = False
            percentage_killed_neurons = 0  # set to 0 if no SC are killed, to 1 if all SC are killed
            save_reservoir_states = False
            save_bot_states = False
            save_decoding = True
            decoder = True  # place_cells, decision_cells, head_direction, None
            reservoir_in_decoder = True

            if cues:
                path_to_save = 'data/' + task + '/cues/'
            else:
                path_to_save = 'data/' + task + '/no_cues/'

            path_to_save += f'seed_{str(seed)}/'

            neurons_to_kill_file = path_to_save + 'reservoir_states/splitter_cells_index.npy'

            if percentage_killed_neurons != 0:
                path_to_save += f'reservoir_states_{percentage_killed_neurons}_killed/'
            else:
                path_to_save += f'reservoir_states/'

            if simulation_mode == 'data_esn':
                data_folder = 'data/R-L/no_cues/'
                model_file = "model_settings/model_RL_no_cues_forced_mode.json"
                #path_to_save = f'data/R-L/no_cues/forced_mode/sr_evol/sr_{spectral_radius}/seed_{str(seed)}/'
                #path_to_save = f'data/R-L/no_cues/forced_mode/co_evol/lr_{lr}/connectivity_{connectivity}/seed_{str(seed)}/'
                path_to_save = f'data/R-L/no_cues/forced_mode/lr_evol/connectivity_{connectivity}/lr_{lr}/seed_{str(seed)}/'
            else:
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
                             decoder=decoder, reservoir_in_decoder=reservoir_in_decoder, connectivity=connectivity,
                             leak_rate=lr,spectral_radius=spectral_radius)

            for i in range(1):
                exp.run(i)

            # Save data after animation completes
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
                print(path_to_save)
                assert decoder is True, "Decoder must be True when save_decoding is enabled."
                np.save(path_to_save + 'decoder_accuracy.npy', exp.model.decoder_accuracy)
                print('Decoder accuracy saved!')

                #add_title = '' if reservoir_in_decoder else 'input_decoder_'
                #np.save(path_to_save + add_title + 'predicted_decisions.npy', exp.bot.all_predicted_dec)
                #np.save(path_to_save + add_title + 'predicted_orientations.npy', exp.bot.all_predicted_or)
                #np.save(path_to_save + add_title + 'predicted_positions.npy', exp.bot.all_predicted_pos)






