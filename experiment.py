from maze import Maze
from bot import Bot
from simulation_visualizer import SimulationVisualizer
import numpy as np
from esn_model import Model


"""
- automatic navigation with walls (Braitenberg + walls) 'walls_directed'
- navigation with data (controlled by generated data) 'data_directed'
- navigation thanks to the models (use ESN) 'esn_directed' 
- input type: cues, no_cues
"""



class Experiment:
    """
    This class run the experiment.
    """
    def __init__(self, seed, model_file, data_folder, simulation_mode,
                 task, cues, save_reservoir_states, save_bot_states,
                 percentage_killed_neurons=0, neurons_to_kill_file=None, decoder=True,
                 reservoir_in_decoder=True, connectivity=None, leak_rate=None,spectral_radius=None):

        self.task = task
        self.simulation_mode = simulation_mode
        self.cues = cues
        self.generated_cues = None
        sensor_size = 60
        self.bot = Bot(save_bot_states, sensor_size=sensor_size, decoder=decoder)
        self.maze = Maze(simulation_mode=simulation_mode)
        self.simulation_visualizer = SimulationVisualizer(self.bot.n_sensors)
        self.model_file = model_file
        self.data_folder = data_folder
        self.save_reservoir_states = save_reservoir_states
        self.decoder = decoder

        if self.simulation_mode == 'data':
            self.output = np.load(self.data_folder + 'output.npy')
            self.positions = np.load(self.data_folder + 'positions.npy')
            self.input = np.load(self.data_folder + 'input.npy')

        elif self.simulation_mode == 'data_esn':

            self.output = np.load(self.data_folder + 'output.npy')
            self.positions = np.load(self.data_folder + 'positions.npy')
            self.input = np.load(self.data_folder + 'input.npy')

            self.model = Model(seed=seed, model_file=self.model_file, data_folder=self.data_folder,
                               save_reservoir_states=self.save_reservoir_states,
                               percentage_killed_neurons=percentage_killed_neurons,
                               neurons_to_kill_file=neurons_to_kill_file, decoder=decoder,
                               connectivity=connectivity, leak_rate=leak_rate,spectral_radius=spectral_radius)
            self.model.nb_train = 0

            self.bot.position = self.positions[self.model.nb_train - 1]
            self.bot.orientation = self.output[self.model.nb_train - 1]

        elif self.simulation_mode == 'esn':
            self.output = np.load(self.data_folder + 'output.npy')
            self.output = self.output.reshape(len(self.output), 1)
            self.input = np.load(self.data_folder + 'input.npy')
            self.model = Model(seed=seed,model_file=self.model_file, data_folder=self.data_folder,
                               simulation_mode=simulation_mode,
                               save_reservoir_states=self.save_reservoir_states,
                               percentage_killed_neurons=percentage_killed_neurons,
                               neurons_to_kill_file=neurons_to_kill_file,
                               decoder=decoder, reservoir_in_decoder=reservoir_in_decoder)

            self.bot.position = self.model.positions[self.model.nb_train-1]
            self.bot.orientation = self.output[self.model.nb_train - 1][0]

        self.maze.draw(self.simulation_visualizer.ax, grid=True, margin=15)
        self.bot.draw(self.simulation_visualizer.ax)

    def run(self, frame):
        #print(frame)

        if self.simulation_mode == 'data':
            self.bot.orientation = self.output[frame]
            self.bot.position = self.positions[frame]

        else:
            if self.cues:
                if frame == 0:
                    cues = self.bot.update_cues(self.task)
                else:
                    cues = self.generated_cues
                    #cues = self.bot.update_cues(self.task)
            else:
                cues = None

            if self.simulation_mode == 'walls':
                self.bot.update_position(maze=self.maze)
                self.bot.compute_orientation()
                if self.task == 'R-L':
                    self.maze.update_walls(self.bot.position)
                elif self.task == 'RR-LL':
                    self.maze.update_walls_RR_LL(self.bot.position)

            elif self.simulation_mode == 'esn':
                self.bot.update_position(maze=self.maze)

                if self.decoder:
                    reservoir_state = self.model.reservoir.state().copy()
                    self.bot.predicted_pos = self.model.decode_position(sensors=self.bot.sensors,  cues=cues)
                    self.bot.all_predicted_pos.append(self.bot.predicted_pos.copy())
                    self.model.reservoir.reset(to_state=reservoir_state)
                    self.bot.orientation_pred = self.model.decode_orientation(sensors=self.bot.sensors, cues=cues)
                    self.bot.all_predicted_or.append(self.bot.orientation_pred)
                    self.model.reservoir.reset(to_state=reservoir_state)
                    self.bot.decision_pred = self.model.decode_decision(sensors=self.bot.sensors, cues=cues)
                    self.bot.all_predicted_dec.append(self.bot.decision_pred)
                    self.model.reservoir.reset(to_state=reservoir_state)

                dtheta, generated_cue = self.model.process(self.bot.sensors, cues)
                self.bot.orientation += dtheta

                if self.cues:
                    self.generated_cues = generated_cue

            elif self.simulation_mode == 'data_esn':
                self.bot.orientation = self.output[self.model.nb_train + frame]
                self.bot.position = self.positions[self.model.nb_train + frame]
                self.bot.update_position(maze=self.maze)
                dtheta = self.model.process(self.bot.sensors, cues)

        self.bot.update(self.maze, cues=self.cues)
        self.simulation_visualizer.update_plot(frame, self.bot.position, self.bot.sensors['value'],
                                               self.bot.predicted_pos)

        return self.bot.artists, self.simulation_visualizer.trace, self.simulation_visualizer.plots





















