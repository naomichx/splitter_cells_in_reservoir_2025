import reservoirpy as rpy
rpy.verbosity(0)
from reservoirpy.nodes import Reservoir, Ridge, Input, RLS, FORCE, Sigmoid, ScikitLearnNode
from reservoirpy.observables import nrmse, rsquare
from sklearn.linear_model import RidgeClassifier
import os, json
import numpy as np
from scipy.sparse import csr_matrix
import math


def generate_connectivity_matrix(connectivity, n_units, sr):
    # Generate a random connectivity matrix
    matrix = np.random.rand(n_units, n_units)
    random_values = np.random.normal(loc=0, scale=1,
                                     size=(n_units, n_units))  # Use n_units for correct shape
    matrix = np.where(matrix > (1 - connectivity), random_values, 0)

    # Compute the spectral radius
    eigenvalues = np.linalg.eigvals(matrix)
    spectral_radius = np.max(np.abs(eigenvalues))

    # Scale the matrix to achieve the desired spectral radius
    if spectral_radius > 0:
        scaling_factor = sr / spectral_radius
        W = matrix * scaling_factor
    else:
        W = matrix  # If spectral radius is 0, just return the original matrix

    return W

def split_train_test(input, output, nb_train, max_len_test=100000):
    """ Splits the input and output lists in two parts: one list for the training phase, one for the testing phase.
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



class Model:
    def __init__(self, seed, model_file, data_folder, save_reservoir_states,simulation_mode,
                 percentage_killed_neurons=0, neurons_to_kill_file=None, decoder=True, reservoir_in_decoder=True,
                 connectivity=None, leak_rate=None, spectral_radius=None):

        self.model_file = model_file
        self.data_folder = data_folder
        self.simulation_mode = simulation_mode
        self.save_reservoir_states = save_reservoir_states

        with open(os.path.join(os.path.dirname(__file__), model_file)) as f:
            _ = json.load(f)

        self.nb_train = _['nb_train']
        self.cues = bool(_['cues'])

        units = _['n_units']
        if leak_rate is None:
            leak_rate = _['leak_rate']
        if spectral_radius is None:
            spectral_radius = _['spectral_radius']
        regularization = _['regularization']
        input_connectivity = _['input_connectivity']
        if connectivity is None:
            connectivity = _['connectivity']
        noise_rc = _['noise_rc']
        warmup = _['warmup']
        #seed = _['seed']
        print(f'Model seed: {seed}')

        self.input = np.load(data_folder + 'input.npy')
        self.output = np.load(data_folder + 'output.npy')

        if self.cues:
            # Self generated cue: output the next cue
            self.cue_decoder_output = np.load(data_folder + 'cue_decoder_output.npy')[:-1]

        self.d_output = []
        for i in range(len(self.output) - 1):
            self.d_output.append(self.output[i + 1] - self.output[i])
        self.input = self.input[:-1]
        self.output = np.array(self.d_output).reshape(len(self.d_output), 1)
        self.positions = np.load(data_folder + 'positions.npy')
        if decoder:
            # Y_train of decoders
            self.output_position = self.positions[1:]
            self.output_decision = np.load(data_folder + 'output_decision.npy')
            self.output_orientation = np.load(data_folder + 'head_direction_zones.npy')

        self.positions = self.positions[:-1]

        if self.cues:
            print('Model with sensors and contextual cues as input...')
        else:
            print('Model with sensors as input...')

        np.random.seed(seed=seed)
        # Initialize the input and bias matrices
        p = 1 - input_connectivity
        # Initialize the input and bias matrices
        Win = np.random.choice([-0.2, 0.2, 0], size=(units, np.shape(self.input)[1]),
                               p=[(1 - p) / 2, (1 - p) / 2, p])
        Wbias = np.random.choice([-0.2, 0.2, 0], size=(units, 1), p=[(1 - p) / 2, (1 - p) / 2, p])
        W = generate_connectivity_matrix(connectivity, units, spectral_radius)

        if self.cues:
            w_cues = 0.1
            w_sensors = 0.25
            Win = np.empty((units, 10))
            Win[:, :8] = np.random.choice([-w_sensors, w_sensors, 0], size=(units, 8), p=[(1 - p) / 2, (1 - p) / 2, p])
            Win[:, 8:] = np.random.choice([-w_cues, w_cues, 0], size=(units, 2), p=[(1 - p) / 2, (1 - p) / 2, p])

        #path = 'data/R-L/no_cues/random_W/'
        #W = np.load(path + 'W_10.npy')

        self.reservoir = Reservoir(units, input_scaling=None, sr=spectral_radius, Win=Win, bias=Wbias,
                                   lr=leak_rate, rc_connectivity=connectivity, W=W,
                                   input_connectivity=None, seed=seed, noise_rc=noise_rc)

        X_train, Y_train, X_test, Y_test = split_train_test(self.input, self.output, self.nb_train)

        readout = Ridge(ridge=regularization)
        self.esn = self.reservoir >> readout

        if self.cues:
            # Build model that predicts the next cues.
            cue_readout = ScikitLearnNode(model=RidgeClassifier, model_hypers={"alpha": 1e-5})
            self.cue_generator = self.reservoir >> cue_readout

        if percentage_killed_neurons != 0:
            print('Killing splitter cells ...')
            self.esn.fit(X_train[0], Y_train[0])
            self.esn.reset()
            #splitter_cells = np.load(neurons_to_kill_file)
            all_sc = []
            for file in neurons_to_kill_file:
                all_sc.append(np.load(file))
            splitter_cells = np.concatenate(all_sc)
            # Remove partially the splitter cells in the model without cues.
            num_to_kill = int(len(splitter_cells) * percentage_killed_neurons)
            neurons_to_kill = np.random.choice(splitter_cells, num_to_kill, replace=False)

            print('Number of neurons to kill:', len(neurons_to_kill))
            W = self.reservoir.params['W']#.todense()
            Win = self.reservoir.params['Win']
            Wbias = self.reservoir.params['bias']  # .todense()

            W[neurons_to_kill, :] = 0
            W[:, neurons_to_kill] = 0
            Win[neurons_to_kill] = 0
            Wbias[neurons_to_kill] = 0

            self.reservoir.params['W'] = csr_matrix(W)
            self.reservoir.params['Win'] = Win
            self.reservoir.params['bias'] = csr_matrix(Wbias)

        if decoder:
            X_train_pos, Y_train_pos, X_test_pos, Y_test_pos = split_train_test(self.input, self.output_position, self.nb_train)
            X_train_orient, Y_train_orient, X_test_orient, Y_test_orient = split_train_test(self.input, self.output_orientation, self.nb_train)
            X_train_decision, Y_train_decision, X_test_decision, Y_test_decision = split_train_test(self.input, self.output_decision[:-1].reshape(len(self.output_decision[:-1]),2), self.nb_train)

            # Training and evaluating decoders: set reservoir_in_decoder to True if the decoder must decode the
            # reservoir states. If the decoder must directly decode from the input, set reservoir_in_decoder to False

            self.decoder_position, accuracy_position = self.train_and_evaluate_decoder(X_train_pos, Y_train_pos,
                                                                                       X_test_pos, Y_test_pos,
                                                                                       ridge=1e-5, name="position",
                                                                                       warmup=warmup,
                                                                                       reservoir=reservoir_in_decoder)
            self.decoder_orient, accuracy_orientation = self.train_and_evaluate_decoder(X_train_orient, Y_train_orient,
                                                                                        X_test_orient, Y_test_orient,
                                                                                        ridge=1e-3, name="orient",
                                                                                        warmup=warmup,
                                                                                        reservoir=reservoir_in_decoder)
            self.decoder_decision, accuracy_decision = self.train_and_evaluate_decoder(X_train_decision, Y_train_decision,
                                                                                       X_test_decision, Y_test_decision,
                                                                                       ridge=1e-3, name="decision",
                                                                                       warmup=warmup,
                                                                                       reservoir=reservoir_in_decoder)
            self.decoder_accuracy = {'position': accuracy_position,
                                'orientation': accuracy_orientation,
                                'decision': accuracy_decision}

        if self.simulation_mode == 'esn':
            print('Offline training...')
            self.esn.fit(X_train, Y_train, warmup=warmup, reset=True)

        else:
            # No training but just activate the reservoir
            self.esn.fit(X_train[0], Y_train[0], reset=True)
            print('No training')

        if self.cues:
            X_train_cue_gen, Y_train_cue_gen, \
            X_test_cue_gen, Y_test_cue_gen = split_train_test(self.input, self.cue_decoder_output, self.nb_train)
            self.cue_generator.fit(X_train_cue_gen, Y_train_cue_gen, reset=True)

        self.save_reservoir_states = save_reservoir_states
        self.reservoir_states = []

    def record_states(self):
        """ Function that records the reservoir state at the given position in the maze.
        Inputs:
        - bot_position: current position of the bot
        - reservoir: reservoir model
        if self.where == None: record the reservoir state everywhere in the maze
        else: record the reservoir state only in the last corridor juste before the decision point.
        """
        s = []
        for val in np.array(self.reservoir.state()[0]):
            s.append(val)
        self.reservoir_states.append(s)

    def process(self, sensors, cues=None):
        if self.cues:
            input = np.concatenate((sensors['value'].ravel(), np.array(cues))).reshape(1, 10)
        else:
            input = np.array(sensors['value']).reshape(1, 8)

        reservoir_state = self.reservoir.state().copy()
        if self.cues:
            generated_cues = np.array(self.cue_generator(input))
            self.reservoir.reset(to_state=reservoir_state)
            #print(generated_cues)
        output = np.array(self.esn(input))[0][0]

        if self.save_reservoir_states:
            self.record_states()
        if self.cues:
            return output, generated_cues
        else:
            return output,None

    def decode_position(self, sensors, cues=None):
        if self.cues:
            input = np.concatenate((sensors['value'].ravel(), np.array(cues))).reshape(1, 10)

        else:
            input = np.array(sensors['value']).reshape(1, 8)

        output = np.array(self.decoder_position(input))[0]
        return output

    def decode_decision(self, sensors, cues=None):
        if self.cues:
            input = np.concatenate((sensors['value'].ravel(), np.array(cues))).reshape(1, 10)
        else:
            input = np.array(sensors['value']).reshape(1, 8)

        output = np.array(self.decoder_decision(input))[0]
        return np.argmax(output)

    def decode_orientation(self, sensors, cues=None):
        orientations = {0: math.pi/4, 1: math.pi/2, 2: 3*math.pi/4, 3: math.pi,
                        4: 5*math.pi/4, 5: 6*math.pi/4, 6: 7*math.pi/4, 7: 8*math.pi/4}
        if self.cues:
            input = np.concatenate((sensors['value'].ravel(), np.array(cues))).reshape(1, 10)
        else:
            input = np.array(sensors['value']).reshape(1, 8)
        output = np.array(self.decoder_orient(input))[0]
        return np.argmax(output)

    def decode_place_cells(self, sensors, cues=None):
        if self.cues:
            input = np.concatenate((sensors['value'].ravel(), np.array(cues))).reshape(1, 10)
        else:
            input = np.array(sensors['value']).reshape(1, 8)

        output = np.array(self.decoder(input))[0]


        return output

    def train_and_evaluate_decoder(self, X_train, Y_train, X_test, Y_test, ridge, name, warmup, reservoir=True):
        """Trains and evaluates a decoder with the given parameters."""
        readout_decoder = Ridge(ridge=ridge)
        if reservoir:
            decoder = self.reservoir >> readout_decoder
            decoder.fit(X_train, Y_train, warmup=warmup, reset=True)
        else:
            decoder = Ridge(ridge=ridge)
            decoder.fit(X_train, Y_train)


        Y_pred = decoder.run(X_test)
        accuracy = np.mean([np.argmax(pred) == np.argmax(true) for pred, true in zip(Y_pred, Y_test)])

        print(f'Score decoder {name}: {accuracy}')
        return decoder, accuracy














