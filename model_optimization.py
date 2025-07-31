import numpy as np
import optuna
import mlflow
from typing import Dict
import os
from reservoirpy.nodes import Reservoir, Ridge, Input, FORCE
from reservoirpy.observables import nrmse, rsquare
import json
from esn_model import split_train_test
SEED = 1



def set_seed(seed):
    """To ensure reproducible runs we fix the seed for different libraries"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    # Deterministic operations for CuDNN, it may impact performances
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_to_disk(path, agent_id, hparams, nrmse):
    try:
        # Create target Directory
        os.mkdir(path + '/'+str(agent_id) + '/')
        print("Directory ", path + '/' + str(agent_id) + '/', " Created ")
    except FileExistsError:
        print("Directory ", path + '/' + str(agent_id) + '/', " already exists")
    with open(path + '/' + str(agent_id) + '/' + 'hparams.json', 'w') as f:
        json.dump(hparams, f)
    np.save(path + '/' + str(agent_id) + '/' + 'nrmse.npy', nrmse)


def get_agent_id(agent_dir) -> str:
    try:
        # Create target Directory
        os.mkdir(agent_dir)
        print("Directory ", agent_dir, " Created ")
    except FileExistsError:
        print("Directory ", agent_dir, " already exists")
    ids = []
    for id in os.listdir(agent_dir):
        try:
            ids.append(int(id))
        except:
            pass
    if ids == []:
        agent_id = 1
    else:
        agent_id = max(ids) + 1
    return str(agent_id)

def sample_hyper_parameters(trial: optuna.trial.Trial) -> Dict:
    # Reservoir
    nb_units = 1000
    warmup = 500
    #noise_rc = 0.
    noise_rc = trial.suggest_categorical("noise_rc", [0, 1e-6, 1e-4, 1e-3, 1e-2])
    seed = 1234
    sr = trial.suggest_float("spectral_radius", 0.5, 1.5)
    #sr = 0.9227733203292644
    #lr = trial.suggest_float("leak_rate", 0.01, 1)
    lr = 1
    #warmup = trial.suggest_int("warmup", 0,1000)
    #lr = 0.3
    ridge = trial.suggest_float("ridge", 1e-6, 100)
    rc_connectivity = trial.suggest_float("rc_connectivity", 0.001, 1.)
    input_scaling = trial.suggest_categorical('input_scaling', [1e-3, 1e-2, 1e-1, 1, 10, 100])
    #rc_connectivity = 0.1
    #input_connectivity = 0.1
    input_connectivity = trial.suggest_float("input_connectivity", 0.01, 1)
    return {
        'reservoir_size': nb_units,
        'spectral_radius': sr,
        'noise_rc': noise_rc,
        'leak_rate': lr,
        'seed': seed,
        'warmup': warmup,
        'ridge': ridge,
        'rc_connectivity': rc_connectivity,
        'input_connectivity': input_connectivity,
        'input_scaling':input_scaling
    }



def positions_to_zones(positions):
    zones = []
    for position in positions:
        if 200 <= position[1] <= 300:
            if 0 < position[0] <= 100:
                zones.append(4)
            elif 100 < position[0] <= 200:
                zones.append(1)
            else:
                zones.append(2)
        elif position[1] > 300:
            zones.append(3)
        else:
            zones.append(5)
    return zones

def objective(trial: optuna.trial.Trial,X_train, Y_train, X_test, Y_test,positions,orientations,nb_train, agent_dir):
    with mlflow.start_run():
        agent_id = get_agent_id(agent_dir)
        mlflow.log_param('agent_id', agent_id)
        # hyper-parameters
        arg = sample_hyper_parameters(trial)
        mlflow.log_params(trial.params)
        set_seed(arg['seed'])

        reservoir = Reservoir(units=arg['reservoir_size'], lr=arg['leak_rate'],
                              sr=arg['spectral_radius'], noise_rc=arg['noise_rc'],
                              input_connectivity=arg['input_connectivity'],
                              input_scaling=arg['input_scaling'],
                              rc_connectivity=arg['rc_connectivity'], seed=SEED)
        readout = Ridge(ridge=arg['ridge'])

        #readout = FORCE()

        esn = reservoir >> readout

        #esn.train(X_train, Y_train)

        esn = esn.fit(X_train, Y_train, warmup=arg['warmup'])
        Y_pred = esn.run(X_test)

        init_pos = positions[nb_train-1]
        init_or = orientations[nb_train-1]
        positions_pred = []
        for y in Y_pred:
            positions_pred.append(init_pos + 2 * np.array([np.cos(init_or+y[0]), np.sin(init_or+y[0])]))
            init_or += y[0]
            init_pos = positions_pred[-1].copy()

        zones_pred = positions_to_zones(positions_pred)
        zones_true = positions_to_zones(positions[nb_train:-1])

        #Y_test = np.array([angle % (2 * np.pi) for angle in Y_test])
        #Y_pred = np.array([angle % (2 * np.pi) for angle in Y_pred])

        error_angle = nrmse(Y_test, Y_pred)
        #error_position = nrmse(positions[nb_train:-1], positions_pred) * 1000
        error_zones = nrmse(zones_true, zones_pred)

        #error_zones = 0
        #for i, zone_p in enumerate(zones_pred):
         #   if zone_p != zones_true[i]:
          #      error_zones += 1
                #index = i
                #break
        #score = len(zones_pred) - index

        #print(zones_true)
        #print(zones_pred)
        #print('error zone', error_zones*0.001)
        #print('penalty walls', penalty_walls * 0.0001)
        #print('angle', error_angle)
        #print('zones', score*0.001)

        rmse = error_zones # error_angle ++ error_position*0.001 ## error_zones*0.001 + penalty_walls*0.00001 #+ error_angle #, norm_value=np.ptp(zones_pred))
        print(rmse)
        #rmse = nrmse(Y_test, Y_pred, norm_value=np.ptp(X_train))*100
        save_to_disk(agent_dir, agent_id, arg, rmse)
        mlflow.log_metric('nrmse', rmse)
    return rmse

def optuna_optim(input, output, positions,orientations, title,nb_train, n_trials = 500):
    print('Start Optuna optimization ...')
    parent_dir = 'model_optimization/'
    SAVED_AGENTS_DIR = parent_dir + 'mlagent/' + title
    MLFLOW_RUNS_DIR = parent_dir + 'mlflows/' + title
    mlflow.set_tracking_uri(MLFLOW_RUNS_DIR)
    mlflow.set_experiment(title)
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(), study_name=title,
                                direction='minimize',
                                load_if_exists=True,
                                storage=f'sqlite:////Users/nchaix/Documents/PhD/code/'
                                                             f'splitter_cells_test/model_optimization/optuna_db/'
                                                             + title + '.db')


    X_train, Y_train, X_test, Y_test = split_train_test(input, output, nb_train)
    func = lambda trial: objective(trial,  X_train, Y_train, X_test, Y_test,positions,orientations,nb_train, agent_dir=SAVED_AGENTS_DIR)
    study.optimize(func, n_trials=n_trials)
    best_trial = study.best_trial
    hparams = {k: best_trial.params[k] for k in best_trial.params if k != 'seed'}



if __name__ == '__main__':
    data_folder = "data/R-L_80/no_cues/"

    input = np.load(data_folder + 'input.npy')
    output = np.load(data_folder + 'output.npy')
    #output = output.reshape(len(output), 1)

    d_output = []
    for i in range(len(output) - 1):
        d_output.append(output[i + 1] - output[i])
    input = input[:-1]
    output = np.array(d_output).reshape(len(d_output), 1)

    orientations = np.load(data_folder + 'output.npy')
    positions = np.load(data_folder + 'positions.npy')

    title = '1000_units_sensors80_zone_offline_lr1'
    nb_train = 7200
    optuna_optim(input, output, positions, orientations, title, nb_train=nb_train, n_trials=700)






