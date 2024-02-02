import time
import os
import copy
import random
import numpy as np
import pandas as pd
from random import sample
from numpy import genfromtxt
from utils.Meta_plain_model_tf2 import MTLPlainModel
from utils.Meta_sparse_model_tf2 import MTLSparseModel
from utils.mlp_plain_model_tf2 import MLPPlainModel
from utils.mlp_sparse_model_tf2 import MLPSparseModel
from utils.general import get_non_zero_indexes, process_training_data
from utils.hyperparameter_tuning import nn_l1_val, hyperparameter_tuning
import warnings
warnings.filterwarnings("ignore")

def meta_training(dir_data, selected_environments, meta_to_train, save_file=True, test_mode=True, max_epoch=1000, meta_sample_percentage=100):
    saving_folder = '{}/Models'.format(os.getcwd())
    total_environments = int(dir_data.split('environments')[0].split('-')[-1])
    whole_data = genfromtxt(dir_data, delimiter=',', skip_header=1)
    (N, n) = whole_data.shape
    n = n - 1
    # delete the zero-performance samples
    non_zero_indexes = get_non_zero_indexes(whole_data, total_environments)
    seed = 2
    N_meta_environments = total_environments - 1
    N_train = int((meta_sample_percentage / 100) * len(non_zero_indexes))
    N_features = n + 1 - total_environments
    random.seed(seed)
    training_index = sample(list(non_zero_indexes), N_train)


    for main_environment in meta_to_train:
        if int(main_environment) in selected_environments:
            for environments in meta_to_train[main_environment]:
                environments = environments[(total_environments - N_meta_environments - 1):]
                N_environment = len(environments)  # the number of environments to group
                saving_file_weights = '{}/weights_{}_M{}_{}.npy'.format(saving_folder, dir_data.split('/')[1].split('.')[0],environments, N_train)
                saving_file_bias = '{}/bias_{}_M{}_{}.npy'.format(saving_folder, dir_data.split('/')[1].split('.')[0],environments, N_train)
                if not (os.path.exists(saving_file_weights) and os.path.exists(saving_file_bias)):
                    layers = []
                    config = []
                    lr_opt = []
                    lambdas = []
                    temp_X = whole_data[training_index, 0:N_features]
                    # scale x
                    temp_max_X = np.amax(temp_X, axis=0)
                    if 0 in temp_max_X:
                        temp_max_X[temp_max_X == 0] = 1
                    temp_X = np.divide(temp_X, temp_max_X)
                    X_train = np.array(temp_X)

                    # Split train data into 2 parts (67-33)
                    N_cross = int(np.ceil(len(temp_X) * 2 / 3))
                    X_train1 = (temp_X[0:N_cross, :])
                    X_train2 = (temp_X[N_cross:len(temp_X), :])

                    for i_environment, environment in enumerate(environments):
                        temp_Y = whole_data[training_index, n - environment][:, np.newaxis]
                        # Scale y
                        temp_max_Y = np.max(temp_Y) / 100
                        if temp_max_Y == 0:
                            temp_max_Y = 1
                        temp_Y = np.divide(temp_Y, temp_max_Y)
                        # Split train data into 2 parts (67-33)
                        if i_environment == 0:
                            Y_train = temp_Y
                            Y_train1 = temp_Y[0:N_cross, :]
                            Y_train2 = temp_Y[N_cross:len(temp_X), :]
                        else:
                            Y_train = np.hstack((Y_train, temp_Y))
                            Y_train1 = np.hstack((Y_train1, temp_Y[0:N_cross, :]))
                            Y_train2 = np.hstack((Y_train2, temp_Y[N_cross:len(temp_X), :]))
                    Y_train = np.array(Y_train)

                    ### training SeMPL
                    if test_mode:
                        # for testing
                        for i in range(0, N_environment):
                            print('\t>> Learning environment {}...'.format(environments[i]))
                            temp_lr_opt = 0.123
                            n_layer_opt = 5
                            lambda_f = 0.123
                            temp_config = dict()
                            temp_config['num_neuron'] = 128
                            temp_config['num_input'] = n - total_environments + 1
                            temp_config['num_layer'] = n_layer_opt
                            temp_config['lambda'] = lambda_f
                            temp_config['verbose'] = 0
                            config.append(temp_config)
                            lr_opt.append(temp_lr_opt)
                            layers.append(n_layer_opt)

                            # print('Training...')
                            # # train SeMPL model
                            if i == 0:
                                SeMPL_model = MTLSparseModel(config[i])
                                SeMPL_model.build_train()
                            else:
                                SeMPL_model = MTLSparseModel(config[i])
                                SeMPL_model.read_weights(weights, bias)
                                SeMPL_model.build_train()

                            SeMPL_model.train(X_train, Y_train[:, i][:, np.newaxis], lr_opt[i], max_epoch=max_epoch)
                            weights, bias = SeMPL_model.get_weights()
                    else:
                        for i in range(0, N_environment):
                            print('\n---Tuning environment {}---'.format(environments[i]))
                            if i == 0:
                                # for testing, start to comment here
                                print('Tuning hyperparameters...')
                                print('Step 1: Tuning the number of layers and the learning rate ...')
                                temp_config = dict()
                                temp_config['num_input'] = n - total_environments + 1
                                temp_config['num_neuron'] = 128
                                temp_config['lambda'] = 'NA'
                                temp_config['decay'] = 'NA'
                                temp_config['verbose'] = 0
                                abs_error_all = np.zeros((20, 4))
                                abs_error_all_train = np.zeros((20, 4))
                                abs_error_layer_lr = np.zeros((20, 2))
                                abs_err_layer_lr_min = 100
                                count = 0
                                layer_range = range(2, 10)
                                lr_range = [0.0001, 0.001, 0.01, 0.1]
                                # lr_range = np.logspace(np.log10(0.0001), np.log10(0.1), 5)
                                for n_layer in layer_range:
                                    temp_config['num_layer'] = n_layer
                                    for lr_index, lr_initial in enumerate(lr_range):
                                        model = MLPPlainModel(temp_config)
                                        model.build_train()
                                        model.train(X_train1, Y_train1[:, i][:, np.newaxis], lr_initial, max_epoch)

                                        Y_pred_train = model.predict(X_train1)
                                        abs_error_train = np.mean(
                                            np.abs(Y_pred_train - Y_train1[:, i][:, np.newaxis]))
                                        abs_error_all_train[int(n_layer), lr_index] = abs_error_train

                                        Y_pred_val = model.predict(X_train2)
                                        abs_error = np.mean(
                                            np.abs(Y_pred_val - Y_train2[:, i][:, np.newaxis]))
                                        abs_error_all[int(n_layer), lr_index] = abs_error

                                    # Pick the learning rate that has the smallest train cost
                                    # Save testing abs_error correspond to the chosen learning_rate
                                    temp = abs_error_all_train[int(n_layer), :] / np.max(
                                        abs_error_all_train)
                                    temp_idx = np.where(abs(temp) < 0.0001)[0]
                                    if len(temp_idx) > 0:
                                        lr_best = lr_range[np.max(temp_idx)]
                                        err_val_best = abs_error_all[int(n_layer), np.max(temp_idx)]
                                    else:
                                        lr_best = lr_range[np.argmin(temp)]
                                        err_val_best = abs_error_all[int(n_layer), np.argmin(temp)]

                                    abs_error_layer_lr[int(n_layer), 0] = err_val_best
                                    abs_error_layer_lr[int(n_layer), 1] = lr_best

                                    if abs_err_layer_lr_min >= abs_error_all[int(n_layer), np.argmin(temp)]:
                                        abs_err_layer_lr_min = abs_error_all[int(n_layer),
                                                                             np.argmin(temp)]
                                        count = 0
                                    else:
                                        count += 1

                                    if count >= 2:
                                        break
                                abs_error_layer_lr = abs_error_layer_lr[abs_error_layer_lr[:, 1] != 0]

                                # Get the optimal number of layers
                                n_layer_opt = layer_range[np.argmin(abs_error_layer_lr[:, 0])] + 5

                                # Find the optimal learning rate of the specific layer
                                temp_config['num_layer'] = n_layer_opt
                                for lr_index, lr_initial in enumerate(lr_range):
                                    model = MLPPlainModel(temp_config)
                                    model.build_train()
                                    model.train(X_train1, Y_train1[:, i][:, np.newaxis], lr_initial, max_epoch)

                                    Y_pred_train = model.predict(X_train1)
                                    abs_error_train = np.mean(
                                        np.abs(Y_pred_train - Y_train1[:, i][:, np.newaxis]))
                                    abs_error_all_train[int(n_layer), lr_index] = abs_error_train

                                    Y_pred_val = model.predict(X_train2)
                                    abs_error = np.mean(np.abs(Y_pred_val - Y_train2[:, i][:, np.newaxis]))
                                    abs_error_all[int(n_layer), lr_index] = abs_error

                                temp = abs_error_all_train[int(n_layer), :] / np.max(abs_error_all_train)
                                temp_idx = np.where(abs(temp) < 0.0001)[0]
                                if len(temp_idx) > 0:
                                    lr_best = lr_range[np.max(temp_idx)]
                                else:
                                    lr_best = lr_range[np.argmin(temp)]

                                temp_lr_opt = lr_best
                                print('The optimal number of layers: {}'.format(n_layer_opt))
                                print('The optimal learning rate: {:.4f}'.format(temp_lr_opt))

                                print('Step 2: Tuning the l1 regularized hyperparameter ...')
                                # Use grid search to find the right value of lambda
                                lambda_range = np.logspace(-2, np.log10(100), 30)
                                error_min = np.zeros((1, len(lambda_range)))
                                rel_error_min = np.zeros((1, len(lambda_range)))
                                decay = 'NA'
                                for idx, lambd in enumerate(lambda_range):
                                    temp_config = dict()
                                    temp_config['num_input'] = X_train1.shape[1]
                                    temp_config['num_layer'] = n_layer_opt
                                    temp_config['num_neuron'] = 128
                                    temp_config['lambda'] = lambd
                                    temp_config['verbose'] = 0
                                    model = MLPSparseModel(temp_config)
                                    model.build_train()
                                    model.train(X_train1, Y_train1[:, i][:, np.newaxis], temp_lr_opt, max_epoch)

                                    Y_pred_val = model.predict(X_train2)
                                    val_abserror = np.mean(
                                        np.abs(Y_pred_val - Y_train2[:, i][:, np.newaxis]))
                                    val_relerror = np.mean(np.abs(
                                        np.divide(Y_train2[:, i][:, np.newaxis] - Y_pred_val,
                                                  Y_train2[:, i][:, np.newaxis])))

                                    error_min[0, idx] = val_abserror
                                    rel_error_min[0, idx] = val_relerror

                                # Find the value of lambda that minimize error_min
                                lambda_f = lambda_range[np.argmin(error_min)]
                                print('The optimal l1 regularizer: {:.4f}'.format(lambda_f))

                                temp_config = dict()
                                temp_config['num_neuron'] = 128
                                temp_config['num_input'] = n - total_environments + 1
                                temp_config['num_layer'] = n_layer_opt
                                temp_config['lambda'] = lambda_f
                                temp_config['verbose'] = 1

                                config.append(temp_config)
                                lr_opt.append(temp_lr_opt)
                                layers.append(n_layer_opt)
                                lambdas.append(lambda_f)

                                print('Training...')
                                # train SeMPL model
                                SeMPL_model = MTLSparseModel(config[i])
                                SeMPL_model.build_train()

                            else:
                                print('Step 1: Tuning the number of layers and the learning rate ...')
                                temp_config = dict()
                                temp_config['num_input'] = n - total_environments + 1
                                temp_config['num_neuron'] = 128
                                temp_config['lambda'] = 'NA'
                                temp_config['decay'] = 'NA'
                                temp_config['verbose'] = 0
                                abs_error_all = np.zeros((20, 4))
                                abs_error_all_train = np.zeros((20, 4))
                                abs_error_layer_lr = np.zeros((20, 2))
                                abs_err_layer_lr_min = 100
                                count = 0
                                layer_range = [layers[0]]
                                lr_range = [0.0001, 0.001, 0.01, 0.1]
                                for n_layer in layer_range:
                                    temp_config['num_layer'] = n_layer
                                    for lr_index, lr_initial in enumerate(lr_range):
                                        model = MTLPlainModel(temp_config)
                                        model.read_weights(weights, bias)
                                        model.build_train()
                                        model.train(X_train1, Y_train1[:, i][:, np.newaxis], lr_initial, max_epoch)

                                        Y_pred_train = model.predict(X_train1)
                                        abs_error_train = np.mean(
                                            np.abs(Y_pred_train - Y_train1[:, i][:, np.newaxis]))
                                        abs_error_all_train[int(n_layer), lr_index] = abs_error_train

                                        Y_pred_val = model.predict(X_train2)
                                        abs_error = np.mean(
                                            np.abs(Y_pred_val - Y_train2[:, i][:, np.newaxis]))
                                        abs_error_all[int(n_layer), lr_index] = abs_error

                                    # Pick the learning rate that has the smallest train cost
                                    # Save testing abs_error correspond to the chosen learning_rate
                                    temp = abs_error_all_train[int(n_layer), :] / np.max(
                                        abs_error_all_train)
                                    temp_idx = np.where(abs(temp) < 0.0001)[0]
                                    if len(temp_idx) > 0:
                                        lr_best = lr_range[np.max(temp_idx)]
                                        err_val_best = abs_error_all[int(n_layer), np.max(temp_idx)]
                                    else:
                                        lr_best = lr_range[np.argmin(temp)]
                                        err_val_best = abs_error_all[int(n_layer), np.argmin(temp)]

                                    abs_error_layer_lr[int(n_layer), 0] = err_val_best
                                    abs_error_layer_lr[int(n_layer), 1] = lr_best

                                    if abs_err_layer_lr_min >= abs_error_all[int(n_layer), np.argmin(temp)]:
                                        abs_err_layer_lr_min = abs_error_all[int(n_layer),
                                                                             np.argmin(temp)]
                                        count = 0
                                    else:
                                        count += 1

                                    if count >= 2:
                                        break
                                abs_error_layer_lr = abs_error_layer_lr[abs_error_layer_lr[:, 1] != 0]

                                # Get the optimal number of layers
                                n_layer_opt = layers[0]

                                # Find the optimal learning rate of the specific layer
                                temp_config['num_layer'] = n_layer_opt
                                for lr_index, lr_initial in enumerate(lr_range):
                                    model = MTLPlainModel(temp_config)
                                    model.read_weights(weights, bias)
                                    model.build_train()
                                    model.train(X_train1, Y_train1[:, i][:, np.newaxis], lr_initial, max_epoch=max_epoch)

                                    Y_pred_train = model.predict(X_train1)
                                    abs_error_train = np.mean(
                                        np.abs(Y_pred_train - Y_train1[:, i][:, np.newaxis]))
                                    abs_error_all_train[int(n_layer), lr_index] = abs_error_train

                                    Y_pred_val = model.predict(X_train2)
                                    abs_error = np.mean(np.abs(Y_pred_val - Y_train2[:, i][:, np.newaxis]))
                                    abs_error_all[int(n_layer), lr_index] = abs_error

                                temp = abs_error_all_train[int(n_layer), :] / np.max(abs_error_all_train)
                                temp_idx = np.where(abs(temp) < 0.0001)[0]
                                if len(temp_idx) > 0:
                                    lr_best = lr_range[np.max(temp_idx)]
                                else:
                                    lr_best = lr_range[np.argmin(temp)]

                                temp_lr_opt = lr_best
                                # print('The optimal number of layers: {}'.format(n_layer_opt))
                                print('The optimal learning rate: {:.4f}'.format(temp_lr_opt))

                                print('Step 2: Tuning the l1 regularized hyperparameter ...')
                                # Use grid search to find the right value of lambda
                                lambda_range = np.logspace(-2, np.log10(100), 30)
                                error_min = np.zeros((1, len(lambda_range)))
                                rel_error_min = np.zeros((1, len(lambda_range)))
                                decay = 'NA'
                                for idx, lambd in enumerate(lambda_range):
                                    temp_config = dict()
                                    temp_config['num_input'] = X_train1.shape[1]
                                    temp_config['num_layer'] = n_layer_opt
                                    temp_config['num_neuron'] = 128
                                    temp_config['lambda'] = lambd
                                    temp_config['verbose'] = 0
                                    model = MTLSparseModel(temp_config)
                                    model.read_weights(weights, bias)
                                    model.build_train()
                                    model.train(X_train1, Y_train1[:, i][:, np.newaxis], temp_lr_opt, max_epoch=max_epoch)

                                    Y_pred_val = model.predict(X_train2)
                                    val_abserror = np.mean(
                                        np.abs(Y_pred_val - Y_train2[:, i][:, np.newaxis]))
                                    val_relerror = np.mean(
                                        np.abs(np.divide(Y_train2[:, i][:, np.newaxis] - Y_pred_val,
                                                         Y_train2[:, i][:, np.newaxis])))

                                    error_min[0, idx] = val_abserror
                                    rel_error_min[0, idx] = val_relerror

                                # Find the value of lambda that minimize error_min
                                lambda_f = lambda_range[np.argmin(error_min)]
                                print('The optimal l1 regularizer: {:.4f}'.format(lambda_f))

                                temp_config = dict()
                                temp_config['num_neuron'] = 128
                                temp_config['num_input'] = n - total_environments + 1
                                temp_config['num_layer'] = n_layer_opt
                                temp_config['lambda'] = lambda_f
                                temp_config['verbose'] = 1

                                config.append(temp_config)
                                lr_opt.append(temp_lr_opt)
                                layers.append(n_layer_opt)
                                lambdas.append(lambda_f)

                                print('Training...')
                                # train SeMPL model
                                SeMPL_model = MTLSparseModel(config[i])
                                SeMPL_model.read_weights(weights, bias)
                                SeMPL_model.build_train()

                                # temp_lr_opt = lr_opt[0]
                                n_layer_opt = layers[0]
                                lambda_f = lambdas[0]
                                temp_config = config[0]
                                config.append(temp_config)
                                lr_opt.append(temp_lr_opt)  # new
                                layers.append(n_layer_opt)
                                lambdas.append(lambda_f)

                            SeMPL_model.train(X_train, Y_train[:, i][:, np.newaxis], lr_opt[i], max_epoch=max_epoch)
                            weights, bias = SeMPL_model.get_weights()

                    if save_file:
                        if not os.path.exists(saving_folder):
                            print('\tCreating folder: {}'.format(saving_folder))
                            os.makedirs(saving_folder)
                        print('\tSaving meta model to: {}'.format(saving_folder))
                        np.save(saving_file_weights, (weights))
                        np.save(saving_file_bias, (bias))
                    return weights, bias
                else:
                    print('\t>> Reading meta model from: {} and {}'.format(saving_file_weights, saving_file_bias))
                    weights = np.load(saving_file_weights, allow_pickle=True)
                    bias = np.load(saving_file_bias, allow_pickle=True)
                    return weights, bias