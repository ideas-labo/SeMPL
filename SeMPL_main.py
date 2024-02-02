import os
import time
import random
import numpy as np
from random import sample
from numpy import genfromtxt
from utils.meta_training import meta_training
from utils.Meta_sparse_model_tf2 import MTLSparseModel
from utils.sequence_selection import sequence_selection
from utils.hyperparameter_tuning import hyperparameter_tuning
from utils.general import get_sizes, get_non_zero_indexes, process_training_data
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    subject_systems = ['deeparch-SizeReduction-3environments.csv','sac_srad_others-5environments.csv', 'sqlite-overwritebatch-4environments.csv', 'nginx-4environments.csv', 'spear-10286-6environments.csv', 'storm-obj2-3environments.csv', 'imagemagick-4environments.csv', 'exastencils-4environments.csv', 'x264-diff_input-10environments.csv']

    ########### experiment parameters ###########
    selected_sys = [6]  # set the subject systems to evaluate
    selected_sizes = [0,1,2,3,4]  # set the training sample sizes to evaluate
    save_MRE = True # save the evaluation results
    test_mode = True # to tune the DNN hyperparameters
    save_best_sequence = False # to save the selected best sequences
    save_meta_model = False # to save the pre-trained meta models
    read_meta_model = True # Load the pre-trained meta model if exists
    seed = 2
    N_experiments = 30
    start = 0
    meta_sample_percentage = 100 # from 0 to 100
    max_epoch = 1000
    learned_environments = [] # to exclude the specified target environments
    learned_meta_models = [] # to exclude the specified meta models
    ########### experiment parameters ###########

    for i_temp_sys, temp_sys in enumerate(subject_systems):
        dir_data = 'data/{}'.format(temp_sys)
        if i_temp_sys in selected_sys:
            system = dir_data.replace('data/', '').replace('.csv', '')
            total_environments = int(dir_data.split('environments')[0].split('-')[-1])
            sample_sizes, environments = get_sizes(dir_data, total_environments) # get the total number of meta-environments and the training sample sizes for evaluations
            print('Dataset: ' + system)
            whole_data = genfromtxt(dir_data, delimiter=',', skip_header=1)
            (N, n) = whole_data.shape
            n = n - 1
            N_features = n + 1 - total_environments
            print('Number of expriments: {} \nTotal sample size: {} \nNumber of features: {} \nTraining sizes: {} \nTotal number of environments: {}'.format(N_experiments, N, N_features, sample_sizes, len(environments)))
            N_meta_environments = total_environments - 1 # exlude the target environment

            for i_size in selected_sizes:
                print('--- Subject system: {}, Size: S_{} ---'.format(system.split('-')[0].split('_')[0], i_size+1))
                non_zero_indexes = get_non_zero_indexes(whole_data, total_environments) # delete the zero-performance samples
                N_train = sample_sizes[i_size]
                N_test = (len(non_zero_indexes) - N_train)
                meta_samples = int((meta_sample_percentage / 100) * len(non_zero_indexes))
                print('Training size: {}, testing size: {}, Meta-training size ({}% samples): {}'.format(N_train, N_test, meta_sample_percentage, meta_samples))
                if read_meta_model: # read the pre-trained meta model
                    start_time_sequence_selection = time.time() # Start measure time
                    print('\n> Sequence selection...')
                    meta_to_train = sequence_selection(dir_data, save_best_sequence)
                    sequence_selection_time = ((time.time() - start_time_sequence_selection) / 60)
                    print('\tTarget_environment: [best sequence] --- {}'.format(meta_to_train))
                    print('\t>> Sequence selection time (min): {}'.format(sequence_selection_time))

                    for main_environment in meta_to_train: # for each target environment to evaluate
                        for meta_environments in meta_to_train[main_environment]: # for each meta model to train
                            if int(main_environment) not in learned_environments and meta_environments not in learned_meta_models:
                                saving_file_name = 'results/SeMPL_{}_T{}_M{}_{}_{}-{}_{}.txt'.format(dir_data.split('/')[1].split('.')[0],
                                    main_environment, meta_environments, seed, N_train, meta_samples,time.strftime('%m-%d_%H-%M-%S',time.localtime(time.time())))
                                main_environment = int(main_environment)
                                if save_MRE:
                                    if not os.path.exists('{}/results'.format(os.getcwd())):
                                        print('\tCreating folder: {}'.format('{}/results'.format(os.getcwd())))
                                        os.makedirs('{}/results'.format(os.getcwd()))
                                    with open(saving_file_name, 'w') as f:  # save the results
                                        f.write('N_train={} N_test={}'.format(N_train, N_test))

                                print('\n> Meta-training in order {} for target environment E_{}...'.format(meta_environments, main_environment))
                                start_time = time.time()  # Start measure time
                                reading_file_weights = 'Models/weights_{}_M{}_{}.npy'.format(dir_data.split('/')[1].split('.')[0],meta_environments, meta_samples)
                                reading_file_bias = 'Models/bias_{}_M{}_{}.npy'.format(dir_data.split('/')[1].split('.')[0],meta_environments, meta_samples)

                                if (os.path.exists(reading_file_weights) and os.path.exists(reading_file_bias)): # if the meta model is pre-trained
                                    print('\t>> Reading meta model from: {}'.format(reading_file_weights))
                                    weights = np.load(reading_file_weights, allow_pickle=True)
                                    bias = np.load(reading_file_bias, allow_pickle=True)
                                else: # meta training
                                    weights, bias = meta_training(dir_data=dir_data, selected_environments=[main_environment], meta_to_train=meta_to_train, save_file=save_meta_model, test_mode=test_mode, max_epoch=max_epoch, meta_sample_percentage=meta_sample_percentage)
                                    weights = np.array(weights)
                                    bias = np.array(bias)
                                meta_training_time = ((time.time() - start_time) / 60) # get the total running time
                                print('\t>> Meta training time (min): {}'.format(meta_training_time))

                                print('\n> Fine-tuning...')
                                for ne in range(start, start + N_experiments):
                                    # print('\tRun {}: '.format(ne + 1))
                                    random.seed(ne * seed)
                                    N_environment = len(meta_environments)  # the number of environments to group
                                    start_time = time.time() # Start measure time
                                    if save_MRE:
                                        with open(saving_file_name, 'a') as f:  # save the results
                                            f.write('\nRun {}'.format(ne + 1))

                                    # generate the training and testing samples
                                    non_zero_indexes = get_non_zero_indexes(whole_data, total_environments)
                                    testing_index = sample(list(non_zero_indexes), N_test)
                                    non_zero_indexes = np.setdiff1d(non_zero_indexes, testing_index)
                                    training_index = sample(list(non_zero_indexes), N_train)

                                    ### process training data
                                    max_X, X_train, X_train1, X_train2, max_Y, Y_train, Y_train1, Y_train2 = process_training_data(whole_data, training_index, N_features, n, main_environment)
                                    ### process testing data
                                    Y_test = whole_data[testing_index, n - main_environment][:, np.newaxis]
                                    X_test = np.divide(whole_data[testing_index, 0:N_features], max_X)

                                    # default hyperparameters, just for testing
                                    if test_mode == True:
                                        lr_opt = 0.123
                                        n_layer_opt = weights.shape[0] - 1
                                        lambda_f = 0.123
                                        config = dict()
                                        config['num_neuron'] = 128
                                        config['num_input'] = N_features
                                        config['num_layer'] = n_layer_opt
                                        config['lambda'] = lambda_f
                                        config['verbose'] = 0
                                    # if not test_mode, tune the hyperparameters
                                    else:
                                        n_layer_opt = weights.shape[0] - 1
                                        lambda_f, lr_opt = hyperparameter_tuning([N_features, X_train1, Y_train1, X_train2, Y_train2, n_layer_opt, max_epoch])
                                        config = dict()
                                        config['num_neuron'] = 128
                                        config['num_input'] = N_features
                                        config['num_layer'] = n_layer_opt
                                        config['lambda'] = lambda_f
                                        config['verbose'] = 0

                                    SeMPL_model = MTLSparseModel(config)
                                    if read_meta_model:
                                        SeMPL_model.read_weights(weights, bias) # load the pre-trained meta model
                                    SeMPL_model.build_train()
                                    SeMPL_model.train(X_train, Y_train, lr_opt, max_epoch=max_epoch)

                                    training_time = ((time.time() - start_time) / 60) # get the total running time
                                    # computing accuracy
                                    rel_error = []
                                    Y_pred_test = max_Y * SeMPL_model.predict(X_test)
                                    rel_error = np.mean(np.abs(np.divide(Y_test.ravel() - Y_pred_test.ravel(), Y_test.ravel()))) * 100
                                    print('\t>> Run{} {} S_{} E_{} MRE: {:.2f}, Training time (min): {}'.format(ne+1, system, i_size+1, main_environment, rel_error, training_time))

                                    if save_MRE:
                                        with open(saving_file_name, 'a') as f:  # save the results
                                            f.write('\nTarget_environment {} SeMPL RE: {}'.format(main_environment, rel_error))
                                            f.write('\ntime (min): {}'.format(training_time))
