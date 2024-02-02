import os
import time
import copy
import random
import warnings
import itertools
import numpy as np
import pandas as pd
from random import sample
from numpy import genfromtxt
from sklearn import preprocessing
warnings.filterwarnings("ignore")
from collections import Counter
from sklearn.linear_model import SGDRegressor
from utils.general import get_non_zero_indexes


def sequence_selection(dir_data, save_file=True, meta_sample_percentage=100):
    total_environments = int(dir_data.split('environments')[0].split('-')[-1])
    seed = 2
    whole_data = genfromtxt(dir_data, delimiter=',', skip_header=1)
    (N, n) = whole_data.shape
    n = n - 1
    non_zero_indexes = get_non_zero_indexes(whole_data, total_environments)  # delete the zero-performance samples
    N_meta_samples = int((meta_sample_percentage / 100) * len(non_zero_indexes))  # the number of meta training samples
    saving_folder = '{}/Best_sequences'.format(os.getcwd())
    saving_best_sequence = '{}/best_sequence_{}_{}_{}.txt'.format(saving_folder, dir_data.split('/')[1].split('.')[0], seed, N_meta_samples)
    if (os.path.exists(saving_best_sequence)):  # if the best sequence is saved
        print('\n> Reading best sequence from {}...'.format(saving_best_sequence))
        with open(saving_best_sequence, 'r') as f:  # save the results
            lines = f.readlines()[0]
            import ast
            meta_to_train = ast.literal_eval(lines)
    else:
        selected_environments = list(range(0,total_environments))
        learned_environments = []
        random.seed(seed)
        temp_N_test = int(N_meta_samples * 3 / 10) # use 30% samples for testing and the others for meta training
        LR_max_iter = 500
        N_experiments = 30
        start = 0
        N_features = n + 1 - total_environments
        random.seed(seed)
        # min_max_scaler = preprocessing.MinMaxScaler(feature_range=(1,100))
        # for environment in range(total_environments): # data normalization
        #     whole_data[:, n-environment] = min_max_scaler.fit_transform(np.array(whole_data[:, n-environment]).reshape(-1, 1))[:, 0]

        meta_to_train = {} # to save the final best sequence to train
        for main_environment in selected_environments: # set each environment to be the target environment, and others to be the meta environments
            if main_environment not in learned_environments:
                groups_all = [] # save all possible sequence of training the environments
                environments = copy.deepcopy(selected_environments)
                environments.remove(main_environment) # set the meta environments
                # get all possible sequence of training the environments
                for i in range(1, 2): # only care about single-environment models here for efficiency
                    iter = itertools.permutations(environments, i)
                    combinations = list(iter)
                    groups_all += combinations
                # print('Target environment: {}, meta-environments: {}'.format(main_environment, groups_all))

                LR_results = {} # to save the test results
                start_time = time.time()
                for temp_main_environment in environments: # Take a environment as the temporary target environment and test it over all data for the remaining meta environments
                    # print('Temp target environment: {}'.format(temp_main_environment))
                    if 'main{}'.format(temp_main_environment) in LR_results:
                        temp_LR_result = LR_results['main{}'.format(temp_main_environment)]
                    else:
                        temp_LR_result = {}
                    for combination in groups_all: # each combination corresponds to a sequence of learning
                        if temp_main_environment not in combination: # leave the sequences with the temporary target environment
                            model = SGDRegressor(max_iter=LR_max_iter, warm_start=True, random_state=seed, penalty='l1')
                            for ne in range(start, start + N_experiments):
                                random.seed(ne * seed)
                                meta_samples = sample(list(non_zero_indexes), N_meta_samples)
                                # train the meta environments in order of this combination
                                for temp_train_environment in combination:
                                    X_train = whole_data[meta_samples, 0:N_features]
                                    Y_train = whole_data[meta_samples, n - temp_train_environment][:, np.newaxis]
                                    model.fit(X_train, Y_train)

                                # fine-tuning using the temporary target environment
                                temp_testing_index = sample(meta_samples, temp_N_test)
                                temp_training_index = np.setdiff1d(meta_samples, temp_testing_index)
                                X_train = whole_data[temp_training_index, 0:N_features]
                                Y_train = whole_data[temp_training_index, n - temp_main_environment][:, np.newaxis]
                                X_test = whole_data[temp_testing_index, 0:N_features]
                                Y_test = whole_data[temp_testing_index, n - temp_main_environment][:, np.newaxis]
                                model.fit(X_train, Y_train)
                                # test the results of the fune-tuned meta model
                                Y_pred_test = model.predict(X_test)
                                rel_error = np.mean(np.abs(np.divide(Y_test.ravel() - Y_pred_test.ravel(), Y_test.ravel()))) * 100
                                # save the test results
                                if 'meta{}'.format(combination) in temp_LR_result:
                                    temp_LR_result['meta{}'.format(combination)].append(rel_error)
                                else:
                                    temp_LR_result['meta{}'.format(combination)] = [rel_error]
                            # normalize the test results
                            min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 100))
                            temp_LR_result['meta{}'.format(combination)] = min_max_scaler.fit_transform(np.array(temp_LR_result['meta{}'.format(combination)]).reshape(-1, 1)).ravel().tolist()

                    LR_results['main{}'.format(temp_main_environment)] = temp_LR_result

                # calculate the overall test results for each sequence (combination)
                temp_meta_results = {}
                for meta in groups_all: # for sequence of learning
                    meta = 'meta{}'.format(meta).replace('(', '').replace(',)', '').replace(')', '')
                    for temp_main in LR_results:
                        for temp_meta in LR_results[temp_main]:
                            if temp_meta.replace('(', '').replace(',)', '').replace(')', '') == meta:
                                temp_meta_new = (temp_meta.replace(',)', '').replace('meta(', '')).replace(')', '')
                                if temp_meta_new not in temp_meta_results:
                                    temp_meta_results[temp_meta_new] = {}
                                temp_meta_results[temp_meta_new][temp_main] = LR_results[temp_main][temp_meta]
                meta_results = {}
                for temp_meta in temp_meta_results:
                    average_results = []
                    for temp_main in temp_meta_results[temp_meta]:
                        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 100))
                        temp_normalised_result = min_max_scaler.fit_transform(
                            np.array(temp_meta_results[temp_meta][temp_main]).reshape(-1, 1))
                        average_results += list(temp_normalised_result.ravel())
                    meta_results[temp_meta] = average_results
                # print(meta_results)

                # use Scott-Knott test to rank each single-environment model to decide the best sequence
                scott_scores = {}
                if len(environments) >= 3: # Scott-Knott test only works for more than 2 groups
                    for temp_main_environment in LR_results:
                        # print('Temp target environment: {}'.format(temp_main_environment))
                        data = pd.DataFrame(LR_results[temp_main_environment])
                        from rpy2.robjects.packages import importr
                        from rpy2.robjects import r, pandas2ri
                        pandas2ri.activate()
                        sk = importr('ScottKnottESD')
                        r_sk = sk.sk_esd(data)  # get the rankings
                        # print(r_sk)
                        environment_sk = np.array(r_sk)[3]
                        groups_sk = np.array(r_sk)[1]
                        max_score = np.max(groups_sk)
                        for i in range(len(groups_sk)):
                            groups_sk[i] = max_score - groups_sk[i] + 1
                        for i, environment in enumerate(environment_sk):
                            temp = r_sk[2][int(environment_sk[i]) - 1]
                            temp_meta = []
                            for j, temp2 in enumerate(temp.replace('meta', '').split('..')):
                                temp2 = temp2.replace('.', '')
                                if temp2 != '':
                                    temp_meta.append(int(temp2))
                            temp_meta = '{}'.format(temp_meta).replace('[', '').replace(']', '')

                            if temp_meta not in scott_scores:
                                scott_scores[temp_meta] = [groups_sk[i]]
                            else:
                                scott_scores[temp_meta].append(groups_sk[i])
                else: # simply use the mean MRE as their rank
                    for temp_meta in meta_results:
                        scott_scores[temp_meta] = np.mean(meta_results[temp_meta])
                for temp_meta in scott_scores:
                    scott_scores[temp_meta] = np.mean(scott_scores[temp_meta])
                # print(scott_scores)

                # rank each meta model in 3 dimensions: SK rank, mean MRE and IQR
                for temp_meta in meta_results:
                    average_results = meta_results[temp_meta]
                    Q1 = np.percentile(average_results, 25, interpolation='midpoint')
                    Q3 = np.percentile(average_results, 75, interpolation='midpoint')
                    IQR = Q3 - Q1
                    temp_meta_results[temp_meta] = [scott_scores[temp_meta], np.mean(average_results), IQR]
                temp_meta_results = sorted(temp_meta_results.items(), key=lambda item: (item[1][0], item[1][1]), reverse=True)
                # print(temp_meta_results)
                best_sequence = []
                for temp_meta in enumerate(temp_meta_results):
                    best_sequence.append(int(temp_meta[1][0].replace('meta[','').replace(']','')))
                meta_to_train[main_environment] = [best_sequence]

        # total_time = (time.time() - start_time) / 60
        # print('Total time cost: {}'.format(total_time))
        if save_file:
            if not os.path.exists(saving_folder):
                print('\tCreating folder: {}'.format(saving_folder))
                os.makedirs(saving_folder)
            import json
            # print(json.dumps(meta_to_train))
            with open(saving_best_sequence, 'w') as f:  # save the results
                f.write(json.dumps(meta_to_train))
            print('\tResults saved to {}'.format(saving_best_sequence))

    return meta_to_train
