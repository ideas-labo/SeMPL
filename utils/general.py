import os
import numpy as np
import math
import warnings
warnings.filterwarnings("ignore")

def get_non_zero_indexes(whole_data, total_tasks):
    (N, n) = whole_data.shape
    n = n - 1
    delete_index = set()
    temp_index = list(range(N))
    for i in range(total_tasks):
        temp_Y = whole_data[:, n - i]
        for j in range(len(temp_Y)):
            if temp_Y[j] == 0:
                delete_index.add(j)
    non_zero_indexes = np.setdiff1d(temp_index, list(delete_index))
    return non_zero_indexes


def process_training_data(whole_data, training_index, N_features, n, main_task):
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

    ### process y
    temp_Y = whole_data[training_index, n - main_task][:, np.newaxis]
    # scale y
    temp_max_Y = np.max(temp_Y) / 100
    if temp_max_Y == 0:
        temp_max_Y = 1
    temp_Y = np.divide(temp_Y, temp_max_Y)
    Y_train = np.array(temp_Y)

    # Split train data into 2 parts (67-33)
    Y_train1 = (temp_Y[0:N_cross, :])
    Y_train2 = (temp_Y[N_cross:len(temp_Y), :])

    return temp_max_X, X_train, X_train1, X_train2, temp_max_Y, Y_train, Y_train1, Y_train2


def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:]
    shuffled_Y = Y[permutation,:]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


def init_dir(dir_name):
    """Creates directory if it does not exists"""
    if dir_name is not None:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)


def get_sizes(dir_data, total_environments):
    if dir_data == 'Data/deeparch-SizeReduction-3environments.csv':
        sample_sizes = [12, 24, 36, 48, 60]
        selected_environments = range(total_environments)
    elif dir_data == 'Data/sqlite-overwritebatch-4environments.csv':
        sample_sizes = [14, 28, 42, 56, 70]
        selected_environments = range(total_environments)
    elif dir_data == 'Data/sac_srad_others-5environments.csv':
        sample_sizes = [58, 116, 174, 232, 290]
        selected_environments = range(total_environments)
    elif dir_data == 'Data/spear-10286-6environments.csv':
        sample_sizes = [14, 28, 42, 56, 70]
        selected_environments = range(total_environments)
    elif dir_data == 'Data/storm-obj2-3environments.csv':
        sample_sizes = [158, 261, 422, 678, 903]
        selected_environments = range(total_environments)
    elif dir_data == 'Data/imagemagick-4environments.csv':
        sample_sizes = [11, 24, 45, 66, 70]
        selected_environments = range(total_environments)
    elif dir_data == 'Data/x264-diff_input-10environments.csv':
        sample_sizes = [24, 53, 81, 122, 141]
        selected_environments = range(total_environments)
    elif dir_data == 'Data/exastencils-4environments.csv':
        sample_sizes = [106, 181, 355, 485, 695]
        selected_environments = range(total_environments)
    elif dir_data == 'Data/nginx-4environments.csv':
        sample_sizes = [16, 32, 48, 64, 80]
        selected_environments = range(total_environments)
    return sample_sizes, selected_environments