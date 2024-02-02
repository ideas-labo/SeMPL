import tensorflow.compat.v1 as tf
import tensorflow as tf2
tf.disable_v2_behavior()
import numpy as np
import sys
from utils.general import init_dir, random_mini_batches
# from tensor_toolbox_yyang import TensorProducer
import math


def neural_net(tf_x, n_layer, n_neuron, lambd, weights, bias):
    """
    Args:
        tf_x: input placeholder
        n_layer: number of layers of hidden layer of the neural network
        lambd: regularized parameter

    """
    if weights==[]:
        layer = tf_x
        for i in range(1, n_layer + 1):
            layer = tf.layers.dense(layer, n_neuron, tf.nn.relu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(seed=1), name=str(i))
        output = tf.layers.dense(layer, 1, name='o')
    else:
        layer = tf_x
        # layer_2 = tf.placeholder("float", [None, 18])
        for i in range(1, n_layer+1):
            # print(np.array(weights[i-1]).shape)
            init_k = tf.constant_initializer(weights[i-1])
            init_b = tf.constant_initializer(bias[i-1])
            # print(n_layer, n_neuron,str(i))

            layer = tf.layers.dense(layer, n_neuron, tf.nn.relu,
                                    kernel_initializer=init_k,
                                    bias_initializer=init_b, name=str(i))
        init_k = tf.constant_initializer(weights[n_layer])
        init_b = tf.constant_initializer(bias[n_layer])
        output = tf.layers.dense(layer, 1,
                                 kernel_initializer=init_k,
                                 bias_initializer=init_b, name='o')
    return output


class MTLPlainModel(object):
    """Generic class for tf l1-sparse MTL models"""

    def __init__(self, config):
        """
        Args:
            config: Config instance defining hyperparams
            dir_ouput: output directory (store model and log files)
        """
        self._config = config
        # self._dir_output = dir_output
        tf.reset_default_graph()  # Saveguard if previous model was defined
        tf.set_random_seed(1)    # Set tensorflow seed for paper replication
        self.weights = []
        self.bias = []



    def build_train(self):
        """Builds model for training"""
        self._add_placeholders_op()
        self._add_pred_op()
        self._add_loss_op()
        self._add_train_op(self.loss)

        self.init_session()


    def build_pred(self):
        """Builds model for predicting"""
        self._add_placeholders_op()
        self._add_pred_op()
        self._add_loss_op()

        self.init_session()


    def _add_placeholders_op(self):
        """ Add placeholder attributes """
        # self.X = tf.placeholder("float", [None, self._config['num_input']])
        self.X = tf.placeholder("float", [None, self._config['num_input']])
        self.Y = tf.placeholder("float", [None, 1])
        #the second output
        # self.Y_2 = tf.placeholder("float", [None, 1])

        self.lr = tf.placeholder("float")  # to schedule learning rate


    def _add_pred_op(self):
        """Defines self.pred"""
        self.output = neural_net(self.X,
                                 self._config['num_layer'],
                                 self._config['num_neuron'],
                                 self._config['lambda'], self.weights, self.bias)


    def _add_loss_op(self):
        """Defines self.loss"""
        # l2_loss = tf.losses.get_regularization_loss()
        # #print(self.output.get_shape())
        # output_1, output_2 = self.output[:,0:1], self.output[:,1:2]
        # #print(output_1.get_shape())
        # self.loss_1 = l2_loss + tf.losses.mean_squared_error(self.Y, output_1)
        # self.loss_2 = l2_loss + tf.losses.mean_squared_error(self.Y_2, output_2)
        # self.loss = self.loss_1 + self.loss_2

        l2_loss = tf.losses.get_regularization_loss()
        self.loss = l2_loss + tf.losses.mean_squared_error(self.Y, self.output)


    def _add_train_op(self, loss):
        """Defines self.train_op that performs an update on a batch

        Args:
            lr: (tf.placeholder) tf.float32, learning rate
            loss: (tensor) tf.float32 loss to minimize

        """
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            grads, vs     = zip(*optimizer.compute_gradients(loss))
            grads, gnorm  = tf.clip_by_global_norm(grads, 0.01)
            self.train_op = optimizer.apply_gradients(zip(grads, vs))


    def init_session(self):
        """Defines self.sess, self.saver and initialize the variables"""
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.compat.v1.train.Saver()


    def train(self, X_matrix, perf_value, lr_initial, max_epoch):
        """Global training procedure

        Calls method self.run_epoch and saves weights if score improves.
        All the epoch-logic must be done in self.run_epoch

        Args:
            X_matrix: Input matrix
            perf_value: Performance value
            lr_initial: Initial learning rate
        """
#        l_old = 0
        lr = lr_initial
        decay = lr_initial/1000

        m = X_matrix.shape[0]
        batch_size = m
        seed = 0    # seed for minibatches
        for epoch in range(1, max_epoch):

            minibatch_loss = 0
            num_minibatches = int(m/batch_size)
            seed += 1
            minibatches = random_mini_batches(X_matrix, perf_value, batch_size, seed)
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _, t_l, pred = self.sess.run([self.train_op, self.loss, self.output],
                                             {self.X: X_matrix, self.Y: perf_value, self.lr: lr})
                minibatch_loss += t_l/num_minibatches

            if epoch % 500 == 0 or epoch == 1:
                # RMSE = []
                # for j in range(len(pred.ravel())):
                #     RMSE.append(pow(perf_value.ravel()[j].ravel() - pred.ravel()[j].ravel(), 2))
                # RMSE = math.sqrt(np.sum(RMSE) / len(pred.ravel()))

                rel_error = np.mean(np.abs(np.divide(perf_value.ravel() - pred.ravel(), perf_value.ravel())))
                # rel_error1 = np.mean(np.abs(np.divide(perf_value.ravel() - pred.ravel()[0], perf_value.ravel())))
                # rel_error2 = np.mean(np.abs(np.divide(perf_value.ravel() - pred.ravel()[1], perf_value.ravel())))
                # rel_error = np.divide(rel_error1 + rel_error2, 2)

                if self._config['verbose']:
                    print("Cost function: {:.4f}", minibatch_loss)
                    print("Train RE: {:.4f}", rel_error)

#                if np.abs(minibatch_loss-l_old)/minibatch_loss < 1e-8:
#                    break;

#            # Store the old cost function
#            l_old = minibatch_loss

            # Decay learning rate
            lr = lr*1/(1 + decay*epoch)



    def get_weights(self):
        dir_model = '/model/'
        weights_read = []
        bias_read = []
        init_dir(dir_model)
        # saving
        path = self.saver.save(self.sess, dir_model + 'model.ckpt')
        reader = tf.train.NewCheckpointReader(path)
        for i in range(1, self._config['num_layer']+1):
            weights = reader.get_tensor(str(i) + '/kernel')  # weight的名字，是由对应层的名字，加上默认的"kernel"组成的
            bias = reader.get_tensor(str(i) + '/bias')  # bias的名字
            weights_read.append(weights)
            bias_read.append(bias)
        weights = reader.get_tensor('o' + '/kernel')  # weight的名字，是由对应层的名字，加上默认的"kernel"组成的
        bias = reader.get_tensor('o' + '/bias')  # bias的名字
        weights_read.append(weights)
        bias_read.append(bias)
        return weights_read, bias_read


    def read_weights(self, weights, bias):
        # print('Reading weights and bias...')
        self.weights = weights
        self.bias = bias


    def predict(self, X_matrix_pred):
        """Predict performance value"""

        Y_pred_val = self.sess.run(self.output, {self.X: X_matrix_pred})
        # print the result
        # print('Sparse model predicted result: {}'.format(Y_pred_val))`

        return Y_pred_val


    # def save_session(self):
    #     """Saves session"""
    #     print('\nSaving model...')
    #     dir_model = './model/'
    #     init_dir(dir_model)
    #
    #     # saving
    #     path = self.saver.save(self.sess, dir_model + 'model.ckpt')
    #
    #     return path
    #
    #
    # def restore_session(self, path):
    #     """Reload weights into session
    #
    #     Args:
    #         sess: tf.Session()
    #         dir_model: dir with weights
    #
    #     """
    #     print('Loading model...')
    #     reader = tf.train.NewCheckpointReader(path)
    #     for i in range(1, self._config['num_layer']+1):
    #         weights = reader.get_tensor(str(i) + '/kernel')  # weight的名字，是由对应层的名字，加上默认的"kernel"组成的
    #         bias = reader.get_tensor(str(i) + '/bias')  # bias的名字
    #         self.weights.append(weights)
    #         self.bias.append(bias)
    #     weights = reader.get_tensor('o' + '/kernel')  # weight的名字，是由对应层的名字，加上默认的"kernel"组成的
    #     bias = reader.get_tensor('o' + '/bias')  # bias的名字
    #     self.weights.append(weights)
    #     self.bias.append(bias)
    #     print('self.weights type: ', type(self.weights))
    #     print('self.bias type: ', type(self.bias))
    #     print('weights type: ', type(weights), weights.shape)
    #     print('bias type: ', type(bias), bias.shape)
    #
    #
    # def decomposition(self, path):
    #     print('Performing Decomposition...')
    #     reader = tf.train.NewCheckpointReader(path)
    #     dcmp_weights = []
    #     for i in range(0, self._config['num_layer']):
    #         weights = reader.get_tensor('1' + '/kernel')  # weight的名字，是由对应层的名字，加上默认的"kernel"组成的
    #         dcmp_weights.append(weights)
    #     # print('Before: ', np.array(dcmp_weights).shape)
    #     dcmp_weights = TensorProducer(np.array(dcmp_weights), 'Tucker', eps_or_k=0.1, return_true_var=False)
    #     # print('After decomposition: ', dcmp_weights)
    #
