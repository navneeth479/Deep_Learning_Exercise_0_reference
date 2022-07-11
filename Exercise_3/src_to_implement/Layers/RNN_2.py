import numpy as np
from .Base import BaseLayer
from .TanH import TanH
from .Sigmoid import Sigmoid
from .FullyConnected import FullyConnected


class RNN(BaseLayer):
    _optimizer: object = None
    _gradient_weights = 0
    _memorize = False
    output_fc_layer_gradient_weights = 0

    regularization_loss = 0
    input_size: None
    hidden_size: None
    output_size: None
    hidden_fc_layer_input_tensor = np.ndarray(shape=None, dtype=np.float64)
    output_fc_layer_input_tensor = np.ndarray(shape=None, dtype=np.float64)
    tan_activation = np.ndarray(shape=None, dtype=np.float64)
    sigmoid_activation = np.ndarray(shape=None, dtype=np.float64)
    input_tensor = np.ndarray(shape=None, dtype=np.float64)
    hidden_state = np.ndarray(shape=None, dtype=np.float64)

    def __init__(self, input_size, hidden_size, output_size):
        self.tan = TanH()
        self.sigmoid = Sigmoid()
        self.trainable = True
        self.input_size = input_size  # the dimension of the input vector
        self.hidden_size = hidden_size  # the dimension of the hidden state
        self.output_size = output_size

        self.hidden_fc_layer = FullyConnected(input_size + hidden_size, hidden_size)
         # Before initializing weights it will initialize randomly
        self.output_fc_layer = FullyConnected(hidden_size, output_size)

        self.hidden_state = np.zeros(shape=(1, self.hidden_size))
        self.output_fc_layer_gradient_weights = np.zeros_like(self.output_fc_layer.weights)

    def initialize(self, weights_initializer, bias_initializer):
        self.hidden_fc_layer.initialize(weights_initializer, bias_initializer)
        self.output_fc_layer.initialize(weights_initializer, bias_initializer)

    def forward(self, input_tensor):
        # input 13 * 9
        # batch 9
        # hidden_state 7
        # output 5
        batch_size = input_tensor.shape[0]
        # Not sure about this!!!should become zero for each sequence?
        self.hidden_fc_layer_input_tensor = np.zeros(shape=(batch_size, self.hidden_size + self.input_size + 1))
        self.output_fc_layer_input_tensor = np.zeros(shape=(batch_size, self.hidden_size + 1))
        self.tan_activation = np.zeros(shape=(batch_size, self.hidden_size))
        self.sigmoid_activation = np.zeros(shape=(batch_size, self.output_size))

        output_tensor = np.ndarray(shape=(batch_size, self.output_size))  # initialize hidden_state

        # self.input_tensor = input_tensor  # 9*13  input tensor

        if not self.memorize:
            hidden_state = np.zeros(shape=(1, self.hidden_size))  # initialize hidden_state with zero
        else:
            hidden_state = self.hidden_state

        for i in range(batch_size):
            # concatenation previous hidden state with corresponding input tensor
            x_tilda = np.concatenate((hidden_state, input_tensor[i, None]), axis=1)  # 1*21

            tan_input = self.hidden_fc_layer.forward(x_tilda)

            # line below: x_tilda, saved for set as input_tensor of hidden layer in the backward mode
            self.hidden_fc_layer_input_tensor[i] = np.transpose(self.hidden_fc_layer.input_tensor)[0]

            # update hidden state, should I save it?
            hidden_state = self.tan.forward(tan_input)  # 1*7

            # saved for set activations in backward mode
            self.tan_activation[i] = self.tan.activations

            # output
            transition_of_hy = self.output_fc_layer.forward(hidden_state)
            # hidden_state with bias, saved for set as input_tensor of output layer in backward
            self.output_fc_layer_input_tensor[i] = np.transpose(self.output_fc_layer.input_tensor)

            output_tensor[i] = self.sigmoid.forward(transition_of_hy)
            #  saved for set activations in backward mode
            self.sigmoid_activation[i] = self.sigmoid.activations

        # save hidden state in the last iteration for initialize hidden state for the next sequence
        self.hidden_state = self.tan_activation[-1, None]
        return output_tensor

    def backward(self, error_tensor):
        # Start from the last element in error_tensor, means the last time_step?
        # Collect gradiant with respect to x and return it at the end #DONE
        # We need input tensor for calculation gradiant with respect to weight,
        #  self.weights = self.hidden_fc_layer.weights
        time_step = error_tensor.shape[0] - 1
        # error tensor pass to the next sequence should have same dimension as input
        error_ = np.zeros(shape=(error_tensor.shape[0], self.input_size))
        gradiant_previous_hidden_state = np.zeros(shape=(1, self.hidden_size))
        self.gradient_weights = np.zeros_like(self.hidden_fc_layer.weights)

        while time_step >= 0:  # can use np.flip(error_tensor, axis=0)
            self.sigmoid.activations = self.sigmoid_activation[time_step, None]
            sigmoid_error = self.sigmoid.backward(error_tensor[time_step, None])  # 1*5

            self.output_fc_layer.input_tensor = np.transpose(
                self.output_fc_layer_input_tensor[time_step, None])  # hidden layer at time step t, 7*1
            output_fc_layer_error = self.output_fc_layer.backward(sigmoid_error)

            # self.output_fc_layer_weights += self.output_fc_layer.weights
            self.output_fc_layer_gradient_weights += self.output_fc_layer.gradient_weights
            #  copy backpropagation
            gradiant_pass_to_tanh = gradiant_previous_hidden_state + output_fc_layer_error

            self.tan.activations = self.tan_activation[time_step, None]
            tanh_error = self.tan.backward(gradiant_pass_to_tanh)

            self.hidden_fc_layer.input_tensor = np.transpose(
                self.hidden_fc_layer_input_tensor[time_step, None])  # x_tilda
            hidden_fc_layer_error = self.hidden_fc_layer.backward(tanh_error)
            # Here the weights are defined as weights which are
            # involved in calculating the hidden state as a stacked tensor
            # self.weights += self.hidden_fc_layer.weights # doesn't need, weights are same everywhere
            self.gradient_weights += self.hidden_fc_layer.gradient_weights

            gradiant_previous_hidden_state = hidden_fc_layer_error[:, :self.hidden_size]
            gradiant_with_res_to_input = hidden_fc_layer_error[:, self.hidden_size:]
            error_[time_step] = gradiant_with_res_to_input

            # calculation the gradiant
            time_step -= 1
        if self.optimizer is not None:
            self.output_fc_layer.weights = self.optimizer.calculate_update(
                self.output_fc_layer.weights, self.output_fc_layer_gradient_weights)
            self.weights = self.optimizer.calculate_update(
                self.weights, self.gradient_weights)
        return error_

    def calculate_regularization_loss(self):
        #  Why we need it?
        if self._optimizer.regularizer:
            self.regularization_loss += self._optimizer.regularizer.norm(self.hidden_fc_layer.weights)
            self.regularization_loss += self._optimizer.regularizer.norm(self.output_fc_layer.weights)
        return self.regularization_loss

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, var):
        self._memorize = var

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, var):
        self._optimizer = var

    @property
    def weights(self):
        return self.hidden_fc_layer.weights

    @weights.setter
    def weights(self, var):
        self.hidden_fc_layer.weights = var

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, var):
        self._gradient_weights = var
