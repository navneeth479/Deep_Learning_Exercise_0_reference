#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 15:30:55 2022

@author: shanur
"""

from Layers import Base
from Layers import FullyConnected
from Layers import TanH
from Layers import Sigmoid
#
#from FullyConnected import FullyConnected
#from TanH import TanH
#from Sigmoid import Sigmoid

#import Base
#import FullyConnected
#import TanH
#import Sigmoid
import copy
import numpy as np


class RNN(Base.BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_state = np.zeros(hidden_size)
        self.trainable = True
        self._mem = False
        self._optimizer = None
        self._gradient_weights = None
        
        
        self.hidden_fc= FullyConnected.FullyConnected(self.input_size + self.hidden_size, self.hidden_size)
        self.otpt_fc  = FullyConnected.FullyConnected(self.hidden_size, self.output_size)
        self.tanh = TanH.TanH()
        self.sigmoid = Sigmoid.Sigmoid()
        self.all_sigmoid_activations = []
        self.all_tanh_activations = []
        self.all_otpt_fc_gradient_weights = []
        self.all_hidden_fc_gradient_weights = []
        self.all_otpt_fc_input_tensors = []
        self.all_hidden_fc_input_tensors = []
        
        
    @property
    def mem(self):
        return self._mem

    @mem.setter
    def memorize(self, var):
        self._mem = var
        
    
    def forward(self, input_tensor):
        self.all_sigmoid_activations = []
        self.all_tanh_activations = []
        self.all_otpt_fc_input_tensors = []
        self.all_hidden_fc_input_tensors = []
        hidden_state = self.hidden_state if self._mem else np.zeros(self.hidden_size)
#        print("Hidden", np.sum(hidden_state))
#        if not self.memorise:
#            hidden_state = np.zeros(self.hidden_size)  # initialize hidden_state with zero
#        else:
#            hidden_state = self.hidden_state
#        self.weights = self.hidden_fc.weights
        
        inpt_2s = []
        
        for i in range(input_tensor.shape[0]):
            inpt_1 = np.concatenate([hidden_state, input_tensor[i]]).reshape(1, -1)

            inpt_1 = self.hidden_fc.forward(inpt_1)
            inpt_1 = self.tanh.forward(inpt_1)
            
#            print("inpt_1", np.sum(inpt_1[0]))
            hidden_state = copy.deepcopy(inpt_1[0])

            inpt_2 = copy.deepcopy(inpt_1)
            
            inpt_2 = self.otpt_fc.forward(inpt_2)
            inpt_2 = self.sigmoid.forward(inpt_2)

            inpt_2s.append(inpt_2[0])
      
        
            self.all_hidden_fc_input_tensors.append(self.hidden_fc.input_tensor)
            self.all_otpt_fc_input_tensors.append(self.otpt_fc.input_tensor)
            self.all_sigmoid_activations.append(self.sigmoid.activations)
            self.all_tanh_activations.append(self.tanh.activations)
        
#        self.hidden_state = hidden_state

    
        self.hidden_state = hidden_state

        
        return np.array(inpt_2s)
    
    
    def initialize(self, weights_initializer, bias_initializer):
        self.hidden_fc.initialize(weights_initializer, bias_initializer)
        self.otpt_fc.initialize(weights_initializer, bias_initializer)
    
    def backward(self, error_tensor):
        
#    
        gradient_weights = []
        otpt_fc_gradient_weights = []
        time_steps = list(range(error_tensor.shape[0]))[::-1]
        gradient_hidden = 0
        gradient_inputs = []
#        print(time_steps)
        for t in time_steps:
            self.sigmoid.activations = self.all_sigmoid_activations[t]
            sigmoid_grad = self.sigmoid.backward(error_tensor[t])
            
            self.otpt_fc.input_tensor = self.all_otpt_fc_input_tensors[t]
            otpt_fc_grad = self.otpt_fc.backward(sigmoid_grad)


            self.tanh.activations = self.all_tanh_activations[t]
            tanh_grad = self.tanh.backward(otpt_fc_grad + gradient_hidden)
            
            self.hidden_fc.input_tensor = self.all_hidden_fc_input_tensors[t]
            hidden_fc_grad = self.hidden_fc.backward(tanh_grad) #x_tild
            
            gradient_hidden = hidden_fc_grad[:, :self.hidden_size]
            gradient_inpt = hidden_fc_grad[:, self.hidden_size:]
            gradient_inputs.append(gradient_inpt[0])


            gradient_weights.append(self.hidden_fc.gradient_weights)
            otpt_fc_gradient_weights.append(self.otpt_fc.gradient_weights)
            
        self.gradient_weights = np.array(gradient_weights).sum(axis = 0)
        self.otpt_fc_gradient_weights = np.array(otpt_fc_gradient_weights).sum(axis = 0)


        if self._optimizer:
            self.otpt_fc.weights = self._optimizer.calculate_update(self.otpt_fc.weights, self.otpt_fc_gradient_weights)
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weights)


        return np.array(gradient_inputs)[::-1]
    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, var):
        self._optimizer = var
#        self._optimizer_otpt_fc = copy.deepcopy(var)
#        self._optimizer_hidden_fc = copy.deepcopy(var)

    @property
    def weights(self):
        return self.hidden_fc.weights

    @weights.setter
    def weights(self, var):
        self.hidden_fc.weights = var

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, var):
        self._gradient_weights = var

    def calculate_regularization_loss(self):
        if self._optimizer.regularizer:
            self.regularization_loss += self._optimizer.regularizer.norm(self.hidden_fc_layer.weights)
            self.regularization_loss += self._optimizer.regularizer.norm(self.output_fc_layer.weights)
        return self.regularization_loss


#self.batch_size = 9
#self.input_size = 13
#self.output_size = 5
#self.hidden_size = 7
#input_tensor = np.random.rand(13, 9).T
##
##
##layer = RNN(13, 7, 5)
##output_tensor = layer.forward(input_tensor)
##error_tensor = layer.backward(output_tensor)
##
##
#layer = RNN(13, 7, 5)
#layer.memorize = True
#output_tensor_first = layer.forward(input_tensor)
#output_tensor_second = layer.forward(input_tensor)
#
#print("diff", np.sum(np.square(output_tensor_first - output_tensor_second)))


