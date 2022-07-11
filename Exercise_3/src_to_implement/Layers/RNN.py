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
import numpy as np


class RNN(Base.BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_state = np.zeros(hidden_size)
        self.trainable = True
        self._memorise = False
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
#        self.otpt_fc_gradient_weights = np.zeros((self.otpt_fc.weights.shape[0] - 1, self.otpt_fc.weights.shape[1]))
        
    @property
    def memorise(self):
        return self._memorise
    @memorise.setter
    def memorise(self, value):
        self._memorise = value
    @memorise.deleter
    def memorise(self):
        del self._memorise
        
    
    def forward(self, input_tensor):
        
        
        hidden_state = self.hidden_state if not(self._memorise) else np.zeros(self.hidden_size)
        self.weights = self.hidden_fc.weights
        
        inpt_2s = []
        for i in range(input_tensor.shape[0]):
            inpt_1 = np.concatenate([hidden_state, input_tensor[i]]).reshape(1, -1)
            self.all_hidden_fc_input_tensors.append(np.concatenate([inpt_1, [[1]]], axis = 1))

            inpt_1 = self.hidden_fc.forward(inpt_1)
            inpt_1 = self.tanh.forward(inpt_1)
            
            hidden_state = inpt_1[0] 

            inpt_2 = inpt_1
            self.all_otpt_fc_input_tensors.append(np.concatenate([inpt_2, [[1]]], axis = 1))
            
            inpt_2 = self.otpt_fc.forward(inpt_2)
            inpt_2 = self.sigmoid.forward(inpt_2)

            inpt_2s.append(inpt_2[0])
      
            self.all_sigmoid_activations.append(self.sigmoid.activations)
            self.all_tanh_activations.append(self.tanh.activations)
        self.hidden_state = hidden_state
        return np.array(inpt_2s)
    
    
    def initialize(self, weights_initializer, bias_initializer):
        self.hidden_fc.initialize(weights_initializer, bias_initializer)
        self.otpt_fc.initialize(weights_initializer, bias_initializer)
    
    def backward(self, error_tensor):
        
#    
        self.gradient_weights = []
        self.otpt_fc_gradient_weights = []
        time_steps = list(range(error_tensor.shape[0]))[::-1]
        gradient_hidden = 0
        gradient_inputs = []
        
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
#            print(gradient_inpt)
            gradient_inputs.append(gradient_inpt[0])
#
            self.gradient_weights.append(self.hidden_fc.gradient_weights)
            self.otpt_fc_gradient_weights.append(self.otpt_fc.gradient_weights)
        self.gradient_weights = np.array(self.gradient_weights).sum(axis = 0)
        self.otpt_fc_gradient_weights = np.array(self.otpt_fc_gradient_weights).sum(axis = 0)


        if self._optimizer:
            self.otpt_fc.weights = self.optimizer.calculate_update(self.otpt_fc.weights, self.otpt_fc_gradient_weights)
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)


        return np.array(gradient_inputs)
    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, var):
        self._optimizer = var

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



