#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 13:25:45 2022

@author: shanur
"""

import random
import numpy as np
from Layers import Base
class FullyConnected(Base.BaseLayer):
    
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size=input_size
        self.output_size=output_size
        #self.weights = np.random.uniform(0,1,size = (input_size, output_size))
#        print("Weights beofre Concatenation", self.weights.shape)
        self.weights = np.random.rand(self.input_size + 1, self.output_size)
        #self.bias = np.random.uniform(0, 1, size=(1, output_size))
#        self.weights = np.x`([self.weights, bias], axis = 1)

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize((self.input_size, self.output_size), self.input_size, self.output_size)
        self.bias = bias_initializer.initialize((1, self.output_size), 1, self.output_size)
        self.weights = np.vstack((self.weights, self.bias))
        
    
    def forward(self, input_tensor):
        self.input_tensor = np.concatenate((input_tensor, np.ones([input_tensor.shape[0], 1])), axis=1)
        return np.matmul(self.input_tensor, self.weights)
        #return prod + self.bias
    
    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, val):
        self._optimizer = val
        
    def backward(self, error_tensor):
#        np.matmul()
#        print("Error Tensor", error_tensor.shape)
#        print("Input Tensor", self.input_tensor.shape)
#        
        self.gradient_inputs = np.matmul(error_tensor, np.transpose(self.weights))
        self.gradient_weights = np.matmul(np.transpose(self.input_tensor), error_tensor)
#        self.gradient_weights -= np.ones(self.gradient_weights.shape)
        
#        print("Gradient_inputs", self.gradient_inputs.shape)
#        print("Error TEsnor", error_tensor.shape)
#        print("Bias SHape", self.bias.shape)
        
        try:
            opt = self.optimizer
#            print("Weights shape", self.weights.shape)
#            print("Gradient Weights", self.gradient_weights.shape)
            self.weights = opt.calculate_update(self.weights, self.gradient_weights)
            #self.bias = opt.calculate_update(self.bias, np.matmul(np.ones((1, error_tensor.shape[0])), error_tensor))

#            self.weights = opt.calculate_update(self.weights, self.gradient_weights)

#            self.weights -= np.ones(self.weights.shape)
            
        except AttributeError:
            pass
        
        return self.gradient_inputs[:,:-1]
    