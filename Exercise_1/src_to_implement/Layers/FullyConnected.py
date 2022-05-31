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
        self.weights = np.random.uniform(0,1,size = (input_size, output_size))

    
    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        return np.matmul(self.input_tensor, self.weights)
    
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
        
#        print("Gradient_inputs", self.gradient_inputs.shape)
#        print("Gradient Weights", self.gradient_weights.shape)
        
        return self.gradient_inputs
    
    def calculate_update(weight_tensor, gradient_tensor):
        pass