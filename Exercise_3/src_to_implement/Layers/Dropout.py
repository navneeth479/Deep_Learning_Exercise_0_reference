#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 15:32:50 2022

@author: pooja
"""
import numpy as np
from Layers import Base
class Dropout(Base.BaseLayer):
    def __init__(self, p):
        super().__init__()
        self.probability = p
        self.switches = None
        
    def forward(self, input_tensor):
        if self.testing_phase:
            return input_tensor
        self.switches = np.random.rand(*input_tensor.shape) < self.probability

        return (input_tensor * self.switches) / self.probability
    
    def backward(self, error_tensor):
        return (self.switches * error_tensor) / self.probability
