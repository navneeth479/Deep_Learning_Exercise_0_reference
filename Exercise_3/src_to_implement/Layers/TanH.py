#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 15:28:51 2022

@author: shanur
"""

from Layers import Base
import numpy as np
#import Base

class TanH(Base.BaseLayer):
    def __init__(self):
        super().__init__()
        self.activations = None
        
    def forward(self,input_tensor):
        self.activations = np.tanh(input_tensor)
        return self.activations
    
    def backward(self, error_tensor):
         return (1 - np.power(self.activations,2))* error_tensor
