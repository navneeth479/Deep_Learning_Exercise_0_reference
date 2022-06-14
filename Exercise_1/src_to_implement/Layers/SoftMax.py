#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 13:25:45 2022

@author: shanur
"""
from Layers import Base
import numpy as np

class SoftMax(Base.BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self,input_tensor):
        self.input_tensor=input_tensor
        
        input_tensor=input_tensor-np.max(input_tensor)

        exponent_sums = np.sum(np.exp(input_tensor), axis = 1).reshape((input_tensor.shape[0], 1))

        self.y_hat = np.exp(input_tensor) / exponent_sums
        return self.y_hat

    def backward(self,error_tensor):

        diff =  error_tensor - np.sum((error_tensor*self.y_hat), axis = 1).reshape((error_tensor.shape[0], 1))        
        
        return self.y_hat * diff