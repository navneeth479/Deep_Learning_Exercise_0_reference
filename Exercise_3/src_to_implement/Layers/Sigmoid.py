#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 15:30:24 2022

@author: shanur
"""

from Layers import Base
import numpy as np
#import Base

class Sigmoid(Base.BaseLayer):
    def __init__(self):
        super().__init__()
        self.activations= None
    def forward(self,input_tensor):
        self.activations = 1/ (1+np.exp(- input_tensor))
        return self.activations
    def backward(self,error_tensor):
        return (self.activations * (1 - self.activations)) * error_tensor