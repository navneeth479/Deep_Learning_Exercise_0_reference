#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 15:30:24 2022

@author: shanur
"""

from Layers import Base
import numpy as np

class Sigmoid(Base.BaseLayer):
    def __init__(self):
        super().__init__()
        self.result= None
    def forward(self,input_tensor):
        self.result = 1/ (1+np.exp(- input_tensor))
        return self.result
    def backward(self,error_tensor):
        return (self.result * (1 - self.result)) * error_tensor