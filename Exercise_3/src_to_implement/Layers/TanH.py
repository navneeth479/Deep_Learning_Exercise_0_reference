#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 15:28:51 2022

@author: shanur
"""

from Layers import Base
import numpy as np

class TanH(Base.BaseLayer):
    def __init__(self):
        super().__init__()
        self. res = None
    def forward(self,input_tensor):
        self.res = np.tanh(input_tensor)
        return self.res
    def backward(self, error_tensor):
         return (1 - np.power(self.res,2))* error_tensor
