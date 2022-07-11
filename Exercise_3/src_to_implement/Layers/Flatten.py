#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 23:59:50 2022

@author: shanur
"""
from Layers import Base

class Flatten(Base.BaseLayer):
    def __init__(self):
        super().__init__()
    
    def forward(self, input_tensor):
        self.shape = input_tensor.shape
        if len(input_tensor.shape) == 2:
            batch_size, _ = self.shape
        elif len(input_tensor.shape) == 4:
            batch_size, _, _, _ = self.shape
        return input_tensor.reshape(batch_size, -1)
    
    def backward(self, error_tensor):
        return error_tensor.reshape(self.shape)
    