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
        self.y=np.divide(np.exp(input_tensor),np.sum(np.exp(input_tensor)))
        return self.y

    def backward(self,error_tensor):
        #return (error_tensor - np.sum(error_tensor*self.y))
        return error_tensor*(1-error_tensor)