#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 13:25:45 2022

@author: shanur
"""
from Layers import Base
import numpy as np

class ReLU(Base.BaseLayer):
    def __init__(self):
        super().__init__()
    
    def forward(self,input_tensor):
        self.input_tensor=input_tensor
        return np.maximum(0,input_tensor)

    def backward(self,error_tensor):
        return np.where(self.input_tensor<=0,0,error_tensor)