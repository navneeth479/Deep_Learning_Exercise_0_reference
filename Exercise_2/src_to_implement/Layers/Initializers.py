#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 23:58:54 2022

@author: pooja
"""

import numpy as np

class Constant:
    def __init__(self, weight_initialization=0.1):
        self.weight_initialization = weight_initialization

    def initialize(self, weights_shape, fan_in, fan_out):
        return (np.zeros((fan_in, fan_out)) + self.weight_initialization)



class UniformRandom():
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        return np.random.rand(fan_in, fan_out)

class Xavier():
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        return np.random.normal(0, (2 / (fan_out + fan_in))**(1/2), weights_shape)



class He():
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        return np.random.normal(0, (2 / fan_in)**(1/2), weights_shape)

