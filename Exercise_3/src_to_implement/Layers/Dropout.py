#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 15:32:50 2022

@author: pooja
"""
<<<<<<< HEAD
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
    
    
#input_tensor = np.ones((10000, 10))
#drop_layer = Dropout(0.5)
#drop_layer.forward(input_tensor)
#output = drop_layer.backward(input_tensor)
#self.assertEqual(np.max(output), 2)
#self.assertEqual(np.min(output), 0)
=======
from Layers import Base
import numpy as np

class Dropout(Base.BaseLayer):
    def __init__(self,probability):
        super().__init__()
        self.probability=probability

    def forward(self,input_tensor):
        self.input_tensor=input_tensor
        if self.testing_phase == True:
            return self.input_tensor

        self.mask = (np.random.rand(self.input_tensor.shape[0],self.input_tensor.shape[1]) < self.probability)/self.probability
        out=self.input_tensor*self.mask
        return out

    def backward(self,error_tensor):
        return (error_tensor * self.mask)
>>>>>>> c4729ecc3c76785aea9d2f1cf1edf6bc704ebc80
