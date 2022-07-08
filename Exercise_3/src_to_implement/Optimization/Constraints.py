#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 15:34:53 2022

@author: shanur
"""
import numpy as np

class L2_Regularizer:
    def __init__(self, a):
        self.alpha =a
        
    def norm(self, weights):
        return self.alpha * np.sum(weights**2)
#        return self.alpha*np.sqrt(np.sum((weights ** 2)))
        
#        return self.alpha * np.linalg.norm(weights)
    
    def calculate_gradient(self, weights):
        return self.alpha * weights
    

    

class L1_Regularizer:
    def __init__(self, a):
        self.alpha =a
    
    def norm(self, weights):
        return self.alpha * np.sum(np.abs(weights))
    
    def calculate_gradient(self, weights):
        return self.alpha * np.sign(weights)
        
    
#regularizer = L2_Regularizer(1337)
#
#weights_tensor = np.ones((4, 5))
#weights_tensor[1:3, 2:4] += 1
#norm = regularizer.norm(weights_tensor)
#self.assertAlmostEqual(norm, 32 * 1337)
        
    
#regularizer = L1_Regularizer(1337)
#
#weights_tensor = np.ones((4, 5))
#weights_tensor[1:3, 2:4] *= -2
#norm = regularizer.norm(weights_tensor)
#self.assertAlmostEqual(norm, 32 * 1337)