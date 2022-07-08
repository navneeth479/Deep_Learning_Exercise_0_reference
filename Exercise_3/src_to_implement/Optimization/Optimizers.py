#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 15:45:01 2022

@author: shanur
"""


class Optimizer:
    def __init__(self):
        self.regularizer = None
        
    def add_regularizer(self, regularizer):
        self.regularizer = regularizer
        
    
class Sgd(Optimizer):
    def __init__(self, lr):
        super().__init__()
        self.learning_rate = lr
        
    def calculate_update(self, weight_tensor, gradient_tensor):
        new_weights = weight_tensor - self.learning_rate * gradient_tensor
        try:
            return new_weights - self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)
        except AttributeError:
            return new_weights
    
class SgdWithMomentum(Optimizer):
    def __init__(self, lr, m):
        super().__init__()
        self.learning_rate = lr
        self.momentum = m
        self.prev_velocity = 0
        
    def calculate_update(self, weight_tensor, gradient_tensor):
        new_velocity = self.momentum * self.prev_velocity - self.learning_rate* gradient_tensor
        new_weights = weight_tensor + new_velocity
        self.prev_velocity = new_velocity
        try:
            return new_weights - self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)
        except AttributeError:    
            return new_weights

class Adam(Optimizer):
    def __init__(self, lr, b1, b2):
        super().__init__()
        self.learning_rate = lr
        self.k = 1
        self.mu = b1
        self.rho = b2
        self.prev_velocity = 0
        self.prev_position = 0
    
    def calculate_update(self, weight_tensor, gradient_tensor):
        new_velocity = self.mu * (self.prev_velocity) + (1 - self.mu) * gradient_tensor
        new_position = self.rho * (self.prev_position) + (1 - self.rho) * gradient_tensor * gradient_tensor
        
        
        new_velocity_bias = new_velocity / (1 - self.mu**self.k)
        new_position_bias = new_position / (1 - self.rho** self.k)
        self.prev_position = new_position
        self.prev_velocity = new_velocity
        self.k+=1
        new_weights = weight_tensor - self.learning_rate * (new_velocity_bias / (new_position_bias**0.5 + 0.000000000000001))

#        print(new_velocity_bias, new_position_bias)
        try:
            return new_weights - self.learning_rate*self.regularizer.calculate_gradient(weight_tensor)
        except AttributeError:
            return new_weights
    

