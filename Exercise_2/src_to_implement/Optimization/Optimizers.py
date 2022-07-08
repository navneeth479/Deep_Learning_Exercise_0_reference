#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class Sgd:
    def __init__(self, lr):
        self.learning_rate = lr
        
    def calculate_update(self, weight_tensor, gradient_tensor):
        return weight_tensor - self.learning_rate * gradient_tensor
    
class SgdWithMomentum:
    def __init__(self, lr, m):
        self.learning_rate = lr
        self.momentum = m
        self.prev_velocity = 0
        
    def calculate_update(self, weight_tensor, gradient_tensor):
        new_velocity = self.momentum * self.prev_velocity - self.learning_rate* gradient_tensor
        new_weights = weight_tensor + new_velocity
        self.prev_velocity = new_velocity
        return new_weights

class Adam:
    def __init__(self, lr, b1, b2):
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
        
        
#        print(new_velocity_bias, new_position_bias)

        new_weights = weight_tensor - self.learning_rate * (new_velocity_bias / (new_position_bias**0.5 + 0.000000000000001))
        self.prev_position = new_position
        self.prev_velocity = new_velocity
        self.k+=1
        return new_weights
    
