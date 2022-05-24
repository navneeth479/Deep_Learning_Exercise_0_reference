#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class Sgd:
    def __init__(self, lr):
        self.learning_rate = lr
        
    def calculate_update(self, weight_tensor, gradient_tensor):
        return weight_tensor - self.learning_rate * gradient_tensor
    
class Loss:
    pass