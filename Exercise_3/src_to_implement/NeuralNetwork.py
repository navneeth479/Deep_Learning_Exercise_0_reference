#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 22:58:35 2022

@author: shanur
"""

import copy
import numpy as np
from Layers import SoftMax, Base

class NeuralNetwork(Base.BaseLayer):
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.weights_initializer=weights_initializer
        self.bias_initializer=bias_initializer
    
    
    @property
    def phase(self):
        return self.testing_phase
    
    @phase.setter
    def phase(self, val):
        self.testing_phase = val
    
    def forward(self):

        self.input_tensor, self.label_tensor = self.data_layer.next()   
        reg_loss = 0
        for layer in self.layers:
            self.input_tensor = layer.forward(self.input_tensor)
            try:
                reg_loss += self.optimizer.regularizer.norm(layer.weights)  
            except AttributeError:
                pass
        return self.loss_layer.forward(self.input_tensor+reg_loss, self.label_tensor)
    
    def backward(self):
        layer_backward_otpt = self.loss_layer.backward(self.label_tensor)
        
        for layer in self.layers[::-1]:
            layer_backward_otpt = layer.backward(layer_backward_otpt)

    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
            layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)
    
    def train(self, iterations):
        
        for _ in range(iterations):
            loss = self.forward()
            self.loss.append(loss)
            
            self.backward()

    
    def test(self, input_tensor):
        softmax = SoftMax.SoftMax()
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        
        return softmax.forward(input_tensor)
        