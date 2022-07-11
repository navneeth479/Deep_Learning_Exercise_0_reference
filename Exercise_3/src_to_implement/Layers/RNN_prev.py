#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 15:30:55 2022

@author: shanur
"""

from Layers import Base
from Layers import FullyConnected
from Layers import TanH
from Layers import Sigmoid
#
#from FullyConnected import FullyConnected
#from TanH import TanH
#from Sigmoid import Sigmoid
import numpy as np


class RNN(Base.BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_state = np.zeros(hidden_size)
        self.trainable = True
        self._memorise = False
        self.all_layer1s = []
        self.all_layer2s = []
        
    @property
    def memorise(self):
        return self._memorise
    @memorise.setter
    def memorise(self, value):
        self._memorise = value
    @memorise.deleter
    def memorise(self):
        del self._memorise
        
    
    def forward(self, input_tensor):
        
        
        hidden_state = self.hidden_state if not(self._memorise) else np.zeros(self.hidden_size)
        inpt_2s = []
        for i in range(input_tensor.shape[0]):
            inpt_1 = np.concatenate([hidden_state, input_tensor[i]]).reshape(1, -1)
            
            FC = FullyConnected.FullyConnected(self.input_size + self.hidden_size, self.hidden_size)
            FC.initialize(self.weights_initializer, self.bias_initializer)
            tanh = TanH.TanH()
            layers_1 = [FC, tanh]
            for l in layers_1:
                inpt_1 = l.forward(inpt_1)
            
            hidden_state = inpt_1[0]
        
            FC_2 = FullyConnected.FullyConnected(self.hidden_size, self.output_size)
            FC_2.initialize(self.weights_initializer, self.bias_initializer)
            sig = Sigmoid.Sigmoid()
            layers_2 = [FC_2, sig]
            
            inpt_2 = inpt_1
            for l in layers_2:
                inpt_2 = l.forward(inpt_2)
            
            inpt_2s.append(inpt_2[0])
            
            self.all_layer1s.append(layers_1)
            self.all_layer2s.append(layers_2)

        self.hidden_state = hidden_state
        return np.array(inpt_2s)
    
    
    def initialize(self, weights_initializer, bias_initializer):
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
    
    def backward(self, error_tensor):
        
#        print(error_tensor.shape)
        
#        gradient_input = np.zeros(shape=(error_tensor.shape[0], self.input_size))
        
        self.gradient_weights = np.zeros_like(self.all_layer1s[0][0].weights)
        
        time_steps = list(range(len(self.all_layer1s)))[::-1]
        
        gradient_hidden = 0
        
        gradient_inputs = []
        
        for t in time_steps:
#            print(t)
#            print("error", error_tensor[t].shape)
            sigmoid_grad = self.all_layer2s[t][1].backward(error_tensor[t]) #sigmoid
#            print("sigm", sigmoid_grad.shape)
            sigmoid_grad *= self.all_layer2s[t][1].activation * ( 1 - self.all_layer2s[t][1].activation)
#            print("sigoid_grad", sigmoid_grad.shape)
            fc2_grad = self.all_layer2s[t][0].backward(sigmoid_grad)
            
            
            tanh_grad = self.all_layer1s[t][1].backward(fc2_grad + gradient_hidden)
            tanh_grad *= (1 - np.power(self.all_layer1s[t][1].activation, 2))
            fc1_grad = self.all_layer1s[t][0].backward(tanh_grad) #x_tild
            
            gradient_hidden = fc1_grad[:, :self.hidden_size]
            gradient_inpt = fc1_grad[:, self.hidden_size:]
            gradient_inputs.append(gradient_inpt[0])
#            break
            
            self.gradient_weights += self.all_layer1s[t][0].gradient_weights

        if self.optimizer is not None:
#            self.output_fc_layer.weights = self.optimizer.calculate_update(self.output_fc_layer.weights, self.output_fc_layer_gradient_weights)
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)


        return np.array(gradient_inputs)
            

#layer = RNN(13, 7, 5)
#
#input_vector = np.random.rand(13, 1).T
#input_tensor = np.tile(input_vector, (2, 1))
#
#output_tensor = layer.forward(input_tensor)

#self.assertNotEqual(np.sum(np.square(output_tensor[0, :] - output_tensor[1, :])), 0)
        
#    def setUp(self):
#        self.batch_size = 9
#        self.input_size = 13
#        self.output_size = 5
#        self.hidden_size = 7
#        self.input_tensor = np.random.rand(self.input_size, self.batch_size).T
#
#        self.categories = 4
#        self.label_tensor = np.zeros([self.categories, self.batch_size]).T
#        for i in range(self.batch_size):
#            self.label_tensor[i, np.random.randint(0, self.categories)] = 1

#    def test_trainable(self):
#        layer = RNN.RNN(self.input_size, self.hidden_size, self.output_size)
#        self.assertTrue(layer.trainable)
#
#    def test_forward_size(self):
#layer = RNN(13, 7, 5)
#input_tensor = np.random.rand(13, 9).T
#output_tensor = layer.forward(input_tensor)
#        self.assertEqual(output_tensor.shape[1], self.output_size)
#        self.assertEqual(output_tensor.shape[0], self.batch_size)

#    def test_forward_stateful(self):
#        layer = RNN.RNN(self.input_size, self.hidden_size, self.output_size)
#
#        input_vector = np.random.rand(self.input_size, 1).T
#        input_tensor = np.tile(input_vector, (2, 1))
#
#        output_tensor = layer.forward(input_tensor)
#
#        self.assertNotEqual(np.sum(np.square(output_tensor[0, :] - output_tensor[1, :])), 0)
        

    
    
#layer = RNN(13, 7, 5)
#input_tensor = np.random.rand(13, 9).T
#output_tensor = layer.forward(input_tensor)
#error_tensor = layer.backward(output_tensor)


#        self.assertEqual(error_tensor.shape[1], self.input_size)
#        self.assertEqual(error_tensor.shape[0], self.batch_size)
#error_tensor = layer.backward(output_tensor)

#self.assertEqual(error_tensor.shape[1], self.input_size)
#self.assertEqual(error_tensor.shape[0], self.batch_size)