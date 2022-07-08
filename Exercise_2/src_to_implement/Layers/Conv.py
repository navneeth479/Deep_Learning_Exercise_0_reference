#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 23:57:58 2022

@author: shanur
"""
import numpy as np
from scipy import signal
from Layers import Base

class Conv(Base.BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        
        self.weights = np.random.uniform(0, 1, size = (num_kernels, *convolution_shape))
        
        self.bias = np.random.rand(num_kernels)
        self._optimizer = None 
        self._bias_optimizer = None
        self.trainable = True
    def iterate_regions(self, inpt):
        h, w = inpt.shape
    
        for i in range(h // self.stride_shape[0]):
            for j in range(w // self.stride_shape[1]):
                window = inpt[(i*self.pooling_shape[0]):(i*self.pooling_shape[0]+self.stride_shape[0]), (j*self.pooling_shape[1]):(j*self.pooling_shape[1]+self.stride_shape[1])]
                
                yield window, i, j

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        
        # Becuase of same padding
        output_shape = [int(np.ceil(input_tensor.shape[2]/self.stride_shape[0])), int(np.ceil(input_tensor.shape[3]/self.stride_shape[1]))\
                        if len(self.convolution_shape) == 3 else None]
        if None in output_shape: output_shape.remove(None)
        result = np.zeros(shape = (input_tensor.shape[0], self.num_kernels, *output_shape))

        for one_batch in range(input_tensor.shape[0]):
            
            for krnls in range(self.weights.shape[0]):
                convs = []
                for ch in range(self.weights.shape[1]):
                    convs.append(signal.correlate(input_tensor[one_batch, ch], self.weights[krnls, ch], mode='same', method='direct'))

                convs = np.array(convs).sum(axis=0)


                if len(self.convolution_shape)==3:
                    convs = convs[::self.stride_shape[0], ::self.stride_shape[1]]
                elif len(self.convolution_shape)==2:
                    convs = convs[::self.stride_shape[0]]
    
                result[one_batch, krnls]= convs + self.bias[krnls]
        return result
    
    
    
    def backward(self, error_tensor):
        
        #gradient with respect to bias:
        self.gradient_bias = np.sum(error_tensor, axis = (0, 2, 3)) if len(self.convolution_shape) == 3 else np.sum(error_tensor, axis = (0, 2))

        #gradient with respect to weights
        self.gradient_weights = np.zeros(shape=(self.weights.shape))
        for one_batch in range(self.input_tensor.shape[0]):
            for krnl in range(self.input_tensor.shape[1]):
                for ch in range(self.num_kernels):
                    if len(self.convolution_shape) == 3:
                        upsampled_array  = np.zeros(self.input_tensor.shape[-2:])
                        upsampled_array[::self.stride_shape[0], :: self.stride_shape[1]] = error_tensor[one_batch, ch]
                        
                        left_padding, right_padding = (int(np.ceil((self.convolution_shape[1] - 1)/2)), int(np.floor((self.convolution_shape[1]-1)/2)))
                        up_padding, down_padding = (int(np.ceil((self.convolution_shape[2] - 1)/2)), int(np.floor((self.convolution_shape[2] - 1 )/2)))
                        
                        padded_input_tensor = np.pad(self.input_tensor[one_batch, krnl], [ (left_padding, right_padding), (up_padding, down_padding)] )
                    else:
                        upsampled_array  = np.zeros(self.input_tensor.shape[-1])
                        upsampled_array[::self.stride_shape[0]] = error_tensor[one_batch, ch]
                        left_padding, right_padding = (int(np.ceil((self.convolution_shape[1] - 1)/2)), int(np.ceil((self.convolution_shape[1] - 1)/2)))
                        padded_input_tensor = np.pad(self.input_tensor[one_batch, krnl], [ (left_padding, right_padding)] )
                    
                    self.gradient_weights[ch, krnl] += signal.correlate(padded_input_tensor, upsampled_array, mode = 'valid')



        #gradient with respect to lower layers
        input_gradient = np.zeros(shape = self.input_tensor.shape)
        weights_copy = self.weights.copy()
        weights_copy = np.transpose(weights_copy, (1,0,2,3)) if len(self.convolution_shape) == 3 else np.transpose(weights_copy, (1,0,2))
        for one_batch in range(error_tensor.shape[0]):
            for krnl in range(weights_copy.shape[0]):
                for ch in range(weights_copy.shape[1]):
                    if len(self.convolution_shape) == 3:
                        upsampled_array  = np.zeros(self.input_tensor.shape[-2:])
                        upsampled_array[::self.stride_shape[0], :: self.stride_shape[1]] = error_tensor[one_batch, ch]
                    else:
                        upsampled_array  = np.zeros(self.input_tensor.shape[-1])
                        upsampled_array[::self.stride_shape[0]] = error_tensor[one_batch, ch]
                    input_gradient[one_batch, krnl] += signal.convolve(upsampled_array, weights_copy[krnl, ch], mode = 'same', method = 'direct')
                    
        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
        if self._bias_optimizer:
            self.bias = self._bias_optimizer.calculate_update(self.bias, self._gradient_bias)
        
        return input_gradient
    
    def initialize(self, weights_initializer, bias_initializer):
        if len(self.convolution_shape) == 3:
            self.weights = weights_initializer.initialize((self.num_kernels, self.convolution_shape[0], self.convolution_shape[1], self.convolution_shape[2]),
                                                          self.convolution_shape[0]*self.convolution_shape[1]* self.convolution_shape[2],
                                                          self.num_kernels*self.convolution_shape[1]* self.convolution_shape[2])
            self.bias = bias_initializer.initialize((self.num_kernels), 1, self.num_kernels)
            self.bias = self.bias[-1]

        elif len(self.convolution_shape) == 2:
            self.weights = weights_initializer.initialize((self.num_kernels, self.convolution_shape[0], self.convolution_shape[1]),
                                                          self.convolution_shape[0]*self.convolution_shape[1],
                                                          self.num_kernels*self.convolution_shape[1])
            self.bias = bias_initializer.initialize((1, self.num_kernels), 1, self.num_kernels)
            self.bias = self.bias[-1]


        

    @property
    def gradient_weights(self):
        return self._gradient_weights
    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value
    @gradient_weights.deleter
    def gradient_weights(self):
        del self._gradient_weights


    @property
    def gradient_bias(self):
        return self._gradient_bias
    @gradient_bias.setter
    def gradient_bias(self, value):
        self._gradient_bias = value
    @gradient_bias.deleter
    def gradient_bias(self):
        del self._gradient_bias


    @property
    def optimizer(self):
        return self._optimizer
    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value
    @optimizer.deleter
    def optimizer(self):
        del self._optimizer


    @property
    def bias_optimizer(self):
        return self._bias_optimizer
    @bias_optimizer.setter
    def bias_optimizer(self, value):
        self._bias_optimizer = value
    @bias_optimizer.deleter
    def bias_optimizer(self):
        del self._bias_optimizer

    
#
