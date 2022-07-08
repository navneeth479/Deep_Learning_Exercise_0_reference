#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 23:58:39 2022

@author: shanur
"""
from Layers import Base
#
import numpy as np
class Pooling(Base.BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()

        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
    
    def iterate_regions(self, inpt):
        h, w = inpt.shape
    
        for i in range(h // self.stride_shape[0]):
            for j in range(w // self.stride_shape[1]):
                window = inpt[(i*self.pooling_shape[0]):(i*self.pooling_shape[0]+self.stride_shape[0]), (j*self.pooling_shape[1]):(j*self.pooling_shape[1]+self.stride_shape[1])]
                
                yield window, i, j
    
    def forward(self, input_tensor):
        self.prev_input = input_tensor

        all_batches_output = []
        for one_batch in input_tensor:
            all_channels_output = []
            for one_ch in one_batch:
                h, w = one_ch.shape
                output = np.zeros((h//self.stride_shape[0], w//self.stride_shape[1]))
                for window, i, j in self.iterate_regions(one_ch):
                    if 0 not in window.shape:
                        output[i, j] = np.max(window)
                expected_height = int(np.ceil((h - self.pooling_shape[0] + 1 ) / self.stride_shape[0]))
                expected_width = int(np.ceil((w - self.pooling_shape[1] + 1 ) / self.stride_shape[1]))
                all_channels_output.append(output[:expected_height, :expected_width])
            all_batches_output.append(all_channels_output)
            
        return np.array(all_batches_output)
    
    def backward(self, error_tensor):
        
        error_output = np.zeros(self.prev_input.shape)
        
        for batch_index, one_batch in enumerate(self.prev_input):
            for channel_index, one_ch in enumerate(one_batch):
                
                for window, i, j in self.iterate_regions(one_ch):
                    if 0 not in window.shape:
                        h, w = window.shape
                        mx = np.max(window)
                        for i2 in range(h):
                            for j2 in range(w):
                                if(window[i2, j2] == mx):
                                    error_output[batch_index, channel_index, i*self.pooling_shape[0]+i2, j*self.pooling_shape[0]+j2] += error_tensor[batch_index, channel_index, i, j]
        return error_output
