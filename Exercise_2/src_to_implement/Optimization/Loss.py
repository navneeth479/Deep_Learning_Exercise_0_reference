#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        pass

    def forward(self,prediction_tensor, label_tensor):
        self.prediction_tensor=prediction_tensor
        return -np.sum(np.where(label_tensor==1,np.log(prediction_tensor+np.finfo(prediction_tensor.dtype).eps),0))

    def backward(self,label_tensor):
        return ((-1) * label_tensor/(self.prediction_tensor))