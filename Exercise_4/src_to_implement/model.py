# -*- coding: utf-8 -*-



import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet(nn.Module):
    def __init__(self, in_channels=3, n_out_classes=2):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.resblock_1 = ResBlock(64, 64, 1)
        self.resblock_2 = ResBlock(64, 128, 2)
        self.resblock_3 = ResBlock(128, 256, 2)
        self.resblock_4 = ResBlock(256, 512, 2)
        self.tail_layer = ResTail(512, 2)
        
        
    def forward(self, input_tensor):
        first_output = self.layer0(input_tensor)
        res_out1 = self.resblock_1(first_output)
        res_out2 = self.resblock_2(res_out1)
        res_out3 = self.resblock_3(res_out2)
        res_out4 = self.resblock_4(res_out3)
        output_tensor = self.tail_layer(res_out4)
        return output_tensor


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.res = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU())
        self.one_by_one = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)
        self.batchnorm = nn.BatchNorm2d(out_ch)

    def forward(self, input_tensor):
        res_out = self.res(input_tensor)
        input_tensor = self.one_by_one(input_tensor)
        input_tensor = self.batchnorm(input_tensor)
        output_tensor = res_out + input_tensor
        return output_tensor
    
class ResTail(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fully = nn.Linear(in_ch, out_ch)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        averaged = self.avg(input_tensor)
        flattened = torch.flatten(averaged, 1)
        output_tensor = self.fully(flattened)
        sigmoided = self.sigmoid(output_tensor)
        return sigmoided