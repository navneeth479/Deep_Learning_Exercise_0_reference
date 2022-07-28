# -*- coding: utf-8 -*-



import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet(nn.Module):
    def __init__(self, n_in_channels=3, n_out_classes=2):
        super().__init__()
        self.encoder = ResNetGate(in_channels=n_in_channels, out_channels=64)
        self.resblock1 = ResBlock(in_ch=64, out_ch=64, stride=1)
        self.resblock2 = ResBlock(in_ch=64, out_ch=128, stride=2)
        self.resblock3 = ResBlock(in_ch=128, out_ch=256, stride=2)
        self.resblock4 = ResBlock(in_ch=256, out_ch=512, stride=2)
        self.decoder = ResnetDecoder(in_ch=512, out_ch=n_out_classes)

    def forward(self, input_tensor):
        first_output = self.encoder(input_tensor)
        res_out1 = self.resblock1(first_output)
        res_out2 = self.resblock2(res_out1)
        res_out3 = self.resblock3(res_out2)
        res_out4 = self.resblock4(res_out3)
        output_tensor = self.decoder(res_out4)
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


class ResNetGate(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=out_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def forward(self, input_tensor):
        output_tensor = self.gate(input_tensor)
        return output_tensor


class ResnetDecoder(nn.Module):
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