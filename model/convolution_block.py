#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from torch import nn

from configs.constants import Constants
from utils.weight_scaler import WeightScaler


class ConvolutionBlock(nn.Module):

    def __init__(self, width, height, in_channels, out_channels, stride, padding, batch_on, relu_on):
        super().__init__()
        basic_parameters = Constants.get_basic_parameters()
        unet_parameters = basic_parameters["unet"]
        self.width = width
        self.height = height
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.relu_on = relu_on
        self.batch_on = batch_on
        self.weight_scale_init_method = unet_parameters["weight_scale_init_method"]
        self.conv = nn.Conv2d(in_channels, out_channels, (height, width),
                              stride=stride, padding=padding, bias=False)
        if unet_parameters["batch_normalization"] and self.batch_on:
            self.bnorm = nn.BatchNorm2d(out_channels)
        if self.relu_on:
            self.relu = nn.ReLU()

    def forward(self, x):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        new_weight = WeightScaler.get_weights_scaled(self.conv.weight, self.weight_scale_init_method, self.height,
                                                     self.width, self.in_channels, self.out_channels).to(device=device, dtype=torch.float32)
        x = self.conv._conv_forward(x, new_weight, self.conv.bias)
        if self.batch_on:
            x = self.bnorm(x)
        if self.relu_on:
            x = self.relu(x)
        return x
