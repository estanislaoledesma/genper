#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import nn
import torchvision

from configs.constants import Constants
from utils.weight_scaler import WeightScaler


class ConvolutionTransposeBlock(nn.Module):

    def __init__(self, width, height, in_channels, out_channels, stride, cropping, batch_on, relu_on):
        super().__init__()
        basic_parameters = Constants.get_basic_parameters()
        unet_parameters = basic_parameters["unet"]
        self.width = width
        self.height = height
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_scale_init_method = unet_parameters["weight_scale_init_method"]
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, (width, height),
                                       stride=stride, bias=False, output_padding=cropping)
        if unet_parameters["batch_normalization"] and batch_on:
            self.bnorm = nn.BatchNorm2d(out_channels)
        if relu_on:
            self.relu = nn.ReLU()

    def forward(self, x):
        new_weight = WeightScaler.get_weights_scaled(self.conv.weight, self.weight_scale_init_method, self.height,
                                                     self.width, self.out_channels, self.in_channels)
        x = self.conv._conv_forward(x, new_weight)
        if self.bnorm:
            x = self.bnorm(x)
        if self.relu:
            x = self.relu(x)
        return x
