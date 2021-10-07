#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from configs.constants import Constants


class ConvolutionBlock(nn.Module):

    def __init__(self, width, height, in_channels, out_channels, stride, padding, batch_on, relu_on):
        super().__init__()
        basic_parameters = Constants.get_basic_parameters()
        unet_parameters = basic_parameters["unet"]
        self.conv = nn.Conv2d(in_channels, out_channels, (width, height),
                              stride = stride, padding = padding, bias = False)
        if unet_parameters["batch_normalization"] and batch_on:
            self.bnorm = nn.BatchNorm2d(out_channels)
        if relu_on:
            self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.bnorm:
            x = self.bnorm(x)
        if self.relu:
            x = self.relu(x)
        return x


