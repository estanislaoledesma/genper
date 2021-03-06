#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import nn

from configs.constants import Constants


class ConvolutionTransposeBlock(nn.Module):

    def __init__(self, width, height, in_channels, out_channels, stride, padding, output_padding, batch_on, relu_on):
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
        self.convt = nn.ConvTranspose2d(in_channels, out_channels, (width, height),
                                       stride=stride, bias=False, padding=padding, output_padding=output_padding)
        if unet_parameters["batch_normalization"] and self.batch_on:
            self.bnorm = nn.BatchNorm2d(out_channels)
        if self.relu_on:
            self.relu = nn.ReLU()

    def forward(self, x):
        x = self.convt(x)
        if self.batch_on:
            x = self.bnorm(x)
        if self.relu_on:
            x = self.relu(x)
        return x
