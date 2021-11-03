#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from torch import nn

from configs.constants import Constants
from model.convolution_block import ConvolutionBlock
from model.convolution_transpose_block import ConvolutionTransposeBlock
from model.max_pooling_block import MaxPoolingBlock
from model.reg_toss_block import RegTossBlock
from model.reg_concat_block import RegConcatBlock
from model.registers import Registers


class UNet(nn.Module):

    def __init__(self):
        super().__init__()
        basic_parameters = Constants.get_basic_parameters()
        unet_parameters = basic_parameters["unet"]
        kernel_size = unet_parameters["kernel_size"]
        in_channels = unet_parameters["in_channels"]
        out_channels = unet_parameters["out_channels"]
        stride = unet_parameters["stride"]
        batch_on = unet_parameters["batch_on"]
        relu_on = unet_parameters["relu_on"]
        reg_num = unet_parameters["reg_num"]
        padding = int(kernel_size/2)
        self.registers = Registers(reg_num)
        self.conv1 = ConvolutionBlock(kernel_size, kernel_size, in_channels, 64, stride, padding, batch_on, relu_on)
        self.conv2 = ConvolutionBlock(kernel_size, kernel_size, 64, 64, stride, padding, batch_on, relu_on)
        self.conv3 = ConvolutionBlock(kernel_size, kernel_size, 64, 64, stride, padding, batch_on, relu_on)
        self.reg_toss1 = RegTossBlock(self.registers, 0)
        self.pool1 = MaxPoolingBlock(2, 2, 0)
        self.conv4 = ConvolutionBlock(kernel_size, kernel_size, 64, 128, stride, padding, batch_on, relu_on)
        self.conv5 = ConvolutionBlock(kernel_size, kernel_size, 128, 128, stride, padding, batch_on, relu_on)
        self.reg_toss2 = RegTossBlock(self.registers, 1)
        self.pool2 = MaxPoolingBlock(2, 2, 0)
        self.conv6 = ConvolutionBlock(kernel_size, kernel_size, 128, 256, stride, padding, batch_on, relu_on)
        self.conv7 = ConvolutionBlock(kernel_size, kernel_size, 256, 256, stride, padding, batch_on, relu_on)
        self.convt1 = ConvolutionTransposeBlock(kernel_size, kernel_size, 256, 128, 2, 1, 1, batch_on, relu_on)
        self.reg_concat1 = RegConcatBlock(self.registers, 1)
        self.conv8 = ConvolutionBlock(kernel_size, kernel_size, 256, 128, stride, padding, batch_on, relu_on)
        self.conv9 = ConvolutionBlock(kernel_size, kernel_size, 128, 128, stride, padding, batch_on, relu_on)
        self.convt2 = ConvolutionTransposeBlock(kernel_size, kernel_size, 128, 64, 2, 1, 1, batch_on, relu_on)
        self.reg_concat2 = RegConcatBlock(self.registers, 0)
        self.conv10 = ConvolutionBlock(kernel_size, kernel_size, 128, 64, stride, padding, batch_on, relu_on)
        self.conv11 = ConvolutionBlock(kernel_size, kernel_size, 64, 64, stride, padding, batch_on, relu_on)
        self.conv12 = ConvolutionBlock(1, 1, 64, out_channels, stride, 0, False, False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.reg_toss1(x)
        x = self.pool1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.reg_toss2(x)
        x = self.pool2(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.convt1(x)
        x = self.reg_concat1(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.convt2(x)
        x = self.reg_concat2(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        return x