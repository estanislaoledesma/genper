#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import nn


class MaxPoolingBlock(nn.Module):

    def __init__(self, pooling_size, stride, padding):
        super().__init__()
        self.pool = nn.MaxPool2d(pooling_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.pool(x)


