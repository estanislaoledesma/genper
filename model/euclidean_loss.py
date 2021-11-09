#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from torch import nn


class EuclideanLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, target):
        assert x.shape == target.shape
        target = torch.reshape(target, x.shape)
        dist = (x - target) ** 2
        dist = dist.transpose(2, 3)
        dist = torch.flatten(dist)
        return torch.sqrt(torch.sum(dist))

    def backward(self, x, target, dzdy):
        assert x.shape == target.shape
        target = torch.reshape(target, x.shape)
        assert torch.numel(dzdy) == 1
        return dzdy * (x - target)