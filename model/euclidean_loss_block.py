#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch


class EuclideanLossBlock():

    @staticmethod
    def get_loss(x, target, dzdy=None):
        assert x.shape == target.shape
        target = torch.reshape(target, x.shape)
        if dzdy is None:
            dist = (x - target) ** 2
            dist = dist.transpose(2, 3)
            dist = torch.flatten(dist)
            return torch.sqrt(torch.sum(dist))
        else:
            assert torch.numel(dzdy) == 1
            return dzdy * (x - target)
