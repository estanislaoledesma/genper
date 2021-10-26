#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


class EuclideanLossBlock():

    @staticmethod
    def get_loss(x, target, dzdy=None):
        assert np.size(x) == np.size(target)
        target = np.reshape(target, x.shape)
        if dzdy is None:
            dist = (x - target) ** 2
            dist = np.atleast_2d(dist.flatten("F")).T
            return np.sqrt(np.sum(dist))
        else:
            assert np.size(dzdy) == 1
            return dzdy * (x - target)
