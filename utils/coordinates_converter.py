#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


class CoordinatesConverter:

    @staticmethod
    def cart2pol(self, x, y):
        rho = np.sqrt(x ** 2 + y ** 2)
        phi = np.arctan2(y, x)
        return rho, phi

    @staticmethod
    def pol2cart(self, rho, phi):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return x, y
