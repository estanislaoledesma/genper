#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

import numpy as np
from numpy import pi

from utils.coordinates_converter import CoordinatesConverter


class TestCoordinatesConverter(unittest.TestCase):

    def test_cart2pol_0_0(self):
        x = 0
        y = 0
        rho, phi = CoordinatesConverter.cart2pol(x, y)
        assert rho == 0
        assert phi == 0