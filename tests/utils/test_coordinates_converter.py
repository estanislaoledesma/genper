#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

import torch
from numpy import pi
from utils.coordinates_converter import CoordinatesConverter


class TestCoordinatesConverter(unittest.TestCase):

    def test_cart2pol_0_0(self):
        x = 0
        y = 0
        rho, phi = CoordinatesConverter.cart2pol(x, y)
        assert rho == 0
        assert phi == 0

    def test_cart2pol_4_0(self):
        x = 4
        y = 0
        rho, phi = CoordinatesConverter.cart2pol(x, y)
        assert rho == 4
        assert phi == 0

    def test_cart2pol_0_4(self):
        x = 0
        y = 4
        rho, phi = CoordinatesConverter.cart2pol(x, y)
        assert rho == 4
        assert phi == pi / 2

    def test_cart2pol_minus_4_0(self):
        x = -4
        y = 0
        rho, phi = CoordinatesConverter.cart2pol(x, y)
        assert rho == 4
        assert phi == pi

    def test_cart2pol_0_minus_4(self):
        x = 0
        y = -4
        rho, phi = CoordinatesConverter.cart2pol(x, y)
        assert rho == 4
        assert phi == - pi / 2


    def test_pol2cart_0_0(self):
        x = 0
        y = 0
        rho, phi = CoordinatesConverter.pol2cart(x, y)
        assert rho == 0
        assert phi == 0

    def test_pol2cart_4_0(self):
        rho = 4
        phi = 0
        x, y = CoordinatesConverter.pol2cart(rho, phi)
        assert x == 4
        assert y == 0

    def test_pol2cart_0_4(self):
        rho = 4
        phi = pi / 2
        x, y = CoordinatesConverter.pol2cart(rho, phi)
        assert abs(x - 0) < 1e-10
        assert y == 4

    def test_pol2cart_minus_4_0(self):
        rho = 4
        phi = pi
        x, y = CoordinatesConverter.pol2cart(rho, phi)
        assert x == -4
        assert abs(y - 0) < 1e-10

    def test_pol2cart_0_minus_4(self):
        rho = 4
        phi = - pi / 2
        x, y = CoordinatesConverter.pol2cart(rho, phi)
        assert abs(x - 0) < 1e-10
        assert y == -4
