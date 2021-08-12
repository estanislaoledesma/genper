#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


class Circle:

    def __init__(self, no_of_circles, radius, center_x, center_y, relative_permittivity):
        self.no_of_circles = no_of_circles
        self.radius = radius
        self.center_x = center_x
        self.center_y = center_y
        self.relative_permittivity = relative_permittivity

    def get_relative_permittivity(self):
        return self.relative_permittivity

    def get_center_x(self):
        return self.center_x

    def get_center_y(self):
        return self.center_y

    def get_radius(self):
        return self.radius