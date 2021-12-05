#!/usr/bin/env python
# -*- coding: utf-8 -*-

class Shape:

    def __init__(self, center_x, center_y, relative_permittivity):
        self.center_x = center_x
        self.center_y = center_y
        self.relative_permittivity = relative_permittivity

    def get_relative_permittivity(self):
        return self.relative_permittivity

    def get_center_x(self):
        return self.center_x

    def get_center_y(self):
        return self.center_y

    def check_if_pixels_belong_to_shape(self, x_domain, y_domain):
        pass
