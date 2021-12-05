#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

from dataloader.shapes.shape import Shape


class Circle(Shape):

    def __init__(self, radius, center_x, center_y, relative_permittivity):
        super().__init__(center_x, center_y, relative_permittivity)
        self.radius = radius

    def get_radius(self):
        return self.radius

    def check_if_pixels_belong_to_shape(self, x_domain, y_domain):
        center_x = self.get_center_x()
        center_y = self.get_center_y()
        radius = self.get_radius()
        dist_to_center = np.sqrt(np.power(x_domain - center_x, 2) + np.power(y_domain - center_y, 2))
        return dist_to_center < radius
