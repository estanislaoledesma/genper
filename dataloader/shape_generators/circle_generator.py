#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from configs.constants import Constants
from dataloader.shapes.circle import Circle
from dataloader.shape_generators.shape_generator import ShapeGenerator


class CircleGenerator(ShapeGenerator):

    def __init__(self):
        super().__init__()
        basic_parameters = Constants.get_basic_parameters()
        images_parameters = basic_parameters["images"]
        self.min_radius = images_parameters["min_radius"]
        self.max_radius = images_parameters["max_radius"]

    def generate_shapes(self, no_of_shapes, test, image_i):
        circles = []
        for i in range(no_of_shapes):
            if test:
                radius = self.min_radius + (self.max_radius - self.min_radius) * super().TEST_RANDOM_VALUES[image_i - 1][i]
                center_x, center_y, relative_permittivity = super().get_test_shape_parameters(radius, image_i, i)
            else:
                radius = self.min_radius + (self.max_radius - self.min_radius) * np.random.uniform()
                center_x, center_y, relative_permittivity = super().get_shape_parameters(radius)
            circle = Circle(radius, center_x, center_y, relative_permittivity)
            circles.append(circle)

        return circles

    def get_shape_name(self):
        return 'circles'


