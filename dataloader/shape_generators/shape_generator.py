#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

from configs.constants import Constants


class ShapeGenerator:

    TEST_RANDOM_VALUES = [[0, 0.5], [0.3, 0.9], [0, 0.8], [0.5, 0.6], [0.5, 0.9],
                          [0.8, 0], [0.5, 0.1], [1, 0.4], [0.9, 0.2], [1, 0]]

    def __init__(self):
        basic_parameters = Constants.get_basic_parameters()
        physics_parameters = basic_parameters["physics"]
        self.min_permittivity = physics_parameters["min_permittivity"]
        self.max_permittivity = physics_parameters["max_permittivity"]

    def generate_shapes(self, no_of_shapes, test, image_i):
        pass

    def get_test_shape_parameters(self, max_side, image_i, shape_i):
        center_range = [-1 + max_side + 0.05, 1 - max_side - 0.05]
        center_x = center_range[0] + (center_range[1] - center_range[0]) * self.TEST_RANDOM_VALUES[image_i - 1][shape_i]
        center_y = center_range[0] + (center_range[1] - center_range[0]) * self.TEST_RANDOM_VALUES[image_i - 1][shape_i]
        relative_permittivity = self.min_permittivity + (
                self.max_permittivity - self.min_permittivity) * self.TEST_RANDOM_VALUES[image_i - 1][shape_i]

        return center_x, center_y, relative_permittivity

    def get_shape_parameters(self, max_side):
        center_range = [-1 + max_side + 0.05, 1 - max_side - 0.05]
        center_x = center_range[0] + (center_range[1] - center_range[0]) * np.random.uniform()
        center_y = center_range[0] + (center_range[1] - center_range[0]) * np.random.uniform()
        relative_permittivity = self.min_permittivity + (
                    self.max_permittivity - self.min_permittivity) * np.random.uniform()

        return center_x, center_y, relative_permittivity

    def get_shape_name(self):
        pass
