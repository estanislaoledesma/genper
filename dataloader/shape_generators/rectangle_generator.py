#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

from configs.constants import Constants
from dataloader.shapes.rectangle import Rectangle
from dataloader.shape_generators.shape_generator import ShapeGenerator


class RectangleGenerator(ShapeGenerator):

    def __init__(self):
        super().__init__()
        basic_parameters = Constants.get_basic_parameters()
        images_parameters = basic_parameters["images"]
        self.min_side = images_parameters["min_side"]
        self.max_side = images_parameters["max_side"]

    def generate_shapes(self, no_of_shapes, test, image_i):
        rectangles = []
        for i in range(no_of_shapes):
            if test:
                width = self.min_side + (self.max_side - self.min_side) * super().TEST_RANDOM_VALUES[image_i - 1][i]
                height = self.min_side + (self.max_side - self.min_side) * super().TEST_RANDOM_VALUES[image_i - 1][i - 1]
                max_side = max(height, width)
                center_x, center_y, relative_permittivity = super().get_test_shape_parameters(max_side, image_i, i)
            else:
                width = self.min_side + (self.max_side - self.min_side) * np.random.uniform()
                height = self.min_side + (self.max_side - self.min_side) * np.random.uniform()
                max_side = max(height, width)
                center_x, center_y, relative_permittivity = super().get_shape_parameters(max_side)
            rectangle = Rectangle(width, height, center_x, center_y, relative_permittivity)
            rectangles.append(rectangle)

        return rectangles

    def get_shape_name(self):
        return 'rectangles'
