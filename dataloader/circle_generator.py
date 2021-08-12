#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from configs.constants import Constants
from dataloader.circle import Circle


class CircleGenerator:

    def __init__(self):
        basic_parameters = Constants.get_basic_parameters()
        physics_parameters = basic_parameters["physics"]
        images_parameters = basic_parameters["images"]
        self.min_permittivity = physics_parameters["min_permittivity"]
        self.max_permittivity = physics_parameters["max_permittivity"]
        self.min_radius = images_parameters["min_radius"]
        self.max_radius = images_parameters["max_radius"]

    def generate_circles(self, no_of_circles):
        circles = []

        for i in range(no_of_circles):
            radius = self.min_radius + (self.max_radius - self.max_radius) * np.random.uniform()
            center_range = [-1 + radius + 0.05, 1 - radius - 0.05]
            center_x = center_range[0] + (center_range[1] - center_range[0]) * np.random.uniform()
            center_y = center_range[0] + (center_range[1] - center_range[0]) * np.random.uniform()
            relative_permittivity = self.min_permittivity + (self.max_permittivity - self.min_permittivity) * np.random.uniform()
            circle = Circle(no_of_circles, radius, center_x, center_y, relative_permittivity)
            circles.append(circle)

        return circles


