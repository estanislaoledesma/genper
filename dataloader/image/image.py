#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from utils.plotter import Plotter


class Image:

    def __init__(self):
        self.plotter = Plotter()

    def generate_relative_permittivities(self, x_domain, y_domain, shapes):
        self.relative_permittivities = np.ones(np.shape(y_domain))
        for shape in shapes:
            relative_permittivity = shape.get_relative_permittivity()
            pixel_belongs_to_shape = shape.check_if_pixels_belong_to_shape(x_domain, y_domain)
            self.relative_permittivities[pixel_belongs_to_shape] = relative_permittivity

    def set_relative_permittivities(self, relative_permittivities):
        self.relative_permittivities = relative_permittivities

    def get_relative_permittivities(self):
        return self.relative_permittivities

    def set_electric_field(self, electric_field):
        self.electric_field = electric_field

    def get_electric_field(self):
        return self.electric_field

    def plot(self, image_i, path):
        plot_title = "Generated image {}".format(image_i)
        self.plotter.plot_comparison(plot_title, path, self.relative_permittivities)

    def set_preprocessor_guess(self, preprocessor_guess):
        self.preprocessor_guess = preprocessor_guess

    def get_preprocessor_guess(self):
        return self.preprocessor_guess

    def plot_with_preprocessor_guess(self, image_i, path):
        plot_title = "Image {}".format(image_i)
        self.plotter.plot_comparison(plot_title, path, self.relative_permittivities, self.preprocessor_guess)

    def check_if_pixels_belong_to_rectangle(self, x_domain, y_domain, rectangle):
        center_x = rectangle.get_center_x()
        center_y = rectangle.get_center_y()
        width = rectangle.get_width()
        height = rectangle.get_height()
        min_x = center_x - width / 2
        max_x = center_x + width / 2
        min_y = center_y - height / 2
        max_y = center_y + height / 2
        return (x_domain >= min_x) & (x_domain <= max_x) & (y_domain >= min_y) & (y_domain <= max_y)