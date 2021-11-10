#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from configs.constants import Constants
from utils.plotter import Plotter


class Image:

    def __init__(self, x_domain, y_domain, circles):
        self.circles = circles
        self.relative_permittivities = np.ones(np.shape(y_domain))
        for circle in circles:
            center_x = circle.get_center_x()
            center_y = circle.get_center_y()
            radius = circle.get_radius()
            relative_permittivity = circle.get_relative_permittivity()
            dist_to_center = np.sqrt(np.power(x_domain - center_x, 2) + np.power(y_domain - center_y, 2))
            pixel_belongs_to_circle = dist_to_center < radius
            self.relative_permittivities[pixel_belongs_to_circle] = relative_permittivity
        self.plotter = Plotter()

    def get_relative_permittivities(self):
        return self.relative_permittivities

    def set_electric_field(self, electric_field):
        self.electric_field = electric_field

    def get_electric_field(self):
        return self.electric_field

    def plot(self, image_i, path):
        plot_title = "Imagen generada {}".format(image_i)
        self.plotter.plot_comparison(plot_title, path, self.relative_permittivities)

    def set_preprocessor_guess(self, preprocessor_guess):
        self.preprocessor_guess = preprocessor_guess

    def get_preprocessor_guess(self):
        return self.preprocessor_guess

    def plot_with_preprocessor_guess(self, image_i, path):
        plot_title = "Imagen {}".format(image_i)
        self.plotter.plot_comparison(plot_title, path, self.relative_permittivities, self.preprocessor_guess)