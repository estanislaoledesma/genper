#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from configs.constants import Constants


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

    def get_relative_permittivities(self):
        return self.relative_permittivities

    def set_electric_field(self, electric_field):
        self.electric_field = electric_field

    def get_electric_field(self):
        return self.electric_field

    def plot(self, image_i, path):
        basic_parameters = Constants.get_basic_parameters()
        images_parameters = basic_parameters["images"]
        max_diameter = images_parameters["max_diameter"]
        x_max = np.shape(self.relative_permittivities)[0] - 1
        y_max = np.shape(self.relative_permittivities)[1] - 1
        plt.close("all")
        ax = plt.axes()
        sns.heatmap(self.relative_permittivities, cmap="rocket", cbar_kws={"label": "Permitividades relativas"},)
        ax.set_xticks(np.linspace(0, x_max, 5))
        ax.set_xticklabels(np.linspace(-max_diameter, max_diameter, 5))
        ax.set_yticks(np.linspace(y_max, 0, 5))
        ax.set_yticklabels(np.linspace(-max_diameter, max_diameter, 5))
        ax.set_title("Imagen " + str(image_i) + " que contiene " + str(len(self.circles)) + " círculo/s")
        plt.pause(0.01)
        plt.savefig(path)

    def set_preprocessor_guess(self, preprocessor_guess):
        self.preprocessor_guess = preprocessor_guess

    def get_preprocessor_guess(self):
        return self.preprocessor_guess

    def plot_with_preprocessor_guess(self, image_i, path):
        basic_parameters = Constants.get_basic_parameters()
        images_parameters = basic_parameters["images"]
        max_diameter = images_parameters["max_diameter"]
        x_max = np.shape(self.relative_permittivities)[0] - 1
        y_max = np.shape(self.relative_permittivities)[1] - 1
        plt.close("all")
        figure, axis = plt.subplots(1, 2, figsize=(15,15))
        figure.suptitle("Imagen {}".format(image_i))
        sns.heatmap(ax=axis[0], data=self.relative_permittivities, cmap="rocket", cbar_kws={"label": "Permitividades relativas"})
        axis[0].set_xticks(np.linspace(0, x_max, 5))
        axis[0].set_xticklabels(np.linspace(-max_diameter, max_diameter, 5))
        axis[0].set_yticks(np.linspace(y_max, 0, 5))
        axis[0].set_yticklabels(np.linspace(-max_diameter, max_diameter, 5))
        axis[0].set_title("Imagen original")
        sns.heatmap(ax=axis[1], data=self.preprocessor_guess, cmap="rocket", cbar_kws={"label": "Permitividades relativas"})
        axis[1].set_xticks(np.linspace(0, x_max, 5))
        axis[1].set_xticklabels(np.linspace(-max_diameter, max_diameter, 5))
        axis[1].set_yticks(np.linspace(y_max, 0, 5))
        axis[1].set_yticklabels(np.linspace(-max_diameter, max_diameter, 5))
        axis[1].set_title("Imagen obtenida del preprocesador")
        plt.pause(0.01)
        plt.savefig(path)