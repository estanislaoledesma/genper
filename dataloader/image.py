#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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
        plt.close("all")
        ax = plt.axes()
        sns.heatmap(self.relative_permittivities, cmap="copper", cbar_kws={"label": "Permitividades relativas"},)
        ax.set_title("Imagen " + str(image_i) + " que contiene " + str(len(self.circles)) + " círculo/s")
        plt.pause(0.01)
        plt.savefig(path)

    def set_preprocessor_guess(self, preprocessor_guess):
        self.preprocessor_guess = preprocessor_guess

    def get_preprocessor_guess(self):
        return self.preprocessor_guess

    def plot_with_preprocessor_guess(self, image_i, path):
        plt.close("all")
        figure, axis = plt.subplots(1, 2)
        figure.suptitle("Imagen {}".format(image_i))
        sns.heatmap(ax=axis[0], data=self.relative_permittivities, cmap="copper", cbar_kws={"label": "Permitividades relativas"},)
        axis[0].set_title("Imagen original")
        sns.heatmap(ax=axis[1], data=self.preprocessor_guess, cmap="copper", cbar_kws={"label": "Permitividades relativas"}, )
        axis[1].set_title("Imagen obtenida del preprocesador")
        plt.pause(0.01)
        plt.savefig(path)