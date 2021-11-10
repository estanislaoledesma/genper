#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from configs.constants import Constants


class Plotter:

    ORIGINAL_TITLE = "Imagen original"
    PREPROCESSOR_TITLE = "Imagen obtenida por el preprocesador"
    PREDICTION_TITLE = "Imagen obtenida por la red neuronal"

    def __init__(self):
        basic_parameters = Constants.get_basic_parameters()
        images_parameters = basic_parameters["images"]
        self.max_diameter = images_parameters["max_diameter"]
        self.x_max = images_parameters["no_of_pixels"] - 1
        self.y_max = self.x_max

    def plot_comparison(self, plot_title, path, labels, images = None, prediction = None):
        plt.close("all")
        no_of_plots = 1
        if images is not None:
            no_of_plots = 2
        if prediction is not None:
            no_of_plots = 3
        figure, axis = plt.subplots(1, no_of_plots, figsize=(15, 15))
        figure.suptitle(plot_title)

        titles = [self.ORIGINAL_TITLE, self.PREPROCESSOR_TITLE, self.PREDICTION_TITLE]
        data = [labels, images, prediction]

        for plot in range(no_of_plots):
            title = titles[plot]
            data_to_plot = data[plot]
            if no_of_plots > 1:
                self.plot(axis[plot], title, data_to_plot)
            else:
                self.plot(axis, title, data_to_plot)
        plt.pause(0.01)
        plt.savefig(path)

    def plot(self, axis, title, data):
        sns.heatmap(ax=axis, data=data, cmap="rocket",
                    cbar_kws={"label": "Permitividades relativas"})
        axis.set_xticks(np.linspace(0, self.x_max, 5))
        axis.set_xticklabels(np.linspace(-self.max_diameter, self.max_diameter, 5))
        axis.set_yticks(np.linspace(self.y_max, 0, 5))
        axis.set_yticklabels(np.linspace(-self.max_diameter, self.max_diameter, 5))
        axis.set_title(title)
