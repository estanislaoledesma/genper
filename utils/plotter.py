#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from configs.constants import Constants


class Plotter:

    ORIGINAL_TITLE = "Original Image"
    PREPROCESSOR_TITLE = "Image from preprocessor"
    PREDICTION_TITLE = "Image from unet\n" \
                       "(Error: {:.2E})"

    def __init__(self):
        basic_parameters = Constants.get_basic_parameters()
        images_parameters = basic_parameters["images"]
        self.max_diameter = images_parameters["max_diameter"]
        self.x_max = images_parameters["no_of_pixels"] - 1
        self.y_max = self.x_max

    def plot_comparison(self, plot_title, path, display, labels, images = None, prediction = None, loss = 0):
        plt.close("all")
        no_of_plots = 1
        if images is not None:
            no_of_plots = 2
        if prediction is not None:
            no_of_plots = 3
        figure, axis = plt.subplots(1, no_of_plots, figsize=(15, 15))
        figure.suptitle(plot_title)

        titles = [self.ORIGINAL_TITLE, self.PREPROCESSOR_TITLE, self.PREDICTION_TITLE.format(loss)]
        data = [labels, images, prediction]

        for plot in range(no_of_plots):
            title = titles[plot]
            data_to_plot = data[plot]
            if no_of_plots > 1:
                self.plot(axis[plot], title, data_to_plot, plot == 0)
            else:
                self.plot(axis, title, data_to_plot, True)
        if display:
            plt.pause(0.01)
        plt.savefig(path)
        plt.close(figure)

    def plot(self, axis, title, data, add_y_axis):
        sns.heatmap(ax=axis, data=data, cmap="rocket",
                    yticklabels=add_y_axis, cbar_kws={"label": "Permitividades relativas",
                                                      "orientation": "horizontal"})
        axis.set_aspect('equal', adjustable='box')
        axis.set_xticks(np.linspace(0, self.x_max, 5))
        axis.set_xticklabels(np.linspace(-self.max_diameter, self.max_diameter, 5))
        axis.set_xlabel("x (m)")
        if add_y_axis:
            axis.set_yticks(np.linspace(self.y_max, 0, 5))
            axis.set_yticklabels(np.linspace(-self.max_diameter, self.max_diameter, 5))
            axis.set_ylabel("y (m)")
        axis.set_title(title)

    def plot_errors(self, training_errors, validation_errors, path, display):
        figure= plt.figure(figsize=(15, 15))
        plt.plot(*zip(*sorted(training_errors.items())), label="Training error")
        plt.plot(*zip(*sorted(validation_errors.items())), label="Validation error")
        plt.title("Model Loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        if display:
            plt.pause(0.01)
        plt.savefig(path)
        plt.close(figure)
