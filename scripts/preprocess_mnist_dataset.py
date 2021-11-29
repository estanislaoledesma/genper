#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataloader.preprocessor import Preprocessor

if __name__ == "__main__":
    generated_training_images_path_prefix = "/data/mnist_dataset_generator/mnist_training_images/"
    training_logs_plots_path_prefix = "/logs/preprocessor/mnist_preprocessed_images/mnist_training_images/"
    preprocessed_training_images_path_prefix = "/data/preprocessor/mnist_training_images/"
    training_plot_interval = 500
    preprocessor = Preprocessor(False, generated_training_images_path_prefix)
    preprocessor.preprocess(False, training_plot_interval, training_logs_plots_path_prefix,
                            preprocessed_training_images_path_prefix)

    generated_testing_images_path_prefix = "/data/mnist_dataset_generator/mnist_testing_images/"
    testing_logs_plots_path_prefix = "/logs/preprocessor/mnist_preprocessed_images/mnist_testing_images/"
    preprocessed_testing_images_path_prefix = "/data/preprocessor/mnist_testing_images/"
    testing_plot_interval = 1000
    preprocessor = Preprocessor(False, generated_testing_images_path_prefix)
    preprocessor.preprocess(False, testing_plot_interval, testing_logs_plots_path_prefix,
                            preprocessed_testing_images_path_prefix)
