#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from executor.trainer import Trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--load", help="Load latest checkpoint", action='store_true')
    args = parser.parse_args()
    load = args.load

    preprocessed_images_path_prefix = "/data/preprocessor/mnist_training_images/"
    checkpoint_path_prefix = "/data/trainer/mnist/"
    training_logs_plots_path_prefix = "/logs/trainer/mnist/training_images/"
    validation_logs_plots_path_prefix = "/logs/trainer/mnist/validation_images/"
    error_logs_plots_path_prefix = "/logs/trainer/mnist/"
    plot_interval = 5000
    trainer = Trainer(False, True, preprocessed_images_path_prefix, checkpoint_path_prefix)
    trainer.train(False, load, plot_interval, training_logs_plots_path_prefix, validation_logs_plots_path_prefix,
                  error_logs_plots_path_prefix)
