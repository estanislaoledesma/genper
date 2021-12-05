#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from executor.trainer import Trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test", help="Run in test mode", action='store_true')
    parser.add_argument("-l", "--load", help="Load latest checkpoint", action='store_true')
    args = parser.parse_args()
    test = args.test
    load = args.load

    preprocessed_images_path_prefix = "/data/preprocessor/"
    checkpoint_path_prefix = "/data/trainer/"
    training_logs_plots_path_prefix = "/logs/trainer/training_images/"
    validation_logs_plots_path_prefix = "/logs/trainer/validation_images/"
    error_logs_plots_path_prefix = "/logs/trainer/"
    plot_interval = 50
    trainer = Trainer(test, False, preprocessed_images_path_prefix, checkpoint_path_prefix)
    trainer.train(test, load, plot_interval, training_logs_plots_path_prefix, validation_logs_plots_path_prefix,
                  error_logs_plots_path_prefix)
