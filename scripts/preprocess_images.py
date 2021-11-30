#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataloader.preprocessor import Preprocessor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test", help="Run in test mode", action='store_true')
    args = parser.parse_args()
    test = args.test

    generated_images_path_prefix = "/data/image_generator/"
    logs_plots_path_prefix = "/logs/preprocessor/preprocessed_images/"
    preprocessed_images_path_prefix = "/data/preprocessor/"
    plot_interval = 50
    preprocessor = Preprocessor(test, generated_images_path_prefix)
    preprocessor.preprocess(test, plot_interval, logs_plots_path_prefix, preprocessed_images_path_prefix)
