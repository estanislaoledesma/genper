#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation.tester import Tester


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test", help="Run in test mode", action='store_true')
    args = parser.parse_args()
    test = args.test

    trained_model_path_prefix = "/data/trainer/"
    test_images_path_prefix = "/data/preprocessor/"
    testing_logs_plots_path_prefix = "/logs/tester/testing_images/"
    plot_interval = 5
    tester = Tester(test, False, trained_model_path_prefix, test_images_path_prefix)
    tester.test(test, plot_interval, testing_logs_plots_path_prefix)
