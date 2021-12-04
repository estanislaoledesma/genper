#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation.tester import Tester


if __name__ == "__main__":
    trained_model_path_prefix = "/data/trainer/matlab/"
    test_images_path_prefix = "/data/preprocessor/matlab_images/"
    testing_logs_plots_path_prefix = "/logs/tester/matlab/testing_images/"
    plot_interval = 5
    tester = Tester(False, False, trained_model_path_prefix, test_images_path_prefix)
    tester.test(False, plot_interval, testing_logs_plots_path_prefix)
