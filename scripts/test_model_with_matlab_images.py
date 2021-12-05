#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation.tester import Tester
from evaluation.matlab_validator import MatlabValidator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="Matlab (.mat) file to load")
    args = parser.parse_args()
    matlab_file_name = args.file

    trained_model_path_prefix = "/data/trainer/matlab/"
    test_images_path_prefix = "/data/preprocessor/matlab_images/"
    testing_logs_plots_path_prefix = "/logs/tester/matlab/testing_images/"
    error_logs_plots_path_prefix = "/logs/matlab_validator/"
    plot_interval = 5
    tester = Tester(False, False, trained_model_path_prefix, test_images_path_prefix)
    training_errors, validation_errors, testing_errors = tester.test(False, plot_interval,
                                                                     testing_logs_plots_path_prefix)

    matlab_validator = MatlabValidator(matlab_file_name)
    matlab_validator.compare_results(training_errors, validation_errors, testing_errors, error_logs_plots_path_prefix)