#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from datetime import datetime

import numpy as np
import scipy.io

from configs.logger import Logger
from utils.plotter import Plotter

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

LOG = Logger.get_root_logger(
    os.environ.get('ROOT_LOGGER', 'root'),
    filename=os.path.join(ROOT_PATH + "/logs/matlab_validator/", '{:%Y-%m-%d}.log'.format(datetime.now()))
)


class MatlabValidator:

    def __init__(self, matlab_network_results_file):
        self.plotter = Plotter()
        matlab_network_results_file_path = ROOT_PATH + f'''/data/matlab_validator/{matlab_network_results_file}'''
        LOG.info(f'''Going to load matlab file from {matlab_network_results_file_path}''')
        self.matlab_network_results_file_content = scipy.io.loadmat(matlab_network_results_file_path)
        LOG.info("Contents from the file successfully loaded")

    def compare_results(self, training_errors, validation_errors, testing_errors, error_logs_plots_path_prefix):
        LOG.info("Comparing local execution to matlab results")
        matlab_training_errors = self.matlab_network_results_file_content['training_results']['error'][0][0].T
        matlab_validation_errors = self.matlab_network_results_file_content['validation_results']['error'][0][0].T
        matlab_testing_errors = self.matlab_network_results_file_content['err_rec'][0].T
        LOG.info(f'''Training mean error:
        Local execution: {np.array(list(training_errors.values())).mean():.2E} vs. Matlab execution: {matlab_training_errors.mean():.2E} ''')
        LOG.info(f'''Validation mean error:
        Local execution: {np.array(list(validation_errors.values())).mean():.2E} vs. Matlab execution: {matlab_validation_errors.mean():.2E} ''')
        LOG.info(f'''Testing mean error:
        Local execution: {np.array(list(testing_errors.values())).mean():.2E} vs. Matlab execution: {matlab_testing_errors.mean():.2E} ''')

        matlab_training_errors = dict(enumerate(matlab_training_errors.flatten(), 1))
        matlab_validation_errors = dict(enumerate(matlab_validation_errors.flatten(), 1))
        matlab_testing_errors = dict(enumerate(matlab_testing_errors.flatten(), 1))
        path = ROOT_PATH + error_logs_plots_path_prefix + "training_errors.png"
        self.plotter.plot_errors("Training Loss", training_errors, "Local execution", matlab_training_errors,
                                 "Matlab execution", "epoch", path)
        path = ROOT_PATH + error_logs_plots_path_prefix + "validation_errors.png"
        self.plotter.plot_errors("Validation Loss", validation_errors, "Local execution",
                                 matlab_validation_errors, "Matlab execution", "epoch", path)
        path = ROOT_PATH + error_logs_plots_path_prefix + "testing_errors.png"
        self.plotter.plot_errors_as_bar_plot("Testing Loss", testing_errors, "Local execution",
                                 matlab_testing_errors, "Matlab execution", "image", path)

        LOG.info("Finishing comparison of local execution to matlab results")
