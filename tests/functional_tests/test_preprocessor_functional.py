import os
import unittest

import numpy as np
import pandas as pd

from dataloader.preprocessor.preprocessor import Preprocessor

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))


class TestPreprocessorFunctional(unittest.TestCase):

    def setUp(self):
        generated_images_path_prefix = "/data/image_generator/"
        logs_plots_path_prefix = "/logs/preprocessor/preprocessed_images/"
        preprocessed_images_path_prefix = "/data/preprocessor/"
        plot_interval = 50
        preprocessor = Preprocessor(True, generated_images_path_prefix)
        self.gs_matrix, self.gd_matrix, self.images = preprocessor.preprocess(True, plot_interval,
                                                                              logs_plots_path_prefix,
                                                                              preprocessed_images_path_prefix)

    def test_equal_preprocessor_images(self):
        i = 1
        for image in self.images:
            preprocessor_image_python = image.get_preprocessor_guess()
            preprocessor_image_matlab_file_name = \
                ROOT_PATH + "/tests/functional_tests/matlab_files/preprocessor_image_{}_matlab.csv".format(i)
            preprocessor_image_matlab = np.genfromtxt(preprocessor_image_matlab_file_name, delimiter=",")
            are_close = np.allclose(preprocessor_image_python, preprocessor_image_matlab)
            self.assertTrue(are_close)
            i += 1

    def test_similar_gs_matrix(self):
        gs_matrix_matlab_file_name = ROOT_PATH + "/tests/functional_tests/matlab_files/gs_matrix_matlab.csv"
        gs_matrix_matlab = pd.read_csv(gs_matrix_matlab_file_name, sep=",", header=None)
        gs_matrix_matlab = gs_matrix_matlab.applymap(lambda s: complex(s.replace('i', 'j'))).values
        are_close = np.allclose(self.gs_matrix, gs_matrix_matlab)
        self.assertTrue(are_close)

    def test_similar_gd_matrix(self):
        self.assertTrue(np.shape(self.gd_matrix) [0] == 4096)
        self.assertTrue(np.shape(self.gd_matrix) [1] == 4096)
