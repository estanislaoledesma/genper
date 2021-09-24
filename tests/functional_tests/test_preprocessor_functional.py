import os
import unittest

import numpy as np
import pandas as pd

from dataloader.preprocessor import Preprocessor

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))


class TestPreprocessorFunctional(unittest.TestCase):

    def setUp(self):
        preprocessor = Preprocessor(True)
        self.gs_matrix, self.gd_matrix, self.images = preprocessor.preprocess(True)

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
        gs_matrix_matlab = gs_matrix_matlab.applymap(lambda s: np.complex(s.replace('i', 'j'))).values
        are_close = np.allclose(self.gs_matrix, gs_matrix_matlab)
        self.assertTrue(are_close)

    def test_similar_gd_matrix(self):
        gd_matrix_matlab_file_name = ROOT_PATH + "/tests/functional_tests/matlab_files/gd_matrix_matlab.csv"
        gd_matrix_matlab = pd.read_csv(gd_matrix_matlab_file_name, sep=",", header=None)
        gd_matrix_matlab = gd_matrix_matlab.applymap(lambda s: np.complex(s.replace('i', 'j'))).values
        are_close = np.allclose(self.gd_matrix, gd_matrix_matlab)
        self.assertTrue(are_close)
