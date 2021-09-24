#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import unittest

import numpy as np
import pandas as pd

from dataloader.image_generator import ImageGenerator

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))


class TestImageGeneratorFunctional(unittest.TestCase):

    def setUp(self):
        image_generator = ImageGenerator(True)
        self.images = image_generator.generate_images(True)

    def test_equal_images(self):
        i = 1
        for image in self.images:
            image_python = image.get_relative_permittivities()
            image_matlab_file_name = ROOT_PATH + "/tests/functional_tests/matlab_files/image_{}_matlab.csv".format(i)
            image_matlab = np.genfromtxt(image_matlab_file_name, delimiter=",")
            are_equal = np.array_equal(image_python, image_matlab)
            self.assertTrue(are_equal)
            i += 1

    def test_similar_electric_fields(self):
        i = 1
        for image in self.images:
            electric_field_python = image.get_electric_field().get_electric_field()
            electric_field_matlab_file_name = ROOT_PATH + "/tests/functional_tests/matlab_files/electric_field_image_{}_matlab.csv".format(i)
            electric_field_matlab = pd.read_csv(electric_field_matlab_file_name, sep=",", header=None)
            electric_field_matlab = electric_field_matlab.applymap(lambda s: np.complex(s.replace('i', 'j'))).values
            are_close = np.allclose(electric_field_python, electric_field_matlab)
            self.assertTrue(are_close)
            i += 1
