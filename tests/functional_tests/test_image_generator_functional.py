#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import unittest

import numpy as np
import pandas as pd

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

from dataloader.circle_generator import CircleGenerator

class TestImageGeneratorFuncitonal(unittest.TestCase):

    def test_equal_images(self):
        image_python = np.genfromtxt(ROOT_PATH + "/tests/functional_tests/image_python.csv", delimiter=",")
        image_matlab = np.genfromtxt(ROOT_PATH + "/tests/functional_tests/image_matlab.csv", delimiter=",")
        are_equal = np.array_equal(image_python, image_matlab)
        self.assertTrue(are_equal)

    def test_similar_electric_fields(self):
        electric_field_matlab = pd.read_csv(ROOT_PATH + "/tests/functional_tests/electric_field_matlab.csv", sep=",", header=None)
        electric_field_matlab = electric_field_matlab.applymap(lambda s: np.complex(s.replace('i', 'j'))).values
        electric_field_python = np.loadtxt(ROOT_PATH + "/tests/functional_tests/electric_field_python.txt").view(complex)
        dif = electric_field_python - electric_field_matlab
        relative_error = np.linalg.norm(dif) / np.linalg.norm(electric_field_python)
        self.assertTrue(relative_error < 10 ** -5)

