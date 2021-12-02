#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import unittest

import numpy as np

from dataloader.image.image import Image
from utils.file_manager import FileManager

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))


class TestFileManager(unittest.TestCase):

    def test_save_and_load(self):
        file_path = ROOT_PATH + "/tests/utils/data/test_data.pkl"
        image1 = Image()
        relative_permittivities1 = np.random.rand(100, 200, 500)
        image1.set_relative_permittivities(relative_permittivities1)
        image2 = Image()
        relative_permittivities2 = np.random.rand(10, 20, 5000)
        image2.set_relative_permittivities(relative_permittivities2)
        obj = [image1, image2]
        try:
            FileManager.save(obj, file_path)
            obj_loaded = FileManager.load(file_path)
            assert obj_loaded is not None
            assert len(obj_loaded) == len(obj)
            assert obj_loaded[0] is not None
            assert np.array_equal(obj_loaded[0].get_relative_permittivities(), obj[0].get_relative_permittivities())
            assert obj_loaded[1] is not None
            assert np.array_equal(obj_loaded[1].get_relative_permittivities(), obj[1].get_relative_permittivities())
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
