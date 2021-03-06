#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tests.configs.test_constants import TestConstants
from tests.dataloader.test_circle_generator import TestCircleGenerator
from tests.dataloader.test_image import TestImage
from tests.dataloader.test_image_generator import TestImageGenerator
from tests.functional_tests.test_image_generator_functional import TestImageGeneratorFunctional
from tests.functional_tests.test_preprocessor_functional import TestPreprocessorFunctional
from tests.utils.test_coordinates_converter import TestCoordinatesConverter
from tests.utils.test_file_manager import TestFileManager

if __name__ == '__main__':
    unittest.main()