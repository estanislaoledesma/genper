import unittest

import numpy as np

from dataloader.circle_generator import CircleGenerator


class TestCircleGenerator(unittest.TestCase):

    def setUp(self):
        self.circle_generator = CircleGenerator()

    def test_generate_circles(self):
        image_domain = np.linspace(-1, 1, 64)
        x_domain, y_domain = np.meshgrid(image_domain, -image_domain)
        self.circle_generator.generate_circles(x_domain, y_domain, 1)
