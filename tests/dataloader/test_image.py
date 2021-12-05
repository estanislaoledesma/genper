#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

import numpy as np

from configs.constants import Constants
from dataloader.image.image import Image
from dataloader.shape_generators.circle_generator import CircleGenerator
from dataloader.shape_generators.rectangle_generator import RectangleGenerator


class TestImage(unittest.TestCase):

    def setUp(self):
        basic_parameters = Constants.get_basic_parameters()
        images_parameters = basic_parameters["images"]
        self.image = Image()
        self.rectangle_generator = RectangleGenerator()
        self.circle_generator = CircleGenerator()
        self.max_diameter = images_parameters["max_diameter"]
        self.no_of_pixels = images_parameters["no_of_pixels"]
        image_domain = np.linspace(-self.max_diameter, self.max_diameter, self.no_of_pixels)
        self.x_domain, self.y_domain = np.meshgrid(image_domain, -image_domain)

    def test_generate_relative_permittivities_with_rectangles(self):
        rectangles = self.rectangle_generator.generate_shapes(2, True, 2)
        self.image.generate_relative_permittivities(self.x_domain, self.y_domain, rectangles)
        max_relative_permittivity = max(rectangles[0].get_relative_permittivity(), rectangles[1].get_relative_permittivity())
        min_relative_permittivity = min(rectangles[0].get_relative_permittivity(), rectangles[1].get_relative_permittivity())
        assert self.image.get_relative_permittivities().shape == (self.no_of_pixels, self.no_of_pixels)
        assert self.image.get_relative_permittivities().max() == max_relative_permittivity
        assert self.image.get_relative_permittivities().min() == 1
        assert np.any(self.image.get_relative_permittivities() == min_relative_permittivity)

    def test_generate_relative_permittivities_with_circles(self):
        circles = self.circle_generator.generate_shapes(2, True, 2)
        self.image.generate_relative_permittivities(self.x_domain, self.y_domain, circles)
        max_relative_permittivity = max(circles[0].get_relative_permittivity(), circles[1].get_relative_permittivity())
        min_relative_permittivity = min(circles[0].get_relative_permittivity(), circles[1].get_relative_permittivity())
        assert self.image.get_relative_permittivities().shape == (self.no_of_pixels, self.no_of_pixels)
        assert self.image.get_relative_permittivities().max() == max_relative_permittivity
        assert self.image.get_relative_permittivities().min() == 1
        assert np.any(self.image.get_relative_permittivities() == min_relative_permittivity)