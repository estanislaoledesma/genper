#!/usr/bin/env python
# -*- coding: utf-8 -*-
from math import pi

import numpy as np
from scipy.special import hankel1

from configs.constants import Constants
from dataloader.circle_generator import CircleGenerator
from dataloader.electric_field_generator import ElectricFieldGenerator
from dataloader.image import Image
from utils.coordinates_converter import CoordinatesConverter


class ImageGenerator:

    def __init__(self):
        basic_parameters = Constants.get_basic_parameters()
        images_parameters = basic_parameters["images"]
        self.no_of_images = images_parameters["no_of_images"]
        self.max_diameter = images_parameters["max_diameter"]
        self.no_of_pixels = images_parameters["no_of_pixels"]
        self.circle_generator = CircleGenerator()
        self.electric_field_generator = ElectricFieldGenerator()

    def generate_images(self):
        images = []

        for image_i in range(self.no_of_images):
            image_domain = np.linspace(-self.max_diameter, self.max_diameter, self.no_of_pixels)
            x_domain, y_domain = np.meshgrid(image_domain, -image_domain)

            no_of_circles = int(np.ceil((3 * np.random.uniform()) + 1e-2))
            circles = self.circle_generator.generate_circles(no_of_circles)
            image = Image(x_domain, y_domain, circles)
            electric_field = self.electric_field_generator.generate_electric_field(image, x_domain, y_domain)
            image.set_electric_field(electric_field)
            images.append(image)
