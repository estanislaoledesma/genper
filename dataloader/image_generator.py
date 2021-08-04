#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from configs.constants import Constants


class ImageGenerator:

    def __init__(self):
        basic_parameters = Constants.get_basic_parameters()
        physics_parameters = basic_parameters["physics"]
        images_parameters = basic_parameters["images"]
        self.no_of_receivers = physics_parameters["no_of_receivers"]
        self.no_of_transmitters = physics_parameters["no_of_transmitters"]
        self.no_of_images = images_parameters["no_of_images"]
        self.max_diameter = images_parameters["max_diameter"]
        self.no_of_pixels = images_parameters["no_of_pixels"]

    def generate_images(self):
        electric_field = np.zeros((self.no_of_receivers, self.no_of_transmitters, self.no_of_images))
        image_parameters = np.zeros((1, self.no_of_images))
        for image_i in range(self.no_of_images):
            image_domain = np.linspace(-self.max_diameter, self.max_diameter, self.no_of_pixels)
            x_domain, y_domain = np.meshgrid(image_domain, -image_domain)
            pixel_total = self.no_of_pixels * self.no_of_pixels

            no_of_circles = np.ceil((3 * np.random.uniform()) + 1e-2)
            print(no_of_circles)

