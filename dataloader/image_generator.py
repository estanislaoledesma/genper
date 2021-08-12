#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from configs.constants import Constants
from dataloader.circle_generator import CircleGenerator
from dataloader.image import Image


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
        self.circle_generator = CircleGenerator()
        self.wave_number = 2 * np.pi / physics_parameters["wavelength"]
        self.angular_frequency = self.wave_number * physics_parameters["speed_of_light"]
        self.vacuum_permittivity = physics_parameters["vacuum_permittivity"]
        self.pixel_length = 2 * self.max_diameter / (self.no_of_pixels - 1)
        self.pixel_area = self.pixel_length ^ 2

    def generate_images(self):
        electric_field = np.zeros((self.no_of_receivers, self.no_of_transmitters, self.no_of_images))
        image_parameters = np.zeros((1, self.no_of_images))
        images = []

        for image_i in range(self.no_of_images):
            image_domain = np.linspace(-self.max_diameter, self.max_diameter, self.no_of_pixels)
            x_domain, y_domain = np.meshgrid(image_domain, -image_domain)
            pixel_total = self.no_of_pixels * self.no_of_pixels

            no_of_circles = int(np.ceil((3 * np.random.uniform()) + 1e-2))
            circles = self.circle_generator.generate_circles(no_of_circles)
            image = Image(x_domain, y_domain, circles)
            images.append(image)
            relative_permittivities = image.get_relative_permittivities()

            complex_relative_permittivities = -1j * self.angular_frequency * (relative_permittivities - 1) \
                                              * self.vacuum_permittivity * self.pixel_area
            pixels_with_circle = relative_permittivities == 1
            pixels_without_circle = np.where(pixels_with_circle == False)
            x_domain[pixels_with_circle] = []
            y_domain[pixels_with_circle] = []
            complex_relative_permittivities[pixels_with_circle] = []
            no_of_pixels_with_circle = max(np.shape(x_domain))

            receiver_angles = np.linspace(0, 2 * np.pi, self.no_of_receivers)
            receiver_angles = receiver_angles[:-1]




