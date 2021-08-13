#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from configs.constants import Constants
from dataloader.circle_generator import CircleGenerator
from dataloader.image import Image
from utils.coordinates_converter import CoordinatesConverter


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
        self.receiver_radius = physics_parameters["receiver_radius"]
        self.transmitter_radius = physics_parameters["transmitter_radius"]
        self.wave_incidence = physics_parameters["wave_incidence"]
        self.wave_type = physics_parameters["wave_type"]
        self.impedance_of_free_space = physics_parameters["impedance_of_free_space"]

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
            x_domain[pixels_with_circle] = []
            x_domain = x_domain.T
            y_domain[pixels_with_circle] = []
            y_domain = y_domain.T
            complex_relative_permittivities[pixels_with_circle] = []
            complex_relative_permittivities = complex_relative_permittivities.T
            no_of_pixels_with_circle = max(np.shape(x_domain))

            receiver_angles = np.linspace(0, 2 * np.pi, self.no_of_receivers)
            receiver_angles = receiver_angles[:-1]
            receiver_angles = receiver_angles.T
            receiver_angles, receiver_radii = np.meshgrid(receiver_angles, self.receiver_radius)
            receiver_angles = receiver_angles.T
            receiver_radii = receiver_radii.T
            x_receivers, y_receivers = CoordinatesConverter.pol2cart(receiver_angles, receiver_radii)

            transmitter_angles = np.linspace(0, 2 * np.pi, self.no_of_receivers)
            transmitter_angles = transmitter_angles[:-1]
            transmitter_angles = transmitter_angles.T
            if self.wave_type == self.wave_incidence ["plane_wave"]:
                wave_number_x = self.wave_number * np.cos(transmitter_angles)
                wave_number_y = self.wave_number * np.sin(transmitter_angles)
                incident_electric_field = np.exp(1j * x_domain * wave_number_x + 1j * y_domain * wave_number_y)
            else:
                transmitter_angles, transmitter_radii = np.meshgrid(transmitter_angles, self.receiver_radius)
                transmitter_angles = transmitter_angles.T
                transmitter_radii = transmitter_radii.T
                x_transmitters, y_transmitters = CoordinatesConverter.pol2cart(transmitter_angles, transmitter_radii)
                circle_x, transmitter_x = np.meshgrid(x_domain, x_transmitters)
                circle_y, transmitter_y = np.meshgrid(y_domain, y_transmitters)
                dist_transmitter_circles = np.sqrt((circle_x - transmitter_x) ** 2 + (circle_y - transmitter_y) ** 2)
                transposed_electric_field = 1j * self.wave_number * self.impedance_of_free_space * 1j / 4 






