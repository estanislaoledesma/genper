#!/usr/bin/env python
# -*- coding: utf-8 -*-
from math import pi

import numpy as np
from scipy.special import hankel1

from configs.constants import Constants
from dataloader.electric_field import ElectricField
from utils.coordinates_converter import CoordinatesConverter


class ElectricFieldGenerator:

    def __init__(self):
        basic_parameters = Constants.get_basic_parameters()
        physics_parameters = basic_parameters["physics"]
        images_parameters = basic_parameters["images"]
        self.no_of_pixels = images_parameters["no_of_pixels"]
        self.no_of_receivers = physics_parameters["no_of_receivers"]
        self.no_of_transmitters = physics_parameters["no_of_transmitters"]
        self.max_diameter = images_parameters["max_diameter"]
        self.wave_number = 2 * np.pi / physics_parameters["wavelength"]
        self.angular_frequency = self.wave_number * physics_parameters["speed_of_light"]
        self.vacuum_permittivity = physics_parameters["vacuum_permittivity"]
        self.pixel_length = 2 * self.max_diameter / (self.no_of_pixels - 1)
        self.pixel_area = self.pixel_length ** 2
        self.receiver_radius = physics_parameters["receiver_radius"]
        self.transmitter_radius = physics_parameters["transmitter_radius"]
        self.wave_incidence = physics_parameters["wave_incidence"]
        self.wave_type = physics_parameters["wave_type"]
        self.impedance_of_free_space = physics_parameters["impedance_of_free_space"]
        self.electric_field_coefficient = 1j * self.wave_number * self.impedance_of_free_space
        self.equivalent_radius = np.sqrt(self.pixel_area / pi)

    def generate_electric_field(self, image, x_domain, y_domain):
        relative_permittivities = image.get_relative_permittivities()

        complex_relative_permittivities = \
            -1j * self.angular_frequency * (relative_permittivities - 1) * self.vacuum_permittivity * self.pixel_area
        pixels_with_circle = relative_permittivities != 1
        x_domain = x_domain[pixels_with_circle]
        x_domain = np.atleast_2d(x_domain).T
        y_domain = y_domain[pixels_with_circle]
        y_domain = np.atleast_2d(y_domain).T
        complex_relative_permittivities = complex_relative_permittivities[pixels_with_circle]
        complex_relative_permittivities = complex_relative_permittivities.T
        no_of_pixels_with_circle = max(np.shape(x_domain))

        receiver_angles = np.linspace(0, 2 * np.pi, self.no_of_receivers + 1)
        receiver_angles = receiver_angles[:-1]
        receiver_angles = np.atleast_2d(receiver_angles).T
        receiver_angles, receiver_radii = np.meshgrid(receiver_angles, self.receiver_radius)
        receiver_angles = np.atleast_2d(receiver_angles).T
        receiver_radii = np.atleast_2d(receiver_radii).T
        x_receivers, y_receivers = CoordinatesConverter.pol2cart(receiver_radii, receiver_angles)

        transmitter_angles = np.linspace(0, 2 * np.pi, self.no_of_transmitters + 1)
        transmitter_angles = transmitter_angles[:-1]
        transmitter_angles = np.atleast_2d(transmitter_angles).T
        if self.wave_type == self.wave_incidence["plane_wave"]:
            wave_number_x = self.wave_number * np.cos(transmitter_angles)
            wave_number_y = self.wave_number * np.sin(transmitter_angles)
            incident_electric_field = np.exp(
                np.matmul(1j * x_domain, wave_number_x).T + np.matmul(1j * y_domain, wave_number_y).T)
        else:
            transmitter_angles, transmitter_radii = np.meshgrid(transmitter_angles, self.receiver_radius)
            transmitter_angles = np.atleast_2d(transmitter_angles).T
            transmitter_radii = np.atleast_2d(transmitter_radii).T
            x_transmitters, y_transmitters = CoordinatesConverter.pol2cart(transmitter_radii, transmitter_angles)
            circle_x, transmitter_x = np.meshgrid(x_domain, x_transmitters)
            circle_y, transmitter_y = np.meshgrid(y_domain, y_transmitters)
            dist_transmitter_circles = np.sqrt((circle_x - transmitter_x) ** 2 + (circle_y - transmitter_y) ** 2)
            transposed_electric_field = \
                1j * self.wave_number * self.impedance_of_free_space * 1j / 4 * \
                hankel1(0, self.wave_number * dist_transmitter_circles)
            incident_electric_field = transposed_electric_field.T

        x_domain_with_circles, x_domain_with_circles_2 = np.meshgrid(x_domain, x_domain)
        y_domain_with_circles, y_domain_with_circles_2 = np.meshgrid(y_domain, y_domain)
        dist_between_pixels_with_circles = np.sqrt((x_domain_with_circles - x_domain_with_circles_2) ** 2 +
                                                   (y_domain_with_circles - y_domain_with_circles_2) ** 2)
        dist_between_pixels_with_circles = dist_between_pixels_with_circles + np.identity(no_of_pixels_with_circle)
        integral_1 = 1j / 4 * hankel1(0, self.wave_number * dist_between_pixels_with_circles)
        phi = self.electric_field_coefficient * integral_1
        phi = phi * (np.ones(no_of_pixels_with_circle) - np.identity(no_of_pixels_with_circle))
        integral_2 = 1j / 4 * (2 / (self.wave_number * self.equivalent_radius) *
                               hankel1(1, self.wave_number * self.equivalent_radius) +
                               4 * 1j / ((self.wave_number ** 2) * self.pixel_area))
        phi = phi + self.electric_field_coefficient * integral_2 * np.identity(no_of_pixels_with_circle)
        total_electric_field_transmitters = \
            np.linalg.solve(
                (np.identity(no_of_pixels_with_circle) - np.matmul(phi, np.diag(complex_relative_permittivities))),
                incident_electric_field)

        x_circles, x_receivers = np.meshgrid(x_domain, x_receivers)
        y_circles, y_receivers = np.meshgrid(y_domain, y_receivers)
        dist_receivers_circles = np.sqrt((x_circles - x_receivers) ** 2 + (y_circles - y_receivers) ** 2)
        integral_receivers = \
            self.electric_field_coefficient * (1j / 4) * hankel1(0, self.wave_number * dist_receivers_circles)
        total_electric_field = np.matmul(np.matmul(integral_receivers, np.diag(complex_relative_permittivities)),
                                         total_electric_field_transmitters)
        return ElectricField(total_electric_field)
