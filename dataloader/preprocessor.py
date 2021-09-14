#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from numpy import pi
from scipy.special import hankel1

from configs.constants import Constants
from dataloader.electric_field_generator import ElectricFieldGenerator


class Preprocessor:

    def __init__(self):
        basic_parameters = Constants.get_basic_parameters()
        physics_parameters = basic_parameters["physics"]
        images_parameters = basic_parameters["images"]
        self.max_diameter = images_parameters["max_diameter"]
        self.no_of_pixels = images_parameters["no_of_pixels"]
        self.total_no_of_pixels = self.no_of_pixels * self.no_of_pixels
        self.pixel_length = 2 * self.max_diameter / (self.no_of_pixels - 1)
        self.pixel_area = self.pixel_length ** 2
        self.wave_number = 2 * np.pi / physics_parameters["wavelength"]
        self.impedance_of_free_space = physics_parameters["impedance_of_free_space"]
        self.equivalent_radius = np.sqrt(self.pixel_area / pi)
        self.electric_field_coefficient = 1j * self.wave_number * self.impedance_of_free_space
        self.no_of_receivers = physics_parameters["no_of_receivers"]
        self.no_of_transmitters = physics_parameters["no_of_transmitters"]
        self.receiver_radius = physics_parameters["receiver_radius"]
        self.electric_field_generator = ElectricFieldGenerator()
        self.noise_level = physics_parameters["noise_level"]

    def preprocess(self, images):
        image_domain = np.linspace(-self.max_diameter, self.max_diameter, self.no_of_pixels)
        x_domain, y_domain = np.meshgrid(image_domain, -image_domain)
        incident_electric_field = self.electric_field_generator.generate_incident_electric_field(x_domain, y_domain)

        x_domain = np.atleast_2d(x_domain).T
        y_domain = np.atleast_2d(y_domain).T

        gs_matrix = self.generate_gs_matrix(x_domain, y_domain)
        gd_matrix = self.generate_gd_matrix(x_domain, y_domain)

        for image in images:
            electric_field = image.get_electric_field()
            rand_real = np.random.randn(self.no_of_receivers, self.no_of_transmitters)
            rand_imag = np.random.randn(self.no_of_receivers, self.no_of_transmitters)
            gaussian_electric_field = np.matmul(1 / np.sqrt(2) *
                                                np.sqrt(1 / self.no_of_receivers / self.no_of_transmitters) *
                                                np.linalg.norm(np.atleast_2d(electric_field).T, 2) * self.noise_level,
                                                (rand_real + 1j * rand_imag))
            noisy_electric_field = electric_field + gaussian_electric_field

            induced_current = np.zeros((self.total_no_of_pixels, self.no_of_transmitters))
            for i in self.no_of_transmitters:
                gamma = np.linalg.solve(np.matmul(gs_matrix, gs_matrix * noisy_electric_field[:, i]),
                                        noisy_electric_field[:, i])
                induced_current[:, i] = gamma * (gs_matrix * noisy_electric_field[:, i])

            total_electric_field_init = incident_electric_field + np.matmul(gd_matrix, induced_current)

    def generate_gs_matrix(self, x_domain, y_domain):
        x_receivers, y_receivers, _ = \
            self.electric_field_generator.get_antennas_coordinates(self.no_of_receivers, self.receiver_radius)
        x_domain_tmp, x_receivers = np.meshgrid(x_domain, x_receivers)
        y_domain_tmp, y_receivers = np.meshgrid(y_domain, y_receivers)
        dist_domain_receivers = np.sqrt((x_domain_tmp - x_receivers) ** 2 + (y_domain_tmp - y_receivers) ** 2)

        gs_matrix = 1j * self.wave_number * self.impedance_of_free_space * \
                    (1j / 4) * hankel1(0, self.wave_number * dist_domain_receivers)
        return gs_matrix

    def generate_gd_matrix(self, x_domain, y_domain):
        x_domain_cell, x_domain_cell_2 = np.meshgrid(x_domain, x_domain)
        x_dist_between_pixels = (x_domain_cell - x_domain_cell_2) ** 2
        y_domain_cell, y_domain_cell_2 = np.meshgrid(y_domain, y_domain)
        y_dist_between_pixels = (y_domain_cell - y_domain_cell_2) ** 2
        dist_between_pixels = np.sqrt(x_dist_between_pixels + y_dist_between_pixels)
        dist_between_pixels = dist_between_pixels + np.identity(total_no_of_pixels)

        phi = 1j * self.wave_number * self.impedance_of_free_space * \
              (1j / 4) * hankel1(0, self.wave_number * dist_between_pixels)
        diag_zero = np.ones(total_no_of_pixels) - np.identity(total_no_of_pixels)
        phi = phi * diag_zero
        integral_receivers = (1j / 4) * (2 / (self.wave_number * self.equivalent_radius) *
                                         hankel1(1, self.wave_number * self.equivalent_radius) +
                                         4 * 1j / ((self.wave_number ** 2) * self.pixel_area))

        gs_matrix = phi + self.electric_field_coefficient * integral_receivers * np.identity(total_no_of_pixels)
        return gs_matrix

