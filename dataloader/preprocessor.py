#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from datetime import datetime

import numpy as np
from numpy import pi
from scipy.special import hankel1
import deepdish as dd

from configs.constants import Constants
from configs.logger import Logger
from dataloader.electric_field_generator import ElectricFieldGenerator

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

LOG = Logger.get_root_logger(
    os.environ.get('ROOT_LOGGER', 'root'),
    filename=os.path.join(ROOT_PATH + "/logs/preprocessor/", '{:%Y-%m-%d}.log'.format(datetime.now()))
)


class Preprocessor:

    def __init__(self, test):
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
        self.angular_frequency = self.wave_number * physics_parameters["speed_of_light"]
        self.vacuum_permittivity = physics_parameters["vacuum_permittivity"]
        if test:
            LOG.info("Starting preprocessor in testing mode")
            images_path = ROOT_PATH + "/data/image_generator/test/images.h5"
        else:
            LOG.info("Starting preprocessor in standard mode")
            images_path = ROOT_PATH + "/data/image_generator/images.h5"
        LOG.info("Loading images from file %s", images_path)
        self.images = dd.io.load(images_path)
        LOG.info("%d images loaded", np.size(self.images))

    def preprocess(self, test, display):
        image_domain = np.linspace(-self.max_diameter, self.max_diameter, self.no_of_pixels)
        x_domain, y_domain = np.meshgrid(image_domain, -image_domain)
        incident_electric_field = self.electric_field_generator.generate_incident_electric_field(x_domain, y_domain)

        x_domain = np.atleast_2d(x_domain.flatten("F")).T
        y_domain = np.atleast_2d(y_domain.flatten("F")).T

        gs_matrix = self.generate_gs_matrix(x_domain, y_domain)
        gd_matrix = self.generate_gd_matrix(x_domain, y_domain)

        image_i = 0
        for image in self.images:
            LOG.info("Preprocessing image no. %d/%d", image_i, np.size(self.images))
            electric_field = image.get_electric_field().get_electric_field()
            rand_real, rand_imag = self.get_rand_real_imag(test)
            electric_field_for_norm = np.atleast_2d(electric_field.flatten("F")).T
            gaussian_electric_field = 1 / np.sqrt(2) * np.sqrt(1 / self.no_of_receivers / self.no_of_transmitters) * \
                                      np.linalg.norm(electric_field_for_norm, 2) * \
                                      self.noise_level * (rand_real + 1j * rand_imag)
            noisy_electric_field = electric_field + gaussian_electric_field

            induced_current = np.zeros((self.total_no_of_pixels, self.no_of_transmitters), dtype=complex)
            for i in range(self.no_of_transmitters):
                first_operand = np.matmul(gs_matrix, np.matmul(gs_matrix.conj().T, np.atleast_2d(noisy_electric_field[:, i]).T))
                second_operand = np.atleast_2d(noisy_electric_field[:, i]).T
                gamma = np.linalg.lstsq(first_operand, second_operand, rcond=None)[0][0]
                i_induced_current = gamma * np.matmul(gs_matrix.conj().T, np.atleast_2d(noisy_electric_field[:, i]).T)
                induced_current[:, i] = i_induced_current.flatten()

            total_electric_field_init = incident_electric_field + np.matmul(gd_matrix, induced_current)
            min_square_num = np.sum(np.conj(total_electric_field_init) * induced_current, axis=1)
            min_square_den = np.sum(np.conj(total_electric_field_init) * total_electric_field_init, axis=1)
            epsilon = np.imag(min_square_num / min_square_den) / \
                      (-self.angular_frequency * self.vacuum_permittivity * self.pixel_length ** 2) + 1
            permittivities = np.reshape(epsilon, (self.no_of_pixels, self.no_of_pixels), order="F")
            image.set_preprocessor_guess(permittivities)
            if image_i % 50 == 0 and not test:
                image.plot_with_preprocessor_guess(image_i, ROOT_PATH +
                                                   "/logs/preprocessor/preprocessed_images/preprocessed_image_{}"
                                                   .format(image_i), display)
            if test:
                image.plot_with_preprocessor_guess(image_i, ROOT_PATH +
                                                   "/logs/preprocessor/preprocessed_images/test/preprocessed_image_{}"
                                                   .format(image_i), display)
            image_i += 1

        if test:
            images_file = ROOT_PATH + "/data/preprocessor/test/preprocessed_images.h5"
        else:
            images_file = ROOT_PATH + "/data/preprocessor/preprocessed_images.h5"
        LOG.info("Saving %d preprocessed images to file %s", np.size(self.images), images_file)
        dd.io.save(images_file, self.images)
        return gs_matrix, gd_matrix, self.images

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
        dist_between_pixels = dist_between_pixels + np.identity(self.total_no_of_pixels)

        phi = 1j * self.wave_number * self.impedance_of_free_space * \
              (1j / 4) * hankel1(0, self.wave_number * dist_between_pixels)
        diag_zero = np.ones(self.total_no_of_pixels) - np.identity(self.total_no_of_pixels)
        phi = phi * diag_zero
        integral_receivers = (1j / 4) * (2 / (self.wave_number * self.equivalent_radius) *
                                         hankel1(1, self.wave_number * self.equivalent_radius) +
                                         4 * 1j / ((self.wave_number ** 2) * self.pixel_area))

        gs_matrix = phi + self.electric_field_coefficient * integral_receivers * np.identity(self.total_no_of_pixels)
        return gs_matrix

    def get_rand_real_imag(self, test):
        if test:
            rand_real = np.ones((self.no_of_receivers, self.no_of_transmitters))
            rand_imag = np.ones((self.no_of_receivers, self.no_of_transmitters))
        else:
            rand_real = np.random.randn(self.no_of_receivers, self.no_of_transmitters)
            rand_imag = np.random.randn(self.no_of_receivers, self.no_of_transmitters)
        return rand_real, rand_imag
