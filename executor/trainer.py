#!/usr/bin/env python
# -*- coding: utf-8 -*-
from configs.constants import Constants


class Trainer:

    def __init__(self):
        basic_parameters = Constants.get_basic_parameters()
        images_parameters = basic_parameters["images"]
        physics_parameters = basic_parameters["physics"]
        self.no_of_pixels = images_parameters["no_of_pixels"]
        self.no_of_images = images_parameters["no_of_images"]
        self.test_percentage = images_parameters["test_percentage"]
        self.no_of_test_images = self.no_of_images * self.test_percentage
        self.permittivity_coefficient = physics_parameters["permittivity_coefficient"]

