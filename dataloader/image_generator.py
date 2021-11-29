#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from datetime import datetime

import numpy as np

from configs.constants import Constants
from configs.logger import Logger
from dataloader.circle_generator import CircleGenerator
from dataloader.electric_field_generator import ElectricFieldGenerator
from dataloader.image import Image
import deepdish as dd

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

LOG = Logger.get_root_logger(
    os.environ.get('ROOT_LOGGER', 'root'),
    filename=os.path.join(ROOT_PATH + "/logs/image_generator/", '{:%Y-%m-%d}.log'.format(datetime.now()))
)


class ImageGenerator:

    def __init__(self, test):
        basic_parameters = Constants.get_basic_parameters()
        images_parameters = basic_parameters["images"]
        if test:
            LOG.info("Starting image generator in testing mode")
            self.no_of_images = 10
        else:
            LOG.info("Starting image generator in standard mode")
            self.no_of_images = images_parameters["no_of_images"]
        self.max_diameter = images_parameters["max_diameter"]
        self.no_of_pixels = images_parameters["no_of_pixels"]
        self.circle_generator = CircleGenerator()
        self.electric_field_generator = ElectricFieldGenerator()

    def generate_images(self, test):
        images = []

        if test:
            LOG.info("%d images with 2 circles will be generated", self.no_of_images)
        else:
            LOG.info("%d images with random number of circles (between 1 and 3) will be generated", self.no_of_images)

        for image_i in range(1, self.no_of_images + 1):
            LOG.info("Generating image no. %d/%d", image_i, self.no_of_images)
            image_domain = np.linspace(-self.max_diameter, self.max_diameter, self.no_of_pixels)
            x_domain, y_domain = np.meshgrid(image_domain, -image_domain)

            if test:
                no_of_circles = 2
            else:
                no_of_circles = int(np.ceil((3 * np.random.uniform()) + 1e-2))
            LOG.info("The image will have %d circles", no_of_circles)
            circles = self.circle_generator.generate_circles(no_of_circles, test, image_i)
            image = Image()
            image.generate_relative_permittivities(x_domain, y_domain, circles)
            electric_field = self.electric_field_generator.generate_electric_field(image, x_domain, y_domain)
            image.set_electric_field(electric_field)
            images.append(image)
            if image_i % 50 == 0 and not test:
                image_path = ROOT_PATH + "/logs/image_generator/images/image_{}.png".format(image_i)
                LOG.info(f'''Saving generated image plot to path {image_path}''')
                image.plot(image_i, image_path)
            if test:
                image_path = ROOT_PATH + "/logs/image_generator/images/test/image_{}.png".format(image_i)
                LOG.info(f'''Saving generated image plot to path {image_path}''')
                image.plot(image_i, image_path)

        images = np.array(images)
        if test:
            images_file = ROOT_PATH + "/data/image_generator/test/images.h5"
        else:
            images_file = ROOT_PATH + "/data/image_generator/images.h5"
        LOG.info("Saving %d images to file %s", self.no_of_images, images_file)
        dd.io.save(images_file, images)
        return images
