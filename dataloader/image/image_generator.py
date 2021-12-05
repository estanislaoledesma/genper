#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from datetime import datetime

import numpy as np

from configs.constants import Constants
from configs.logger import Logger
from dataloader.image.image import Image
from dataloader.shape_generators.circle_generator import CircleGenerator
from dataloader.electric_field.electric_field_generator import ElectricFieldGenerator
from dataloader.shape_generators.rectangle_generator import RectangleGenerator

from utils.file_manager import FileManager

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

LOG = Logger.get_root_logger(
    os.environ.get('ROOT_LOGGER', 'root'),
    filename=os.path.join(ROOT_PATH + "/logs/image_generator/", '{:%Y-%m-%d}.log'.format(datetime.now()))
)


class ImageGenerator:

    def __init__(self, test, rectangles):
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
        if rectangles:
            self.shape_generator = RectangleGenerator()
        else:
            self.shape_generator = CircleGenerator()
        self.electric_field_generator = ElectricFieldGenerator()

    def generate_images(self, test):
        shape_name = self.shape_generator.get_shape_name()
        images = []

        if test:
            LOG.info(f'''{self.no_of_images} images with 2 {shape_name} will be generated''')
        else:
            LOG.info(f'''{self.no_of_images} images with random number of {shape_name} (between 1 and 3) will be generated''')

        for image_i in range(1, self.no_of_images + 1):
            LOG.info(f'''Generating image no. {image_i}/{self.no_of_images}''')
            image_domain = np.linspace(-self.max_diameter, self.max_diameter, self.no_of_pixels)
            x_domain, y_domain = np.meshgrid(image_domain, -image_domain)

            if test:
                no_of_shapes = 2
            else:
                no_of_shapes = int(np.ceil((3 * np.random.uniform()) + 1e-2))
            LOG.info(f'''The image will have {no_of_shapes} {shape_name}''')
            shapes = self.shape_generator.generate_shapes(no_of_shapes, test, image_i)
            image = Image()
            image.generate_relative_permittivities(x_domain, y_domain, shapes)
            electric_field = self.electric_field_generator.generate_electric_field(image, x_domain, y_domain)
            image.set_electric_field(electric_field)
            images.append(image)
            if image_i % 50 == 0 and not test:
                image_path = ROOT_PATH + f'''/logs/image_generator/images/image_{image_i}.png'''
                LOG.info(f'''Saving generated image plot to path {image_path}''')
                image.plot(image_i, image_path)
            if test:
                image_path = ROOT_PATH + f'''/logs/image_generator/images/test/image_{image_i}.png'''
                LOG.info(f'''Saving generated image plot to path {image_path}''')
                image.plot(image_i, image_path)

        images = np.array(images)
        if test:
            images_file = ROOT_PATH + "/data/image_generator/test/images.pkl"
        else:
            images_file = ROOT_PATH + "/data/image_generator/images.pkl"
        LOG.info(f'''Saving {self.no_of_images} images to file {images_file}''')
        FileManager.save(images, images_file)
        return images
