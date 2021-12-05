#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from datetime import datetime

import numpy as np
import scipy.io

from configs.constants import Constants
from configs.logger import Logger
from dataloader.electric_field.electric_field import ElectricField
from dataloader.electric_field.electric_field_generator import ElectricFieldGenerator
from dataloader.image.image import Image
from dataloader.shapes.circle import Circle
from utils.file_manager import FileManager

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

LOG = Logger.get_root_logger(
    os.environ.get('ROOT_LOGGER', 'root'),
    filename=os.path.join(ROOT_PATH + "/logs/matlab_image_generator/", '{:%Y-%m-%d}.log'.format(datetime.now()))
)

class MatlabImageGenerator:

    def __init__(self, matlab_file_name):
        basic_parameters = Constants.get_basic_parameters()
        images_parameters = basic_parameters["images"]
        matlab_file_path = ROOT_PATH + f'''/data/matlab_image_generator/{matlab_file_name}'''
        LOG.info(f'''Going to load matlab file from {matlab_file_path}''')
        self.matlab_file_content = scipy.io.loadmat(matlab_file_path)
        LOG.info("Contents from the file successfully loaded")
        self.max_diameter = images_parameters["max_diameter"]
        self.no_of_pixels = images_parameters["no_of_pixels"]
        self.electric_field_generator = ElectricFieldGenerator()

    def generate_images(self):
        electric_fields = self.matlab_file_content['E_s'].T
        images_parameters = self.matlab_file_content['Pro_Para'].T
        image_domain = np.linspace(-self.max_diameter, self.max_diameter, self.no_of_pixels)
        x_domain, y_domain = np.meshgrid(image_domain, -image_domain)
        images = []
        for i, image_parameters in enumerate(images_parameters):
            LOG.info(f'''Generating image no. {i + 1}/{images_parameters.shape[0]}''')
            image_parameters = image_parameters[0]
            circles = []
            LOG.info(f'''This image has {image_parameters.shape[0]} circles''')
            for circle_parameters in image_parameters:
                radius = circle_parameters[1]
                center_x = circle_parameters[2]
                center_y = circle_parameters[3]
                relative_permittivity = circle_parameters[4]
                circle = Circle(radius, center_x, center_y, relative_permittivity)
                circles.append(circle)
            image = Image()
            image.generate_relative_permittivities(x_domain, y_domain, circles)
            electric_field = ElectricField(electric_fields[i, ...].T)
            image.set_electric_field(electric_field)
            images.append(image)
            if (i + 1) % 50 == 0:
                image_path = ROOT_PATH + f'''/logs/matlab_image_generator/images/image_{i + 1}.png'''
                LOG.info(f'''Saving generated image plot to path {image_path}''')
                image.plot(i + 1, image_path)
        images = np.array(images)
        images_file = ROOT_PATH + "/data/matlab_image_generator/images.pkl"
        LOG.info(f'''Saving {images_parameters.shape[0]} images to file {images_file}''')
        FileManager.save(images, images_file)
        return images
