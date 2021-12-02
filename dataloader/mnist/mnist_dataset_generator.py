#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import random
from datetime import datetime

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

from configs.constants import Constants
from configs.logger import Logger
from dataloader.electric_field.electric_field_generator import ElectricFieldGenerator
from dataloader.image.image import Image

from utils.file_manager import FileManager

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

LOG = Logger.get_root_logger(
    os.environ.get('ROOT_LOGGER', 'root'),
    filename=os.path.join(ROOT_PATH + "/logs/mnist_dataset_generator/", '{:%Y-%m-%d}.log'.format(datetime.now()))
)


class MNISTDatasetGenerator:

    PADDING = 40
    FILL = 1

    def __init__(self):
        torch.multiprocessing.set_sharing_strategy('file_system')
        LOG.info("Starting MNIST dataset generator")
        basic_parameters = Constants.get_basic_parameters()
        images_parameters = basic_parameters["images"]
        physics_parameters = basic_parameters["physics"]
        unet_parameters = basic_parameters["unet"]
        self.max_diameter = images_parameters["max_diameter"]
        self.no_of_pixels = images_parameters["no_of_pixels"]
        self.num_workers = unet_parameters["num_workers"]
        max_permittivity = physics_parameters["max_permittivity"]
        mean = 1 / (1 - max_permittivity)
        std = - mean
        dataset_path = ROOT_PATH + "/data/mnist_dataset_generator/"
        LOG.info(f'''Going to download MNIST dataset to {dataset_path}''')
        loader_args = dict(num_workers=self.num_workers, pin_memory=True)
        train_set = torchvision.datasets.MNIST(dataset_path, train=True, download=True,
                                               transform=torchvision.transforms.Compose([
                                                   torchvision.transforms.ToTensor(),
                                                   torchvision.transforms.Normalize(mean, std),
                                                   torchvision.transforms.Pad(self.PADDING, self.FILL),
                                                   torchvision.transforms.RandomCrop(
                                                       (self.no_of_pixels, self.no_of_pixels))
                                               ]))
        self.train_loader = DataLoader(train_set, shuffle=False, **loader_args)
        self.train_loader_overlap = DataLoader(train_set, shuffle=True, **loader_args)
        test_set = torchvision.datasets.MNIST(dataset_path, train=False, download=True,
                                              transform=torchvision.transforms.Compose([
                                                  torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize(mean, std),
                                                  torchvision.transforms.Pad(self.PADDING, self.FILL),
                                                  torchvision.transforms.RandomCrop(
                                                      (self.no_of_pixels, self.no_of_pixels))
                                              ]))
        self.test_loader = DataLoader(test_set, shuffle=False, **loader_args)
        self.test_loader_overlap = DataLoader(test_set, shuffle=True, **loader_args)
        image_domain = np.linspace(-self.max_diameter, self.max_diameter, self.no_of_pixels)
        self.x_domain, self.y_domain = np.meshgrid(image_domain, -image_domain)
        self.electric_field_generator = ElectricFieldGenerator()

    def generate_datasets(self):
        LOG.info(f'''Going to generate {len(self.train_loader)} of MNIST images for training''')

        training_image_path = ROOT_PATH + "/logs/mnist_dataset_generator/mnist_training_images/"
        training_images_path = ROOT_PATH + "/data/mnist_dataset_generator/mnist_training_images/images.pkl"
        training_images = self.generate_dataset(self.train_loader, self.train_loader_overlap, training_image_path, training_images_path)
        LOG.info(f'''Finishing generation of MNIST images for training''')

        LOG.info(f'''Going to generate {len(self.test_loader)} of MNIST images for testing''')
        training_image_path = ROOT_PATH + "/logs/mnist_dataset_generator/mnist_testing_images/"
        training_images_path = ROOT_PATH + "/data/mnist_dataset_generator/mnist_testing_images/images.pkl"
        testing_images = self.generate_dataset(self.test_loader, self.test_loader_overlap, training_image_path, training_images_path)
        LOG.info(f'''Finishing generation of MNIST images for testing''')

        LOG.info("Finishing generation of MNIST datasets")
        return training_images, testing_images

    def generate_dataset(self, loader, loader_overlap, image_path, images_path):
        images = []
        overlap_iterator = iter(loader_overlap)
        image_i = 1
        for mnist_image, _ in loader:
            LOG.info("Generating MNIST image no. %d/%d", image_i, len(loader))
            image_overlap, _ = next(overlap_iterator)
            new_image = self.overlap_images(mnist_image, image_overlap)
            image = Image()
            image.set_relative_permittivities(torch.squeeze(new_image).detach().numpy())
            electric_field = self.electric_field_generator.generate_electric_field(image, self.x_domain, self.y_domain)
            image.set_electric_field(electric_field)
            images.append(image)
            if image_i % 500 == 0:
                plot_image_path = image_path + "mnist_image_{}.png".format(image_i)
                LOG.info(f'''Saving generated image plot to path {plot_image_path}''')
                image.plot(image_i, plot_image_path)
            image_i += 1
        images = np.array(images)
        LOG.info("Saving %d MNIST images to file %s", len(loader), images_path)
        FileManager.save(images, images_path)
        return images

    def overlap_images(self, image, image_overlap):
        rand = random.uniform(0, 1)
        if rand < 0.5:
            new_image = torch.where((image == self.FILL) & (image_overlap != self.FILL),
                                    image + image_overlap,
                                    image)
        else:
            new_image = image
        return new_image