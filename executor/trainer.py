#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from datetime import datetime

import numpy as np

from configs.constants import Constants
from configs.logger import Logger
from model.unet import UNet
import deepdish as dd

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

LOG = Logger.get_root_logger(
    os.environ.get('ROOT_LOGGER', 'root'),
    filename=os.path.join(ROOT_PATH + "/logs/trainer/", '{:%Y-%m-%d}.log'.format(datetime.now()))
)


class Trainer:

    def __init__(self, test):
        basic_parameters = Constants.get_basic_parameters()
        unet_parameters = basic_parameters["unet"]
        self.test_proportion = unet_parameters["test_proportion"]
        self.unet = UNet()
        if test:
            LOG.info("Starting preprocessor in testing mode")
            images_path = ROOT_PATH + "/data/preprocessor/test/preprocessed_images.h5"
        else:
            LOG.info("Starting preprocessor in standard mode")
            images_path = ROOT_PATH + "/data/preprocessor/preprocessed_images.h5"
        LOG.info("Loading preprocessed images from file %s", images_path)
        self.preprocessed_images = dd.io.load(images_path)
        LOG.info("%d preprocessed images loaded", np.size(self.preprocessed_images))

    def train(self):
        return


