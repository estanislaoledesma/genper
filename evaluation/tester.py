#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from datetime import datetime

import torch
from torch import nn

from configs.logger import Logger
from model.unet import UNet
from utils.checkpoint_manager import CheckpointManager
from utils.plotter import Plotter

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

LOG = Logger.get_root_logger(
    os.environ.get('ROOT_LOGGER', 'root'),
    filename=os.path.join(ROOT_PATH + "/logs/trainer/", '{:%Y-%m-%d}.log'.format(datetime.now()))
)


class Tester:

    def __init__(self, test):
        if test:
            LOG.info("Starting tester in testing mode")
            self.checkpoint_path = ROOT_PATH + "/data/trainer/trained_model.pt"
            test_images_file = ROOT_PATH + "/data/trainer/test/datasets.pt"
        else:
            LOG.info("Starting tester in standard mode")
            self.checkpoint_path = ROOT_PATH + "/data/trainer/trained_model.pt"
            test_images_file = ROOT_PATH + "/data/trainer/datasets.pt"
        self.unet = UNet()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        LOG.info(f'''Using device: {self.device}''')
        self.unet.to(device=self.device)
        LOG.info(f'''Going to load model from {self.checkpoint_path}''')
        self.unet, _, _, _, _, _, _ = \
            CheckpointManager.load_checkpoint(self.unet, self.checkpoint_path)
        self.criterion = nn.MSELoss()
        _, _, self.testing_loader = CheckpointManager.load_datasets(test_images_file)
        LOG.info("%d testing images loaded", len(self.testing_loader))
        self.plotter = Plotter()

    def test(self, test):
        LOG.info(f'''Going to test model for {len(self.testing_loader)} images''')
        testing_loss = 0.0
        with torch.no_grad():
            self.unet.eval()
            for ix, (images, labels) in enumerate(self.testing_loader):
                images = images.to(device=self.device, dtype=torch.float32)
                labels = labels.to(device=self.device, dtype=torch.float32)
                prediction = self.unet(images)
                loss = self.criterion(prediction, labels)
                testing_loss += loss.item()

                if ix % 5 == 0 and not test:
                    plot_title = "Testing - Batch {}".format(ix)
                    path = ROOT_PATH + "/logs/tester/testing_images/testing_image_{}".format(ix)
                    LOG.info(f'''Saving testing image plot to path {path}''')
                    self.plotter.plot_comparison_with_tensors(plot_title, path, labels,
                                                 images, prediction, loss.item())
                if test:
                    plot_title = "Testing - Batch {}".format(ix)
                    path = ROOT_PATH + "/logs/tester/testing_images/test/testing_image_{}".format(ix)
                    LOG.info(f'''Saving testing image plot to path {path}''')
                    self.plotter.plot_comparison_with_tensors(plot_title, path, labels,
                                                 images, prediction,  loss.item())

        LOG.info(f'''Tested model for {len(self.testing_loader)} images with a total loss of {testing_loss:.2E}''')
