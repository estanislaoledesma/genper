#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from configs.constants import Constants
from configs.logger import Logger
from dataloader.image_dataset import ImageDataset
from model.unet import UNet
from utils.checkpoint_manager import CheckpointManager
from utils.file_manager import FileManager
from utils.plotter import Plotter

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

LOG = Logger.get_root_logger(
    os.environ.get('ROOT_LOGGER', 'root'),
    filename=os.path.join(ROOT_PATH + "/logs/tester/", '{:%Y-%m-%d}.log'.format(datetime.now()))
)


class Tester:

    def __init__(self, test, mnist, trained_model_path_prefix, test_images_path_prefix):
        basic_parameters = Constants.get_basic_parameters()
        unet_parameters = basic_parameters["unet"]
        self.batch_size = unet_parameters["batch_size"]
        self.checkpoint_path = ROOT_PATH + trained_model_path_prefix + "trained_model.pt"
        if test:
            LOG.info("Starting tester in testing mode")
            test_images_file = ROOT_PATH + test_images_path_prefix + "test/datasets.pt"
        else:
            LOG.info("Starting tester in standard mode")
            test_images_file = ROOT_PATH + test_images_path_prefix + "datasets.pt"
        self.unet = UNet()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        LOG.info(f'''Using device: {self.device}''')
        self.unet.to(device=self.device)
        LOG.info(f'''Going to load model from {self.checkpoint_path}''')
        self.unet, _, _, _, _, _, _ = \
            CheckpointManager.load_checkpoint(self.unet, self.checkpoint_path, self.device)
        self.criterion = nn.MSELoss()
        if mnist:
            test_images_file = test_images_path_prefix + "preprocessed_images.pkl"
            LOG.info("Loading MNIST preprocessed images from file %s", test_images_file)
            transform = transforms.ToTensor()
            preprocessed_images = ImageDataset(FileManager.load(test_images_file), transform=transform)
            LOG.info("%d MNIST preprocessed images loaded", len(preprocessed_images))
            loader_args = dict(batch_size=self.batch_size, num_workers=4, pin_memory=True)
            self.testing_loader = DataLoader(preprocessed_images, shuffle=True, drop_last=True, **loader_args)
        else:
            _, _, self.testing_loader = CheckpointManager.load_datasets(test_images_file, self.device)
        LOG.info("%d testing images loaded", len(self.testing_loader))
        self.plotter = Plotter()

    def test(self, test, plot_interval, testing_logs_plots_path_prefix):
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

                if ix % plot_interval == 0 and not test:
                    plot_title = "Testing - Batch {}".format(ix)
                    path = ROOT_PATH + testing_logs_plots_path_prefix + "testing_image_{}".format(ix)
                    LOG.info(f'''Saving testing image plot to path {path}''')
                    self.plotter.plot_comparison_with_tensors(plot_title, path, labels,
                                                 images, prediction, loss.item())
                if test:
                    plot_title = "Testing - Batch {}".format(ix)
                    path = ROOT_PATH + testing_logs_plots_path_prefix + "test/testing_image_{}".format(ix)
                    LOG.info(f'''Saving testing image plot to path {path}''')
                    self.plotter.plot_comparison_with_tensors(plot_title, path, labels,
                                                 images, prediction,  loss.item())

        LOG.info(f'''Tested model for {len(self.testing_loader)} images with a total loss of {testing_loss:.2E}''')
