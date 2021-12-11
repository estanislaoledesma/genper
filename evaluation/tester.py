#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
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
        self.manual_seed = unet_parameters["manual_seed"]
        self.num_workers = unet_parameters["num_workers"]
        self.val_proportion = unet_parameters["val_proportion"]
        self.test_proportion = unet_parameters["test_proportion"]
        self.checkpoint_path = ROOT_PATH + trained_model_path_prefix + "trained_model.pt"
        if test:
            LOG.info("Starting tester in testing mode")
            test_images_file = ROOT_PATH + test_images_path_prefix + "test/preprocessed_images.pkl"
        else:
            LOG.info("Starting tester in standard mode")
            test_images_file = ROOT_PATH + test_images_path_prefix + "preprocessed_images.pkl"
        self.unet = UNet()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        LOG.info(f'''Using device: {self.device}''')
        self.unet.to(device=self.device)
        LOG.info(f'''Going to load model from {self.checkpoint_path}''')
        self.unet, _, _, _, self.training_errors, self.validation_errors, _ = \
            CheckpointManager.load_checkpoint(self.unet, self.checkpoint_path, self.device)
        self.criterion = nn.MSELoss()
        LOG.info("Loading preprocessed images from file %s", test_images_file)
        transform = transforms.ToTensor()
        preprocessed_images = ImageDataset(FileManager.load(test_images_file), transform=transform)
        if mnist:
            test_set = preprocessed_images
            LOG.info("%d MNIST preprocessed images loaded", len(test_set))
        else:
            n_val = int(len(preprocessed_images) * self.val_proportion)
            n_test = int(len(preprocessed_images) * self.test_proportion)
            n_train = len(preprocessed_images) - n_val - n_test
            _, _, test_set = random_split(preprocessed_images, [n_train, n_val, n_test],
                                          generator=torch.Generator().manual_seed(self.manual_seed))
        loader_args = dict(batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
        self.testing_loader = DataLoader(test_set, shuffle=True, drop_last=True, **loader_args)
        LOG.info("%d testing images loaded", len(self.testing_loader) * self.batch_size)
        self.plotter = Plotter()

    def test(self, test, plot_interval, testing_logs_plots_path_prefix):
        LOG.info(f'''Going to test model for {len(self.testing_loader) * self.batch_size} images''')
        testing_loss = OrderedDict()
        with torch.no_grad():
            self.unet.eval()
            for ix, (images, labels) in enumerate(self.testing_loader):
                images = images.to(device=self.device, dtype=torch.float32)
                labels = labels.to(device=self.device, dtype=torch.float32)
                prediction = self.unet(images)
                loss = self.criterion(prediction, labels)
                testing_loss[ix + 1] = loss.item()

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

        testing_loss_list = np.array(list(testing_loss.values()))
        LOG.info(f'''Tested model for {len(self.testing_loader) * self.batch_size} images with a total loss of {testing_loss_list.sum():.2E}, average loss of {testing_loss_list.mean():.2E} and standard deviation {testing_loss_list.std():.2E}''')
        return self.training_errors, self.validation_errors, testing_loss
