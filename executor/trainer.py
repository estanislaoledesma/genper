#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from datetime import datetime

import torch
from torchvision.transforms import transforms
from tqdm import tqdm

from configs.constants import Constants
from configs.logger import Logger
from dataloader.image_dataset import ImageDataset
from model.euclidean_loss_block import EuclideanLossBlock
from model.unet import UNet
import deepdish as dd
from torch.utils.data import random_split, DataLoader

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

LOG = Logger.get_root_logger(
    os.environ.get('ROOT_LOGGER', 'root'),
    filename=os.path.join(ROOT_PATH + "/logs/trainer/", '{:%Y-%m-%d}.log'.format(datetime.now()))
)


class Trainer:

    def __init__(self, test):
        basic_parameters = Constants.get_basic_parameters()
        unet_parameters = basic_parameters["unet"]
        images_parameters = basic_parameters["images"]
        self.val_proportion = unet_parameters["val_proportion"]
        self.batch_size = unet_parameters["batch_size"]
        self.num_sub_batches = unet_parameters["num_sub_batches"]
        self.no_of_pixels = images_parameters["no_of_pixels"]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.unet = UNet()
        self.unet.to(device=self.device)
        if test:
            LOG.info("Starting preprocessor in testing mode")
            images_path = ROOT_PATH + "/data/preprocessor/test/preprocessed_images.h5"
        else:
            LOG.info("Starting preprocessor in standard mode")
            images_path = ROOT_PATH + "/data/preprocessor/preprocessed_images.h5"
        LOG.info("Loading preprocessed images from file %s", images_path)
        transform = transforms.ToTensor()
        preprocessed_images = ImageDataset(dd.io.load(images_path), transform=transform)
        LOG.info("%d preprocessed images loaded", len(preprocessed_images))
        self.n_val = int(len(preprocessed_images) * self.val_proportion)
        self.n_train = len(preprocessed_images) - self.n_val
        train_set, test_set = random_split(preprocessed_images, [self.n_train, self.n_val],
                                                     generator=torch.Generator().manual_seed(0))
        loader_args = dict(batch_size=self.batch_size, num_workers=0, pin_memory=False)
        self.train_loader = DataLoader(train_set, shuffle=False, **loader_args)
        self.val_loader = DataLoader(test_set, shuffle=False, drop_last=True, **loader_args)
        LOG.info("Splitting image set into validation and train with %d % and %d %, respectively ",
                 self.val_proportion * 100, (1 - self.val_proportion) * 100)
        LOG.info("Train set has %d images", self.n_train)
        LOG.info("Validation set has %d images", self.n_val)
        self.num_epochs = unet_parameters["num_epochs"]

    def train(self, test):
        LOG.info("Going to iterate for %d epochs", self.num_epochs)
        for epoch in range(self.num_epochs):
            self.unet.train()
            epoch_loss = 0
            with tqdm(total=self.n_train, desc=f'Epoch {epoch + 1}/{self.num_epochs}', unit='img') as pbar:
                for images, labels in self.train_loader:
                    images = images.to(device=self.device, dtype=torch.float32)
                    prediction = self.unet(images)
                    loss = EuclideanLossBlock.get_loss(prediction, labels)
                    pbar.update(len(images))
                    epoch_loss += loss
                    pbar.set_postfix(**{'loss (batch)': loss})



