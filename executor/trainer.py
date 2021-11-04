#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from datetime import datetime

import numpy as np
import torch
from torch import optim, nn
from matplotlib import pyplot as plt
from torchvision.transforms import transforms
from tqdm import tqdm
import seaborn as sns

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
        self.learning_rate = 1e-2
        self.optimizer = optim.RMSprop(self.unet.parameters(), lr=self.learning_rate, weight_decay=1e-8, momentum=0.9)
        self.criterion = nn.MSELoss()

    def train(self, test):
        LOG.info("Going to iterate for %d epochs", self.num_epochs)
        for epoch in range(self.num_epochs):
            self.unet.train()
            epoch_loss = 0
            with tqdm(total=self.n_train, desc=f'Epoch {epoch + 1}/{self.num_epochs}', unit='img') as pbar:
                for images, labels in self.train_loader:
                    images = images.to(device=self.device, dtype=torch.float32)
                    labels = labels.to(device=self.device, dtype=torch.float32)
                    self.optimizer.zero_grad(set_to_none=True)
                    prediction = self.unet(images)
                    loss = self.criterion(prediction, labels)
                    epoch_loss += loss.item()
                    loss.backward()
                    self.optimizer.step()
                    pbar.update(len(images))

                    pbar.set_postfix(**{'loss (batch)': loss})
                    basic_parameters = Constants.get_basic_parameters()
                    images_parameters = basic_parameters["images"]
                    max_diameter = images_parameters["max_diameter"]
                    x_max = 63
                    y_max = 63
                    plt.close("all")
                    figure, axis = plt.subplots(1, 2, figsize=(15, 15))
                    sns.heatmap(ax=axis[0], data=labels[-1, -1, :, :].detach().numpy(), cmap="rocket",
                                cbar_kws={"label": "Permitividades relativas"})
                    axis[0].set_xticks(np.linspace(0, x_max, 5))
                    axis[0].set_xticklabels(np.linspace(-max_diameter, max_diameter, 5))
                    axis[0].set_yticks(np.linspace(y_max, 0, 5))
                    axis[0].set_yticklabels(np.linspace(-max_diameter, max_diameter, 5))
                    axis[0].set_title("Imagen original")
                    sns.heatmap(ax=axis[1], data=prediction[-1, -1, :, :].detach().numpy(), cmap="rocket",
                                cbar_kws={"label": "Permitividades relativas"})
                    axis[1].set_xticks(np.linspace(0, x_max, 5))
                    axis[1].set_xticklabels(np.linspace(-max_diameter, max_diameter, 5))
                    axis[1].set_yticks(np.linspace(y_max, 0, 5))
                    axis[1].set_yticklabels(np.linspace(-max_diameter, max_diameter, 5))
                    axis[1].set_title("Imagen obtenida de la red neuronal")
                    plt.pause(0.01)



