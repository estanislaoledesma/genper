#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from datetime import datetime

import numpy as np
import torch
from torch import optim, nn
from torchsummary import summary
from torchvision.transforms import transforms
from tqdm import tqdm

from configs.constants import Constants
from configs.logger import Logger
from dataloader.image_dataset import ImageDataset
from model.unet import UNet
import deepdish as dd
from torch.utils.data import random_split, DataLoader

from utils.checkpoint_manager import CheckpointManager
from utils.plotter import Plotter

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
        self.batch_size = unet_parameters["batch_size"]
        self.accumulation_steps = unet_parameters["accumulation_steps"]
        self.num_sub_batches = unet_parameters["num_sub_batches"]
        self.no_of_pixels = images_parameters["no_of_pixels"]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.unet = UNet()
        LOG.info("Summary of the model:")
        LOG.info(summary(self.unet, (unet_parameters["in_channels"], self.no_of_pixels, self.no_of_pixels)))
        self.unet.to(device=self.device)
        if test:
            self.val_proportion = 0.2
            LOG.info("Starting preprocessor in testing mode")
            images_path = ROOT_PATH + "/data/preprocessor/test/preprocessed_images.h5"
            self.checkpoint_path = ROOT_PATH + "/data/trainer/trained_model.pth"
        else:
            self.val_proportion = unet_parameters["val_proportion"]
            LOG.info("Starting preprocessor in standard mode")
            images_path = ROOT_PATH + "/data/preprocessor/preprocessed_images.h5"
            self.checkpoint_path = ROOT_PATH + "/data/trainer/trained_model.pth"
        LOG.info("Loading preprocessed images from file %s", images_path)
        transform = transforms.ToTensor()
        preprocessed_images = ImageDataset(dd.io.load(images_path), transform=transform)
        LOG.info("%d preprocessed images loaded", len(preprocessed_images))
        self.n_val = int(len(preprocessed_images) * self.val_proportion)
        self.n_train = len(preprocessed_images) - self.n_val
        train_set, test_set = random_split(preprocessed_images, [self.n_train, self.n_val],
                                           generator=torch.Generator().manual_seed(0))
        loader_args = dict(batch_size=self.batch_size, num_workers=4, pin_memory=False)
        self.train_loader = DataLoader(train_set, shuffle=False, **loader_args)
        self.val_loader = DataLoader(test_set, shuffle=False, drop_last=True, **loader_args)
        LOG.info("Splitting image set into validation and train with %d %% and %d %%, respectively ",
                 self.val_proportion * 100, (1 - self.val_proportion) * 100)
        LOG.info("Train set has %d images", self.n_train)
        LOG.info("Validation set has %d images", self.n_val)
        self.num_epochs = unet_parameters["num_epochs"]
        self.learning_rate = unet_parameters["learning_rate"]
        weight_decay = unet_parameters["weight_decay"]
        self.optimizer = optim.AdamW(self.unet.parameters(), lr=self.learning_rate, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()
        self.plotter = Plotter()

    def train(self, test, load, display):
        init_epoch = 0
        min_valid_loss = np.inf
        if load:
            LOG.info(f'''Going to load model from {self.checkpoint_path}''')
            self.unet, self.optimizer, init_epoch, min_valid_loss = \
                CheckpointManager.load_checkpoint(self.unet, self.optimizer, self.checkpoint_path)

        LOG.info(f'''Starting training:
                            Epochs:          {self.num_epochs - init_epoch}
                            Batch size:      {self.batch_size}
                            Learning rate:   {self.learning_rate}
                            Training size:   {self.n_train}
                            Validation size: {self.n_val}
                        ''')

        for epoch in range(init_epoch, self.num_epochs):
            self.unet.train()
            epoch_loss = 0.0
            self.optimizer.zero_grad(set_to_none=True)
            with tqdm(total=self.n_train, desc=f'Epoch {epoch + 1}/{self.num_epochs}', unit='img') as pbar:
                for ix, (images, labels) in enumerate(self.train_loader):
                    images = images.to(device=self.device, dtype=torch.float32)
                    labels = labels.to(device=self.device, dtype=torch.float32)
                    prediction = self.unet(images)
                    loss = self.criterion(prediction, labels)
                    loss = loss / self.accumulation_steps
                    epoch_loss += loss.item()
                    loss.backward()
                    if (ix + 1) % self.accumulation_steps == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                    pbar.update(len(images))
                    pbar.set_postfix(**{'loss (batch)': loss})

                    if ix % 50 == 0 and not test:
                        plot_title = "Epoch {} - Batch {}".format(epoch + 1, ix)
                        path = ROOT_PATH + "/logs/trainer/trained_images/trained_image_{}_{}".format(epoch + 1, ix)
                        self.plotter.plot_comparison(plot_title, path, display, labels[-1, -1, :, :].detach().numpy(),
                                                     images[-1, -1, :, :].detach().numpy(),
                                                     prediction[-1, -1, :, :].detach().numpy(),
                                                     loss.item())
                    if test:
                        plot_title = "Epoch {} - Batch {}".format(epoch + 1, ix)
                        path = ROOT_PATH + "/logs/trainer/trained_images/test/trained_image_{}_{}".format(epoch + 1, ix)
                        self.plotter.plot_comparison(plot_title, path, display, labels[-1, -1, :, :].detach().numpy(),
                                                     images[-1, -1, :, :].detach().numpy(),
                                                     prediction[-1, -1, :, :].detach().numpy(),
                                                     loss.item())

            validation_loss = 0.0
            self.unet.eval()
            for images, labels in self.val_loader:
                images = images.to(device=self.device, dtype=torch.float32)
                labels = labels.to(device=self.device, dtype=torch.float32)
                prediction = self.unet(images)
                loss = self.criterion(prediction, labels)
                validation_loss = loss.item()

            LOG.info(f'''Statistics of epoch {epoch + 1}/{self.num_epochs}:
                                Validation loss: {validation_loss:.6f}
                                Min validation loss: {min_valid_loss:.6f}''')
            if min_valid_loss > validation_loss:
                min_valid_loss = validation_loss
                CheckpointManager.save_checkpoint(self.unet, self.optimizer, self.checkpoint_path, epoch,
                                                  min_valid_loss)
                LOG.info(f'''Saving progress for epoch {epoch} with loss {epoch_loss:.6f}''')

    LOG.info(f'''Finishing training of the network''')