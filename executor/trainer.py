#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from collections import OrderedDict
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
        self.val_proportion = unet_parameters["val_proportion"]
        self.test_proportion = unet_parameters["test_proportion"]
        if test:
            LOG.info("Starting preprocessor in testing mode")
            images_path = ROOT_PATH + "/data/preprocessor/test/preprocessed_images.h5"
            self.checkpoint_path = ROOT_PATH + "/data/trainer/trained_model.pth"
            test_images_file = ROOT_PATH + "/data/trainer/test/test_images.pth"
        else:
            LOG.info("Starting preprocessor in standard mode")
            images_path = ROOT_PATH + "/data/preprocessor/preprocessed_images.h5"
            self.checkpoint_path = ROOT_PATH + "/data/trainer/trained_model.pth"
            test_images_file = ROOT_PATH + "/data/trainer/test_images.pth"
        LOG.info("Loading preprocessed images from file %s", images_path)
        transform = transforms.ToTensor()
        preprocessed_images = ImageDataset(dd.io.load(images_path), transform=transform)
        LOG.info("%d preprocessed images loaded", len(preprocessed_images))
        self.n_val = int(len(preprocessed_images) * self.val_proportion)
        self.n_test = int(len(preprocessed_images) * self.test_proportion)
        self.n_train = len(preprocessed_images) - self.n_val - self.n_test
        train_set, val_set, test_set = random_split(preprocessed_images, [self.n_train, self.n_val, self.n_test],
                                           generator=torch.Generator().manual_seed(0))
        loader_args = dict(batch_size=self.batch_size, num_workers=4, pin_memory=False)
        self.train_loader = DataLoader(train_set, shuffle=False, **loader_args)
        self.val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
        self.test_loader = DataLoader(test_set, shuffle=False, drop_last=True, **loader_args)
        LOG.info("Splitting image set into test, validation and train with %d %%, %d %% and %d %%, respectively ",
                 self.test_proportion * 100, self.val_proportion * 100, (1 - self.val_proportion) * 100)
        LOG.info("Train set has %d images", self.n_train)
        LOG.info("Validation set has %d images", self.n_val)
        LOG.info("Test set has %d images", self.n_test)
        self.num_epochs = unet_parameters["num_epochs"]
        self.learning_rate = unet_parameters["learning_rate"]
        weight_decay = unet_parameters["weight_decay"]
        self.optimizer = optim.AdamW(self.unet.parameters(), lr=self.learning_rate, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()
        self.plotter = Plotter()
        LOG.info("Saving %d test images set to file %s", self.n_test, test_images_file)
        CheckpointManager.save_dataset(test_images_file, self.test_loader)

    def train(self, test, load, display):
        init_epoch = 1
        min_valid_loss = np.inf
        training_errors = OrderedDict()
        validation_errors = OrderedDict()
        if load:
            LOG.info(f'''Going to load model from {self.checkpoint_path}''')
            self.unet, self.optimizer, init_epoch, min_valid_loss, training_errors, validation_errors = \
                CheckpointManager.load_checkpoint(self.unet, self.optimizer, self.checkpoint_path)

        LOG.info(f'''Starting training:
                            Epochs:          {self.num_epochs - init_epoch}
                            Batch size:      {self.batch_size}
                            Learning rate:   {self.learning_rate}
                            Training size:   {self.n_train}
                            Validation size: {self.n_val}
                            Testing size: {self.n_test}
                        ''')

        for epoch in range(init_epoch, self.num_epochs + 1):
            self.unet.train()
            training_loss = 0.0
            self.optimizer.zero_grad(set_to_none=True)
            with tqdm(total=self.n_train, desc=f'Epoch {epoch}/{self.num_epochs}', unit='img') as pbar:
                for ix, (images, labels) in enumerate(self.train_loader):
                    images = images.to(device=self.device, dtype=torch.float32)
                    labels = labels.to(device=self.device, dtype=torch.float32)
                    prediction = self.unet(images)
                    loss = self.criterion(prediction, labels)
                    loss = loss / self.accumulation_steps
                    training_loss += loss.item()
                    loss.backward()
                    if (ix + 1) % self.accumulation_steps == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                    pbar.update(len(images))
                    pbar.set_postfix(**{'loss (batch)': loss.item()})

                    if ix % 50 == 0 and not test:
                        plot_title = "Training - Epoch {} - Batch {}".format(epoch, ix)
                        path = ROOT_PATH + "/logs/trainer/trained_images/trained_image_{}_{}".format(epoch, ix)
                        self.plotter.plot_comparison(plot_title, path, display, labels[-1, -1, :, :].detach().numpy(),
                                                     images[-1, -1, :, :].detach().numpy(),
                                                     prediction[-1, -1, :, :].detach().numpy(),
                                                     loss.item())
                    if test:
                        plot_title = "Training - Epoch {} - Batch {}".format(epoch, ix)
                        path = ROOT_PATH + "/logs/trainer/trained_images/test/trained_image_{}_{}".format(epoch, ix)
                        self.plotter.plot_comparison(plot_title, path, display, labels[-1, -1, :, :].detach().numpy(),
                                                     images[-1, -1, :, :].detach().numpy(),
                                                     prediction[-1, -1, :, :].detach().numpy(),
                                                     loss.item())

            training_loss = training_loss / self.n_train
            training_errors [epoch] = training_loss
            validation_loss = 0.0
            self.unet.eval()
            for ix, (images, labels) in enumerate(self.val_loader):
                images = images.to(device=self.device, dtype=torch.float32)
                labels = labels.to(device=self.device, dtype=torch.float32)
                prediction = self.unet(images)
                loss = self.criterion(prediction, labels)
                validation_loss += loss.item()

                if ix % 5 == 0 and not test:
                    plot_title = "Validation - Epoch {} - Batch {}".format(epoch, ix)
                    path = ROOT_PATH + "/logs/trainer/validation_images/validation_image_{}_{}".format(epoch, ix)
                    self.plotter.plot_comparison(plot_title, path, display, labels[-1, -1, :, :].detach().numpy(),
                                                 images[-1, -1, :, :].detach().numpy(),
                                                 prediction[-1, -1, :, :].detach().numpy(),
                                                 loss.item())
                if test:
                    plot_title = "Validation - Epoch {} - Batch {}".format(epoch, ix)
                    path = ROOT_PATH + "/logs/trainer/validation_images/test/validation_image_{}_{}".format(epoch, ix)
                    self.plotter.plot_comparison(plot_title, path, display, labels[-1, -1, :, :].detach().numpy(),
                                                 images[-1, -1, :, :].detach().numpy(),
                                                 prediction[-1, -1, :, :].detach().numpy(),
                                                 loss.item())
            validation_loss = validation_loss / self.n_val
            validation_errors[epoch] = validation_loss

            LOG.info(f'''Statistics of epoch {epoch}/{self.num_epochs}:
                                Training loss: {training_loss:.6f}
                                Validation loss: {validation_loss:.6f}
                                Min validation loss: {min_valid_loss:.6f}''')
            if min_valid_loss > validation_loss:
                min_valid_loss = validation_loss
                CheckpointManager.save_checkpoint(self.unet, self.optimizer, self.checkpoint_path, epoch,
                                                  min_valid_loss, training_errors, validation_errors)
                LOG.info(f'''Saving progress for epoch {epoch} with loss {validation_loss:.6f}''')

        LOG.info(f'''Finishing training of the network''')

        if test:
            path = ROOT_PATH + "/logs/trainer/test_errors_{:%Y-%m-%d_%H:%M:%S}".format(datetime.now())
        else:
            path = ROOT_PATH + "/logs/trainer/errors_{:%Y-%m-%d_%H:%M:%S}".format(datetime.now())

        self.plotter.plot_errors(training_errors, validation_errors, path, display)
