#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
from torch import optim, nn
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

    def __init__(self, test, load):
        basic_parameters = Constants.get_basic_parameters()
        unet_parameters = basic_parameters["unet"]
        images_parameters = basic_parameters["images"]
        self.batch_size = unet_parameters["batch_size"]
        self.accumulation_steps = unet_parameters["accumulation_steps"]
        self.num_sub_batches = unet_parameters["num_sub_batches"]
        self.no_of_pixels = images_parameters["no_of_pixels"]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.unet = UNet()
        self.unet.to(device=self.device)
        self.val_proportion = unet_parameters["val_proportion"]
        self.test_proportion = unet_parameters["test_proportion"]
        if test:
            LOG.info("Starting trainer in testing mode")
            preprocessed_images_path = ROOT_PATH + "/data/preprocessor/test/preprocessed_images.h5"
            self.checkpoint_path = ROOT_PATH + "/data/trainer/test/trained_model.pt"
            datasets_path = ROOT_PATH + "/data/trainer/test/datasets.pt"
        else:
            LOG.info("Starting trainer in standard mode")
            preprocessed_images_path = ROOT_PATH + "/data/preprocessor/preprocessed_images.h5"
            self.checkpoint_path = ROOT_PATH + "/data/trainer/trained_model.pt"
            datasets_path = ROOT_PATH + "/data/trainer/datasets.pt"
        LOG.info(f'''Using device: {self.device}''')
        self.load_datasets(load, preprocessed_images_path, datasets_path)
        self.num_epochs = unet_parameters["num_epochs"]
        self.learning_rate = unet_parameters["learning_rate"]
        weight_decay = unet_parameters["weight_decay"]
        self.optimizer = optim.AdamW(self.unet.parameters(), lr=self.learning_rate, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()
        self.plotter = Plotter()

    def train(self, test, load):
        init_epoch = 0
        min_valid_loss = np.inf
        training_errors = OrderedDict()
        validation_errors = OrderedDict()
        start_epoch_time = datetime.now()
        time_elapsed = start_epoch_time - start_epoch_time
        if load:
            LOG.info(f'''Going to load model from {self.checkpoint_path}''')
            self.unet, self.optimizer, init_epoch, min_valid_loss, training_errors, validation_errors, time_elapsed = \
                CheckpointManager.load_checkpoint(self.unet, self.checkpoint_path, optimizer=self.optimizer)

        LOG.info(f'''Starting training:
                            Total epochs:    {self.num_epochs}
                            Batch size:      {self.batch_size}
                            Learning rate:   {self.learning_rate}
                            Training size:   {len(self.train_loader)}
                            Validation size: {len(self.val_loader)}
                            Time elapsed:    {time_elapsed}
                        ''')
        for epoch in range(init_epoch + 1, self.num_epochs + 1):
            self.unet.train()
            training_loss = 0.0
            self.optimizer.zero_grad(set_to_none=True)
            with tqdm(total=len(self.train_loader), desc=f'Epoch {epoch}/{self.num_epochs}', unit='img') as pbar:
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
                        path = ROOT_PATH + "/logs/trainer/training_images/trained_image_{}_{}.png".format(epoch, ix)
                        LOG.info(f'''Saving trained image plot to path {path}''')
                        self.plotter.plot_comparison_with_tensors(plot_title, path, labels,
                                                     images, prediction, loss.item())
                    if test:
                        plot_title = "Training - Epoch {} - Batch {}".format(epoch, ix)
                        path = ROOT_PATH + "/logs/trainer/training_images/test/trained_image_{}_{}.png".format(epoch, ix)
                        LOG.info(f'''Saving trained image plot to path {path}''')
                        self.plotter.plot_comparison_with_tensors(plot_title, path, labels,
                                                     images, prediction, loss.item())

            training_loss = training_loss / len(self.train_loader)
            training_errors [epoch] = training_loss
            validation_loss = self.validate(test, epoch)
            validation_errors[epoch] = validation_loss

            LOG.info(f'''Statistics of epoch {epoch}/{self.num_epochs}:
                                Training loss: {training_loss:.2E}
                                Validation loss: {validation_loss:.2E}
                                Min validation loss: {min_valid_loss:.2E}''')
            time_elapsed += (datetime.now() - start_epoch_time)
            start_epoch_time = datetime.now()
            if min_valid_loss > validation_loss:
                min_valid_loss = validation_loss
                LOG.info(f'''Saving progress for epoch {epoch} with loss {validation_loss:.2E} to path {self.checkpoint_path}''')
                CheckpointManager.save_checkpoint(self.unet, self.optimizer, self.checkpoint_path, epoch,
                                                  min_valid_loss, training_errors, validation_errors, time_elapsed)
            else:
                LOG.info(f'''Updating checkpoint with new epoch value ({epoch}) in path {self.checkpoint_path}''')
                CheckpointManager.update_epoch(self.checkpoint_path, epoch, training_errors, validation_errors, time_elapsed)
        LOG.info(f'''Finishing training of the network''')
        LOG.info(f'''Total duration of the training was {time_elapsed}''')

        if test:
            path = ROOT_PATH + "/logs/trainer/test_errors_{:%Y-%m-%d_%H:%M:%S}.png".format(datetime.now())
        else:
            path = ROOT_PATH + "/logs/trainer/errors_{:%Y-%m-%d_%H:%M:%S}.png".format(datetime.now())

        LOG.info(f'''Saving per epoch training/validation errors plot to path {path}''')
        self.plotter.plot_errors(training_errors, validation_errors, path)

    def load_datasets(self, load, images_path, datasets_path):
        if load:
            LOG.info(f'''Loading training and validation testing datasets from {datasets_path}''')
            self.train_loader, self.val_loader, _ = CheckpointManager.load_datasets(datasets_path)
        else:
            LOG.info("Loading preprocessed images from file %s", images_path)
            transform = transforms.ToTensor()
            preprocessed_images = ImageDataset(dd.io.load(images_path), transform=transform)
            LOG.info("%d preprocessed images loaded", len(preprocessed_images))
            n_val = int(len(preprocessed_images) * self.val_proportion)
            n_test = int(len(preprocessed_images) * self.test_proportion)
            n_train = len(preprocessed_images) - n_val - n_test
            LOG.info("Train set has %d images", n_train)
            LOG.info("Validation set has %d images", n_val)
            LOG.info("Test set has %d images", n_test)
            train_set, val_set, test_set = random_split(preprocessed_images, [n_train, n_val, n_test],
                                                        generator=torch.Generator().manual_seed(0))
            loader_args = dict(batch_size=self.batch_size, num_workers=4, pin_memory=False)
            self.train_loader = DataLoader(train_set, shuffle=True, **loader_args)
            self.val_loader = DataLoader(val_set, shuffle=True, drop_last=True, **loader_args)
            test_loader = DataLoader(test_set, shuffle=True, drop_last=True, **loader_args)
            CheckpointManager.save_datasets(self.train_loader, self.val_loader, test_loader, datasets_path)
            LOG.info(f'''Saving training, validation and testing datasets to {datasets_path}''')

    def validate(self, test, epoch):
        LOG.info(f'''Validating model for epoch {epoch}''')
        validation_loss = 0.0
        self.unet.eval()
        with torch.no_grad():
            for ix, (images, labels) in enumerate(self.val_loader):
                images = images.to(device=self.device, dtype=torch.float32)
                labels = labels.to(device=self.device, dtype=torch.float32)
                prediction = self.unet(images)
                loss = self.criterion(prediction, labels)
                validation_loss += loss.item()

                if ix % 5 == 0 and not test:
                    plot_title = "Validation - Epoch {} - Batch {}".format(epoch, ix)
                    path = ROOT_PATH + "/logs/trainer/validation_images/validation_image_{}_{}.png".format(epoch, ix)
                    LOG.info(f'''Saving validation image plot to path {path}''')
                    self.plotter.plot_comparison_with_tensors(plot_title, path, labels,
                                                              images, prediction, loss.item())
                if test:
                    plot_title = "Validation - Epoch {} - Batch {}".format(epoch, ix)
                    path = ROOT_PATH + "/logs/trainer/validation_images/test/validation_image_{}_{}.png".format(epoch, ix)
                    LOG.info(f'''Saving validation image plot to path {path}''')
                    self.plotter.plot_comparison_with_tensors(plot_title, path, labels, images,
                                                 prediction, loss.item())
        return validation_loss / len(self.val_loader)