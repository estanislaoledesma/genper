#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from torch.utils.data import Dataset


class ImageDataset(Dataset):

    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        current_image = self.images[idx]
        image = current_image.get_preprocessor_guess()
        label = current_image.get_relative_permittivities()
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        return image, label

