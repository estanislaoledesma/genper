#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataloader.mnist.mnist_dataset_generator import MNISTDatasetGenerator

if __name__ == "__main__":
    mnist_dataset_generator = MNISTDatasetGenerator()
    mnist_dataset_generator.generate_datasets()
