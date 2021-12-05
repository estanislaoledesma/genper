#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
import sys


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataloader.matlab.matlab_image_generator import MatlabImageGenerator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="Matlab (.mat) file to load")
    args = parser.parse_args()
    matlab_file_name = args.file

    matlab_image_generator = MatlabImageGenerator(matlab_file_name)
    matlab_image_generator.generate_images()
