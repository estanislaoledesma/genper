#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

from dataloader.image_generator import ImageGenerator


class TestImageGenerator(unittest.TestCase):
    def setUp(self):
        self.image_generator = ImageGenerator()
