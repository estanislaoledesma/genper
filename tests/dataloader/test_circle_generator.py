#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

from dataloader.shape_generators.circle_generator import CircleGenerator


class TestCircleGenerator(unittest.TestCase):

    def setUp(self):
        self.circle_generator = CircleGenerator()
