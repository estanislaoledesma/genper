#!/usr/bin/env python
# -*- coding: utf-8 -*-

from unittest import TestCase
from configs.constants import Constants


class TestConstants(TestCase):
    def test_get_basic_parameters(self):
        basic_parameters = Constants.get_basic_parameters()
        assert len(basic_parameters) != 0
        assert len(basic_parameters["physics"]) != 0
        assert len(basic_parameters["images"]) != 0

