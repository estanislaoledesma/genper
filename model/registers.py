#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import torch


class Registers:

    def __init__(self, register_number):
        self.x_registers = np.zeros(register_number).tolist()
        self.dzdx_registers = np.zeros(register_number).tolist()

    def set_x(self, register_number, x):
        self.x_registers[register_number] = x

    def get_x(self, register_number):
        return self.x_registers[register_number]

    def set_dzdx(self, register_number, dzdx):
        self.dzdx_registers[register_number] = dzdx

    def get_dzdx(self, register_number):
        return self.dzdx_registers[register_number]
