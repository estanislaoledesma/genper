#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import nn


class RegTossBlock(nn.Module):

    def __init__(self, registers, reg_number):
        super().__init__()
        self.registers = registers
        self.reg_number = reg_number

    def forward(self, x):
        self.registers.set_x(self.reg_number, x)
        return x

    def backward(self, dzdx):
        return dzdx + self.registers.get_dzdx(self.reg_number)


