#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from torch import nn


class RegConcatBlock(nn.Module):

    def __init__(self, registers, reg_set):
        super().__init__()
        self.registers = registers
        self.reg_set = reg_set

    def forward(self, x):
        reg_concat_temp = np.zeros(self.registers.get_x(self.reg_set).shape[0],
                                   self.registers.get_x(self.reg_set).shape[1], x.shape[2] * len(self.regset) + 1,
                                   x.shape[3])

        for i in range(len(self.reg_set)):
            reg_concat_temp[:, :, (i - 1) * x.shape[2] + np.arrange(0, x.shape[2]), :] = self.registers.get_x(i)

        dif_dim = self.registers.get_x(self.reg_set).shape - x.shape
        if dif_dim[0] % 2 == 0:
            tmp = np.pad(x, [(dif_dim[0] / 2,), (0,), (0,)])
        else:
            tmp = np.pad(x, [(np.fix(dif_dim[0] / 2),), (0,), (0,)])
            tmp = np.pad(x, [(1, 0), (0, 0), (0, 0)])

        if dif_dim[1] % 2 == 0:
            tmp = np.pad(x, [(0,), (dif_dim[1] / 2,), (0,)])
        else:
            tmp = np.pad(x, [(0,), (np.fix(dif_dim[1] / 2),), (0,)])
            tmp = np.pad(x, [(0, 0), (1, 0), (0, 0)])
        reg_concat_temp[:, :, (len(self.reg_set) - 1) * x.shape[2] + np.arrange(0, x.shape[2]), :] = tmp
        return reg_concat_temp

    def backward(self, dzdx):
        self.registers.set_dzdx(self.reg_set, dzdx)
        return dzdx
