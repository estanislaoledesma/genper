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
        reg_concat_temp = np.zeros((x.shape[0],
                                   x.shape[1] * 2, self.registers.get_x(self.reg_set).shape[2],
                                   self.registers.get_x(self.reg_set).shape[3]))

        reg_concat_temp[:, np.arange(0, x.shape[1]), :, :] = \
                self.registers.get_x(self.reg_set).cpu().detach().numpy()

        dif_dim = np.array(self.registers.get_x(self.reg_set).shape) - np.array(x.shape)
        if dif_dim[0] % 2 == 0:
            tmp = np.pad(x.detach().numpy(), [(int(dif_dim[0] / 2),), (0,), (0,)])
        else:
            tmp = np.pad(x.detach().numpy(), [(np.fix(dif_dim[0] / 2),), (0,), (0,)])
            tmp = np.pad(x.detach().numpy(), [(1, 0), (0, 0), (0, 0)])

        if dif_dim[1] % 2 == 0:
            tmp = np.pad(x.detach().numpy(), [(0,), (int(dif_dim[1] / 2),), (0,)])
        else:
            tmp = np.pad(x.detach().numpy(), [(0,), (np.fix(dif_dim[1] / 2),), (0,)])
            tmp = np.pad(x.detach().numpy(), [(0, 0), (1, 0), (0, 0)])
        reg_concat_temp[:, np.arrange(0, x.shape[1]), :, :] = tmp
        return reg_concat_temp

    def backward(self, dzdx):
        self.registers.set_dzdx(self.reg_set, dzdx)
        return dzdx
