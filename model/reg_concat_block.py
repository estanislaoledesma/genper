#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class RegConcatBlock(nn.Module):

    def __init__(self, registers, reg_set):
        super().__init__()
        self.registers = registers
        self.reg_set = reg_set

    def forward(self, x):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        reg_concat_temp = torch.zeros(x.shape[0],
                                   x.shape[1] * 2, self.registers.get_x(self.reg_set).shape[2],
                                   self.registers.get_x(self.reg_set).shape[3]).to(device=device, dtype=torch.float32)

        reg_concat_temp[:, torch.arange(0, x.shape[1]).to(device=device), :, :] = \
                self.registers.get_x(self.reg_set)

        dif_dim = np.array(self.registers.get_x(self.reg_set).shape) - np.array(x.shape)
        if dif_dim[2] % 2 == 0:
            padding = int(dif_dim[2] / 2)
            tmp = F.pad(x, (0, 0, padding, padding, 0, 0, 0, 0)).to(device=device, dtype=torch.float32)
        else:
            padding = torch.fix(dif_dim[2] / 2)
            tmp = F.pad(x, (0, 0, padding, padding, 0, 0, 0, 0)).to(device=device, dtype=torch.float32)
            tmp = F.pad(tmp, (0, 0, 1, 0, 0, 0, 0, 0)).to(device=device, dtype=torch.float32)

        if dif_dim[3] % 2 == 0:
            padding = int(dif_dim[3] / 2)
            tmp = F.pad(tmp, (padding, padding, 0, 0, 0, 0, 0, 0)).to(device=device, dtype=torch.float32)
        else:
            padding = torch.fix(dif_dim[1] / 2)
            tmp = F.pad(tmp, (padding, padding, 0, 0, 0, 0, 0, 0)).to(device=device, dtype=torch.float32)
            tmp = F.pad(tmp, (1, 0, 0, 0, 0, 0, 0, 0)).to(device=device, dtype=torch.float32)
        reg_concat_temp[:, torch.arange(0, x.shape[1]).to(device=device), :, :] = tmp
        return reg_concat_temp

    def backward(self, dzdx):
        self.registers.set_dzdx(self.reg_set, dzdx)
        return dzdx
