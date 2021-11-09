#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from torch import nn

from configs.constants import Constants


class WeightScaler:

    @staticmethod
    def set_weights_scaled(layer, weight_scale_init_method, height, width, in_channels, out_channels):
        basic_parameters = Constants.get_basic_parameters()
        unet_parameters = basic_parameters["unet"]
        scale = unet_parameters["scale"]
        if weight_scale_init_method == "gaussian":
            scale = 0.01/scale
            nn.init.normal_(layer.weight, scale)
        elif weight_scale_init_method == "xavier":
            scale = np.sqrt(3/(height*width*in_channels))
            nn.init.xavier_uniform_(layer.weight, scale)
        else:
            scale = np.sqrt(2/(height*width*out_channels))
            nn.init.xavier_normal_(layer.weight, scale)