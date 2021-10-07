#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

from configs.constants import Constants


class WeightScaler:

    @staticmethod
    def get_weight_scale(weight_scale_init_method, h, w, in_channels, out_channels):
        basic_parameters = Constants.get_basic_parameters()
        unet_parameters = basic_parameters["unet"]
        scale = unet_parameters["scale"]
        if weight_scale_init_method == "gaussian":
            return 0.01/scale
        elif weight_scale_init_method == "xavier":
            return np.sqrt(3/(h*w*in_channels))
        else:
            return np.sqrt(2/(h*w*out_channels))