#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

from configs.constants import Constants


class WeightScaler:

    @staticmethod
    def get_weights_scaled(weights, weight_scale_init_method, height, width, in_channels, out_channels):
        basic_parameters = Constants.get_basic_parameters()
        unet_parameters = basic_parameters["unet"]
        scale = unet_parameters["scale"]
        if weight_scale_init_method == "gaussian":
            scale = 0.01/scale
            return weights * scale
        elif weight_scale_init_method == "xavier":
            scale = np.sqrt(3/(height*width*in_channels))
            return (weights * 2 - 1) * scale
        else:
            scale = np.sqrt(2/(height*width*out_channels))
            return weights * scale