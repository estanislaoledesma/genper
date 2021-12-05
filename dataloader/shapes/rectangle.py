#!/usr/bin/env python
# -*- coding: utf-8 -*-

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataloader.shapes.shape import Shape


class Rectangle(Shape):

    def __init__(self, width, height, center_x, center_y, relative_permittivity):
        super().__init__(center_x, center_y, relative_permittivity)
        self.width = width
        self.height = height

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height

    def check_if_pixels_belong_to_shape(self, x_domain, y_domain):
        center_x = self.get_center_x()
        center_y = self.get_center_y()
        width = self.get_width()
        height = self.get_height()
        min_x = center_x - width / 2
        max_x = center_x + width / 2
        min_y = center_y - height / 2
        max_y = center_y + height / 2
        return (x_domain >= min_x) & (x_domain <= max_x) & (y_domain >= min_y) & (y_domain <= max_y)

