#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pickle


class FileManager:

    @staticmethod
    def save(obj, file_path):
        with open(file_path, 'wb') as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(file_path):
        with open(file_path, 'rb') as handle:
            return pickle.load(handle)
