#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from executor.trainer import Trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test", help="Run in test mode", action='store_true')
    parser.add_argument("-l", "--load", help="Load latest checkpoint", action='store_true')
    args = parser.parse_args()
    test = args.test
    load = args.load

    trainer = Trainer(test)
    trainer.train(test, load)
