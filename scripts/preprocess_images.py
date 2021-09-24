import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataloader.preprocessor import Preprocessor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test", help="Run in test mode", action='store_true')
    args = parser.parse_args()
    test = args.test

    preprocessor = Preprocessor(test)
    preprocessor.preprocess(test)
