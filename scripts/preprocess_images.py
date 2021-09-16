import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataloader.preprocessor import Preprocessor

if __name__ == "__main__":
    preprocessor = Preprocessor()
    preprocessor.preprocess()