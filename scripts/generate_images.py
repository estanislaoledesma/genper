import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataloader.image_generator import ImageGenerator

if __name__ == "__main__":
    image_generator = ImageGenerator()
    image_generator.generate_images()