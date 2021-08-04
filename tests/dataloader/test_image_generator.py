import unittest

from dataloader.image_generator import ImageGenerator


class TestImageGenerator(unittest.TestCase):
    def setUp(self):
        self.image_generator = ImageGenerator()

    def test_generate_images(self):
        self.image_generator.generate_images()
