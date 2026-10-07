import os
import tempfile
import unittest

import cv2
import numpy as np

from filters.overlay import apply_overlay
from filters.pixelate import apply_pixelate


class AlphaHandlingTests(unittest.TestCase):
    def test_pixelate_preserves_rgba_channels(self):
        image = np.zeros((4, 4, 4), dtype=np.uint8)
        image[:, :2, :3] = [255, 0, 0]
        image[:, :2, 3] = 128

        result = apply_pixelate(image, pixel_size=2)

        self.assertEqual(result.shape, image.shape)
        self.assertEqual(result.dtype, np.uint8)
        np.testing.assert_array_equal(result[:, :2, 3], 128)
        np.testing.assert_array_equal(result[:, 2:, 3], 0)

    def test_overlay_composites_over_transparent_rgba_background(self):
        sticker_bgra = np.array([[[0, 0, 255, 128]]], dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as sticker:
            sticker_path = sticker.name
        try:
            self.assertTrue(cv2.imwrite(sticker_path, sticker_bgra))
            background = np.zeros((4, 4, 4), dtype=np.uint8)

            result = apply_overlay(background, sticker_path, scale=0.25, position="center")

            np.testing.assert_array_equal(result[1, 1], [255, 0, 0, 128])
            np.testing.assert_array_equal(result[0, 0], [0, 0, 0, 0])
        finally:
            os.unlink(sticker_path)

    def test_overlay_updates_alpha_using_source_over(self):
        sticker_bgra = np.array([[[0, 0, 255, 128]]], dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as sticker:
            sticker_path = sticker.name
        try:
            self.assertTrue(cv2.imwrite(sticker_path, sticker_bgra))
            background = np.array([[[0, 255, 0, 128]]], dtype=np.uint8)

            result = apply_overlay(background, sticker_path, scale=1, position="top-left")

            self.assertEqual(int(result[0, 0, 3]), 192)
            np.testing.assert_array_equal(result[0, 0, :3], [170, 85, 0])
        finally:
            os.unlink(sticker_path)


if __name__ == "__main__":
    unittest.main()
