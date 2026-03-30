import sys
import unittest
from pathlib import Path

import numpy as np


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.segmentation_service import _build_binary_details, _build_binary_mask


def _make_synthetic_lesion_image(seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """Create a bright tissue-like image with many dark lesions and edge vignette."""
    height, width = 768, 512
    y, x = np.mgrid[0:height, 0:width]

    image = 0.78 + 0.15 * np.cos((x - width / 2) / width * np.pi)
    image += 0.06 * np.sin(y / height * np.pi * 2)

    dx = (x - width / 2) / (width / 2)
    dy = (y - height / 2) / (height / 2)
    image -= 0.22 * (dx * dx + dy * dy)

    lesion_mask = np.zeros((height, width), dtype=bool)
    rng = np.random.default_rng(seed)

    for _ in range(130):
        cy = rng.integers(40, height - 40)
        cx = rng.integers(40, width - 40)
        radius = rng.uniform(2.0, 10.0)
        amplitude = rng.uniform(0.10, 0.30)

        distance_sq = (x - cx) ** 2 + (y - cy) ** 2
        lesion_blob = np.exp(-distance_sq / (2 * radius * radius))

        image -= amplitude * lesion_blob
        lesion_mask |= lesion_blob > 0.4

    return np.clip(image, 0.0, 1.0), lesion_mask


class SegmentationMaskTests(unittest.TestCase):
    def test_binary_mask_keeps_small_dark_lesions(self) -> None:
        image, expected_mask = _make_synthetic_lesion_image()
        predicted_mask = _build_binary_mask(image)

        true_positive = np.logical_and(predicted_mask, expected_mask).sum()
        false_positive = np.logical_and(predicted_mask, ~expected_mask).sum()
        false_negative = np.logical_and(~predicted_mask, expected_mask).sum()

        recall = true_positive / (true_positive + false_negative + 1e-9)
        precision = true_positive / (true_positive + false_positive + 1e-9)

        self.assertGreater(recall, 0.85)
        self.assertGreater(precision, 0.80)

    def test_binary_details_include_timing_and_resolution(self) -> None:
        image = np.zeros((120, 80), dtype=np.float64)

        details = _build_binary_details(image, 45.0394)

        self.assertEqual(details["title"], "Binary Mask Details")
        self.assertEqual(details["average_filtering_time_ms"], 45.0394)
        self.assertEqual(details["width"], 80)
        self.assertEqual(details["height"], 120)
        self.assertEqual(details["pixel_count"], 9600)
        self.assertIn("80 x 120", details["timing_note"])


if __name__ == "__main__":
    unittest.main()
