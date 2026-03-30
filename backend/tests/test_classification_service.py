import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services import classification_service


class DummyModel:
    def predict(self, image_batch, verbose=0):
        assert image_batch.shape == (1, 224, 224, 3)
        return np.array([[0.05, 0.83, 0.07, 0.05]], dtype=np.float32)


def _make_test_image_bytes() -> bytes:
    image = Image.new("RGB", (48, 48), color=(188, 144, 120))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class ClassificationServiceTests(unittest.TestCase):
    def test_classify_image_returns_top_label(self) -> None:
        bundle = classification_service.ClassifierBundle(
            model=DummyModel(),
            class_names=["Healthy", "Monkeypox", "Chickenpox", "Measles"],
            image_size=(224, 224),
            model_path="dummy.keras",
        )

        with patch(
            "app.services.classification_service._load_classifier_bundle",
            return_value=bundle,
        ):
            result = classification_service.classify_image(_make_test_image_bytes())

        self.assertEqual(result["predicted_label"], "Monkeypox")
        self.assertEqual(result["class_index"], 1)
        self.assertAlmostEqual(result["confidence"], 0.83, places=5)


if __name__ == "__main__":
    unittest.main()
