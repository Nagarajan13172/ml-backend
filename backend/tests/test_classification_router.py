import io
import sys
import asyncio
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.routers.classification import classify_uploaded_image


def _make_test_upload() -> bytes:
    image = Image.new("RGB", (32, 32), color=(200, 150, 130))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class DummyUploadFile:
    def __init__(self, filename: str, content_type: str, payload: bytes):
        self.filename = filename
        self.content_type = content_type
        self._payload = payload

    async def read(self) -> bytes:
        return self._payload


class ClassificationRouterTests(unittest.TestCase):
    def test_classify_endpoint_returns_predicted_label(self) -> None:
        payload = {
            "predicted_label": "Monkeypox",
            "confidence": 0.91,
            "class_index": 0,
        }
        upload = DummyUploadFile("sample.png", "image/png", _make_test_upload())

        with patch(
            "app.routers.classification.classification_service.classify_image",
            return_value=payload,
        ):
            response = asyncio.run(classify_uploaded_image(upload))

        self.assertEqual(response.predicted_label, "Monkeypox")
        self.assertAlmostEqual(response.confidence, 0.91, places=5)


if __name__ == "__main__":
    unittest.main()
