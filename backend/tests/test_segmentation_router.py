import asyncio
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.routers.segmentation import segment_image


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


class SegmentationRouterTests(unittest.TestCase):
    def test_segment_endpoint_returns_expected_outputs(self) -> None:
        payload = {
            "original_image": "original",
            "segmented_image": "segmented",
            "binary_image": "binary",
            "masked_image": "masked",
            "binary_details": {
                "title": "Binary Mask Details",
                "description": "Binary details",
                "average_filtering_time_ms": 12.3,
                "timing_note": "Timing note",
                "width": 32,
                "height": 32,
                "pixel_count": 1024,
            },
        }
        upload = DummyUploadFile("sample.png", "image/png", _make_test_upload())

        with patch(
            "app.routers.segmentation.segmentation_service.process_image",
            return_value=payload,
        ):
            response = asyncio.run(segment_image(upload))

        self.assertEqual(response.original_image, "original")
        self.assertEqual(response.segmented_image, "segmented")
        self.assertEqual(response.binary_image, "binary")
        self.assertEqual(response.masked_image, "masked")
        self.assertEqual(response.binary_details.title, "Binary Mask Details")


if __name__ == "__main__":
    unittest.main()
