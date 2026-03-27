import io
import base64
import numpy as np
import skfuzzy as fuzz
from skimage import io as skio, color
from skimage.filters import threshold_otsu
from skimage.util import img_as_ubyte
from PIL import Image


def _encode_image_to_base64(image_array: np.ndarray, normalize: bool = False) -> str:
    """
    Convert a numpy image array to a base64-encoded PNG string.

    Args:
        image_array: 2D numpy array (grayscale image).
        normalize: If True, normalize the array to [0, 255] before encoding.

    Returns:
        Base64-encoded PNG string (without data URI prefix).
    """
    arr = image_array.copy().astype(np.float64)

    if normalize:
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max - arr_min > 0:
            arr = (arr - arr_min) / (arr_max - arr_min) * 255.0
        else:
            arr = np.zeros_like(arr)

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    pil_image = Image.fromarray(arr, mode="L")

    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)

    return base64.b64encode(buffer.read()).decode("utf-8")


def process_image(file_bytes: bytes) -> dict:
    """
    Run fuzzy image segmentation + Otsu thresholding on raw image bytes.

    Pipeline:
        1. Decode bytes → grayscale numpy array
        2. Compute histogram (3 bins)
        3. Build fuzzy membership functions (trimf)
        4. Apply fuzzy segmentation
        5. Apply Otsu threshold to produce binary image
        6. Encode all three images as base64 PNG strings

    Args:
        file_bytes: Raw bytes of the uploaded image file.

    Returns:
        dict with keys: original_image, segmented_image, binary_image (all base64 PNG strings)
    """
    # -----------------------------------------------------------------
    # Step 1: Decode image → grayscale float [0, 1]
    # -----------------------------------------------------------------
    np_arr = np.frombuffer(file_bytes, dtype=np.uint8)
    image = skio.imread(io.BytesIO(file_bytes), as_gray=False)

    # Handle different channel counts
    if image.ndim == 3 and image.shape[2] == 4:
        # RGBA → RGB first
        image = image[:, :, :3]
    if image.ndim == 3:
        gray_image = color.rgb2gray(image)   # float [0, 1]
    else:
        gray_image = image.astype(np.float64)
        if gray_image.max() > 1.0:
            gray_image = gray_image / 255.0  # normalise to [0, 1]

    # -----------------------------------------------------------------
    # Step 2: Generate histogram (3 bins over [0, 1] range)
    # -----------------------------------------------------------------
    hist, bin_edges = np.histogram(gray_image, bins=3)

    # -----------------------------------------------------------------
    # Step 3: Build fuzzy membership functions (trimf over 0-255 scale)
    # -----------------------------------------------------------------
    level = 0.5  # fuzziness level
    x_range = np.arange(0, 256)
    membership_functions = []

    for i in range(len(bin_edges) - 1):
        lo = bin_edges[i] * 255 - level
        mid = bin_edges[i] * 255
        hi = bin_edges[i + 1] * 255 + level
        membership_functions.append(fuzz.trimf(x_range, [lo, mid, hi]))

    # -----------------------------------------------------------------
    # Step 4: Apply fuzzy segmentation
    # -----------------------------------------------------------------
    img_scaled = (gray_image * 255).astype(np.float64)  # scale to [0, 255]
    segmented_image = np.zeros_like(img_scaled)

    for i, mf in enumerate(membership_functions):
        membership_vals = fuzz.interp_membership(x_range, mf, img_scaled)
        segmented_image += membership_vals * (i + 1)

    # -----------------------------------------------------------------
    # Step 5: Otsu thresholding → binary image
    # -----------------------------------------------------------------
    thresh = threshold_otsu(segmented_image)
    binary_image = (segmented_image > thresh).astype(np.uint8) * 255

    # -----------------------------------------------------------------
    # Step 6: Encode all three to base64 PNG
    # -----------------------------------------------------------------
    original_b64 = _encode_image_to_base64(gray_image * 255.0)
    segmented_b64 = _encode_image_to_base64(segmented_image, normalize=True)
    binary_b64 = _encode_image_to_base64(binary_image)

    return {
        "original_image": original_b64,
        "segmented_image": segmented_b64,
        "binary_image": binary_b64,
    }
