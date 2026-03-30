import io
import base64
import numpy as np
import skfuzzy as fuzz
from skimage import io as skio, color
from skimage.filters import threshold_otsu, threshold_local, gaussian
from skimage.morphology import (
    binary_opening, binary_closing,
    remove_small_objects, remove_small_holes,
    disk
)
from skimage.segmentation import clear_border
from PIL import Image


def _encode_image_to_base64(image_array: np.ndarray, normalize: bool = False) -> str:
    """
    Convert a numpy image array to a base64-encoded PNG string.
    Handles both grayscale (2D) and RGBA (2D with alpha already masked).
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


def _encode_rgba_to_base64(rgba_array: np.ndarray) -> str:
    """Encode an RGBA numpy array (H, W, 4) to base64 PNG (preserves transparency)."""
    pil_image = Image.fromarray(rgba_array.astype(np.uint8), mode="RGBA")
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def _clean_binary_mask(binary_mask: np.ndarray) -> np.ndarray:
    """
    Clean up a noisy binary mask using morphological operations:
      1. Lightly connect fragmented lesion pixels
      2. Remove tiny isolated noise without deleting lesion-sized blobs
      3. Fill very small holes inside the retained lesion regions
    """
    total_pixels = binary_mask.shape[0] * binary_mask.shape[1]
    min_size = max(8, total_pixels // 45000)
    hole_area = max(8, total_pixels // 50000)

    # Preserve small lesion structures by skipping an aggressive opening pass.
    cleaned = binary_closing(binary_mask, disk(1))
    cleaned = remove_small_objects(cleaned.astype(bool), min_size=min_size)
    cleaned = remove_small_holes(cleaned, area_threshold=hole_area)

    return cleaned.astype(bool)


def _odd_window_size(size: int, minimum: int = 35) -> int:
    """Return an odd-valued window size suitable for local thresholding."""
    value = max(minimum, int(size))
    return value if value % 2 == 1 else value + 1


def _build_binary_mask(gray_image: np.ndarray) -> np.ndarray:
    """
    Build a robust binary lesion mask from grayscale image.

    Strategy:
      1. Smooth image to reduce sensor noise
      2. Estimate slow-varying background illumination (large Gaussian)
      3. Build a normalized "dark lesion score" = (background - image) / background
      4. Detect locally dark pixels with adaptive thresholding
      5. Keep only locally dark pixels that also have sufficient contrast
      6. Remove border-connected artifacts and gently clean morphology
    """
    image = np.clip(gray_image.astype(np.float64), 0.0, 1.0)
    height, width = image.shape

    smoothed = gaussian(image, sigma=1.0, preserve_range=True)
    bg_sigma = max(8.0, min(height, width) / 10.0)
    background = gaussian(smoothed, sigma=bg_sigma, preserve_range=True)

    lesion_score = np.clip(background - smoothed, 0.0, None)
    if lesion_score.max() <= 1e-8:
        lesion_score = 1.0 - smoothed

    relative_score = lesion_score / np.maximum(background, 1e-3)
    positive_scores = relative_score[relative_score > 0]
    if positive_scores.size == 0:
        return np.zeros_like(image, dtype=bool)

    local_window = _odd_window_size(min(height, width) // 8, minimum=35)
    local_offset = float(np.clip(smoothed.std() * 0.12, 0.012, 0.03))
    local_threshold = threshold_local(
        smoothed,
        block_size=local_window,
        method="gaussian",
        offset=local_offset,
    )
    adaptive_mask = smoothed < local_threshold

    strong_cutoff = float(threshold_otsu(positive_scores))
    soft_cutoff = float(np.percentile(positive_scores, 45))

    binary_mask = adaptive_mask & (relative_score > soft_cutoff)
    binary_mask |= relative_score > strong_cutoff
    binary_mask = clear_border(binary_mask)
    binary_mask = _clean_binary_mask(binary_mask)

    # Safety checks: tighten or relax the mask if illumination skews it too far.
    foreground_ratio = float(binary_mask.mean())
    if foreground_ratio > 0.18:
        stricter_cutoff = float(np.percentile(positive_scores, 70))
        binary_mask = adaptive_mask & (relative_score > stricter_cutoff)
        binary_mask = clear_border(binary_mask)
        binary_mask = binary_opening(binary_mask, disk(1))
        binary_mask = _clean_binary_mask(binary_mask)
    elif foreground_ratio < 0.004:
        relaxed_cutoff = float(np.percentile(positive_scores, 35))
        binary_mask = adaptive_mask & (relative_score > relaxed_cutoff)
        binary_mask |= relative_score > (strong_cutoff * 0.85)
        binary_mask = clear_border(binary_mask)
        binary_mask = _clean_binary_mask(binary_mask)

    return binary_mask.astype(bool)


def process_image(file_bytes: bytes) -> dict:
    """
    Run fuzzy image segmentation + hybrid local/Otsu lesion masking on raw image bytes.

    Pipeline:
        1. Decode bytes → grayscale float [0, 1]
        2. Compute histogram (3 bins)
        3. Build fuzzy membership functions (trimf)
        4. Apply fuzzy segmentation
        5. Build a hybrid adaptive/Otsu binary lesion mask
        6. Remove background → transparent RGBA masked image
        7. Encode all outputs as base64 PNG strings

    Returns:
        dict with keys: original_image, segmented_image, binary_image, masked_image
    """
    # -----------------------------------------------------------------
    # Step 1: Decode image → grayscale float [0, 1]
    # -----------------------------------------------------------------
    image = skio.imread(io.BytesIO(file_bytes), as_gray=False)

    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]

    if image.ndim == 3:
        gray_image = color.rgb2gray(image)      # float [0, 1]
    else:
        gray_image = image.astype(np.float64)
        if gray_image.max() > 1.0:
            gray_image /= 255.0

    # -----------------------------------------------------------------
    # Step 2: Generate histogram (3 bins)
    # -----------------------------------------------------------------
    hist, bin_edges = np.histogram(gray_image, bins=3)

    # -----------------------------------------------------------------
    # Step 3: Build fuzzy membership functions
    # -----------------------------------------------------------------
    level = 0.5
    x_range = np.arange(0, 256)
    membership_functions = []

    for i in range(len(bin_edges) - 1):
        lo  = bin_edges[i] * 255 - level
        mid = bin_edges[i] * 255
        hi  = bin_edges[i + 1] * 255 + level
        membership_functions.append(fuzz.trimf(x_range, [lo, mid, hi]))

    # -----------------------------------------------------------------
    # Step 4: Apply fuzzy segmentation
    # -----------------------------------------------------------------
    img_scaled = (gray_image * 255).astype(np.float64)
    segmented_image = np.zeros_like(img_scaled)

    for i, mf in enumerate(membership_functions):
        membership_vals = fuzz.interp_membership(x_range, mf, img_scaled)
        segmented_image += membership_vals * (i + 1)

    # -----------------------------------------------------------------
    # Step 5: Build robust binary mask directly from grayscale image
    # -----------------------------------------------------------------
    clean_binary = _build_binary_mask(gray_image)

    binary_uint8 = clean_binary.astype(np.uint8) * 255

    # -----------------------------------------------------------------
    # Step 6: Background removal → RGBA masked image
    #   White pixels stay where the mask is TRUE, background → transparent
    # -----------------------------------------------------------------
    gray_uint8 = np.clip(gray_image * 255, 0, 255).astype(np.uint8)
    alpha_channel = binary_uint8  # 255 = keep, 0 = remove

    rgba = np.stack([
        gray_uint8,
        gray_uint8,
        gray_uint8,
        alpha_channel,
    ], axis=2)  # shape (H, W, 4)

    # -----------------------------------------------------------------
    # Step 7: Encode everything
    # -----------------------------------------------------------------
    return {
        "original_image":   _encode_image_to_base64(gray_image * 255.0),
        "segmented_image":  _encode_image_to_base64(segmented_image, normalize=True),
        "binary_image":     _encode_image_to_base64(binary_uint8),
        "masked_image":     _encode_rgba_to_base64(rgba),
    }
