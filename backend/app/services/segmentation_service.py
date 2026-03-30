import base64
import io
from time import perf_counter

import numpy as np
import skfuzzy as fuzz
from PIL import Image
from skimage import color, io as skio
from skimage.filters import gaussian, threshold_local, threshold_otsu
from skimage.morphology import (
    binary_closing,
    binary_opening,
    disk,
    remove_small_holes,
    remove_small_objects,
)
from skimage.segmentation import clear_border


GRADCAM_BANDS = (
    {
        "key": "low",
        "label": "Low affected area",
        "color": "#22c55e",
        "threshold_min": 0.35,
        "threshold_max": 0.58,
        "rgb": (34, 197, 94),
        "overlay_alpha": 0.36,
    },
    {
        "key": "medium",
        "label": "Medium affected area",
        "color": "#facc15",
        "threshold_min": 0.58,
        "threshold_max": 0.78,
        "rgb": (250, 204, 21),
        "overlay_alpha": 0.56,
    },
    {
        "key": "high",
        "label": "High affected area",
        "color": "#ef4444",
        "threshold_min": 0.78,
        "threshold_max": 1.00,
        "rgb": (239, 68, 68),
        "overlay_alpha": 0.72,
    },
)


def _encode_image_to_base64(image_array: np.ndarray, normalize: bool = False) -> str:
    """
    Convert a numpy image array to a base64-encoded PNG string.
    Handles both grayscale (2D) and RGB/RGBA images.
    """
    arr = image_array.copy().astype(np.float64)

    if normalize:
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max - arr_min > 0:
            arr = (arr - arr_min) / (arr_max - arr_min) * 255.0
        else:
            arr = np.zeros_like(arr)

    arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.ndim == 2:
        pil_image = Image.fromarray(arr)
    elif arr.ndim == 3 and arr.shape[2] in {3, 4}:
        pil_image = Image.fromarray(arr)
    else:
        raise ValueError("Unsupported image shape for base64 encoding.")
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def _encode_rgba_to_base64(rgba_array: np.ndarray) -> str:
    """Encode an RGBA numpy array (H, W, 4) to base64 PNG (preserves transparency)."""
    pil_image = Image.fromarray(rgba_array.astype(np.uint8))
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

    cleaned = binary_closing(binary_mask, disk(1))
    cleaned = remove_small_objects(cleaned.astype(bool), min_size=min_size)
    cleaned = remove_small_holes(cleaned, area_threshold=hole_area)

    return cleaned.astype(bool)


def _odd_window_size(size: int, minimum: int = 35) -> int:
    """Return an odd-valued window size suitable for local thresholding."""
    value = max(minimum, int(size))
    return value if value % 2 == 1 else value + 1


def _compute_lesion_score(gray_image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the dark-lesion score map used by the segmentation pipeline.

    Returns:
        smoothed image,
        relative lesion score,
        positive score samples
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

    return smoothed, relative_score, positive_scores


def _build_binary_mask_from_score(
    smoothed: np.ndarray,
    relative_score: np.ndarray,
    positive_scores: np.ndarray,
) -> np.ndarray:
    """Build a robust binary lesion mask from a precomputed lesion score."""
    if positive_scores.size == 0:
        return np.zeros_like(smoothed, dtype=bool)

    height, width = smoothed.shape
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


def _build_binary_mask(gray_image: np.ndarray) -> np.ndarray:
    """
    Build a robust binary lesion mask from grayscale image.

    Strategy:
      1. Smooth image to reduce sensor noise
      2. Estimate slow-varying background illumination (large Gaussian)
      3. Build a normalized dark-lesion score
      4. Detect locally dark pixels with adaptive thresholding
      5. Combine score and adaptive thresholding
      6. Clean morphology and suppress border-connected artifacts
    """
    smoothed, relative_score, positive_scores = _compute_lesion_score(gray_image)
    return _build_binary_mask_from_score(smoothed, relative_score, positive_scores)


def _normalize_attention_score(
    relative_score: np.ndarray,
    positive_scores: np.ndarray,
    binary_mask: np.ndarray,
) -> np.ndarray:
    """Normalize the lesion score into a stable 0..1 attention map."""
    if positive_scores.size == 0:
        return np.zeros_like(relative_score, dtype=np.float32)

    lesion_support = binary_closing(binary_mask.astype(bool), disk(1))
    lesion_support = remove_small_objects(
        lesion_support,
        min_size=max(10, int(binary_mask.size // 50000)),
    )

    if not np.any(lesion_support):
        return np.zeros_like(relative_score, dtype=np.float32)

    support_scores = relative_score[lesion_support]

    low = float(np.percentile(support_scores, 20))
    high = float(np.percentile(support_scores, 99))
    if high - low <= 1e-8:
        normalized = relative_score / (relative_score.max() + 1e-8)
    else:
        normalized = np.clip((relative_score - low) / (high - low), 0.0, 1.0)

    normalized = normalized.astype(np.float32)

    normalized *= lesion_support.astype(np.float32)

    normalized = gaussian(normalized, sigma=0.7, preserve_range=True).astype(np.float32)

    max_value = float(normalized.max())
    if max_value > 1e-8:
        normalized = normalized / max_value

    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def _build_gradcam_visuals(
    gray_image: np.ndarray,
    attention_map: np.ndarray,
    binary_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build an RGB overlay and a discrete three-band map from the normalized attention map.
    """
    base_gray = np.clip(gray_image * 255.0, 0, 255).astype(np.uint8)
    base_rgb = np.stack([base_gray, base_gray, base_gray], axis=2)

    overlay = base_rgb.astype(np.float32).copy()
    banded = np.zeros_like(base_rgb, dtype=np.uint8)

    lesion_support = binary_closing(binary_mask.astype(bool), disk(1))
    lesion_support = remove_small_objects(
        lesion_support,
        min_size=max(10, int(binary_mask.size // 50000)),
    )

    if not np.any(lesion_support):
        return np.clip(overlay, 0, 255).astype(np.uint8), banded

    support_values = attention_map[lesion_support]
    medium_cutoff = float(np.percentile(support_values, 60))
    high_cutoff = float(np.percentile(support_values, 85))

    medium_cutoff = float(np.clip(medium_cutoff, 0.18, 0.75))
    high_cutoff = float(np.clip(high_cutoff, medium_cutoff + 0.05, 0.92))

    low_mask = lesion_support & (attention_map < medium_cutoff)
    medium_mask = lesion_support & (attention_map >= medium_cutoff) & (attention_map < high_cutoff)
    high_mask = lesion_support & (attention_map >= high_cutoff)

    band_masks = {
        "low": low_mask,
        "medium": medium_mask,
        "high": high_mask,
    }

    for band in GRADCAM_BANDS:
        mask = band_masks.get(band["key"])

        if mask is None or not np.any(mask):
            continue

        color = np.asarray(band["rgb"], dtype=np.uint8)
        alpha = float(band["overlay_alpha"])
        banded[mask] = color
        overlay[mask] = overlay[mask] * (1.0 - alpha) + color * alpha

    return np.clip(overlay, 0, 255).astype(np.uint8), banded


def _build_binary_details(gray_image: np.ndarray, average_filtering_time_ms: float) -> dict:
    """Build explanatory metadata for the binary mask result."""
    height, width = gray_image.shape
    pixel_count = int(height * width)

    return {
        "title": "Binary Mask Details",
        "description": (
            "The binary mask is created by smoothing the grayscale lesion image, "
            "measuring darker-than-background regions, combining adaptive and Otsu "
            "thresholding, and then cleaning the mask with light morphology."
        ),
        "average_filtering_time_ms": round(float(average_filtering_time_ms), 4),
        "timing_note": (
            "Measured on this upload for the binary-mask filtering stage. "
            f"This sample is {width} x {height}, for a total of {pixel_count:,} pixels."
        ),
        "width": int(width),
        "height": int(height),
        "pixel_count": pixel_count,
    }


def _build_gradcam_details(gray_image: np.ndarray, average_filtering_time_ms: float) -> dict:
    """Build explanatory metadata for the three-band segmentation attention map."""
    height, width = gray_image.shape
    pixel_count = int(height * width)

    return {
        "title": "Segmentation Grad-CAM Details",
        "description": (
            "This three-band attention map is derived from the lesion score used by the "
            "segmentation pipeline. Green marks lightly affected pixels, yellow marks "
            "medium-strength lesion evidence, and red marks the strongest affected areas."
        ),
        "average_filtering_time_ms": round(float(average_filtering_time_ms), 4),
        "timing_note": (
            "Measured on this upload for the segmentation attention-map stage. "
            f"This sample is {width} x {height}, for a total of {pixel_count:,} pixels."
        ),
        "width": int(width),
        "height": int(height),
        "pixel_count": pixel_count,
    }


def process_image(file_bytes: bytes) -> dict:
    """
    Run fuzzy image segmentation + hybrid local/Otsu lesion masking on raw image bytes.

    Pipeline:
        1. Decode bytes → grayscale float [0, 1]
        2. Compute histogram (3 bins)
        3. Build fuzzy membership functions (trimf)
        4. Apply fuzzy segmentation
        5. Build a hybrid adaptive/Otsu binary lesion mask
        6. Build a three-band segmentation Grad-CAM-style attention map
        7. Remove background → transparent RGBA masked image
        8. Encode all outputs as base64 PNG strings

    Returns:
        dict with keys: original_image, segmented_image, binary_image,
        gradcam_overlay_image, gradcam_banded_image, masked_image,
        binary_details, gradcam_details
    """
    image = skio.imread(io.BytesIO(file_bytes), as_gray=False)

    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]

    if image.ndim == 3:
        gray_image = color.rgb2gray(image)
    else:
        gray_image = image.astype(np.float64)
        if gray_image.max() > 1.0:
            gray_image /= 255.0

    hist, bin_edges = np.histogram(gray_image, bins=3)
    del hist

    level = 0.5
    x_range = np.arange(0, 256)
    membership_functions = []

    for index in range(len(bin_edges) - 1):
        lo = bin_edges[index] * 255 - level
        mid = bin_edges[index] * 255
        hi = bin_edges[index + 1] * 255 + level
        membership_functions.append(fuzz.trimf(x_range, [lo, mid, hi]))

    img_scaled = (gray_image * 255).astype(np.float64)
    segmented_image = np.zeros_like(img_scaled)

    for index, membership_function in enumerate(membership_functions):
        membership_vals = fuzz.interp_membership(x_range, membership_function, img_scaled)
        segmented_image += membership_vals * (index + 1)

    binary_start = perf_counter()
    smoothed, relative_score, positive_scores = _compute_lesion_score(gray_image)
    clean_binary = _build_binary_mask_from_score(smoothed, relative_score, positive_scores)
    average_filtering_time_ms = (perf_counter() - binary_start) * 1000.0

    gradcam_start = perf_counter()
    attention_map = _normalize_attention_score(relative_score, positive_scores, clean_binary)
    gradcam_overlay, gradcam_banded = _build_gradcam_visuals(gray_image, attention_map, clean_binary)
    average_gradcam_time_ms = (perf_counter() - gradcam_start) * 1000.0

    binary_uint8 = clean_binary.astype(np.uint8) * 255
    gray_uint8 = np.clip(gray_image * 255, 0, 255).astype(np.uint8)
    alpha_channel = binary_uint8

    rgba = np.stack([
        gray_uint8,
        gray_uint8,
        gray_uint8,
        alpha_channel,
    ], axis=2)

    return {
        "original_image": _encode_image_to_base64(gray_image * 255.0),
        "segmented_image": _encode_image_to_base64(segmented_image, normalize=True),
        "binary_image": _encode_image_to_base64(binary_uint8),
        "gradcam_overlay_image": _encode_image_to_base64(gradcam_overlay),
        "gradcam_banded_image": _encode_image_to_base64(gradcam_banded),
        "masked_image": _encode_rgba_to_base64(rgba),
        "binary_details": _build_binary_details(gray_image, average_filtering_time_ms),
        "gradcam_details": _build_gradcam_details(gray_image, average_gradcam_time_ms),
    }
