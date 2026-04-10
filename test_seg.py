import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.filters import gaussian, threshold_otsu, threshold_local
from skimage.morphology import disk, white_tophat, black_tophat, remove_small_objects

def _odd_window_size(size: int, minimum: int = 35) -> int:
    value = max(minimum, int(size))
    return value if value % 2 == 1 else value + 1

def analyze_image(filepath, out_dir):
    filename = os.path.basename(filepath)
    try:
        image = io.imread(filepath)
        if image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, :3]
            
        height, width = image.shape[:2]
        
        if image.ndim == 3:
            # Traditional grayscale
            gray = color.rgb2gray(image)
            
            # CIELAB color space for redness (erythema)
            # a* channel represents red-green axis. Positive = red.
            lab = color.rgb2lab(image)
            a_channel = lab[:, :, 1]
            # Normalize a_channel simply for visualization/processing
            a_norm = np.clip((a_channel - a_channel.min()) / (a_channel.max() - a_channel.min() + 1e-8), 0, 1)
            
            # For purely redness-driven diseases like measles, the a* channel 
            # is often a better "base" image than grayscale. 
            # We can combine them: bumps (gray contrast) + redness (a* intensity)
            base_img = a_norm * 0.7 + gray * 0.3 # just testing combinations
        else:
            gray = image.astype(np.float64) / 255.0
            a_norm = gray
            base_img = gray
            
        gray = np.clip(gray, 0.0, 1.0)
        base_img = np.clip(base_img, 0.0, 1.0)
        
        smoothed = gaussian(base_img, sigma=1.0, preserve_range=True)
        
        selem_radius = max(3, min(height, width) // 25)
        selem = disk(selem_radius)
        
        # We can extract bumps like before
        bright_spots = white_tophat(smoothed, selem)
        dark_spots = black_tophat(smoothed, selem)
        lesion_score = bright_spots + dark_spots
        
        # But for flattened rashes (measles), tophat doesn't catch them.
        # We also need to add back raw redness variance!
        bg_sigma = max(8.0, min(height, width) / 10.0)
        background = gaussian(smoothed, sigma=bg_sigma, preserve_range=True)
        
        # Combine structural score (tophat) with intensity score (redness outlier)
        intensity_score = np.clip(smoothed - background, 0, None)
        lesion_score = lesion_score + intensity_score*0.5
        
        relative_score = lesion_score / np.maximum(background, 1e-3)
        positive_scores = relative_score[relative_score > 0]
        
        if positive_scores.size == 0:
            mask = np.zeros_like(gray, dtype=bool)
        else:
            local_window = _odd_window_size(min(height, width) // 8, minimum=35)
            local_offset = float(np.clip(relative_score.std() * 0.12, 0.01, 0.05))
            local_thresh = threshold_local(
                relative_score, block_size=local_window, method="gaussian", offset=-local_offset
            )
            adaptive_mask = relative_score > local_thresh
            
            strong_cutoff = float(threshold_otsu(positive_scores))
            soft_cutoff = float(np.percentile(positive_scores, 45))
            
            mask = adaptive_mask & (relative_score > soft_cutoff)
            mask |= relative_score > strong_cutoff
            mask = remove_small_objects(mask, min_size=15)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(image)
        axes[0].set_title("Original")
        axes[0].axis('off')
        
        axes[1].imshow(a_norm, cmap='gray')
        axes[1].set_title("Redness (a* channel)")
        axes[1].axis('off')
        
        axes[2].imshow(relative_score, cmap='hot')
        axes[2].set_title("Combined Score map")
        axes[2].axis('off')
        
        overlay = np.stack([gray, gray, gray], axis=-1)
        overlay[mask] = [1.0, 0.0, 0.0]
        axes[3].imshow(overlay)
        axes[3].set_title("Mask Overlay")
        axes[3].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"measles_{filename}"))
        plt.close()
        
    except Exception as e:
        print(f"Failed {filename}: {e}")

img_dir = "Monkeypox Images/Measles/"
out_dir = "test_masks2"
os.makedirs(out_dir, exist_ok=True)

images = glob.glob(os.path.join(img_dir, "*.png")) + glob.glob(os.path.join(img_dir, "*.jpg"))
if len(images) > 0:
    np.random.seed(42)
    sample_size = min(5, len(images))
    sample_images = np.random.choice(images, sample_size, replace=False)

    for img_path in sample_images:
        analyze_image(img_path, out_dir)
        print(f"Processed {os.path.basename(img_path)}")
else:
    print(f"No images found in {img_dir}")

