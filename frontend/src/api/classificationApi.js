const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

/**
 * Upload an image file to the classification API.
 * @param {File} file
 * @returns {Promise<{predicted_label: string, confidence: number, class_index: number, message: string}>}
 */
export async function classifyImage(file) {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE}/api/classify`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `Server error: ${response.status}`);
  }

  return response.json();
}

/**
 * Stream 100-epoch TTA classification via SSE.
 * Async generator — yields event objects:
 *   { epoch, predicted_label, confidence, done }   — progress
 *   { error }                                       — failure from server
 *
 * @param {File} file
 * @yields {{ epoch: number, predicted_label: string, confidence: number, done: boolean } | { error: string }}
 */
export async function* classifyImageStream(file) {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE}/api/classify/stream`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || `Server error: ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop(); // keep incomplete line

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        try {
          yield JSON.parse(line.slice(6));
        } catch {
          // skip malformed line
        }
      }
    }
  }
}

/**
 * Classify an image and generate Grad-CAM heatmap + overlay.
 * Calls POST /api/classify/gradcam (single request, not streaming).
 *
 * @param {File} file
 * @returns {Promise<{
 *   predicted_label: string,
 *   confidence: number,
 *   class_index: number,
 *   gradcam_heatmap_image: string|null,
 *   gradcam_overlay_image: string|null,
 *   gradcam_available: boolean
 * }>}
 */
export async function classifyWithGradcam(file) {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE}/api/classify/gradcam`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || `Server error: ${response.status}`);
  }

  return response.json();
}

/**
 * Format bytes to human-readable string.
 */
export function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
