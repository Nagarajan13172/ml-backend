import { useState, useRef, useCallback, useEffect, useImperativeHandle, forwardRef } from 'react';
import { formatBytes } from '../api/classificationApi';

/**
 * Drag-and-drop image upload zone.
 * Props:
 *   onFileSelect(file) - called when a valid image file is chosen
 *   file - current selected file (controlled)
 *   onClear() - called when user removes the file
 */
const UploadZone = forwardRef(function UploadZone(
  { onFileSelect, onScanCapture, file, onClear },
  ref
) {
  const [dragOver, setDragOver] = useState(false);
  const [isCameraOpen, setIsCameraOpen] = useState(false);
  const [cameraError, setCameraError] = useState('');
  const [isCapturing, setIsCapturing] = useState(false);
  const uploadInputRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const previewUrl = file ? URL.createObjectURL(file) : null;

  const handleFile = useCallback((f) => {
    if (!f || !f.type.startsWith('image/')) return;
    onFileSelect(f);
  }, [onFileSelect]);

  const handleScanFile = useCallback((f) => {
    if (!f || !f.type.startsWith('image/')) return;
    if (onScanCapture) {
      onScanCapture(f);
    } else {
      onFileSelect(f);
    }
  }, [onFileSelect, onScanCapture]);

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files?.[0];
    handleFile(f);
  };

  const handleChange = (e) => {
    const f = e.target.files?.[0];
    handleFile(f);
    // Reset input so same file can be re-selected
    e.target.value = '';
  };

  const openUploadPicker = useCallback(() => {
    uploadInputRef.current?.click();
  }, []);

  const openCamera = useCallback((e) => {
    if (e?.stopPropagation) {
      e.stopPropagation();
    }
    setCameraError('');
    setIsCameraOpen(true);
  }, []);

  const closeCamera = useCallback(() => {
    setIsCameraOpen(false);
    setCameraError('');
  }, []);

  useEffect(() => {
    if (!isCameraOpen) {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }
      return;
    }

    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraError('Camera access is not supported in this browser.');
      return;
    }

    let cancelled = false;

    navigator.mediaDevices
      .getUserMedia({ video: { facingMode: { ideal: 'environment' } }, audio: false })
      .then((stream) => {
        if (cancelled) return;
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.setAttribute('playsinline', 'true');
          videoRef.current.setAttribute('muted', 'true');
          videoRef.current.onloadedmetadata = () => {
            videoRef.current?.play().catch(() => undefined);
          };
        }
      })
      .catch(() => {
        if (cancelled) return;
        setCameraError('Unable to access the camera. Check permissions.');
      });

    return () => {
      cancelled = true;
    };
  }, [isCameraOpen]);

  const captureFromCamera = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const width = video.videoWidth || 1280;
    const height = video.videoHeight || 720;
    const squareSize = Math.min(width, height);
    const sx = Math.max(0, Math.floor((width - squareSize) / 2));
    const sy = Math.max(0, Math.floor((height - squareSize) / 2));

    canvas.width = squareSize;
    canvas.height = squareSize;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    setIsCapturing(true);
    ctx.drawImage(video, sx, sy, squareSize, squareSize, 0, 0, squareSize, squareSize);

    canvas.toBlob((blob) => {
      setIsCapturing(false);
      if (!blob) return;
      const fileName = `scan-${Date.now()}.jpg`;
      const scannedFile = new File([blob], fileName, { type: 'image/jpeg' });
      handleScanFile(scannedFile);
      closeCamera();
    }, 'image/jpeg', 0.92);
  }, [closeCamera, handleScanFile]);

  useImperativeHandle(ref, () => ({
    openCamera,
    closeCamera,
  }), [openCamera, closeCamera]);

  return (
    <>
      {file ? (
        <div className="upload-zone has-file">
          <div className="preview-wrapper">
            <img src={previewUrl} alt="preview" className="preview-thumb" />
            <div className="preview-info">
              <div className="fname">{file.name}</div>
              <div className="fsize">{formatBytes(file.size)} · {file.type}</div>
            </div>
            <div className="preview-actions">
              <button
                className="btn btn-ghost"
                onClick={openCamera}
                title="Capture a new image"
                type="button"
              >
                📷 Scan again
              </button>
              <button
                className="btn btn-ghost"
                onClick={onClear}
                title="Remove image"
                type="button"
              >
                ✕ Remove
              </button>
            </div>
          </div>
        </div>
      ) : (
        <div
          className={`upload-zone${dragOver ? ' drag-over' : ''}`}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={openUploadPicker}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault();
              openUploadPicker();
            }
          }}
        >
          <input
            ref={uploadInputRef}
            className="upload-input"
            type="file"
            accept="image/png,image/jpeg,image/bmp,image/tiff"
            onChange={handleChange}
            id="image-upload-input"
          />
          <div className="upload-content">
            <div className="upload-icon">
              {dragOver ? '📂' : '🖼️'}
            </div>
            <div className="upload-text" style={{ textAlign: 'center' }}>
              <h3>{dragOver ? 'Drop your image here' : 'Upload Lesion Image'}</h3>
              <p>Drag, drop, or click to upload, or scan with your camera.</p>
            </div>
            <div className="upload-badge">
              {['PNG', 'JPEG', 'BMP', 'TIFF'].map(fmt => (
                <span key={fmt} className="badge">{fmt}</span>
              ))}
            </div>
          </div>
        </div>
      )}
      {isCameraOpen && (
        <div className="camera-backdrop" onClick={closeCamera}>
          <div className="camera-panel" onClick={(e) => e.stopPropagation()}>
            <div className="camera-header">
              <div>
                <div className="camera-kicker">Live Scan</div>
                <div className="camera-title">Position the lesion in frame</div>
              </div>
              <button className="camera-close" type="button" onClick={closeCamera}>
                ✕
              </button>
            </div>
            <div className="camera-stage">
              {cameraError ? (
                <div className="camera-error">{cameraError}</div>
              ) : (
                <>
                  <video ref={videoRef} autoPlay playsInline muted disablePictureInPicture />
                  <div className="camera-overlay">
                    <div className="camera-guide">
                      <span className="camera-guide-dot" />
                      Center the lesion in the frame
                    </div>
                  </div>
                </>
              )}
            </div>
            <div className="camera-actions">
              <button
                className="btn btn-ghost"
                type="button"
                onClick={closeCamera}
              >
                Cancel
              </button>
              <button
                className="camera-shutter"
                type="button"
                onClick={captureFromCamera}
                disabled={!!cameraError || isCapturing}
                aria-label="Capture photo"
              >
                <span className="camera-shutter-ring" />
              </button>
              <button
                className="btn btn-primary"
                type="button"
                onClick={captureFromCamera}
                disabled={!!cameraError || isCapturing}
              >
                {isCapturing ? 'Capturing…' : 'Capture'}
              </button>
            </div>
            <canvas ref={canvasRef} className="camera-canvas" />
          </div>
        </div>
      )}
    </>
  );
});

export default UploadZone;
