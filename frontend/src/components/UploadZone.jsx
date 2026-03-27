import { useState, useRef, useCallback } from 'react';
import { formatBytes } from '../api/segmentationApi';

/**
 * Drag-and-drop image upload zone.
 * Props:
 *   onFileSelect(file) - called when a valid image file is chosen
 *   file - current selected file (controlled)
 *   onClear() - called when user removes the file
 */
export default function UploadZone({ onFileSelect, file, onClear }) {
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef(null);
  const previewUrl = file ? URL.createObjectURL(file) : null;

  const handleFile = useCallback((f) => {
    if (!f || !f.type.startsWith('image/')) return;
    onFileSelect(f);
  }, [onFileSelect]);

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

  if (file) {
    return (
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
              onClick={onClear}
              title="Remove image"
            >
              ✕ Remove
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div
      className={`upload-zone${dragOver ? ' drag-over' : ''}`}
      onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
    >
      <input
        ref={inputRef}
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
          <h3>{dragOver ? 'Drop your image here' : 'Upload Disease Image'}</h3>
          <p>Drag & drop or click to browse</p>
        </div>
        <div className="upload-badge">
          {['PNG', 'JPEG', 'BMP', 'TIFF'].map(fmt => (
            <span key={fmt} className="badge">{fmt}</span>
          ))}
        </div>
      </div>
    </div>
  );
}
