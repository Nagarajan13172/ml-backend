import { useEffect, useMemo } from 'react';
import { formatBytes } from '../api/classificationApi';

function formatLabel(label) {
  return label
    .replace(/[_-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function getTone(label) {
  const normalized = label.toLowerCase();

  if (/(normal|healthy|negative|benign)/.test(normalized)) {
    return 'normal';
  }

  if (/(monkeypox|mpox|positive|infected)/.test(normalized)) {
    return 'alert';
  }

  return 'review';
}

function getSupportText(tone) {
  if (tone === 'normal') {
    return 'The uploaded image matched most strongly with a normal or non-infected class.';
  }

  if (tone === 'alert') {
    return 'The uploaded image matched most strongly with a disease-positive class in the trained model.';
  }

  return 'The classifier found this label to be the closest match for the uploaded image.';
}

export default function ClassificationResult({ file, result, processedAt }) {
  const previewUrl = useMemo(() => (file ? URL.createObjectURL(file) : ''), [file]);

  useEffect(() => (
    () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    }
  ), [previewUrl]);

  const confidence = Math.max(0, Math.min(100, Math.round((result.confidence || 0) * 100)));
  const displayLabel = formatLabel(result.predicted_label || 'Unknown');
  const tone = getTone(displayLabel);

  return (
    <div className="classification-layout">
      <div className="preview-panel">
        <div className="preview-panel-header">
          <div>
            <div className="panel-title">Uploaded Image</div>
            <div className="panel-subtitle">The same image sent to the FastAPI classifier</div>
          </div>
          <span className="panel-chip">Input</span>
        </div>

        <div className="preview-panel-image">
          <img src={previewUrl} alt={file?.name || 'Uploaded preview'} />
        </div>

        <div className="preview-panel-footer">
          <div>
            <div className="file-label">{file?.name}</div>
            <div className="file-meta">{formatBytes(file?.size || 0)} · {file?.type || 'image/*'}</div>
          </div>
          {processedAt && <span className="mini-chip">Processed at {processedAt}</span>}
        </div>
      </div>

      <div className={`diagnosis-panel diagnosis-panel-${tone}`}>
        <div className="diagnosis-kicker">Predicted Label</div>
        <h2 className="diagnosis-label">{displayLabel}</h2>
        <p className="diagnosis-text">{getSupportText(tone)}</p>

        <div className="confidence-card">
          <div className="confidence-row">
            <span>Model confidence</span>
            <strong>{confidence}%</strong>
          </div>
          <div className="confidence-track" aria-hidden="true">
            <div className={`confidence-fill confidence-fill-${tone}`} style={{ width: `${Math.max(confidence, 8)}%` }} />
          </div>
        </div>

        <div className="diagnosis-note">
          This screen intentionally keeps the result simple, with the label as the main outcome instead of charts or training plots.
        </div>
      </div>
    </div>
  );
}
