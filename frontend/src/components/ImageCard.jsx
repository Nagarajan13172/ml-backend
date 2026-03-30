import { downloadBase64Image } from '../api/segmentationApi';

const CONFIG = {
  original:     { icon: '🔬', iconClass: 'original',  label: null },
  segmented:    { icon: '🌈', iconClass: 'segmented', label: null },
  binary:       { icon: '⬛', iconClass: 'binary',    label: 'CLEANED' },
  masked:       { icon: '✂️', iconClass: 'masked',    label: 'KEY OUTPUT' },
  gradcam:      { icon: '🔥', iconClass: 'gradcam',   label: 'HEATMAP' },
  gradcamBands: { icon: '🎯', iconClass: 'gradcam-bands', label: '3 BANDS' },
};

function InfoIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path
        d="M12 10.25v5.25m0-8.75h.01M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18Z"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

/**
 * A single result image card with title, image, and download button.
 * Props:
 *   type        - 'original' | 'segmented' | 'binary' | 'masked' | 'gradcam' | 'gradcamBands'
 *   title       - display title
 *   description - subtitle
 *   b64         - base64-encoded PNG string
 *   info        - optional metadata object to describe the image
 *   onInfoClick - optional click handler for opening an info modal
 */
export default function ImageCard({ type, title, description, b64, info, onInfoClick }) {
  const { icon, iconClass, label } = CONFIG[type] || CONFIG.original;
  const isBinary  = type === 'binary';
  const isMasked  = type === 'masked';
  const isGradcam = type === 'gradcam' || type === 'gradcamBands';
  const hasInfo = Boolean(info && onInfoClick);
  const src = `data:image/png;base64,${b64}`;

  function handleDownload() {
    downloadBase64Image(b64, `${type}_image.png`);
  }

  return (
    <div className={`image-card${isBinary ? ' binary-card' : ''}${isMasked ? ' masked-card' : ''}${isGradcam ? ' gradcam-card' : ''}`}>
      {/* Card Header */}
      <div className="card-header">
        <div className={`card-icon ${iconClass}`}>{icon}</div>
        <div className="card-copy">
          <div className="card-title">{title}</div>
          <div className="card-meta">{description}</div>
        </div>
        <div className="card-actions">
          {hasInfo && (
            <button
              type="button"
              className="card-info-button"
              onClick={onInfoClick}
              aria-label={`Show details for ${title}`}
              title={`Show details for ${title}`}
            >
              <InfoIcon />
            </button>
          )}
          {label && (
            <span className={`binary-tag${isMasked ? ' masked-tag' : ''}`}>{label}</span>
          )}
        </div>
      </div>

      {/* Image — transparent bg for masked output */}
      <div className={`card-img-wrap${isMasked ? ' transparent-bg' : ''}`}>
        <img src={src} alt={title} />
      </div>

      {/* Footer */}
      <div className="card-footer">
        <span>{isMasked ? 'PNG · Transparent' : isGradcam ? 'PNG · Color' : 'PNG · Grayscale'}</span>
        <button
          onClick={handleDownload}
          className={`download-btn${isBinary ? ' binary' : ''}${isMasked ? ' masked' : ''}${isGradcam ? ' gradcam' : ''}`}
          title={`Download ${title}`}
        >
          ⬇ Download
        </button>
      </div>
    </div>
  );
}
