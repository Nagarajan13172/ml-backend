import { downloadBase64Image } from '../api/segmentationApi';

const CONFIG = {
  original:  { icon: '🔬', iconClass: 'original',  label: null },
  segmented: { icon: '🌈', iconClass: 'segmented', label: null },
  binary:    { icon: '⬛', iconClass: 'binary',    label: 'CLEANED' },
  masked:    { icon: '✂️', iconClass: 'masked',    label: 'KEY OUTPUT' },
};

/**
 * A single result image card with title, image, and download button.
 * Props:
 *   type        - 'original' | 'segmented' | 'binary' | 'masked'
 *   title       - display title
 *   description - subtitle
 *   b64         - base64-encoded PNG string
 */
export default function ImageCard({ type, title, description, b64 }) {
  const { icon, iconClass, label } = CONFIG[type] || CONFIG.original;
  const isBinary  = type === 'binary';
  const isMasked  = type === 'masked';
  const src = `data:image/png;base64,${b64}`;

  function handleDownload() {
    downloadBase64Image(b64, `${type}_image.png`);
  }

  return (
    <div className={`image-card${isBinary ? ' binary-card' : ''}${isMasked ? ' masked-card' : ''}`}>
      {/* Card Header */}
      <div className="card-header">
        <div className={`card-icon ${iconClass}`}>{icon}</div>
        <div>
          <div className="card-title">{title}</div>
          <div className="card-meta">{description}</div>
        </div>
        {label && (
          <span className={`binary-tag${isMasked ? ' masked-tag' : ''}`}>{label}</span>
        )}
      </div>

      {/* Image — transparent bg for masked output */}
      <div className={`card-img-wrap${isMasked ? ' transparent-bg' : ''}`}>
        <img src={src} alt={title} />
      </div>

      {/* Footer */}
      <div className="card-footer">
        <span>{isMasked ? 'PNG · Transparent' : 'PNG · Grayscale'}</span>
        <button
          onClick={handleDownload}
          className={`download-btn${isBinary ? ' binary' : ''}${isMasked ? ' masked' : ''}`}
          title={`Download ${title}`}
        >
          ⬇ Download
        </button>
      </div>
    </div>
  );
}
