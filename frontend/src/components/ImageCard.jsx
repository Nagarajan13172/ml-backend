import { downloadBase64Image } from '../api/segmentationApi';

/**
 * A single result image card with title, image, and download button.
 */
export default function ImageCard({ type, title, description, b64, highlight }) {
  const icons = { original: '🔬', segmented: '🌈', binary: '⬛' };
  const iconClass = { original: 'original', segmented: 'segmented', binary: 'binary' };
  const src = `data:image/png;base64,${b64}`;

  function handleDownload() {
    downloadBase64Image(b64, `${type}_image.png`);
  }

  const isBinary = type === 'binary';

  return (
    <div className={`image-card${isBinary ? ' binary-card' : ''}${highlight ? ' highlight' : ''}`}>
      {/* Card Header */}
      <div className="card-header">
        <div className={`card-icon ${iconClass[type]}`}>{icons[type]}</div>
        <div>
          <div className="card-title">{title}</div>
          <div className="card-meta">{description}</div>
        </div>
        {isBinary && <span className="binary-tag">Key Output</span>}
      </div>

      {/* Image */}
      <div className="card-img-wrap">
        <img src={src} alt={title} />
      </div>

      {/* Footer */}
      <div className="card-footer">
        <span>PNG · Processed</span>
        <button
          onClick={handleDownload}
          className={`download-btn${isBinary ? ' binary' : ''}`}
          title={`Download ${title}`}
        >
          ⬇ Download
        </button>
      </div>
    </div>
  );
}
