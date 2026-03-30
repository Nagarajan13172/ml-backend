import { useEffect } from 'react';

function CloseIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path
        d="M6 6l12 12M18 6 6 18"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function formatNumber(value) {
  return typeof value === 'number' ? value.toLocaleString() : value;
}

function formatMilliseconds(value) {
  return typeof value === 'number' ? `${value.toFixed(4)} ms` : value;
}

export default function InfoModal({ details, onClose }) {
  useEffect(() => {
    if (!details) {
      return undefined;
    }

    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';

    function handleKeyDown(event) {
      if (event.key === 'Escape') {
        onClose();
      }
    }

    window.addEventListener('keydown', handleKeyDown);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [details, onClose]);

  if (!details) {
    return null;
  }

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal-panel"
        role="dialog"
        aria-modal="true"
        aria-labelledby="binary-details-title"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="modal-header">
          <div>
            <div className="modal-kicker">Segmentation Insight</div>
            <h3 id="binary-details-title">{details.title}</h3>
          </div>
          <button
            type="button"
            className="modal-close"
            onClick={onClose}
            aria-label="Close binary mask details"
            title="Close"
          >
            <CloseIcon />
          </button>
        </div>

        <p className="modal-description">{details.description}</p>

        <div className="modal-grid">
          <div className="modal-stat">
            <span className="modal-stat-label">Average Filtering Time</span>
            <strong>{formatMilliseconds(details.average_filtering_time_ms)}</strong>
          </div>
          <div className="modal-stat">
            <span className="modal-stat-label">Resolution</span>
            <strong>{details.width} x {details.height}</strong>
          </div>
          <div className="modal-stat">
            <span className="modal-stat-label">Pixels</span>
            <strong>{formatNumber(details.pixel_count)}</strong>
          </div>
        </div>

        <div className="modal-note">{details.timing_note}</div>
      </div>
    </div>
  );
}
