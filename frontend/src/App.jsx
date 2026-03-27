import { useState, useCallback } from 'react';
import './index.css';
import UploadZone from './components/UploadZone';
import ImageCard from './components/ImageCard';
import { segmentImage } from './api/segmentationApi';

const CARDS = [
  {
    type: 'original',
    title: 'Original Image',
    description: 'Grayscale input',
  },
  {
    type: 'segmented',
    title: 'Segmented Image',
    description: 'Fuzzy membership-weighted',
  },
  {
    type: 'binary',
    title: 'Binary Image',
    description: 'Otsu thresholded · Key Diagnostic Output',
    highlight: true,
  },
];

export default function App() {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState('idle'); // idle | loading | success | error
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [processedAt, setProcessedAt] = useState(null);

  const handleFileSelect = useCallback((f) => {
    setFile(f);
    setResult(null);
    setError('');
    setStatus('idle');
  }, []);

  const handleClear = useCallback(() => {
    setFile(null);
    setResult(null);
    setError('');
    setStatus('idle');
    setProcessedAt(null);
  }, []);

  const handleAnalyse = async () => {
    if (!file) return;
    setStatus('loading');
    setError('');
    setResult(null);

    try {
      const data = await segmentImage(file);
      setResult(data);
      setStatus('success');
      setProcessedAt(new Date().toLocaleTimeString());
    } catch (err) {
      setError(err.message || 'Unexpected error occurred.');
      setStatus('error');
    }
  };

  return (
    <div className="app">
      {/* ── Header ── */}
      <header className="header">
        <div className="header-inner">
          <div className="header-logo">🧬</div>
          <span className="header-title">MonkeyPox Segmentation AI</span>
          <span className="header-subtitle">Fuzzy + Otsu Pipeline</span>
        </div>
      </header>

      {/* ── Main ── */}
      <main className="main">
        {/* Hero */}
        <div className="hero">
          <h1>Disease Image Segmentation</h1>
          <p>Upload a monkeypox or skin lesion image to extract original, fuzzy-segmented, and binary diagnostic views.</p>
        </div>

        {/* Upload Zone */}
        <UploadZone
          file={file}
          onFileSelect={handleFileSelect}
          onClear={handleClear}
        />

        {/* Analyse Button */}
        {file && status !== 'loading' && (
          <button
            className="btn btn-primary btn-large"
            onClick={handleAnalyse}
            disabled={status === 'loading'}
            id="analyse-btn"
          >
            🔬 Run Segmentation Analysis
          </button>
        )}

        {/* Loading State */}
        {status === 'loading' && (
          <div style={{ marginTop: '1rem' }}>
            <button className="btn btn-primary btn-large" disabled>
              <div className="spinner" />
              Analysing image…
            </button>
            <div className="progress-bar-wrap">
              <div className="progress-bar" />
            </div>
          </div>
        )}

        {/* Error Banner */}
        {status === 'error' && (
          <div className="error-banner" style={{ marginTop: '1.25rem' }}>
            <span className="error-icon">⚠️</span>
            <div>
              <strong>Segmentation failed</strong>
              <div style={{ marginTop: 4 }}>{error}</div>
            </div>
          </div>
        )}

        {/* ── Results ── */}
        {status === 'success' && result && (
          <div className="results-section" style={{ marginTop: '2.5rem' }}>
            {/* Results Header */}
            <div className="results-header">
              <h2>Analysis Results</h2>
              {processedAt && (
                <div className="success-pill">
                  <span className="dot" />
                  Completed at {processedAt}
                </div>
              )}
            </div>

            {/* Image Cards */}
            <div className="image-grid">
              {CARDS.map(card => (
                <ImageCard
                  key={card.type}
                  type={card.type}
                  title={card.title}
                  description={card.description}
                  b64={result[`${card.type}_image`]}
                  highlight={card.highlight}
                />
              ))}
            </div>

            {/* Info Row */}
            <div className="section-divider">
              <div className="divider-line" />
              <h3>Processing Info</h3>
              <div className="divider-line" />
            </div>

            <div className="info-grid">
              <div className="info-card">
                <div className="info-card-icon">📁</div>
                <div className="info-card-text">
                  <div className="label">Source File</div>
                  <div className="value">{file.name}</div>
                </div>
              </div>
              <div className="info-card">
                <div className="info-card-icon">⚙️</div>
                <div className="info-card-text">
                  <div className="label">Algorithm</div>
                  <div className="value">Fuzzy + Otsu</div>
                </div>
              </div>
              <div className="info-card">
                <div className="info-card-icon">🎯</div>
                <div className="info-card-text">
                  <div className="label">Bins</div>
                  <div className="value">3 (Fuzzy memberships)</div>
                </div>
              </div>
              <div className="info-card">
                <div className="info-card-icon">✅</div>
                <div className="info-card-text">
                  <div className="label">Status</div>
                  <div className="value" style={{ color: 'var(--success)' }}>Complete</div>
                </div>
              </div>
            </div>

            {/* Analyse another */}
            <div style={{ textAlign: 'center', marginTop: '2rem' }}>
              <button className="btn btn-ghost" onClick={handleClear}>
                ↩ Analyse Another Image
              </button>
            </div>
          </div>
        )}
      </main>

      {/* ── Footer ── */}
      <footer className="footer">
        ML_DEMO · Fuzzy Image Segmentation API · FastAPI + React
      </footer>
    </div>
  );
}
