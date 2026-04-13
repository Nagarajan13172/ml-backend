import { useEffect, useState, useCallback } from 'react';
import './index.css';
import UploadZone from './components/UploadZone';
import ImageCard from './components/ImageCard';
import InfoModal from './components/InfoModal';
import { classifyWithGradcam } from './api/classificationApi';
import { segmentImage } from './api/segmentationApi';

const THEME_STORAGE_KEY = 'ml-demo-theme';

function getInitialTheme() {
  if (typeof window === 'undefined') {
    return 'dark';
  }

  const storedTheme = window.localStorage.getItem(THEME_STORAGE_KEY);
  if (storedTheme === 'light' || storedTheme === 'dark') {
    return storedTheme;
  }

  return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
}

function ThemeToggleIcon({ theme }) {
  if (theme === 'dark') {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path
          d="M12 4.5V2m0 20v-2.5m7.5-7.5H22m-20 0h2.5m12.803-5.303 1.768-1.768M4.929 19.071l1.768-1.768m0-10.606L4.93 4.929m14.142 14.142-1.768-1.768M12 17a5 5 0 1 0 0-10 5 5 0 0 0 0 10Z"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.8"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    );
  }

  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path
        d="M20.354 15.354A9 9 0 0 1 8.646 3.646a9 9 0 1 0 11.708 11.708Z"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export default function App() {
  const [file, setFile] = useState(null);
  const [theme, setTheme] = useState(getInitialTheme);

  // Classification state
  const [classStatus, setClassStatus] = useState('idle'); // idle|loading|success|error
  const [classResult, setClassResult] = useState(null);
  const [classError, setClassError]   = useState('');

  // Segmentation state
  const [segStatus, setSegStatus]   = useState('idle'); // idle|loading|success|error
  const [segResult, setSegResult]   = useState(null);
  const [segError, setSegError]     = useState('');
  const [infoDetails, setInfoDetails] = useState(null);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    window.localStorage.setItem(THEME_STORAGE_KEY, theme);
  }, [theme]);

  const handleFileSelect = useCallback((selectedFile) => {
    setFile(selectedFile);
    setClassStatus('idle'); setClassResult(null); setClassError('');
    setSegStatus('idle');   setSegResult(null);   setSegError('');
    setInfoDetails(null);
  }, []);

  const handleClear = useCallback(() => {
    setFile(null);
    setClassStatus('idle'); setClassResult(null); setClassError('');
    setSegStatus('idle');   setSegResult(null);   setSegError('');
    setInfoDetails(null);
  }, []);

  // ── Classify: single request → result + Grad-CAM ─────────────────────
  const handleClassify = async () => {
    if (!file) return;
    setClassStatus('loading');
    setClassResult(null);
    setClassError('');

    try {
      const data = await classifyWithGradcam(file);
      setClassResult(data);
      setClassStatus('success');
    } catch (err) {
      setClassError(err.message || 'Classification failed.');
      setClassStatus('error');
    }
  };

  // ── Segment ───────────────────────────────────────────────────────────
  const handleSegment = async () => {
    if (!file) return;
    setSegStatus('loading');
    setSegResult(null);
    setSegError('');
    setInfoDetails(null);

    try {
      const data = await segmentImage(file);
      setSegResult(data);
      setSegStatus('success');
    } catch (err) {
      setSegError(err.message || 'Segmentation failed.');
      setSegStatus('error');
    }
  };

  const nextTheme = theme === 'dark' ? 'light' : 'dark';

  return (
    <div className="app">
      <header className="header">
        <div className="header-inner">
          <div className="header-logo">AI</div>
          <span className="header-title">POXAI</span>
          <div className="header-actions">
            <span className="header-subtitle">Classification · Segmentation</span>
            <button
              type="button"
              className="theme-toggle"
              onClick={() => setTheme(nextTheme)}
              aria-label={`Switch to ${nextTheme} mode`}
              title={`Switch to ${nextTheme} mode`}
            >
              <ThemeToggleIcon theme={theme} />
            </button>
          </div>
        </div>
      </header>

      <main className="main">
        <div className="hero">
          <h1>Disease Detection &amp; Lesion Segmentation</h1>
          <p>Upload a lesion image, then run classification or segmentation independently.</p>
          {/* <div className="hero-badges">
            <span className="hero-badge">100-Epoch TTA</span>
            <span className="hero-badge">Fuzzy Logic</span>
            <span className="hero-badge">Background Removal</span>
            <span className="hero-badge">TensorFlow</span>
          </div> */}
        </div>

        <UploadZone file={file} onFileSelect={handleFileSelect} onClear={handleClear} />

        {/* ── Action buttons (only when a file is ready) ── */}
        {file && (
          <div className="action-buttons">
            <button
              className="btn btn-classify btn-large"
              onClick={handleClassify}
              disabled={classStatus === 'loading' || segStatus === 'loading'}
            >
              {classStatus === 'loading' ? (
                <><div className="spinner" /> Classifying…</>
              ) : 'Classify'}
            </button>

            <button
              className="btn btn-segment btn-large"
              onClick={handleSegment}
              disabled={segStatus === 'loading' || classStatus === 'loading'}
            >
              {segStatus === 'loading' ? (
                <><div className="spinner" /> Segmenting…</>
              ) : 'Segment'}
            </button>
          </div>
        )}

        {/* ── Classification error ── */}
        {classStatus === 'error' && (
          <div className="error-banner" style={{ marginTop: '1.25rem' }}>
            <span className="error-icon">!</span>
            <div>
              <strong>Classification failed</strong>
              <div style={{ marginTop: 4 }}>{classError}</div>
            </div>
          </div>
        )}

        {/* ── Classification result + Grad-CAM ── */}
        {classStatus === 'success' && classResult && (
          <div className="results-section" style={{ marginTop: '2rem' }}>
            <div className="results-header">
              <h2>Classification Result</h2>
              <div className="success-pill"><span className="dot" />Done</div>
            </div>
            <div className="label-card">
              <div className="label-kicker">Predicted Diagnosis</div>
              <div className="label-text">{classResult.predicted_label}</div>
            </div>

            {classResult.gradcam_available && classResult.predicted_label !== 'Normal' && (
              <div style={{ marginTop: '1.5rem' }}>
                <div className="results-header">
                  <h2>Grad-CAM</h2>
                  <div className="success-pill"><span className="dot" />Computed</div>
                </div>
                <div className="gradcam-description">
                  Regions the model focused on to predict <strong>{classResult.predicted_label}</strong>.
                  Blue = low importance · red = high importance.
                </div>
                <div className="image-grid">
                  <ImageCard
                    type="classGradcam"
                    title="Grad-CAM"
                    description="Jet colormap — standalone activation map"
                    b64={classResult.gradcam_heatmap_image}
                  />
                </div>
              </div>
            )}
          </div>
        )}

        {/* ── Segmentation error ── */}
        {segStatus === 'error' && (
          <div className="error-banner" style={{ marginTop: '1.25rem' }}>
            <span className="error-icon">!</span>
            <div>
              <strong>Segmentation failed</strong>
              <div style={{ marginTop: 4 }}>{segError}</div>
            </div>
          </div>
        )}

        {/* ── Segmentation results ── */}
        {segStatus === 'success' && segResult && (
          <div className="results-section" style={{ marginTop: '2rem' }}>
            <div className="results-header">
              <h2>Segmentation Results</h2>
              {segResult.is_normal
                ? <div className="success-pill" style={{ background: 'rgba(16,185,129,0.15)', color: '#10b981' }}><span className="dot" style={{ background: '#10b981' }} />Normal</div>
                : <div className="success-pill"><span className="dot" />Done</div>
              }
            </div>

            {segResult.is_normal ? (
              /* ── Normal skin: show original + clear message ── */
              <>
                <div className="error-banner" style={{ marginTop: '1rem', borderColor: 'rgba(16,185,129,0.3)', background: 'rgba(16,185,129,0.08)' }}>
                  <span className="error-icon" style={{ background: 'rgba(16,185,129,0.2)', color: '#10b981' }}>✓</span>
                  <div>
                    <strong style={{ color: '#10b981' }}>No lesion detected</strong>
                    <div style={{ marginTop: 4, color: 'var(--text-secondary)' }}>
                      This image was classified as <strong>Normal</strong> skin. No affected areas or lesion masks to display.
                    </div>
                  </div>
                </div>
                <div className="image-grid" style={{ marginTop: '1.25rem' }}>
                  <ImageCard
                    type="original"
                    title="Original (Grayscale)"
                    description="Input converted to grayscale"
                    b64={segResult.original_image}
                  />
                </div>
              </>
            ) : (
              /* ── Diseased: show original + binary mask ── */
              <div className="image-grid">
                <ImageCard
                  type="original"
                  title="Original (Grayscale)"
                  description="Input converted to grayscale"
                  b64={segResult.original_image}
                />
                <ImageCard
                  type="binary"
                  title="Binary Mask"
                  description="Adaptive / Otsu threshold"
                  b64={segResult.binary_image}
                  info={segResult.binary_details}
                  onInfoClick={() => setInfoDetails(segResult.binary_details)}
                />
              </div>
            )}
          </div>
        )}

        {(classStatus === 'success' || segStatus === 'success') && (
          <div style={{ textAlign: 'center', marginTop: '1.5rem' }}>
            <button className="btn btn-ghost" onClick={handleClear}>
              Analyse Another Image
            </button>
          </div>
        )}
      </main>

      <footer className="footer">
        Classification + Segmentation
      </footer>

      <InfoModal details={infoDetails} onClose={() => setInfoDetails(null)} />
    </div>
  );
}
