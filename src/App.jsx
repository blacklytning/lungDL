import './App.css'
import { useState, useRef } from 'react'

function prettyProb(p) {
  return (p * 100).toFixed(1) + '%'
}

export default function App() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const inputRef = useRef(null)

  async function uploadImage() {
    if (!file) return
    setLoading(true)
    setResult(null)
    const form = new FormData()
    form.append('file', file)

    try {
        const BASE = (import.meta.env && import.meta.env.VITE_API_BASE) ? import.meta.env.VITE_API_BASE : 'http://localhost:8000'
        const res = await fetch(`${BASE.replace(/\/$/, '')}/predict`, { method: 'POST', body: form })
      if (!res.ok) throw new Error('upload failed')
      const data = await res.json()
      setResult(data)
    } catch (err) {
      setResult({ error: String(err) })
    } finally {
      setLoading(false)
    }
  }

  function onFileChange(e) {
    const f = e.target.files[0]
    if (!f) return
    setFile(f)
    setPreview(URL.createObjectURL(f))
    setResult(null)
  }

  function clearAll() {
    setFile(null)
    setPreview(null)
    setResult(null)
    inputRef.current.value = null
  }

  return (
    <div className="app-root">
      <header className="app-header">
        <h1>Lung X-ray Grad-CAM</h1>
        <p className="subtitle">Upload an X-ray to get a model prediction and Grad-CAM visualization</p>
      </header>

      <main className="card">
        <section className="upload-area">
          <label className="file-input" htmlFor="file">
            <div className="dropzone">
              {preview ? (
                <img src={preview} alt="preview" className="preview" />
              ) : (
                <div className="placeholder">
                  <strong>Choose an image</strong>
                  <span>or drag & drop here</span>
                </div>
              )}
            </div>
          </label>
          <input id="file" ref={inputRef} type="file" accept="image/*" onChange={onFileChange} hidden />

          <div className="controls">
            <button className="btn primary" onClick={uploadImage} disabled={!file || loading}>
              {loading ? 'Processing...' : 'Analyze'}
            </button>
            <button className="btn" onClick={clearAll} disabled={loading}>
              Clear
            </button>
          </div>
        </section>

        <section className="results">
          {result ? (
            result.error ? (
              <div className="error">{result.error}</div>
            ) : (
              <div className="result-grid">
                <div className="predictions">
                  <h3>Prediction</h3>
                  <div className="pred-box">
                    <div className="pred-index">#{result.prediction_index}</div>
                    <div className="pred-prob">Top: {prettyProb(Math.max(...(result.probabilities || [0])))}</div>
                      {result.prediction_label ? <div className="label-name">{result.prediction_label}</div> : null}
                    </div>

                  <h4>Probabilities</h4>
                  <ul className="prob-list">
                    {(result.probabilities || []).map((p, i) => {
                      const label = result.labels?.[i] ?? `Class ${i}`
                      return (
                        <li key={i}>
                          <span className="label">{label}</span>
                          <span className="bar">
                            <span className="fill" style={{ width: `${p * 100}%` }} />
                          </span>
                          <span className="value">{prettyProb(p)}</span>
                        </li>
                      )
                    })}
                  </ul>
                </div>

                <div className="overlay-preview">
                  <h3>Grad-CAM Overlay</h3>
                  {result.gradcam_overlay_base64 ? (
                    <img
                      src={`data:image/png;base64,${result.gradcam_overlay_base64}`}
                      alt="gradcam"
                      className="overlay-img"
                    />
                  ) : (
                    <div className="no-overlay">No overlay available</div>
                  )}
                </div>
              </div>
            )
          ) : (
            <div className="hint">Upload an image and click Analyze to see prediction + Grad-CAM</div>
          )}
        </section>
      </main>

      <footer className="footer">
        <small>Made with care — backend served at /predict</small>
      </footer>
    </div>
  )
}
