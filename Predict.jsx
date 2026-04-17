import { useState, useEffect } from 'react'
import { predict, getMeta } from '../utils/api'

const DEFAULTS = {
  age: 55, sex: 1, cp: 1, trestbps: 130, chol: 250,
  fbs: 0, restecg: 1, thalach: 150, exang: 0,
  oldpeak: 1.5, slope: 1, ca: 0, thal: 2,
}

// High-risk example patient (for demo)
const HIGH_RISK = {
  age: 63, sex: 1, cp: 3, trestbps: 145, chol: 233,
  fbs: 1, restecg: 0, thalach: 150, exang: 0,
  oldpeak: 2.3, slope: 0, ca: 0, thal: 1,
}
const LOW_RISK = {
  age: 41, sex: 0, cp: 1, trestbps: 130, chol: 204,
  fbs: 0, restecg: 0, thalach: 172, exang: 0,
  oldpeak: 1.4, slope: 2, ca: 0, thal: 2,
}

export default function Predict() {
  const [meta, setMeta]         = useState(null)
  const [form, setForm]         = useState(DEFAULTS)
  const [result, setResult]     = useState(null)
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState(null)

  useEffect(() => { getMeta().then(setMeta).catch(() => {}) }, [])

  const handleChange = (key, val) => {
    setForm(f => ({ ...f, [key]: parseFloat(val) }))
  }

  const handleSubmit = async () => {
    setLoading(true); setError(null); setResult(null)
    try {
      const res = await predict(form)
      setResult(res)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const applyPreset = (preset) => { setForm(preset); setResult(null) }

  const featureRanges = meta?.feature_ranges || {}
  const featureDesc   = meta?.feature_descriptions || {}

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Patient Risk Prediction</h1>
        <p className="page-sub">Enter patient clinical features to get risk predictions from both models</p>
      </div>

      {/* Presets */}
      <div style={{ display: 'flex', gap: 10, marginBottom: 20, alignItems: 'center' }}>
        <span style={{ fontSize: 12, color: 'var(--muted)' }}>Load example:</span>
        <button className="btn" onClick={() => applyPreset(HIGH_RISK)}>High-risk patient</button>
        <button className="btn" onClick={() => applyPreset(LOW_RISK)}>Low-risk patient</button>
        <button className="btn" onClick={() => applyPreset(DEFAULTS)}>Reset</button>
      </div>

      <div className="grid-2" style={{ gap: 20, alignItems: 'start' }}>
        {/* Input form */}
        <div className="card">
          <div className="card-title">Patient Features</div>
          <div className="form-grid">
            {Object.entries(DEFAULTS).map(([key]) => {
              const r   = featureRanges[key] || {}
              const val = form[key]
              const isFloat = r.type === 'float'
              return (
                <div className="form-row" key={key}>
                  <label className="form-label">
                    {featureDesc[key] || key}
                  </label>
                  <div className="slider-row">
                    <input
                      type="range"
                      min={r.min ?? 0} max={r.max ?? 10}
                      step={r.step ?? 1}
                      value={val}
                      onChange={e => handleChange(key, e.target.value)}
                    />
                    <span className="slider-val">{isFloat ? parseFloat(val).toFixed(1) : val}</span>
                  </div>
                </div>
              )
            })}
          </div>
          <div style={{ marginTop: 20 }}>
            <button className="btn primary" onClick={handleSubmit} disabled={loading} style={{ width: '100%', justifyContent: 'center' }}>
              {loading ? <><div className="spinner" /> Predicting…</> : 'Predict Risk'}
            </button>
          </div>
          {error && <div style={{ marginTop: 12, color: 'var(--danger)', fontSize: 12 }}>{error}</div>}
        </div>

        {/* Results */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          {result ? (
            <>
              {/* Consensus */}
              <div className="card">
                <div className="card-title">Risk Assessment</div>
                <div className="risk-display">
                  <div>
                    <div className={`risk-label ${result.gasvm.prediction === 1 ? 'high' : 'low'}`}>
                      {result.gasvm.label}
                    </div>
                    <div style={{ fontSize: 12, color: 'var(--muted)', marginTop: 4 }}>
                      Consensus: {result.consensus ? '✓ Both models agree' : '⚠ Models disagree'}
                    </div>
                  </div>
                  <div className="risk-bar-wrap">
                    <div style={{ fontSize: 11, color: 'var(--muted)', marginBottom: 4, display: 'flex', justifyContent: 'space-between' }}>
                      <span>Risk score</span>
                      <span style={{ fontFamily: 'Space Mono, monospace' }}>{(result.risk_score * 100).toFixed(1)}%</span>
                    </div>
                    <div className="risk-bar-track">
                      <div className={`risk-bar-fill ${result.risk_score > 0.5 ? 'high' : 'low'}`}
                           style={{ width: `${result.risk_score * 100}%` }} />
                    </div>
                  </div>
                </div>
              </div>

              {/* Side-by-side model results */}
              <div className="grid-2" style={{ gap: 12 }}>
                <ModelResult label="Baseline SVM" data={result.baseline} color="var(--accent2)" />
                <ModelResult label="GA-SVM" data={result.gasvm} color="var(--accent)" isGA />
              </div>

              {/* Feature comparison */}
              {result.gasvm.feature_mask && meta && (
                <div className="card">
                  <div className="card-title">Features used by GA-SVM ({result.gasvm.features_used.length}/13)</div>
                  <div className="feature-chips">
                    {meta.feature_names.map((name, i) => (
                      <span key={name} className={`feature-chip ${result.gasvm.feature_mask[i] ? 'on' : 'off'}`}
                            title={meta.feature_descriptions?.[name]}>
                        {name}
                      </span>
                    ))}
                  </div>
                  <div style={{ marginTop: 12, fontSize: 12, color: 'var(--muted)' }}>
                    Optimized: C = {result.gasvm.C}, γ = {result.gasvm.gamma}
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="card" style={{ color: 'var(--muted)', textAlign: 'center', padding: 40 }}>
              <div style={{ fontSize: 32, marginBottom: 12 }}>⬅</div>
              Adjust patient features and click Predict Risk
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function ModelResult({ label, data, color, isGA }) {
  const prob = data.probability
  return (
    <div className="card">
      <div className="card-title">{label}</div>
      <div style={{ marginBottom: 12 }}>
        <span className={`badge ${data.prediction === 1 ? 'high-risk' : 'low-risk'}`}>{data.label}</span>
      </div>
      <div style={{ fontSize: 13, color: 'var(--muted)', marginBottom: 4 }}>Probability</div>
      <div style={{ fontFamily: 'Space Mono, monospace', fontSize: 20, fontWeight: 700, color, marginBottom: 8 }}>
        {(prob * 100).toFixed(1)}%
      </div>
      <div className="risk-bar-track">
        <div className="risk-bar-fill" style={{
          width: `${prob * 100}%`,
          background: color,
          transition: 'width 0.6s ease'
        }} />
      </div>
      {isGA && (
        <div style={{ marginTop: 10, fontSize: 11, color: 'var(--muted)' }}>
          {data.features_used?.length} features · C={data.C} · γ={data.gamma}
        </div>
      )}
      {!isGA && (
        <div style={{ marginTop: 10, fontSize: 11, color: 'var(--muted)' }}>
          13 features · default hyperparameters
        </div>
      )}
    </div>
  )
}
