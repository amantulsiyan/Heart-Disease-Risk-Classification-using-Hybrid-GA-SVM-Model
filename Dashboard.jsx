import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { getComparison, getGPUInfo } from '../utils/api'

export default function Dashboard() {
  const [cmp, setCmp]       = useState(null)
  const [gpuInfo, setGpu]   = useState(null)
  const [error, setError]   = useState(null)

  useEffect(() => {
    getComparison().then(setCmp).catch((e) => setError(e.message))
    getGPUInfo().then(setGpu).catch(() => {})
  }, [])

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Overview</h1>
        <p className="page-sub">Heart Disease Risk Classification using Hybrid GA-SVM</p>
      </div>

      {error && (
        <div className="card" style={{ borderColor: 'var(--danger)', marginBottom: 20, color: 'var(--danger)' }}>
          {error} — run the Python training scripts first, then restart the API.
        </div>
      )}

      {/* GPU info strip */}
      {gpuInfo && (
        <div className="card" style={{ marginBottom: 16, display: 'flex', alignItems: 'center', gap: 16 }}>
          <div className="badge" style={{ background: gpuInfo.cuda_available ? 'rgba(63,185,80,0.15)' : 'rgba(139,148,158,0.15)', color: gpuInfo.cuda_available ? 'var(--success)' : 'var(--muted)' }}>
            {gpuInfo.cuda_available ? '● GPU' : '● CPU only'}
          </div>
          {gpuInfo.device_name && <span style={{ fontSize: 12, color: 'var(--muted)' }}>{gpuInfo.device_name}</span>}
          {gpuInfo.cuml_available && <span className="badge info">cuML ready</span>}
          {!gpuInfo.cuda_available && <span style={{ fontSize: 12, color: 'var(--muted)' }}>Install RAPIDS cuML + CuPy for GPU acceleration</span>}
        </div>
      )}

      {/* Metric summary cards */}
      {cmp ? (
        <>
          <div className="metric-grid">
            <MetricCard label="GA-SVM Accuracy"  value={pct(cmp.gasvm?.accuracy)}  sub={delta(cmp.gasvm?.accuracy, cmp.baseline?.accuracy, 'vs baseline')} color="accent" />
            <MetricCard label="GA-SVM F1 Score"  value={pct(cmp.gasvm?.f1)}        sub={delta(cmp.gasvm?.f1, cmp.baseline?.f1)} color="green" />
            <MetricCard label="GA-MLP Accuracy"  value={pct(cmp.gamlp?.accuracy)}  sub={delta(cmp.gamlp?.accuracy, cmp.baseline?.accuracy, 'vs baseline')} color="blue" />
            <MetricCard label="GA-MLP F1 Score"  value={pct(cmp.gamlp?.f1)}        sub={cmp.gamlp ? `${cmp.gamlp.depth} layers x ${cmp.gamlp.hidden_size} units` : 'not trained'} />
          </div>

          <div className="grid-2" style={{ gap: 16 }}>
            <div className="card">
              <div className="card-title">Model comparison</div>
              <table className="tbl">
                <thead>
                  <tr>
                    <th>Metric</th>
                    <th>Baseline SVM</th>
                    <th>GA-SVM</th>
                    {cmp.gamlp && <th style={{ color: '#55A868' }}>GA-MLP</th>}
                  </tr>
                </thead>
                <tbody>
                  {[['Accuracy', 'accuracy'], ['F1 Score', 'f1'], ['AUC-ROC', 'auc']].map(([label, key]) => (
                    <tr key={key}>
                      <td>{label}</td>
                      <td style={{ fontFamily: 'Space Mono, monospace' }}>{fmt4(cmp.baseline?.[key])}</td>
                      <td style={{ fontFamily: 'Space Mono, monospace', color: 'var(--accent)' }}>{fmt4(cmp.gasvm?.[key])}</td>
                      {cmp.gamlp && <td style={{ fontFamily: 'Space Mono, monospace', color: '#55A868' }}>{fmt4(cmp.gamlp?.[key])}</td>}
                    </tr>
                  ))}
                  <tr>
                    <td>Features used</td>
                    <td style={{ fontFamily: 'Space Mono, monospace' }}>13 / 13</td>
                    <td style={{ fontFamily: 'Space Mono, monospace', color: 'var(--accent)' }}>{cmp.gasvm?.n_features} / 13</td>
                    {cmp.gamlp && <td style={{ fontFamily: 'Space Mono, monospace', color: '#55A868' }}>{cmp.gamlp?.n_features} / 13</td>}
                  </tr>
                  <tr>
                    <td>Device</td>
                    <td style={{ fontFamily: 'Space Mono, monospace', color: 'var(--muted)' }}>CPU</td>
                    <td style={{ fontFamily: 'Space Mono, monospace', color: 'var(--muted)' }}>CPU</td>
                    {cmp.gamlp && <td style={{ fontFamily: 'Space Mono, monospace', color: cmp.gamlp?.device?.includes('cuda') ? 'var(--success)' : 'var(--muted)' }}>{cmp.gamlp?.device ?? 'CPU'}</td>}
                  </tr>
                </tbody>
              </table>
            </div>

            <div className="card">
              <div className="card-title">GA-selected features</div>
              <FeatureMaskDisplay names={cmp.feature_names} mask={cmp.gasvm?.feature_mask} />
              <div style={{ marginTop: 20 }}>
                <div className="card-title">Quick actions</div>
                <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
                  <Link to="/predict" className="btn primary">Run Prediction</Link>
                  <Link to="/training" className="btn">Retrain GA</Link>
                  <Link to="/results" className="btn">Full Results</Link>
                </div>
              </div>
            </div>
          </div>

          {/* GA convergence mini-chart */}
          {cmp.ga_history?.length > 0 && (
            <div className="card" style={{ marginTop: 16 }}>
              <div className="card-title">GA convergence ({cmp.ga_history.length} generations)</div>
              <MiniLineChart data={cmp.ga_history} />
            </div>
          )}
        </>
      ) : !error ? (
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, color: 'var(--muted)' }}>
          <div className="spinner" /> Loading results…
        </div>
      ) : null}
    </div>
  )
}

function MetricCard({ label, value, sub, color }) {
  const colorMap = { accent: 'var(--accent)', blue: 'var(--accent2)', green: 'var(--success)' }
  return (
    <div className="metric-card">
      <div className="metric-label">{label}</div>
      <div className="metric-value" style={{ color: colorMap[color] || 'var(--text)' }}>{value ?? '—'}</div>
      {sub && <div className="metric-delta" style={{ color: 'var(--muted)' }}>{sub}</div>}
    </div>
  )
}

function FeatureMaskDisplay({ names = [], mask = [] }) {
  return (
    <div className="feature-chips">
      {names.map((name, i) => (
        <span key={name} className={`feature-chip ${mask[i] ? 'on' : 'off'}`}>{name}</span>
      ))}
    </div>
  )
}

function MiniLineChart({ data }) {
  const W = 600, H = 100, PAD = 10
  const best = data.map(d => d.best_fitness)
  const avg  = data.map(d => d.avg_fitness)
  const minY = Math.min(...avg)  * 0.98
  const maxY = Math.max(...best) * 1.01
  const scaleX = i => PAD + (i / (data.length - 1)) * (W - PAD * 2)
  const scaleY = v => H - PAD - ((v - minY) / (maxY - minY)) * (H - PAD * 2)

  const toPath = (arr) => arr.map((v, i) => `${i === 0 ? 'M' : 'L'}${scaleX(i)},${scaleY(v)}`).join(' ')

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: H }}>
      <path d={toPath(avg)}  fill="none" stroke="var(--muted)"  strokeWidth="1.5" strokeDasharray="3 2"/>
      <path d={toPath(best)} fill="none" stroke="var(--accent)" strokeWidth="2" />
    </svg>
  )
}

const pct  = v => v != null ? `${(v * 100).toFixed(1)}%` : '—'
const fmt4 = v => v != null ? v.toFixed(4) : '—'
const delta = (a, b, suffix = '') => {
  if (a == null || b == null) return ''
  const d = ((a - b) * 100).toFixed(2)
  return `${d >= 0 ? '+' : ''}${d}% ${suffix}`
}
