import { useEffect, useState } from 'react'
import { getComparison } from './api'

export default function Results() {
  const [cmp, setCmp]   = useState(null)
  const [tab, setTab]   = useState('metrics')
  const [error, setErr] = useState(null)

  useEffect(() => {
    getComparison().then(setCmp).catch(e => setErr(e.message))
  }, [])

  if (error) return (
    <div>
      <div className="page-header"><h1 className="page-title">Results</h1></div>
      <div className="card" style={{ color: 'var(--danger)' }}>{error}</div>
    </div>
  )
  if (!cmp) return (
    <div>
      <div className="page-header"><h1 className="page-title">Results</h1></div>
      <div style={{ display: 'flex', gap: 12, color: 'var(--muted)' }}><div className="spinner"/>Loading…</div>
    </div>
  )

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Full Results</h1>
        <p className="page-sub">Comprehensive evaluation: GA-SVM vs Baseline SVM</p>
      </div>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 24, borderBottom: '1px solid var(--border)', paddingBottom: 0 }}>
        {['metrics', 'confusion', 'roc', 'features', 'convergence'].map(t => (
          <button key={t} onClick={() => setTab(t)}
            style={{ padding: '8px 16px', background: 'none', border: 'none', cursor: 'pointer',
              borderBottom: tab === t ? '2px solid var(--accent)' : '2px solid transparent',
              color: tab === t ? 'var(--accent)' : 'var(--muted)',
              fontFamily: 'inherit', fontSize: 13, fontWeight: tab === t ? 600 : 400, transition: 'all 0.15s'
            }}>
            {t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>

      {tab === 'metrics'     && <MetricsTab cmp={cmp} />}
      {tab === 'confusion'   && <ConfusionTab cmp={cmp} />}
      {tab === 'roc'         && <ROCTab cmp={cmp} />}
      {tab === 'features'    && <FeaturesTab cmp={cmp} />}
      {tab === 'convergence' && <ConvergenceTab cmp={cmp} />}
    </div>
  )
}

// ── Metrics tab ──────────────────────────────────────────────────────────────
function MetricsTab({ cmp }) {
  const rows = [
    ['Accuracy',       'accuracy',     true],
    ['F1 Score',       'f1',           true],
    ['AUC-ROC',        'auc',          true],
    ['Features used',  'n_features',   false],
    ['Train time (s)', 'train_time_s', false],
  ]
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div className="grid-2" style={{ gap: 12 }}>
        {[
          { label: 'Baseline accuracy', v: cmp.baseline.accuracy, color: 'blue' },
          { label: 'GA-SVM accuracy',   v: cmp.gasvm.accuracy,    color: 'accent' },
          { label: 'Baseline F1',       v: cmp.baseline.f1,       color: 'blue' },
          { label: 'GA-SVM F1',         v: cmp.gasvm.f1,          color: 'accent' },
        ].map(({ label, v, color }) => (
          <div className="metric-card" key={label}>
            <div className="metric-label">{label}</div>
            <div className={`metric-value ${color}`}>{(v * 100).toFixed(1)}%</div>
          </div>
        ))}
      </div>
      <div className="card">
        <div className="card-title">Detailed comparison</div>
        <table className="tbl">
          <thead>
            <tr><th>Metric</th><th>Baseline SVM</th><th>GA-SVM</th><th>Improvement</th><th>Winner</th></tr>
          </thead>
          <tbody>
            {rows.map(([label, key, pct]) => {
              const bv = cmp.baseline[key]
              const gv = cmp.gasvm[key]
              const diff = gv - bv
              const fmt = v => pct ? `${(v*100).toFixed(2)}%` : v
              const win = pct ? diff > 0 : diff < 0
              return (
                <tr key={key}>
                  <td style={{ fontWeight: 500 }}>{label}</td>
                  <td style={{ fontFamily: 'Space Mono, monospace', color: 'var(--accent2)' }}>{fmt(bv)}</td>
                  <td style={{ fontFamily: 'Space Mono, monospace', color: 'var(--accent)'  }}>{fmt(gv)}</td>
                  <td style={{ fontFamily: 'Space Mono, monospace', color: diff >= 0 ? 'var(--success)' : 'var(--danger)' }}>
                    {diff >= 0 ? '+' : ''}{pct ? `${(diff*100).toFixed(2)}%` : diff.toFixed(4)}
                  </td>
                  <td>
                    <span className="badge" style={{ background: win ? 'rgba(240,136,62,0.15)' : 'rgba(88,166,255,0.15)', color: win ? 'var(--accent)' : 'var(--accent2)' }}>
                      {win ? 'GA-SVM' : 'Baseline'}
                    </span>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// ── Confusion matrix tab ─────────────────────────────────────────────────────
function ConfusionTab({ cmp }) {
  return (
    <div className="grid-2" style={{ gap: 20 }}>
      <div className="card">
        <div className="card-title">Baseline SVM</div>
        <ConfusionMatrix cm={cmp.baseline.cm} />
      </div>
      <div className="card">
        <div className="card-title">GA-SVM</div>
        <ConfusionMatrix cm={cmp.gasvm.cm} />
      </div>
    </div>
  )
}

function ConfusionMatrix({ cm }) {
  if (!cm) return <div style={{ color: 'var(--muted)' }}>No data</div>
  const [[tn, fp], [fn, tp]] = cm
  const total = tn + fp + fn + tp
  const cells = [
    { v: tp, label: 'TP', cls: 'cm-tp', caption: 'True Positive' },
    { v: fp, label: 'FP', cls: 'cm-fp', caption: 'False Positive' },
    { v: fn, label: 'FN', cls: 'cm-fn', caption: 'False Negative' },
    { v: tn, label: 'TN', cls: 'cm-tn', caption: 'True Negative' },
  ]
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
        {cells.map(c => (
          <div key={c.label} className={`cm-cell ${c.cls}`} style={{ flexDirection: 'column', gap: 2 }}>
            <div style={{ fontSize: 24 }}>{c.v}</div>
            <div style={{ fontSize: 11, fontFamily: 'Space Mono, monospace', opacity: 0.7 }}>{c.label}</div>
            <div style={{ fontSize: 10, opacity: 0.6 }}>{c.caption}</div>
          </div>
        ))}
      </div>
      <div style={{ fontSize: 12, color: 'var(--muted)', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
        <div>Sensitivity (TPR): <strong style={{ color: 'var(--success)' }}>{(tp/(tp+fn)*100).toFixed(1)}%</strong></div>
        <div>Specificity (TNR): <strong style={{ color: 'var(--accent2)' }}>{(tn/(tn+fp)*100).toFixed(1)}%</strong></div>
        <div>Precision: <strong style={{ color: 'var(--text)' }}>{(tp/(tp+fp)*100).toFixed(1)}%</strong></div>
        <div>Total samples: <strong style={{ color: 'var(--text)' }}>{total}</strong></div>
      </div>
    </div>
  )
}

// ── ROC curve tab ─────────────────────────────────────────────────────────────
function ROCTab({ cmp }) {
  const W = 400, H = 320, P = 40
  const sx = v => P + v * (W - P * 2)
  const sy = v => H - P - v * (H - P * 2)

  const pathFrom = (roc) => {
    if (!roc?.fpr) return ''
    return roc.fpr.map((x, i) => `${i===0?'M':'L'}${sx(x)},${sy(roc.tpr[i])}`).join(' ')
  }

  const bPath = pathFrom(cmp.baseline.roc)
  const gPath = pathFrom(cmp.gasvm.roc)
  const diagonal = `M${sx(0)},${sy(0)} L${sx(1)},${sy(1)}`

  return (
    <div className="card">
      <div className="card-title">ROC curves</div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: W }}>
        {[0,.25,.5,.75,1].map(t => (
          <g key={t}>
            <line x1={sx(0)} y1={sy(t)} x2={sx(1)} y2={sy(t)} stroke="var(--border)" strokeWidth="0.5"/>
            <line x1={sx(t)} y1={sy(0)} x2={sx(t)} y2={sy(1)} stroke="var(--border)" strokeWidth="0.5"/>
            <text x={sx(0)-4} y={sy(t)+4} textAnchor="end" fontSize="9" fill="var(--muted)">{t}</text>
            <text x={sx(t)}   y={sy(0)+14} textAnchor="middle" fontSize="9" fill="var(--muted)">{t}</text>
          </g>
        ))}
        <path d={diagonal} fill="none" stroke="var(--muted)" strokeWidth="1" strokeDasharray="4 3"/>
        <path d={bPath} fill="none" stroke="var(--accent2)" strokeWidth="2.5"/>
        <path d={gPath} fill="none" stroke="var(--accent)"  strokeWidth="2.5"/>
        <text x={W/2} y={H-4} textAnchor="middle" fontSize="11" fill="var(--muted)">False Positive Rate</text>
        <text x={12}  y={H/2} textAnchor="middle" fontSize="11" fill="var(--muted)" transform={`rotate(-90,12,${H/2})`}>True Positive Rate</text>
        <line x1={P} y1={P-10} x2={P+24} y2={P-10} stroke="var(--accent2)" strokeWidth="2.5"/>
        <text x={P+28} y={P-6} fontSize="11" fill="var(--muted)">Baseline AUC={cmp.baseline.auc}</text>
        <line x1={P} y1={P+5} x2={P+24} y2={P+5} stroke="var(--accent)" strokeWidth="2.5"/>
        <text x={P+28} y={P+9} fontSize="11" fill="var(--muted)">GA-SVM AUC={cmp.gasvm.auc}</text>
      </svg>
    </div>
  )
}

// ── Features tab ─────────────────────────────────────────────────────────────
function FeaturesTab({ cmp }) {
  const names = cmp.feature_names || []
  const mask  = cmp.gasvm?.feature_mask || []
  const n = mask.filter(Boolean).length

  const DESCRIPTIONS = {
    age:'Age in years', sex:'Sex (1=M, 0=F)', cp:'Chest pain type',
    trestbps:'Resting BP (mmHg)', chol:'Cholesterol (mg/dl)',
    fbs:'Fasting glucose >120', restecg:'Resting ECG',
    thalach:'Max heart rate', exang:'Exercise angina',
    oldpeak:'ST depression', slope:'ST slope', ca:'Vessels', thal:'Thalassemia'
  }

  return (
    <div className="card">
      <div className="card-title">GA Feature Selection — {n} of {names.length} features selected</div>
      <table className="tbl">
        <thead>
          <tr><th>#</th><th>Feature</th><th>Description</th><th>GA Selected</th></tr>
        </thead>
        <tbody>
          {names.map((name, i) => (
            <tr key={name}>
              <td style={{ fontFamily: 'Space Mono, monospace', color: 'var(--muted)' }}>{i+1}</td>
              <td style={{ fontFamily: 'Space Mono, monospace', fontWeight: 600 }}>{name}</td>
              <td style={{ color: 'var(--muted)' }}>{DESCRIPTIONS[name] || '—'}</td>
              <td>
                {mask[i]
                  ? <span className="badge" style={{ background: 'rgba(240,136,62,0.15)', color: 'var(--accent)' }}>✓ Selected</span>
                  : <span className="badge" style={{ background: 'var(--bg3)', color: 'var(--muted)' }}>✗ Dropped</span>
                }
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <div style={{ marginTop: 14, padding: '12px 16px', background: 'var(--bg3)', borderRadius: 8 }}>
        <div style={{ fontSize: 12, color: 'var(--muted)' }}>
          Optimized hyperparameters: <strong style={{ color: 'var(--accent)', fontFamily: 'Space Mono, monospace' }}>
            C = {cmp.gasvm?.C}
          </strong> · <strong style={{ color: 'var(--accent)', fontFamily: 'Space Mono, monospace' }}>
            γ = {cmp.gasvm?.gamma}
          </strong>
        </div>
        <div style={{ fontSize: 12, color: 'var(--muted)', marginTop: 4 }}>
          GA fitness score: <strong style={{ fontFamily: 'Space Mono, monospace' }}>{cmp.best_chromosome?.fitness?.toFixed(6)}</strong>
        </div>
      </div>
    </div>
  )
}

// ── Convergence tab ───────────────────────────────────────────────────────────
function ConvergenceTab({ cmp }) {
  const history = cmp.ga_history || []
  if (!history.length) return <div className="card" style={{ color: 'var(--muted)' }}>No history data. Run GA training first.</div>

  const W = 560, H = 200, PX = 48, PY = 16
  const best = history.map(d => d.best_fitness)
  const avg  = history.map(d => d.avg_fitness)
  const nfeat = history.map(d => d.best_n_features)
  const minY  = Math.min(...avg) * 0.97
  const maxY  = Math.max(...best) * 1.01
  const sx = i => PX + (i / (history.length - 1)) * (W - PX - 16)
  const sy = v => H - PY - ((v - minY) / (maxY - minY || 1)) * (H - PY * 2)
  const path = (arr) => arr.map((v, i) => `${i===0?'M':'L'}${sx(i)},${sy(v)}`).join(' ')

  const yTicks = [0, 0.25, 0.5, 0.75, 1].map(t => minY + t * (maxY - minY))

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div className="card">
        <div className="card-title">Fitness over generations</div>
        <svg viewBox={`0 0 ${W} ${H+20}`} style={{ width: '100%', height: H + 20 }}>
          {yTicks.map((v, i) => {
            const y = sy(v)
            return (
              <g key={i}>
                <line x1={PX-4} y1={y} x2={W-16} y2={y} stroke="var(--border)" strokeWidth="0.5"/>
                <text x={PX-8} y={y+4} textAnchor="end" fontSize="9" fill="var(--muted)">{v.toFixed(3)}</text>
              </g>
            )
          })}
          <path d={path(avg)}  fill="none" stroke="var(--muted)"  strokeWidth="1.5" strokeDasharray="4 2"/>
          <path d={path(best)} fill="none" stroke="var(--accent)" strokeWidth="2.5"/>
          {/* X axis */}
          {[0, 0.25, 0.5, 0.75, 1].map(t => {
            const i = Math.floor(t * (history.length - 1))
            return <text key={t} x={sx(i)} y={H+14} textAnchor="middle" fontSize="9" fill="var(--muted)">{history[i]?.generation}</text>
          })}
          <text x={(PX + W - 16)/2} y={H+26} textAnchor="middle" fontSize="10" fill="var(--muted)">Generation</text>
        </svg>
      </div>

      <div className="card">
        <div className="card-title">Summary statistics</div>
        <table className="tbl">
          <thead><tr><th>Metric</th><th>Initial</th><th>Final</th><th>Change</th></tr></thead>
          <tbody>
            {[
              ['Best fitness', history[0]?.best_fitness, history[history.length-1]?.best_fitness, true],
              ['Avg fitness',  history[0]?.avg_fitness,  history[history.length-1]?.avg_fitness,  true],
              ['Features (best)', history[0]?.best_n_features, history[history.length-1]?.best_n_features, false],
            ].map(([label, init, fin, pct]) => {
              const diff = fin - init
              return (
                <tr key={label}>
                  <td>{label}</td>
                  <td style={{ fontFamily: 'Space Mono, monospace' }}>{pct ? init?.toFixed(4) : init}</td>
                  <td style={{ fontFamily: 'Space Mono, monospace', color: 'var(--accent)' }}>{pct ? fin?.toFixed(4) : fin}</td>
                  <td style={{ fontFamily: 'Space Mono, monospace', color: diff >= 0 ? 'var(--success)' : 'var(--danger)' }}>
                    {diff >= 0 ? '+' : ''}{pct ? diff?.toFixed(4) : diff}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
