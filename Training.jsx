import { useState, useRef, useEffect } from 'react'
import { streamGATraining, trainBaseline } from './api'

export default function Training() {
  const [cfg, setCfg]             = useState({ pop_size: 60, n_generations: 80, mutation_rate: 0.02, crossover_rate: 0.80, use_gpu: false })
  const [running, setRunning]     = useState(false)
  const [logs, setLogs]           = useState([])
  const [history, setHistory]     = useState([])
  const [best, setBest]           = useState(null)
  const [done, setDone]           = useState(false)
  const [blLoading, setBlLoading] = useState(false)
  const cancelRef   = useRef(null)
  const termRef     = useRef(null)

  useEffect(() => {
    if (termRef.current) termRef.current.scrollTop = termRef.current.scrollHeight
  }, [logs])

  const startGA = () => {
    setRunning(true); setDone(false); setLogs([]); setHistory([]); setBest(null)

    addLog('system', `Starting GA: pop=${cfg.pop_size}, gen=${cfg.n_generations}, mut=${cfg.mutation_rate}`)

    cancelRef.current = streamGATraining(
      cfg,
      (msg) => {
        setHistory(h => [...h, msg])
        setBest(msg.best_chromosome)
        addLog('gen', `Gen ${String(msg.generation).padStart(3)} | best=${msg.best_fitness.toFixed(4)} | avg=${msg.avg_fitness.toFixed(4)} | features=${msg.best_n_features}/13`)
      },
      (bestChr) => {
        setBest(bestChr)
        setRunning(false); setDone(true)
        addLog('fit', `✓ GA complete. Best fitness=${bestChr.fitness.toFixed(4)}, features=${bestChr.n_features}/13, C=${bestChr.C}, γ=${bestChr.gamma}`)
      },
      (err) => {
        setRunning(false)
        addLog('warn', `Error: ${err.message}`)
      }
    )
  }

  const stopGA = () => {
    if (cancelRef.current) cancelRef.current()
    setRunning(false)
    addLog('warn', 'Training cancelled by user.')
  }

  const addLog = (type, msg) => {
    setLogs(l => [...l, { type, msg, t: new Date().toLocaleTimeString() }])
  }

  const handleBaseline = async () => {
    setBlLoading(true)
    try {
      await trainBaseline()
      addLog('fit', 'Baseline SVM training launched in background.')
    } catch(e) {
      addLog('warn', `Baseline error: ${e.message}`)
    } finally {
      setBlLoading(false)
    }
  }

  const progress = history.length > 0 ? (history.length / cfg.n_generations) * 100 : 0

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Model Training</h1>
        <p className="page-sub">Configure and launch the Genetic Algorithm — watch it evolve in real time</p>
      </div>

      <div className="grid-2" style={{ gap: 20, alignItems: 'start' }}>
        {/* Config panel */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          <div className="card">
            <div className="card-title">GA Configuration</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
              <SliderCfg label="Population size" k="pop_size" min={10} max={200} step={10} cfg={cfg} setCfg={setCfg} />
              <SliderCfg label="Generations" k="n_generations" min={10} max={300} step={10} cfg={cfg} setCfg={setCfg} />
              <SliderCfg label="Mutation rate" k="mutation_rate" min={0.001} max={0.2} step={0.001} cfg={cfg} setCfg={setCfg} fmt={v => v.toFixed(3)} />
              <SliderCfg label="Crossover rate" k="crossover_rate" min={0.5} max={1.0} step={0.05} cfg={cfg} setCfg={setCfg} fmt={v => v.toFixed(2)} />
              <div className="form-row">
                <label className="form-label">GPU acceleration</label>
                <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
                  <input type="checkbox" checked={cfg.use_gpu}
                    onChange={e => setCfg(c => ({...c, use_gpu: e.target.checked}))} />
                  <span style={{ fontSize: 12 }}>Enable (requires RAPIDS cuML)</span>
                </label>
              </div>
            </div>

            <div style={{ display: 'flex', gap: 10, marginTop: 20, flexWrap: 'wrap' }}>
              {!running ? (
                <button className="btn primary" onClick={startGA}>Start GA Training</button>
              ) : (
                <button className="btn danger" onClick={stopGA}>Stop</button>
              )}
              <button className="btn" onClick={handleBaseline} disabled={blLoading || running}>
                {blLoading ? <><div className="spinner"/> Launching…</> : 'Train Baseline SVM'}
              </button>
            </div>
          </div>

          {/* Chromosome display */}
          {best && (
            <div className="card">
              <div className="card-title">Best chromosome {done ? '(final)' : '(live)'}</div>
              <ChromosomeDisplay chr={best} />
            </div>
          )}
        </div>

        {/* Right: chart + terminal */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          {/* Progress */}
          {(running || done) && (
            <div className="card">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                <div className="card-title" style={{ marginBottom: 0 }}>
                  {done ? 'Completed' : `Generation ${history.length} / ${cfg.n_generations}`}
                </div>
                <span style={{ fontFamily: 'Space Mono, monospace', fontSize: 11, color: 'var(--muted)' }}>
                  {Math.round(progress)}%
                </span>
              </div>
              <div className="progress-bar">
                <div className="progress-fill" style={{ width: `${progress}%`, background: done ? 'var(--success)' : 'var(--accent)' }} />
              </div>
            </div>
          )}

          {/* Convergence chart */}
          {history.length > 1 && (
            <div className="card">
              <div className="card-title">Fitness convergence</div>
              <ConvergenceChart history={history} total={cfg.n_generations} />
            </div>
          )}

          {/* Terminal */}
          <div className="card" style={{ padding: 0 }}>
            <div style={{ padding: '12px 16px', borderBottom: '1px solid var(--border)', display: 'flex', alignItems: 'center', gap: 10 }}>
              <div className="card-title" style={{ marginBottom: 0 }}>Training log</div>
              {running && <div className="spinner" style={{ width: 14, height: 14 }} />}
              {logs.length > 0 && (
                <button className="btn" style={{ padding: '3px 10px', fontSize: 11, marginLeft: 'auto' }}
                  onClick={() => setLogs([])}>Clear</button>
              )}
            </div>
            <div className="terminal" ref={termRef} style={{ borderRadius: '0 0 var(--radius) var(--radius)' }}>
              {logs.length === 0
                ? <span style={{ color: 'var(--muted)' }}>Ready. Configure and start training above.</span>
                : logs.map((l, i) => (
                  <div key={i}>
                    <span style={{ color: 'var(--muted)' }}>[{l.t}] </span>
                    <span className={l.type}>{l.msg}</span>
                  </div>
                ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function SliderCfg({ label, k, min, max, step, cfg, setCfg, fmt }) {
  const val = cfg[k]
  const display = fmt ? fmt(parseFloat(val)) : val
  return (
    <div className="form-row">
      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
        <label className="form-label">{label}</label>
        <span style={{ fontFamily: 'Space Mono, monospace', fontSize: 11, color: 'var(--accent)' }}>{display}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={val}
        onChange={e => setCfg(c => ({ ...c, [k]: parseFloat(e.target.value) }))} />
    </div>
  )
}

function ChromosomeDisplay({ chr }) {
  if (!chr) return null
  const FEATURE_NAMES = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      <div className="feature-chips">
        {FEATURE_NAMES.map((n, i) => (
          <span key={n} className={`feature-chip ${chr.feature_mask?.[i] ? 'on' : 'off'}`}>{n}</span>
        ))}
      </div>
      <div style={{ display: 'flex', gap: 16, fontSize: 12 }}>
        {[['Fitness', chr.fitness?.toFixed(4)], ['C', chr.C], ['γ', chr.gamma], ['Features', `${chr.n_features}/13`]].map(([k,v]) => (
          <div key={k}>
            <span style={{ color: 'var(--muted)' }}>{k}: </span>
            <span style={{ fontFamily: 'Space Mono, monospace', color: 'var(--accent)' }}>{v}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

function ConvergenceChart({ history, total }) {
  const W = 520, H = 120, PX = 12, PY = 10
  const best = history.map(d => d.best_fitness)
  const avg  = history.map(d => d.avg_fitness)

  const allVals = [...best, ...avg].filter(Boolean)
  const minY = Math.min(...allVals) * 0.97
  const maxY = Math.max(...allVals) * 1.01
  const sx = i => PX + (i / Math.max(total - 1, 1)) * (W - PX * 2)
  const sy = v => H - PY - ((v - minY) / (maxY - minY || 1)) * (H - PY * 2)
  const path = (arr) => arr.map((v, i) => `${i===0?'M':'L'}${sx(i)},${sy(v)}`).join(' ')

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: H }}>
      {/* Grid lines */}
      {[0.25, 0.5, 0.75, 1.0].map(t => {
        const y = PY + t * (H - PY * 2)
        return <line key={t} x1={PX} y1={y} x2={W - PX} y2={y}
          stroke="var(--border)" strokeWidth="0.5" strokeDasharray="3 3"/>
      })}
      <path d={path(avg)}  fill="none" stroke="var(--muted)"  strokeWidth="1.5" strokeDasharray="4 2"/>
      <path d={path(best)} fill="none" stroke="var(--accent)" strokeWidth="2.5"/>
      {/* Endpoint dot */}
      {best.length > 0 && (
        <circle cx={sx(best.length - 1)} cy={sy(best[best.length - 1])} r="4"
          fill="var(--accent)" />
      )}
      {/* Legend */}
      <line x1={W-120} y1={H-4} x2={W-100} y2={H-4} stroke="var(--accent)" strokeWidth="2"/>
      <text x={W-96} y={H-1} fontSize="10" fill="var(--muted)">Best</text>
      <line x1={W-60} y1={H-4} x2={W-40} y2={H-4} stroke="var(--muted)" strokeWidth="1.5" strokeDasharray="4 2"/>
      <text x={W-36} y={H-1} fontSize="10" fill="var(--muted)">Avg</text>
    </svg>
  )
}
