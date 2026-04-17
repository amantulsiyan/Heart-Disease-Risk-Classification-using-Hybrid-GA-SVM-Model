const BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export async function api(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || 'Request failed')
  }
  return res.json()
}

export const getMeta        = () => api('/api/meta')
export const getBaseline    = () => api('/api/results/baseline')
export const getGASVM       = () => api('/api/results/gasvm')
export const getComparison  = () => api('/api/results/comparison')
export const getGAHistory   = () => api('/api/results/ga-history')
export const getGPUInfo     = () => api('/api/gpu-info')

export const predict = (features) =>
  api('/api/predict', { method: 'POST', body: JSON.stringify(features) })

export const trainBaseline = () =>
  api('/api/train/baseline', { method: 'POST', body: JSON.stringify({}) })

// SSE streaming for GA training
export function streamGATraining(config, onMessage, onDone, onError) {
  // Use fetch with ReadableStream to parse SSE
  const controller = new AbortController()

  fetch(`${BASE}/api/train/ga`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
    signal: controller.signal,
  }).then(async (res) => {
    if (!res.ok) throw new Error('Training request failed')
    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop()  // keep incomplete line

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const raw = line.slice(6).trim()
          if (!raw) continue
          try {
            const parsed = JSON.parse(raw)
            if (parsed.ping) continue
            if (parsed.done) { onDone(parsed.best); return }
            onMessage(parsed)
          } catch {}
        }
      }
    }
  }).catch((err) => {
    if (err.name !== 'AbortError') onError(err)
  })

  return () => controller.abort()   // returns cancel fn
}
