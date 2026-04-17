import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import Predict from './pages/Predict'
import Training from './pages/Training'
import Results from './pages/Results'
import './index.css'

export default function App() {
  return (
    <BrowserRouter>
      <div className="app-shell">
        <aside className="sidebar">
          <div className="sidebar-logo">
            <div className="logo-icon">
              <svg viewBox="0 0 32 32" fill="none">
                <path d="M16 3C9.373 3 4 8.373 4 15c0 4.5 2.34 8.46 5.87 10.74L8 29l3.83-1.28A12.93 12.93 0 0016 28c6.627 0 12-5.373 12-12S22.627 3 16 3z"
                  fill="var(--accent)" opacity="0.15"/>
                <path d="M16 3C9.373 3 4 8.373 4 15c0 4.5 2.34 8.46 5.87 10.74L8 29l3.83-1.28A12.93 12.93 0 0016 28c6.627 0 12-5.373 12-12S22.627 3 16 3z"
                  stroke="var(--accent)" strokeWidth="1.5" fill="none"/>
                <circle cx="11" cy="15" r="1.5" fill="var(--accent)"/>
                <circle cx="16" cy="13" r="1.5" fill="var(--accent)"/>
                <circle cx="21" cy="15" r="1.5" fill="var(--accent)"/>
              </svg>
            </div>
            <div>
              <div className="logo-title">CardioGA</div>
              <div className="logo-sub">GA-SVM Classifier</div>
            </div>
          </div>

          <nav className="sidebar-nav">
            <NavLink to="/" end className={({isActive}) => isActive ? 'nav-item active' : 'nav-item'}>
              <IconGrid /> Dashboard
            </NavLink>
            <NavLink to="/predict" className={({isActive}) => isActive ? 'nav-item active' : 'nav-item'}>
              <IconPredict /> Predict
            </NavLink>
            <NavLink to="/training" className={({isActive}) => isActive ? 'nav-item active' : 'nav-item'}>
              <IconTrain /> Training
            </NavLink>
            <NavLink to="/results" className={({isActive}) => isActive ? 'nav-item active' : 'nav-item'}>
              <IconResults /> Results
            </NavLink>
          </nav>

          <div className="sidebar-footer">
            <div className="badge-gpu" id="gpu-badge">Checking GPU…</div>
          </div>
        </aside>

        <main className="main-content">
          <Routes>
            <Route path="/"         element={<Dashboard />} />
            <Route path="/predict"  element={<Predict />} />
            <Route path="/training" element={<Training />} />
            <Route path="/results"  element={<Results />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}

function IconGrid() {
  return <svg viewBox="0 0 20 20" fill="currentColor" width="16" height="16">
    <rect x="2" y="2" width="7" height="7" rx="1.5"/>
    <rect x="11" y="2" width="7" height="7" rx="1.5"/>
    <rect x="2" y="11" width="7" height="7" rx="1.5"/>
    <rect x="11" y="11" width="7" height="7" rx="1.5"/>
  </svg>
}
function IconPredict() {
  return <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.8" width="16" height="16">
    <circle cx="10" cy="10" r="7.5"/>
    <path d="M10 6.5v4l2.5 2" strokeLinecap="round"/>
  </svg>
}
function IconTrain() {
  return <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.8" width="16" height="16">
    <path d="M3 14.5c2-4 4-6 7-6s5 2 7 6" strokeLinecap="round"/>
    <circle cx="10" cy="6" r="2"/>
  </svg>
}
function IconResults() {
  return <svg viewBox="0 0 20 20" fill="currentColor" width="16" height="16">
    <rect x="3" y="10" width="3" height="7" rx="1"/>
    <rect x="8.5" y="6" width="3" height="11" rx="1"/>
    <rect x="14" y="3" width="3" height="14" rx="1"/>
  </svg>
}
