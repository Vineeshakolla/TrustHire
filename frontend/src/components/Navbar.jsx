import React, { useEffect, useState } from 'react';
import { Shield, Activity, Cpu } from 'lucide-react';
import { checkBackendHealth } from '../services/api';

export default function Navbar({ currentPage, setCurrentPage }) {
  const [healthStatus, setHealthStatus] = useState({ online: false, model_loaded: false, checking: true });

  useEffect(() => {
    async function verifyHealth() {
      const res = await checkBackendHealth();
      setHealthStatus({ online: res.online, model_loaded: res.model_loaded, checking: false });
    }
    verifyHealth();
    const interval = setInterval(verifyHealth, 15000); // Poll every 15s
    return () => clearInterval(interval);
  }, []);

  return (
    <nav className="navbar">
      <div className="container nav-container">
        <div className="nav-brand" onClick={() => setCurrentPage('home')}>
          <div className="brand-icon">
            <Shield size={22} />
          </div>
          <span>TrustHire</span>
        </div>

        <ul className="nav-links">
          <li>
            <span
              className={`nav-link ${currentPage === 'home' ? 'active' : ''}`}
              onClick={() => setCurrentPage('home')}
            >
              Home
            </span>
          </li>
          <li>
            <span
              className={`nav-link ${currentPage === 'analyzer' ? 'active' : ''}`}
              onClick={() => setCurrentPage('analyzer')}
            >
              Analyze Job
            </span>
          </li>
          <li>
            <span
              className={`nav-link ${currentPage === 'how-it-works' ? 'active' : ''}`}
              onClick={() => setCurrentPage('how-it-works')}
            >
              How It Works
            </span>
          </li>
          <li>
            <span
              className={`nav-link ${currentPage === 'about' ? 'active' : ''}`}
              onClick={() => setCurrentPage('about')}
            >
              About
            </span>
          </li>
        </ul>

        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div className="nav-status" title={healthStatus.online ? "FastAPI backend server is online" : "FastAPI backend server is offline"}>
            <div className={`status-dot ${healthStatus.online ? 'online' : 'offline'}`}></div>
            <span>{healthStatus.online ? 'TrustHire API Online' : 'Backend Offline'}</span>
          </div>

          <button className="btn btn-primary" onClick={() => setCurrentPage('analyzer')}>
            <Cpu size={16} />
            <span>Analyze a Job</span>
          </button>
        </div>
      </div>
    </nav>
  );
}
