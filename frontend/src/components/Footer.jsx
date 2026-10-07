import React from 'react';
import { Shield } from 'lucide-react';

export default function Footer({ setCurrentPage }) {
  return (
    <footer className="footer">
      <div className="container footer-content">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          <div className="brand-icon" style={{ width: '28px', height: '28px' }}>
            <Shield size={16} />
          </div>
          <span style={{ fontWeight: '700', color: 'var(--text-primary)', fontFamily: 'var(--font-heading)' }}>
            TrustHire
          </span>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            — B.Tech CSE Project
          </span>
        </div>

        <div style={{ display: 'flex', gap: '1.5rem', fontSize: '0.85rem' }}>
          <span style={{ cursor: 'pointer' }} onClick={() => setCurrentPage('home')}>Home</span>
          <span style={{ cursor: 'pointer' }} onClick={() => setCurrentPage('analyzer')}>Analyze Job</span>
          <span style={{ cursor: 'pointer' }} onClick={() => setCurrentPage('how-it-works')}>How It Works</span>
          <span style={{ cursor: 'pointer' }} onClick={() => setCurrentPage('about')}>About</span>
        </div>

        <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textAlign: 'right', maxWidth: '400px' }}>
          Machine learning screening signal system. Results do not constitute legal or absolute proof of fraud.
        </div>
      </div>
    </footer>
  );
}
