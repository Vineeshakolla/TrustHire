import React from 'react';
import { HelpCircle, AlertCircle, CheckCircle } from 'lucide-react';

export default function SignalCard({ signals = [] }) {
  if (!signals || signals.length === 0) {
    return null;
  }

  return (
    <div className="card" style={{ marginBottom: '1.5rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
        <h3 style={{ fontSize: '1.25rem', color: 'var(--text-primary)' }}>What influenced TrustHire's assessment?</h3>
        <span style={{ fontSize: '0.8rem', color: 'var(--accent-blue)', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
          <HelpCircle size={14} /> Explainable ML Features
        </span>
      </div>

      <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
        Linear log-odds feature coefficients driving the ML classification output:
      </p>

      <div className="signals-container">
        {signals.map((sig, idx) => {
          const isRisk = sig.direction === 'risk_associated';
          return (
            <div key={idx} className={`signal-chip ${isRisk ? 'risk' : 'legitimate'}`}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                {isRisk ? (
                  <AlertCircle size={18} color="var(--accent-rose)" />
                ) : (
                  <CheckCircle size={18} color="var(--accent-emerald)" />
                )}
                <div>
                  <span className="signal-name">{sig.feature}</span>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                    {isRisk ? 'Risk-Associated Feature' : 'Legitimate-Associated Feature'}
                  </div>
                </div>
              </div>
              <span className="signal-importance">
                Weight: {Math.abs(sig.importance).toFixed(4)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
