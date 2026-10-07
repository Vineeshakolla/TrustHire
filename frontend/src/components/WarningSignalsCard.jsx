import React from 'react';
import { AlertTriangle, ShieldAlert } from 'lucide-react';

export default function WarningSignalsCard({ warningSignals = [] }) {
  if (!warningSignals || warningSignals.length === 0) return null;

  return (
    <div className="card" style={{ marginBottom: '1.5rem', borderLeft: '4px solid var(--accent-rose)' }}>
      <div className="section-title" style={{ color: 'var(--accent-rose)' }}>
        <AlertTriangle size={18} />
        <span>Potential Warning Signals</span>
      </div>

      <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginBottom: '1rem' }}>
        Rule-based NLP analysis flagged the following details requiring candidate verification:
      </p>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.9rem' }}>
        {warningSignals.map((ws, idx) => (
          <div key={idx} style={{
            background: 'rgba(244, 63, 94, 0.08)',
            border: '1px solid rgba(244, 63, 94, 0.25)',
            borderRadius: 'var(--radius-md)',
            padding: '1rem'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: '700', fontSize: '0.95rem', color: 'var(--accent-rose)', marginBottom: '0.4rem' }}>
              <ShieldAlert size={16} />
              <span>Potential Warning Signal: {ws.signal}</span>
            </div>

            {ws.evidence && (
              <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '0.4rem' }}>
                <strong>Evidence / Matched Wording:</strong> <code style={{ background: 'rgba(15, 23, 42, 0.8)', padding: '0.15rem 0.4rem', borderRadius: '4px', color: 'var(--text-primary)' }}>"{ws.evidence}"</code>
              </div>
            )}

            <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
              <strong>Explanation:</strong> {ws.explanation}
            </div>
          </div>
        ))}
      </div>

      <div style={{ marginTop: '1rem', padding: '0.75rem', borderRadius: 'var(--radius-sm)', background: 'rgba(15, 23, 42, 0.5)', border: '1px solid var(--border-color)', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
        <strong>Important:</strong> Warning signals are informational flags for applicant awareness. They do not constitute proof of fraud. Model fraud classification is determined by the ML risk classifier above.
      </div>
    </div>
  );
}
