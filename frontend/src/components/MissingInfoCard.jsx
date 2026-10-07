import React from 'react';
import { HelpCircle, Info } from 'lucide-react';

export default function MissingInfoCard({ missingInformation = [] }) {
  if (!missingInformation || missingInformation.length === 0) return null;

  return (
    <div className="card" style={{ marginBottom: '1.5rem', borderLeft: '4px solid var(--accent-amber)' }}>
      <div className="section-title" style={{ color: 'var(--accent-amber)' }}>
        <HelpCircle size={18} />
        <span>Information Not Specified</span>
      </div>

      <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginBottom: '0.9rem' }}>
        The following standard metadata fields were not explicitly detailed in the posting text:
      </p>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', marginBottom: '1.25rem' }}>
        {missingInformation.map((item, idx) => (
          <span key={idx} style={{
            background: 'rgba(245, 158, 11, 0.15)',
            border: '1px solid rgba(245, 158, 11, 0.35)',
            color: 'var(--accent-amber)',
            padding: '0.35rem 0.8rem',
            borderRadius: 'var(--radius-md)',
            fontSize: '0.875rem',
            fontWeight: '600'
          }}>
            {item}
          </span>
        ))}
      </div>

      <div style={{
        background: 'rgba(15, 23, 42, 0.6)',
        border: '1px solid var(--border-color)',
        borderRadius: 'var(--radius-sm)',
        padding: '0.75rem 0.9rem',
        fontSize: '0.825rem',
        color: 'var(--text-muted)',
        display: 'flex',
        alignItems: 'flex-start',
        gap: '0.5rem'
      }}>
        <Info size={16} style={{ flexShrink: 0, marginTop: '2px', color: 'var(--accent-amber)' }} />
        <span>
          <strong>Note:</strong> Missing information does not mean the posting is fraudulent. It only means the submitted posting does not explicitly provide that information.
        </span>
      </div>
    </div>
  );
}
