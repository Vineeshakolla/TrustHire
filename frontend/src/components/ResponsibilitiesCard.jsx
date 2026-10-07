import React from 'react';
import { ListChecks } from 'lucide-react';

export default function ResponsibilitiesCard({ responsibilities }) {
  if (!responsibilities || responsibilities.length === 0) return null;

  return (
    <div className="card" style={{ marginBottom: '1.5rem' }}>
      <div className="section-title">
        <ListChecks size={18} />
        <span>Extracted Responsibilities</span>
      </div>

      <ul style={{ paddingLeft: '1.25rem', display: 'flex', flexDirection: 'column', gap: '0.5rem', color: 'var(--text-primary)' }}>
        {responsibilities.map((resp, idx) => (
          <li key={idx} style={{ lineHeight: '1.5', fontSize: '0.95rem' }}>
            {resp}
          </li>
        ))}
      </ul>
    </div>
  );
}
