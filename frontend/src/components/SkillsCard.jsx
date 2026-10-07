import React from 'react';
import { Code } from 'lucide-react';

export default function SkillsCard({ skills }) {
  if (!skills || (!skills.all_skills?.length && !skills.required_skills?.length)) {
    return null;
  }

  const required = skills.required_skills || [];
  const preferred = skills.preferred_skills || [];
  const all = skills.all_skills || [];

  return (
    <div className="card" style={{ marginBottom: '1.5rem' }}>
      <div className="section-title">
        <Code size={18} />
        <span>Technical Skills</span>
      </div>

      {required.length > 0 && (
        <div style={{ marginBottom: '1rem' }}>
          <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)', fontWeight: '600', textTransform: 'uppercase', display: 'block', marginBottom: '0.5rem' }}>
            Required Skills
          </span>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
            {required.map((skill, idx) => (
              <span key={idx} style={{
                background: 'rgba(59, 130, 246, 0.15)',
                border: '1px solid rgba(59, 130, 246, 0.35)',
                color: 'var(--accent-blue)',
                padding: '0.3rem 0.75rem',
                borderRadius: 'var(--radius-full)',
                fontSize: '0.875rem',
                fontWeight: '600'
              }}>
                {skill}
              </span>
            ))}
          </div>
        </div>
      )}

      {preferred.length > 0 && (
        <div style={{ marginBottom: '1rem' }}>
          <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)', fontWeight: '600', textTransform: 'uppercase', display: 'block', marginBottom: '0.5rem' }}>
            Preferred / Bonus Skills
          </span>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
            {preferred.map((skill, idx) => (
              <span key={idx} style={{
                background: 'rgba(6, 182, 212, 0.15)',
                border: '1px solid rgba(6, 182, 212, 0.35)',
                color: 'var(--accent-cyan)',
                padding: '0.3rem 0.75rem',
                borderRadius: 'var(--radius-full)',
                fontSize: '0.875rem',
                fontWeight: '600'
              }}>
                {skill}
              </span>
            ))}
          </div>
        </div>
      )}

      {required.length === 0 && preferred.length === 0 && all.length > 0 && (
        <div>
          <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)', fontWeight: '600', textTransform: 'uppercase', display: 'block', marginBottom: '0.5rem' }}>
            Skills Mentioned in Posting
          </span>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
            {all.map((skill, idx) => (
              <span key={idx} style={{
                background: 'rgba(16, 185, 129, 0.15)',
                border: '1px solid rgba(16, 185, 129, 0.35)',
                color: 'var(--accent-emerald)',
                padding: '0.3rem 0.75rem',
                borderRadius: 'var(--radius-full)',
                fontSize: '0.875rem',
                fontWeight: '600'
              }}>
                {skill}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
