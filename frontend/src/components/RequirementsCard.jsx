import React from 'react';
import { CheckCircle2, GraduationCap, Clock, Code, Heart, FileText } from 'lucide-react';

export default function RequirementsCard({ requirements }) {
  if (!requirements) return null;

  const { education, experience, technical_skills, soft_skills, other_requirements } = requirements;

  return (
    <div className="card" style={{ marginBottom: '1.5rem' }}>
      <div className="section-title">
        <CheckCircle2 size={18} />
        <span>Structured Requirements</span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '1.25rem' }}>
        {education && (
          <div className="result-item" style={{ background: 'rgba(15, 23, 42, 0.5)', padding: '0.9rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-color)' }}>
            <span className="result-label" style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <GraduationCap size={14} /> Education Requirement
            </span>
            <span style={{ fontSize: '0.95rem', fontWeight: '600' }}>{education}</span>
          </div>
        )}

        {experience && (
          <div className="result-item" style={{ background: 'rgba(15, 23, 42, 0.5)', padding: '0.9rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-color)' }}>
            <span className="result-label" style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <Clock size={14} /> Experience Requirement
            </span>
            <span style={{ fontSize: '0.95rem', fontWeight: '600' }}>{experience}</span>
          </div>
        )}

        {technical_skills && technical_skills.length > 0 && (
          <div className="result-item" style={{ background: 'rgba(15, 23, 42, 0.5)', padding: '0.9rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-color)' }}>
            <span className="result-label" style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <Code size={14} /> Technical Prerequisites
            </span>
            <span style={{ fontSize: '0.95rem', color: 'var(--accent-blue)' }}>{technical_skills.join(', ')}</span>
          </div>
        )}

        {soft_skills && soft_skills.length > 0 && (
          <div className="result-item" style={{ background: 'rgba(15, 23, 42, 0.5)', padding: '0.9rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-color)' }}>
            <span className="result-label" style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <Heart size={14} /> Soft Skills & Mindset
            </span>
            <ul style={{ paddingLeft: '1rem', margin: 0, fontSize: '0.9rem' }}>
              {soft_skills.map((s, idx) => (
                <li key={idx}>{s}</li>
              ))}
            </ul>
          </div>
        )}

        {other_requirements && other_requirements.length > 0 && (
          <div className="result-item" style={{ background: 'rgba(15, 23, 42, 0.5)', padding: '0.9rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-color)', gridColumn: '1 / -1' }}>
            <span className="result-label" style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <FileText size={14} /> Additional Stated Requirements
            </span>
            <ul style={{ paddingLeft: '1rem', margin: '0.4rem 0 0 0', fontSize: '0.9rem' }}>
              {other_requirements.map((r, idx) => (
                <li key={idx} style={{ marginBottom: '0.25rem' }}>{r}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}
