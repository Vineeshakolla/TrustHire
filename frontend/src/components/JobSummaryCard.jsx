import React from 'react';
import { FileText, MapPin, Briefcase, DollarSign, Calendar, Clock, GraduationCap, Building } from 'lucide-react';

export default function JobSummaryCard({ summary }) {
  if (!summary) return null;

  return (
    <div className="card" style={{ marginBottom: '1.5rem' }}>
      <div className="section-title">
        <FileText size={18} />
        <span>Job Summary</span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1.25rem' }}>
        <div className="result-item">
          <span className="result-label" style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <Briefcase size={14} /> Role
          </span>
          <span className="result-value" style={{ fontSize: '1.05rem' }}>{summary.role || 'Not specified'}</span>
        </div>

        <div className="result-item">
          <span className="result-label" style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <Building size={14} /> Company
          </span>
          <span className="result-value" style={{ fontSize: '1.05rem' }}>{summary.company || 'Not specified'}</span>
        </div>

        <div className="result-item">
          <span className="result-label" style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <MapPin size={14} /> Location & Mode
          </span>
          <span className="result-value" style={{ fontSize: '1.05rem' }}>
            {summary.location || 'Not specified'} ({summary.work_mode || 'Not specified'})
          </span>
        </div>

        <div className="result-item">
          <span className="result-label" style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <Briefcase size={14} /> Employment Type
          </span>
          <span className="result-value" style={{ fontSize: '1.05rem' }}>{summary.employment_type || 'Not specified'}</span>
        </div>

        <div className="result-item">
          <span className="result-label" style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <DollarSign size={14} /> Salary / Stipend
          </span>
          <span className="result-value" style={{ fontSize: '1.05rem', color: 'var(--accent-emerald)' }}>
            {summary.salary || 'Not specified'}
          </span>
        </div>

        <div className="result-item">
          <span className="result-label" style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <Clock size={14} /> Required Experience
          </span>
          <span className="result-value" style={{ fontSize: '1.05rem' }}>{summary.experience || 'Not specified'}</span>
        </div>

        <div className="result-item">
          <span className="result-label" style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <GraduationCap size={14} /> Required Education
          </span>
          <span className="result-value" style={{ fontSize: '1.05rem' }}>{summary.education || 'Not specified'}</span>
        </div>

        <div className="result-item">
          <span className="result-label" style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <Calendar size={14} /> Duration / Start Date
          </span>
          <span className="result-value" style={{ fontSize: '0.95rem' }}>
            {summary.duration || 'Not specified'} | {summary.start_date || 'Not specified'}
          </span>
        </div>
      </div>
    </div>
  );
}
