import React from 'react';
import { AlertTriangle, CheckCircle2 } from 'lucide-react';

export default function ResultCard({ result }) {
  if (!result) return null;

  const mlPct = result.ml_risk_percentage !== undefined ? result.ml_risk_percentage : result.risk_percentage;
  const finalPct = result.final_risk_percentage !== undefined ? result.final_risk_percentage : result.risk_percentage;
  const assessment = result.trusthire_assessment || result.risk_level;

  const isRisk =
    finalPct >= 35 ||
    (assessment && (assessment.includes('High') || assessment.includes('Moderate'))) ||
    result.prediction === 'Potentially Fraudulent';

  return (
    <div className="result-card">
      <div className="result-item">
        <span className="result-label">TrustHire Assessment</span>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          {isRisk ? (
            <AlertTriangle size={22} color="var(--accent-rose)" />
          ) : (
            <CheckCircle2 size={22} color="var(--accent-emerald)" />
          )}
          <span className="result-value" style={{ color: isRisk ? 'var(--accent-rose)' : 'var(--accent-emerald)' }}>
            {assessment}
          </span>
        </div>
      </div>

      <div className="result-item">
        <span className="result-label">ML Fraud Probability</span>
        <span className="result-value" style={{ color: 'var(--text-secondary)' }}>
          {mlPct}%
        </span>
      </div>

      <div className="result-item">
        <span className="result-label">TrustHire Final Risk</span>
        <span className="result-value" style={{ color: isRisk ? 'var(--accent-rose)' : 'var(--accent-emerald)' }}>
          {finalPct}%
        </span>
      </div>

      <div className="result-item">
        <span className="result-label">Model Engine</span>
        <span className="result-value" style={{ fontSize: '0.95rem' }}>
          {result.model || 'Logistic Regression'}
        </span>
      </div>
    </div>
  );
}
