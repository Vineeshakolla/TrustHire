import React from 'react';

export default function RiskMeter({ riskPercentage = 0, mlRiskPercentage, riskLevel = 'Low Risk', trusthireAssessment }) {
  const clampedPct = Math.max(0, Math.min(100, Number(riskPercentage) || 0));
  const mlPctVal = mlRiskPercentage !== undefined ? Math.max(0, Math.min(100, Number(mlRiskPercentage) || 0)) : clampedPct;

  let rangeLabel = 'High Possibility of Legitimate';
  let badgeClass = 'low';

  if (clampedPct <= 20) {
    rangeLabel = 'High Possibility of Legitimate';
    badgeClass = 'low';
  } else if (clampedPct <= 40) {
    rangeLabel = 'Possibility of Legitimate';
    badgeClass = 'low';
  } else if (clampedPct <= 60) {
    rangeLabel = 'Moderate Risk — Needs Verification';
    badgeClass = 'moderate';
  } else if (clampedPct <= 80) {
    rangeLabel = 'High Risk';
    badgeClass = 'high';
  } else {
    rangeLabel = 'Very High Risk';
    badgeClass = 'high';
  }

  const assessmentDisplay = trusthireAssessment || rangeLabel;

  return (
    <div className="risk-meter-container">
      <div className="risk-meter-header">
        <div>
          <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)', fontWeight: '600', textTransform: 'uppercase' }}>
            TrustHire Screening Risk
          </span>
          <h3 style={{ fontSize: '1.35rem', marginTop: '0.2rem' }}>Continuous Risk-Fused Assessment</h3>
        </div>
        <div style={{ textAlign: 'right' }}>
          <div className={`risk-score-badge ${badgeClass}`}>
            {clampedPct.toFixed(1)}%
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
            ML Model Probability: {mlPctVal.toFixed(1)}%
          </div>
        </div>
      </div>

      {/* CONTINUOUS HORIZONTAL RISK METER TRACK */}
      <div className="risk-meter-track" title={`TrustHire Screening Risk: ${clampedPct.toFixed(1)}%`}>
        <div
          className="risk-meter-indicator"
          style={{ left: `${clampedPct}%` }}
        ></div>
      </div>

      <div className="risk-meter-labels">
        <span>0% (Legitimate)</span>
        <span>50% Decision Threshold</span>
        <span>100% (High Risk)</span>
      </div>

      <div style={{ marginTop: '1.25rem', paddingTop: '1rem', borderTop: '1px solid var(--border-color)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Screening Risk Level:</span>
          <div style={{ fontWeight: '700', fontSize: '1.1rem', color: badgeClass === 'high' ? 'var(--accent-rose)' : badgeClass === 'moderate' ? 'var(--accent-amber)' : 'var(--accent-emerald)' }}>
            {rangeLabel}
          </div>
        </div>
        <div style={{ textAlign: 'right' }}>
          <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>TrustHire Assessment:</span>
          <div style={{ fontWeight: '700', fontSize: '1.1rem', color: badgeClass === 'high' || badgeClass === 'moderate' ? 'var(--accent-rose)' : 'var(--accent-emerald)' }}>
            {assessmentDisplay}
          </div>
        </div>
      </div>
    </div>
  );
}
