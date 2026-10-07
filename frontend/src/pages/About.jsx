import React from 'react';
import { ShieldCheck, Server, Code2, Database, Cpu, CheckCircle2 } from 'lucide-react';

export default function About() {
  const candidateModels = [
    { name: 'Logistic Regression', precision: '0.6301', recall: '0.9012', f1: '0.7416', roc: '0.9857', status: 'Selected Model', selected: true },
    { name: 'Linear SVM', precision: '0.8324', recall: '0.8372', f1: '0.8348', roc: '0.9834', status: 'Candidate', selected: false },
    { name: 'Multinomial Naive Bayes', precision: '0.3664', recall: '0.9012', f1: '0.5210', roc: '0.9720', status: 'Candidate', selected: false },
    { name: 'Random Forest', precision: '0.9798', recall: '0.5640', f1: '0.7159', roc: '0.9890', status: 'Candidate', selected: false },
    { name: 'XGBoost', precision: '0.6535', recall: '0.8663', f1: '0.7450', roc: '0.9899', status: 'Candidate', selected: false },
  ];

  return (
    <main className="container" style={{ padding: '3.5rem 1.5rem 5rem' }}>
      <div style={{ textAlign: 'center', marginBottom: '3.5rem' }}>
        <h1 style={{ fontSize: '2.5rem', marginBottom: '0.75rem' }}>About TrustHire</h1>
        <p style={{ color: 'var(--accent-cyan)', fontSize: '1.2rem', fontWeight: '600', maxWidth: '720px', margin: '0 auto 1rem' }}>
          Machine Learning-Based Detection of Fraudulent Job and Internship Postings
        </p>
        <p style={{ color: 'var(--text-secondary)', fontSize: '1.05rem', maxWidth: '780px', margin: '0 auto' }}>
          TrustHire is an academic machine-learning screening system designed to assist job seekers and university placement cells in evaluating online job and internship postings for potential risk signals.
        </p>
      </div>

      {/* EMSCAD Dataset Section */}
      <section className="card" style={{ marginBottom: '3rem' }}>
        <h2 style={{ fontSize: '1.5rem', marginBottom: '1rem', color: 'var(--accent-cyan)' }}>
          Benchmark Dataset: EMSCAD
        </h2>
        <p style={{ color: 'var(--text-secondary)', lineHeight: '1.7', marginBottom: '1.25rem' }}>
          The machine-learning pipeline was trained on the benchmark <strong>Employment Scam Aegean Dataset (EMSCAD)</strong>, containing <strong>17,880 original job postings</strong>.
        </p>
        <p style={{ color: 'var(--text-secondary)', lineHeight: '1.7', marginBottom: '1.5rem' }}>
          After data validation and duplicate removal, the processed dataset contains <strong>17,645 postings</strong> (16,787 legitimate and 858 fraudulent postings). An 80/20 stratified split yielded 14,116 training samples and 3,529 test samples.
        </p>

        <div style={{ display: 'flex', gap: '2rem', flexWrap: 'wrap', borderTop: '1px solid var(--border-color)', paddingTop: '1.25rem' }}>
          <div>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Raw Postings</span>
            <div style={{ fontSize: '1.5rem', fontWeight: '800', color: 'var(--text-primary)' }}>17,880</div>
          </div>
          <div>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Processed Postings</span>
            <div style={{ fontSize: '1.5rem', fontWeight: '800', color: 'var(--accent-blue)' }}>17,645</div>
          </div>
          <div>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Training Set</span>
            <div style={{ fontSize: '1.5rem', fontWeight: '800', color: 'var(--accent-emerald)' }}>14,116</div>
          </div>
          <div>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Test Evaluation Set</span>
            <div style={{ fontSize: '1.5rem', fontWeight: '800', color: 'var(--accent-amber)' }}>3,529</div>
          </div>
        </div>
      </section>

      {/* Candidate Model Comparison */}
      <section className="card" style={{ marginBottom: '3rem' }}>
        <h2 style={{ fontSize: '1.5rem', marginBottom: '1rem' }}>Candidate Model Evaluation & Selection</h2>
        <p style={{ color: 'var(--text-secondary)', marginBottom: '1.5rem', lineHeight: '1.6' }}>
          Five candidate machine-learning algorithms were trained and systematically evaluated on the benchmark test set (3,529 postings):
        </p>

        <div style={{ overflowX: 'auto', marginBottom: '1.5rem' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.9rem' }}>
            <thead>
              <tr style={{ borderBottom: '2px solid var(--border-color)', color: 'var(--text-muted)' }}>
                <th style={{ padding: '0.75rem 0.5rem' }}>Model</th>
                <th style={{ padding: '0.75rem 0.5rem' }}>Precision</th>
                <th style={{ padding: '0.75rem 0.5rem' }}>Recall</th>
                <th style={{ padding: '0.75rem 0.5rem' }}>F1-Score</th>
                <th style={{ padding: '0.75rem 0.5rem' }}>ROC-AUC</th>
                <th style={{ padding: '0.75rem 0.5rem' }}>Status</th>
              </tr>
            </thead>
            <tbody>
              {candidateModels.map((m, idx) => (
                <tr key={idx} style={{
                  borderBottom: '1px solid var(--border-color)',
                  background: m.selected ? 'rgba(16, 185, 129, 0.1)' : 'transparent',
                  fontWeight: m.selected ? '700' : '400'
                }}>
                  <td style={{ padding: '0.75rem 0.5rem', color: m.selected ? 'var(--accent-emerald)' : 'var(--text-primary)' }}>{m.name}</td>
                  <td style={{ padding: '0.75rem 0.5rem' }}>{m.precision}</td>
                  <td style={{ padding: '0.75rem 0.5rem' }}>{m.recall}</td>
                  <td style={{ padding: '0.75rem 0.5rem' }}>{m.f1}</td>
                  <td style={{ padding: '0.75rem 0.5rem' }}>{m.roc}</td>
                  <td style={{ padding: '0.75rem 0.5rem' }}>
                    {m.selected ? (
                      <span style={{ color: 'var(--accent-emerald)', display: 'inline-flex', alignItems: 'center', gap: '0.3rem' }}>
                        <CheckCircle2 size={14} /> Selected Model
                      </span>
                    ) : (
                      <span style={{ color: 'var(--text-muted)' }}>Candidate</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: '1.25rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-color)' }}>
          <h3 style={{ fontSize: '1.1rem', marginBottom: '0.5rem', color: 'var(--accent-emerald)' }}>
            Why Logistic Regression was Selected
          </h3>
          <ul style={{ listStyleType: 'disc', paddingLeft: '1.25rem', color: 'var(--text-secondary)', fontSize: '0.9rem', lineHeight: '1.7' }}>
            <li><strong>High Fraud Recall (90.12%):</strong> Crucial in fraud detection to catch as many fraudulent postings as possible.</li>
            <li><strong>Sparse Matrix Suitability:</strong> Optimized for high-dimensional sparse TF-IDF vectors (9,150 features).</li>
            <li><strong>Interpretable Log-Odds Coefficients:</strong> Enables direct feature signal explanations ("Why did TrustHire flag this posting?").</li>
            <li><strong>Well-Calibrated Probabilities:</strong> Provides smooth probability scores across 0–100%.</li>
            <li><strong>Computational Efficiency:</strong> Fast runtime inference for real-time API responses.</li>
          </ul>
        </div>
      </section>

      {/* Safety Notice */}
      <div style={{ padding: '1.25rem', borderRadius: 'var(--radius-lg)', background: 'rgba(59, 130, 246, 0.08)', border: '1px solid var(--border-glow)', fontSize: '0.9rem', color: 'var(--text-secondary)', textAlign: 'center' }}>
        <strong>Academic Project Note:</strong> TrustHire is designed as a machine-learning screening system to assist users in reviewing potentially fraudulent job postings. All prediction outputs are probabilistic screening signals and do not replace human verification.
      </div>
    </main>
  );
}
