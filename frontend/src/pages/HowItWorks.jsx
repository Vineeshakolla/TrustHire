import React from 'react';
import { Database, FileText, Cpu, LineChart, Code, CheckCircle2, Sparkles, MessageSquare, ShieldAlert } from 'lucide-react';

export default function HowItWorks({ setCurrentPage }) {
  return (
    <main className="container" style={{ padding: '3.5rem 1.5rem 5rem' }}>
      <div style={{ textAlign: 'center', marginBottom: '3.5rem' }}>
        <h1 style={{ fontSize: '2.5rem', marginBottom: '0.75rem' }}>How TrustHire Works</h1>
        <p style={{ color: 'var(--text-secondary)', fontSize: '1.1rem', maxWidth: '760px', margin: '0 auto' }}>
          An end-to-end technical explanation of how TrustHire combines ML statistical risk screening with local NLP information extraction and deterministic intelligent Q&A.
        </p>
      </div>

      {/* Visual Step-by-Step Architecture Flow */}
      <div className="card" style={{ marginBottom: '3rem' }}>
        <h2 style={{ fontSize: '1.4rem', marginBottom: '1.5rem', textAlign: 'center', color: 'var(--accent-cyan)' }}>
          8-Step End-to-End System Architecture
        </h2>

        <div className="arch-flow" style={{ background: 'transparent', padding: '0', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
          <div className="flow-step">
            <FileText className="flow-step-icon" size={22} />
            <span className="flow-step-text">STEP 1: Job Posting</span>
          </div>
          <div className="flow-step">
            <CheckCircle2 className="flow-step-icon" size={22} />
            <span className="flow-step-text">STEP 2: Cleaning & Preprocessing</span>
          </div>
          <div className="flow-step">
            <Code className="flow-step-icon" size={22} />
            <span className="flow-step-text">STEP 3: TF-IDF + Features</span>
          </div>
          <div className="flow-step">
            <Cpu className="flow-step-icon" size={22} />
            <span className="flow-step-text">STEP 4: Logistic Regression</span>
          </div>
          <div className="flow-step">
            <LineChart className="flow-step-icon" size={22} />
            <span className="flow-step-text">STEP 5: Risk Score</span>
          </div>
          <div className="flow-step">
            <Database className="flow-step-icon" size={22} />
            <span className="flow-step-text">STEP 6: NLP Extraction</span>
          </div>
          <div className="flow-step">
            <Sparkles className="flow-step-icon" size={22} />
            <span className="flow-step-text">STEP 7: Intelligent Analysis</span>
          </div>
          <div className="flow-step">
            <MessageSquare className="flow-step-icon" size={22} />
            <span className="flow-step-text">STEP 8: TrustHire Insights & Q&A</span>
          </div>
        </div>
      </div>

      {/* 3 Core System Layers Explanation */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '2rem', marginBottom: '3rem' }}>
        {/* Layer 1: ML */}
        <div className="card card-hover">
          <div style={{ width: '44px', height: '44px', borderRadius: '10px', background: 'rgba(59, 130, 246, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--accent-blue)', marginBottom: '1.25rem' }}>
            <Cpu size={24} />
          </div>
          <h3 style={{ fontSize: '1.25rem', marginBottom: '0.75rem' }}>1. Machine Learning Layer</h3>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.925rem', lineHeight: '1.7' }}>
            <strong>Purpose:</strong> Potential fraud risk classification.
          </p>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem', marginTop: '0.5rem', lineHeight: '1.6' }}>
            Applies TF-IDF vectorization (5,000 top n-grams) combined with numerical salary bounds, binary flags, and One-Hot encoded categoricals into a 9,150-feature matrix. Evaluated by a calibrated Scikit-Learn Logistic Regression model (<code>class_weight='balanced'</code>).
          </p>
        </div>

        {/* Layer 2: NLP */}
        <div className="card card-hover">
          <div style={{ width: '44px', height: '44px', borderRadius: '10px', background: 'rgba(6, 182, 212, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--accent-cyan)', marginBottom: '1.25rem' }}>
            <FileText size={24} />
          </div>
          <h3 style={{ fontSize: '1.25rem', marginBottom: '0.75rem' }}>2. NLP Extraction Layer</h3>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.925rem', lineHeight: '1.7' }}>
            <strong>Purpose:</strong> Extracting structured job information.
          </p>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem', marginTop: '0.5rem', lineHeight: '1.6' }}>
            Uses regex patterns and keyphrase matching to extract 18 structured fields: responsibilities, technical/soft skills, salary, location, remote work mode, duration, start date, and missing metadata.
          </p>
        </div>

        {/* Layer 3: Intelligent Analysis & Q&A */}
        <div className="card card-hover">
          <div style={{ width: '44px', height: '44px', borderRadius: '10px', background: 'rgba(16, 185, 129, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--accent-emerald)', marginBottom: '1.25rem' }}>
            <Sparkles size={24} />
          </div>
          <h3 style={{ fontSize: '1.25rem', marginBottom: '0.75rem' }}>3. Intelligent Analysis & Local Q&A</h3>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.925rem', lineHeight: '1.7' }}>
            <strong>Purpose:</strong> Summaries, warning signals, and deterministic Q&A.
          </p>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem', marginTop: '0.5rem', lineHeight: '1.6' }}>
            Generates standardized job highlights, detects rule-based caution signals (urgency, fee requests, missing company info), and answers user questions deterministically without external generative AI APIs or hallucinations.
          </p>
        </div>
      </div>

      <div className="card" style={{ textAlign: 'center', padding: '2rem' }}>
        <h3 style={{ fontSize: '1.3rem', marginBottom: '0.75rem' }}>Ready to test a job posting?</h3>
        <p style={{ color: 'var(--text-secondary)', marginBottom: '1.5rem' }}>
          Analyze a sample Software Engineering Intern posting or paste your own job description.
        </p>
        <button className="btn btn-primary" onClick={() => setCurrentPage('analyzer')}>
          <Cpu size={18} />
          <span>Launch Job Analyzer</span>
        </button>
      </div>
    </main>
  );
}
