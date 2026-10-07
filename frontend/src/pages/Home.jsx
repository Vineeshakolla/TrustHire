import React from 'react';
import { ArrowRight, Cpu, FileText, Database, LineChart, ShieldCheck, Sparkles } from 'lucide-react';

export default function Home({ setCurrentPage }) {
  return (
    <main className="container" style={{ paddingBottom: '4rem' }}>
      {/* Hero Section */}
      <section className="hero">
        <div className="hero-badge">
          <Sparkles size={14} />
          <span>B.Tech CSE Project — Machine Learning Screening</span>
        </div>

        <h1 className="hero-title">
          TrustHire
        </h1>

        <p style={{ fontSize: '1.3rem', color: 'var(--accent-cyan)', fontWeight: '600', marginBottom: '0.75rem' }}>
          Machine Learning-Based Detection of Fraudulent Job and Internship Postings
        </p>

        <p className="hero-subtitle">
          Analyze online job and internship postings using NLP, TF-IDF vectorization, structured metadata extraction, and calibrated machine learning models to identify potentially fraudulent patterns.
        </p>

        <div className="hero-actions">
          <button className="btn btn-primary" onClick={() => setCurrentPage('analyzer')}>
            <Cpu size={18} />
            <span>Analyze a Job</span>
            <ArrowRight size={16} />
          </button>

          <button className="btn btn-secondary" onClick={() => setCurrentPage('how-it-works')}>
            <FileText size={18} />
            <span>How It Works</span>
          </button>
        </div>

        {/* Visual Architecture Flow Diagram */}
        <div className="arch-flow">
          <div className="flow-step">
            <FileText className="flow-step-icon" size={24} />
            <span className="flow-step-text">Job Posting Input</span>
          </div>
          <span className="flow-arrow">➔</span>
          <div className="flow-step">
            <Database className="flow-step-icon" size={24} />
            <span className="flow-step-text">NLP & Feature Pipeline</span>
          </div>
          <span className="flow-arrow">➔</span>
          <div className="flow-step">
            <Cpu className="flow-step-icon" size={24} />
            <span className="flow-step-text">Logistic Regression</span>
          </div>
          <span className="flow-arrow">➔</span>
          <div className="flow-step">
            <LineChart className="flow-step-icon" size={24} />
            <span className="flow-step-text">Risk Score & Signals</span>
          </div>
        </div>
      </section>

      {/* Feature Cards Grid */}
      <section className="feature-grid">
        <div className="card card-hover">
          <div style={{ width: '44px', height: '44px', borderRadius: '10px', background: 'rgba(59, 130, 246, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--accent-blue)', marginBottom: '1.25rem' }}>
            <FileText size={24} />
          </div>
          <h3 style={{ fontSize: '1.25rem', marginBottom: '0.5rem' }}>NLP & TF-IDF Analysis</h3>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.925rem' }}>
            Applies Scikit-Learn TF-IDF vectorization across combined job posting titles, company profiles, descriptions, requirements, and benefits to extract 5,000 text n-gram features.
          </p>
        </div>

        <div className="card card-hover">
          <div style={{ width: '44px', height: '44px', borderRadius: '10px', background: 'rgba(6, 182, 212, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--accent-cyan)', marginBottom: '1.25rem' }}>
            <Database size={24} />
          </div>
          <h3 style={{ fontSize: '1.25rem', marginBottom: '0.5rem' }}>Structured Attributes</h3>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.925rem' }}>
            Parses numerical salary bounds, derived metadata lengths, binary indicators (logo, screening questions), and One-Hot encoded categorical attributes (location, industry, function).
          </p>
        </div>

        <div className="card card-hover">
          <div style={{ width: '44px', height: '44px', borderRadius: '10px', background: 'rgba(16, 185, 129, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--accent-emerald)', marginBottom: '1.25rem' }}>
            <ShieldCheck size={24} />
          </div>
          <h3 style={{ fontSize: '1.25rem', marginBottom: '0.5rem' }}>Explainable Model Signals</h3>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.925rem' }}>
            Exposes linear log-odds feature coefficients to provide transparent explanation signals showing specific textual terms and metadata contributing to the risk assessment.
          </p>
        </div>
      </section>
    </main>
  );
}
