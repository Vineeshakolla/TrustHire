import React, { useState } from 'react';
import { Cpu, RefreshCw, AlertCircle, FileText, CheckSquare, Layers, Trash2 } from 'lucide-react';
import { fullAnalysis } from '../services/api';
import RiskMeter from '../components/RiskMeter';
import ResultCard from '../components/ResultCard';
import SignalCard from '../components/SignalCard';
import JobSummaryCard from '../components/JobSummaryCard';
import SkillsCard from '../components/SkillsCard';
import ResponsibilitiesCard from '../components/ResponsibilitiesCard';
import RequirementsCard from '../components/RequirementsCard';
import MissingInfoCard from '../components/MissingInfoCard';
import WarningSignalsCard from '../components/WarningSignalsCard';
import AskTrustHire from '../components/AskTrustHire';

const SAMPLE_JOB_POSTING = {
  title: 'Software Engineering Intern',
  company_profile: 'This role is for one of our clients',
  description: `Industry: Software Development
Seniority level: Junior - Entry Level
Min Experience: 0 years
Location: Bengaluru (Remote)
JobType: full-time

About The Role

We are looking for a motivated and curious Software Engineering Intern to join the team. This is a hands-on role where you'll work alongside experienced engineers, contribute to real projects, and grow your technical skills in a fast-paced environment.

Stipend: ₹30,000 – ₹50,000 per month

Duration: Minimum 2 months, extendable up to 6 months

Start Date: Immediate / As soon as possible

Responsibilities:
- Assist in the design, development, and testing of software features
- Debug issues and contribute to improving system performance
- Learn and adapt quickly to new tools, technologies, and workflows

Requirements:
- Strong problem-solving skills and eagerness to learn
- Good communication and teamwork skills
- Currently pursuing or recently completed a degree in Computer Science, Engineering, or a related field
- Basic understanding of data structures, algorithms, and software development principles
- Proficiency in at least one programming language (e.g., Python, JavaScript, Java, C++)

Nice to Have:
- Familiarity with web frameworks, cloud platforms, or AI/ML tools
- Prior internship experience or personal projects on GitHub
- Exposure to agile development practices`,
  requirements: 'Proficiency in Python, JavaScript, Java, C++, Data Structures, Algorithms. Computer Science or Engineering degree.',
  benefits: 'Stipend ₹30,000–₹50,000/month, Remote flexibility, 1-on-1 mentorship.',
  salary_range: '30000-50000',
  location: 'Bengaluru (Remote)',
  department: 'Software Development',
  employment_type: 'Full-time',
  required_experience: 'Entry level',
  required_education: "Bachelor's Degree",
  industry: 'Software Development',
  function: 'Engineering',
  telecommuting: true,
  has_company_logo: false,
  has_questions: true,
};

export default function Analyzer() {
  const [formData, setFormData] = useState({
    title: '',
    company_profile: '',
    description: '',
    requirements: '',
    benefits: '',
    salary_range: '',
    location: '',
    department: '',
    employment_type: 'Full-time',
    required_experience: 'Entry level',
    required_education: "Bachelor's Degree",
    industry: 'Software Development',
    function: 'Engineering',
    telecommuting: false,
    has_company_logo: false,
    has_questions: false,
  });

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [fullResult, setFullResult] = useState(null);

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value,
    }));
  };

  const handleLoadSample = () => {
    setFormData(SAMPLE_JOB_POSTING);
    setError(null);
  };

  const handleClear = () => {
    setFormData({
      title: '',
      company_profile: '',
      description: '',
      requirements: '',
      benefits: '',
      salary_range: '',
      location: '',
      department: '',
      employment_type: 'Full-time',
      required_experience: 'Entry level',
      required_education: "Bachelor's Degree",
      industry: 'Software Development',
      function: 'Engineering',
      telecommuting: false,
      has_company_logo: false,
      has_questions: false,
    });
    setError(null);
    setFullResult(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!formData.title.trim()) {
      setError('Job Title is required.');
      return;
    }
    if (!formData.description.trim()) {
      setError('Job Description is required.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const payload = {
        title: formData.title,
        description: formData.description,
        company_profile: formData.company_profile,
        requirements: formData.requirements,
        benefits: formData.benefits,
        salary_range: formData.salary_range || null,
        location: formData.location || 'Unknown',
        department: formData.department || 'Unknown',
        employment_type: formData.employment_type || 'Unknown',
        required_experience: formData.required_experience || 'Unknown',
        required_education: formData.required_education || 'Unknown',
        industry: formData.industry || 'Unknown',
        function: formData.function || 'Unknown',
        telecommuting: Boolean(formData.telecommuting),
        has_company_logo: Boolean(formData.has_company_logo),
        has_questions: Boolean(formData.has_questions),
      };

      const res = await fullAnalysis(payload);
      setFullResult(res);
    } catch (err) {
      setError(err.message || 'An error occurred during full job analysis.');
      setFullResult(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="container" style={{ padding: '3rem 1.5rem 5rem' }}>
      <div style={{ textAlign: 'center', marginBottom: '3rem' }}>
        <h1 style={{ fontSize: '2.5rem', marginBottom: '0.5rem' }}>Analyze a Job Posting</h1>
        <p style={{ color: 'var(--text-secondary)', fontSize: '1.05rem', maxWidth: '680px', margin: '0 auto' }}>
          Paste a complete job or internship description to evaluate fraud risk, extract structured information, and interact with local AI Q&A.
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: fullResult ? '1fr 1.1fr' : '1fr', gap: '2rem' }}>
        {/* Form Container */}
        <div className="card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
            <h2 style={{ fontSize: '1.4rem' }}>Job Posting Inputs</h2>
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              <button type="button" className="btn btn-secondary" onClick={handleLoadSample} style={{ padding: '0.4rem 0.85rem', fontSize: '0.85rem' }}>
                <RefreshCw size={14} />
                <span>Load Sample Job</span>
              </button>
              <button type="button" className="btn btn-secondary" onClick={handleClear} style={{ padding: '0.4rem 0.85rem', fontSize: '0.85rem' }}>
                <Trash2 size={14} />
                <span>Clear</span>
              </button>
            </div>
          </div>

          {error && (
            <div style={{ background: 'rgba(244, 63, 94, 0.1)', border: '1px solid var(--accent-rose)', borderRadius: 'var(--radius-md)', padding: '0.9rem', marginBottom: '1.5rem', color: 'var(--accent-rose)', display: 'flex', alignItems: 'center', gap: '0.6rem', fontSize: '0.9rem' }}>
              <AlertCircle size={18} />
              <span>{error}</span>
            </div>
          )}

          <form onSubmit={handleSubmit}>
            {/* SECTION A — BASIC INFORMATION */}
            <div className="form-section">
              <div className="section-title">
                <Layers size={18} />
                <span>Section A — Basic Information</span>
              </div>
              <div className="form-grid">
                <div className="form-group full-width">
                  <label className="form-label">Job Title *</label>
                  <input
                    type="text"
                    name="title"
                    className="form-input"
                    placeholder="e.g. Software Engineering Intern"
                    value={formData.title}
                    onChange={handleChange}
                    required
                  />
                </div>

                <div className="form-group">
                  <label className="form-label">Location</label>
                  <input
                    type="text"
                    name="location"
                    className="form-input"
                    placeholder="e.g. Bengaluru (Remote)"
                    value={formData.location}
                    onChange={handleChange}
                  />
                </div>

                <div className="form-group">
                  <label className="form-label">Department</label>
                  <input
                    type="text"
                    name="department"
                    className="form-input"
                    placeholder="e.g. Engineering, Software Development"
                    value={formData.department}
                    onChange={handleChange}
                  />
                </div>

                <div className="form-group">
                  <label className="form-label">Employment Type</label>
                  <select name="employment_type" className="form-select" value={formData.employment_type} onChange={handleChange}>
                    <option value="Full-time">Full-time</option>
                    <option value="Part-time">Part-time</option>
                    <option value="Contract">Contract</option>
                    <option value="Temporary">Temporary</option>
                    <option value="Internship">Internship</option>
                    <option value="Unknown">Unknown</option>
                  </select>
                </div>

                <div className="form-group">
                  <label className="form-label">Required Experience</label>
                  <select name="required_experience" className="form-select" value={formData.required_experience} onChange={handleChange}>
                    <option value="Entry level">Entry level</option>
                    <option value="Associate">Associate</option>
                    <option value="Mid-Senior level">Mid-Senior level</option>
                    <option value="Director">Director</option>
                    <option value="Executive">Executive</option>
                    <option value="Not Applicable">Not Applicable</option>
                    <option value="Unknown">Unknown</option>
                  </select>
                </div>

                <div className="form-group">
                  <label className="form-label">Required Education</label>
                  <input
                    type="text"
                    name="required_education"
                    className="form-input"
                    placeholder="e.g. Computer Science / Engineering"
                    value={formData.required_education}
                    onChange={handleChange}
                  />
                </div>

                <div className="form-group">
                  <label className="form-label">Industry</label>
                  <input
                    type="text"
                    name="industry"
                    className="form-input"
                    placeholder="e.g. Software Development"
                    value={formData.industry}
                    onChange={handleChange}
                  />
                </div>
              </div>
            </div>

            {/* SECTION B — JOB DESCRIPTION */}
            <div className="form-section">
              <div className="section-title">
                <FileText size={18} />
                <span>Section B — Job Text & Content</span>
              </div>

              <div className="form-group full-width" style={{ marginBottom: '1rem' }}>
                <label className="form-label">Company Profile</label>
                <textarea
                  name="company_profile"
                  className="form-textarea"
                  placeholder="Overview of company history, mission, or recruitment agency notes..."
                  value={formData.company_profile}
                  onChange={handleChange}
                />
              </div>

              <div className="form-group full-width" style={{ marginBottom: '1rem' }}>
                <label className="form-label">Job Description *</label>
                <textarea
                  name="description"
                  className="form-textarea"
                  placeholder="Paste complete job or internship posting text..."
                  value={formData.description}
                  onChange={handleChange}
                  rows={6}
                  required
                />
              </div>

              <div className="form-group full-width" style={{ marginBottom: '1rem' }}>
                <label className="form-label">Requirements</label>
                <textarea
                  name="requirements"
                  className="form-textarea"
                  placeholder="Technical skills, qualifications, degree..."
                  value={formData.requirements}
                  onChange={handleChange}
                />
              </div>

              <div className="form-group full-width">
                <label className="form-label">Benefits</label>
                <textarea
                  name="benefits"
                  className="form-textarea"
                  placeholder="Stipend, perks, learning allowance..."
                  value={formData.benefits}
                  onChange={handleChange}
                />
              </div>
            </div>

            {/* SECTION C — ADDITIONAL ATTRIBUTES */}
            <div className="form-section">
              <div className="section-title">
                <CheckSquare size={18} />
                <span>Section C — Attributes & Salary</span>
              </div>

              <div className="form-grid">
                <div className="form-group">
                  <label className="form-label">Salary Range</label>
                  <input
                    type="text"
                    name="salary_range"
                    className="form-input"
                    placeholder="e.g. 30000-50000"
                    value={formData.salary_range}
                    onChange={handleChange}
                  />
                </div>

                <div className="form-group">
                  <label className="form-label">Flags & Options</label>
                  <div className="toggle-group" style={{ flexDirection: 'column', alignItems: 'flex-start', gap: '0.5rem' }}>
                    <label style={{ fontSize: '0.875rem', display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
                      <input type="checkbox" name="telecommuting" checked={formData.telecommuting} onChange={handleChange} />
                      Telecommuting / Remote Position
                    </label>
                    <label style={{ fontSize: '0.875rem', display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
                      <input type="checkbox" name="has_company_logo" checked={formData.has_company_logo} onChange={handleChange} />
                      Company Logo Present
                    </label>
                    <label style={{ fontSize: '0.875rem', display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
                      <input type="checkbox" name="has_questions" checked={formData.has_questions} onChange={handleChange} />
                      Screening Questions Included
                    </label>
                  </div>
                </div>
              </div>
            </div>

            <button type="submit" className="btn btn-primary" style={{ width: '100%', marginTop: '1.5rem', padding: '0.85rem' }} disabled={loading}>
              {loading ? (
                <>
                  <RefreshCw className="animate-spin" size={18} />
                  <span>Analyzing job posting...</span>
                </>
              ) : (
                <>
                  <Cpu size={18} />
                  <span>Analyze Job Posting</span>
                </>
              )}
            </button>
          </form>
        </div>

        {/* Results Display Area */}
        {fullResult && (
          <div style={{ display: 'flex', flexFlow: 'column' }}>
            <h2 style={{ fontSize: '1.6rem', marginBottom: '1.25rem' }}>TrustHire Analysis Results</h2>

            {/* Top Section & Model Summary Card */}
            <ResultCard result={fullResult} />

            {/* Continuous Risk Meter */}
            <RiskMeter
              riskPercentage={fullResult.final_risk_percentage !== undefined ? fullResult.final_risk_percentage : fullResult.risk_percentage}
              mlRiskPercentage={fullResult.ml_risk_percentage !== undefined ? fullResult.ml_risk_percentage : fullResult.risk_percentage}
              riskLevel={fullResult.risk_level}
              trusthireAssessment={fullResult.trusthire_assessment || fullResult.risk_level}
              predictionLabel={fullResult.prediction}
            />

            {/* Model Feature Signals */}
            {fullResult.feature_signals && fullResult.feature_signals.length > 0 && (
              <SignalCard signals={fullResult.feature_signals} />
            )}

            {/* Intelligent Job Summary */}
            {fullResult.intelligent_analysis?.summary && (
              <JobSummaryCard summary={fullResult.intelligent_analysis.summary} />
            )}

            {/* Skills Breakdown */}
            {fullResult.intelligent_analysis?.skills && (
              <SkillsCard skills={fullResult.intelligent_analysis.skills} />
            )}

            {/* Responsibilities */}
            {fullResult.intelligent_analysis?.responsibilities && (
              <ResponsibilitiesCard responsibilities={fullResult.intelligent_analysis.responsibilities} />
            )}

            {/* Structured Requirements */}
            {fullResult.intelligent_analysis?.requirements && (
              <RequirementsCard requirements={fullResult.intelligent_analysis.requirements} />
            )}

            {/* Missing Information */}
            {fullResult.intelligent_analysis?.missing_information && (
              <MissingInfoCard missingInformation={fullResult.intelligent_analysis.missing_information} />
            )}

            {/* Warning Signals */}
            {fullResult.intelligent_analysis?.warning_signals && (
              <WarningSignalsCard warningSignals={fullResult.intelligent_analysis.warning_signals} />
            )}

            {/* Interactive Ask TrustHire Q&A */}
            <AskTrustHire jobData={formData} />

            {/* Safety Disclaimer */}
            <div style={{ padding: '1rem', borderRadius: 'var(--radius-md)', background: 'rgba(30, 41, 59, 0.5)', border: '1px solid var(--border-color)', fontSize: '0.825rem', color: 'var(--text-muted)', marginBottom: '2rem' }}>
              <strong>Safety Disclaimer:</strong> {fullResult.disclaimer}
            </div>
          </div>
        )}
      </div>
    </main>
  );
}
