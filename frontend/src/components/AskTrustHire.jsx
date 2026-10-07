import React, { useState } from 'react';
import { MessageSquare, Send, Bot, User, Sparkles, Loader2 } from 'lucide-react';
import { askQuestion } from '../services/api';

const SUGGESTED_QUESTIONS = [
  "What is the salary?",
  "What skills are required?",
  "Is this remote?",
  "What experience is required?",
  "What are the responsibilities?",
  "Is this an internship?",
  "What information is missing?",
  "What warning signals were detected?"
];

export default function AskTrustHire({ jobData }) {
  const [question, setQuestion] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleAsk = async (qText) => {
    const textToAsk = qText || question;
    if (!textToAsk || !textToAsk.strip ? !textToAsk.trim() : !textToAsk) return;

    const userQ = textToAsk.trim();
    setQuestion('');
    setLoading(true);
    setError(null);

    // Add user message to conversation list
    setChatHistory((prev) => [...prev, { sender: 'user', text: userQ }]);

    try {
      const res = await askQuestion(jobData, userQ);
      setChatHistory((prev) => [
        ...prev,
        { sender: 'bot', text: res.answer, confidence: res.confidence, source: res.source }
      ]);
    } catch (err) {
      setError(err.message || 'Failed to get answer from local Q&A engine.');
      setChatHistory((prev) => [
        ...prev,
        { sender: 'bot', text: 'Sorry, I encountered an error answering your question. Please try again.', isError: true }
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card" style={{ marginBottom: '2rem', border: '1px solid var(--border-glow)' }}>
      <div className="section-title">
        <MessageSquare size={18} />
        <span>Ask TrustHire — Intelligent Local Q&A</span>
      </div>

      <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginBottom: '1rem' }}>
        Ask natural language questions about this job posting. All answers are generated locally from the submitted text without hallucination.
      </p>

      {/* Suggested Questions Chips */}
      <div style={{ marginBottom: '1.25rem' }}>
        <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: '600', textTransform: 'uppercase', display: 'block', marginBottom: '0.5rem' }}>
          Suggested Questions:
        </span>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem' }}>
          {SUGGESTED_QUESTIONS.map((sq, idx) => (
            <button
              key={idx}
              type="button"
              onClick={() => handleAsk(sq)}
              disabled={loading}
              style={{
                background: 'rgba(30, 41, 59, 0.8)',
                border: '1px solid var(--border-color)',
                color: 'var(--text-secondary)',
                padding: '0.3rem 0.65rem',
                borderRadius: 'var(--radius-full)',
                fontSize: '0.8rem',
                cursor: 'pointer',
                transition: 'all 0.15s ease',
                display: 'inline-flex',
                alignItems: 'center',
                gap: '0.3rem'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.borderColor = 'var(--accent-blue)';
                e.currentTarget.style.color = 'white';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = 'var(--border-color)';
                e.currentTarget.style.color = 'var(--text-secondary)';
              }}
            >
              <Sparkles size={12} style={{ color: 'var(--accent-blue)' }} />
              <span>{sq}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Conversation Dialog Area */}
      {chatHistory.length > 0 && (
        <div style={{
          background: '#0b1329',
          border: '1px solid var(--border-color)',
          borderRadius: 'var(--radius-md)',
          padding: '1rem',
          maxHeight: '320px',
          overflowY: 'auto',
          display: 'flex',
          flexDirection: 'column',
          gap: '0.9rem',
          marginBottom: '1rem'
        }}>
          {chatHistory.map((msg, idx) => (
            <div
              key={idx}
              style={{
                display: 'flex',
                gap: '0.75rem',
                alignSelf: msg.sender === 'user' ? 'flex-end' : 'flex-start',
                maxWidth: '85%'
              }}
            >
              {msg.sender === 'bot' && (
                <div style={{
                  width: '32px',
                  height: '32px',
                  borderRadius: '50%',
                  background: 'linear-gradient(135deg, var(--accent-blue), var(--accent-cyan))',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'white',
                  flexShrink: 0
                }}>
                  <Bot size={16} />
                </div>
              )}

              <div style={{
                background: msg.sender === 'user' ? 'var(--accent-blue)' : 'var(--bg-card)',
                color: msg.sender === 'user' ? 'white' : 'var(--text-primary)',
                border: msg.sender === 'bot' ? '1px solid var(--border-color)' : 'none',
                padding: '0.75rem 1rem',
                borderRadius: 'var(--radius-md)',
                fontSize: '0.9rem',
                lineHeight: '1.45'
              }}>
                <div>{msg.text}</div>
                {msg.sender === 'bot' && !msg.isError && (
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.4rem', borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: '0.3rem' }}>
                    Source: {msg.source} | Engine: Local Deterministic NLP
                  </div>
                )}
              </div>

              {msg.sender === 'user' && (
                <div style={{
                  width: '32px',
                  height: '32px',
                  borderRadius: '50%',
                  background: 'var(--bg-card)',
                  border: '1px solid var(--border-color)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'var(--text-secondary)',
                  flexShrink: 0
                }}>
                  <User size={16} />
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Input Control Box */}
      <form onSubmit={(e) => { e.preventDefault(); handleAsk(); }} style={{ display: 'flex', gap: '0.5rem' }}>
        <input
          type="text"
          className="form-input"
          placeholder="Ask something about this job (e.g. What is the salary?)"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          disabled={loading}
          style={{ flex: 1 }}
        />
        <button
          type="submit"
          className="btn btn-primary"
          disabled={loading || !question.trim()}
          style={{ padding: '0.65rem 1.25rem' }}
        >
          {loading ? <Loader2 className="animate-spin" size={18} /> : <Send size={18} />}
        </button>
      </form>
    </div>
  );
}
