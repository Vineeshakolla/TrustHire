/**
 * TrustHire API Service Module
 * Handles all communication with the FastAPI backend at http://127.0.0.1:8000.
 */

const API_BASE_URL = 'http://127.0.0.1:8000';

/**
 * Checks FastAPI backend health status and model availability.
 * @returns {Promise<{online: boolean, status: string, model_loaded: boolean}>}
 */
export async function checkBackendHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}/health`, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
    });

    if (!response.ok) {
      return { online: false, status: 'degraded', model_loaded: false };
    }

    const data = await response.json();
    return {
      online: true,
      status: data.status,
      model_loaded: data.model_loaded,
    };
  } catch (error) {
    return { online: false, status: 'offline', model_loaded: false };
  }
}

/**
 * Sends job posting payload to FastAPI backend for ML prediction screening.
 * @param {Object} jobData
 * @returns {Promise<Object>} API Prediction response
 */
export async function predictJobPosting(jobData) {
  try {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify(jobData),
    });

    const data = await response.json();

    if (!response.ok) {
      const errorMsg = data?.detail?.error || data?.detail || 'Prediction request failed.';
      throw new Error(errorMsg);
    }

    return data;
  } catch (error) {
    if (error.name === 'TypeError' && error.message.includes('fetch')) {
      throw new Error('Unable to connect to TrustHire backend server (http://127.0.0.1:8000). Please ensure the FastAPI backend is running.');
    }
    throw error;
  }
}

/**
 * Sends job posting payload to FastAPI backend for NLP information extraction.
 * @param {Object} jobData
 * @returns {Promise<Object>} NLP Extraction response
 */
export async function analyzeJobPosting(jobData) {
  try {
    const response = await fetch(`${API_BASE_URL}/analyze`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify(jobData),
    });

    const data = await response.json();

    if (!response.ok) {
      const errorMsg = data?.detail?.error || data?.detail || 'NLP extraction request failed.';
      throw new Error(errorMsg);
    }

    return data;
  } catch (error) {
    if (error.name === 'TypeError' && error.message.includes('fetch')) {
      throw new Error('Unable to connect to TrustHire backend server (http://127.0.0.1:8000). Please ensure the FastAPI backend is running.');
    }
    throw error;
  }
}

/**
 * Sends job posting payload for intelligent summary & warning signal analysis.
 * @param {Object} jobData
 * @returns {Promise<Object>} Intelligent Analysis response
 */
export async function intelligentAnalysis(jobData) {
  try {
    const response = await fetch(`${API_BASE_URL}/analyze/intelligent`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify(jobData),
    });

    const data = await response.json();

    if (!response.ok) {
      const errorMsg = data?.detail?.error || data?.detail || 'Intelligent analysis request failed.';
      throw new Error(errorMsg);
    }

    return data;
  } catch (error) {
    if (error.name === 'TypeError' && error.message.includes('fetch')) {
      throw new Error('Unable to connect to TrustHire backend server (http://127.0.0.1:8000). Please ensure the FastAPI backend is running.');
    }
    throw error;
  }
}

/**
 * Answers natural language questions about a job posting using local intent matching.
 * @param {Object} jobData
 * @param {string} question
 * @returns {Promise<Object>} Q&A response
 */
export async function askQuestion(jobData, question) {
  try {
    const response = await fetch(`${API_BASE_URL}/analyze/ask`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify({
        job_posting: jobData,
        question: question,
      }),
    });

    const data = await response.json();

    if (!response.ok) {
      const errorMsg = data?.detail?.error || data?.detail || 'Q&A request failed.';
      throw new Error(errorMsg);
    }

    return data;
  } catch (error) {
    if (error.name === 'TypeError' && error.message.includes('fetch')) {
      throw new Error('Unable to connect to TrustHire backend server (http://127.0.0.1:8000). Please ensure the FastAPI backend is running.');
    }
    throw error;
  }
}

/**
 * Combined end-to-end endpoint executing ML screening, NLP extraction, and intelligent analysis.
 * @param {Object} jobData
 * @returns {Promise<Object>} Full Analysis response
 */
export async function fullAnalysis(jobData) {
  try {
    const response = await fetch(`${API_BASE_URL}/analyze/full`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify(jobData),
    });

    const data = await response.json();

    if (!response.ok) {
      const errorMsg = data?.detail?.error || data?.detail || 'Full analysis request failed.';
      throw new Error(errorMsg);
    }

    return data;
  } catch (error) {
    if (error.name === 'TypeError' && error.message.includes('fetch')) {
      throw new Error('Unable to connect to TrustHire backend server (http://127.0.0.1:8000). Please ensure the FastAPI backend is running.');
    }
    throw error;
  }
}
