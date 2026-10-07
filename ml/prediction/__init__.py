"""
TrustHire Prediction Package
Exports predict_job_posting for single job evaluation.
"""

from ml.prediction.predict import predict_job_posting, load_prediction_artifacts

__all__ = ["predict_job_posting", "load_prediction_artifacts"]
