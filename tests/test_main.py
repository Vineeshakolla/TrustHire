"""
TrustHire - Backend Health Endpoint Tests
"""

from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_health_endpoint():
    """
    Test GET /health returns 200 OK and expected status response.
    """
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "ok"
    assert data.get("project") == "TrustHire"
