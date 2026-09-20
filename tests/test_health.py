from fastapi.testclient import TestClient
from image_classifier.app import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_unknown_path_returns_404():
    response = client.get("/hello")
    assert response.status_code == 404