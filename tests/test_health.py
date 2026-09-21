import io

from fastapi.testclient import TestClient
from PIL import Image

from image_classifier.app import app

client = TestClient(app)


def make_test_image() -> bytes:
    image = Image.new("RGB", (64, 64), color="red")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_a_label():
    response = client.post(
        "/predict",
        files={"file": ("test.png", make_test_image(), "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["label"], str)
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_rejects_non_image():
    response = client.post(
        "/predict",
        files={"file": ("notes.txt", b"this is not an image", "text/plain")},
    )
    assert response.status_code == 400