import os
import pytest
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY", "")


def test_predict(client):
    response = client.post(
        "/api/v1/predict",
        json={
            "features": [0, 0, 0, 0]
        },
        headers={"X-API-Key": API_KEY}
    )

    assert response.status_code == 200


def test_predict_invalid_features(client):
    response = client.post(
        "/api/v1/predict",
        json={
            "features": [0, 0]
        },
        headers={"X-API-Key": API_KEY}
    )

    assert response.status_code == 422


def test_predict_missing_features(client):
    response = client.post(
        "/api/v1/predict",
        json={},
        headers={"X-API-Key": API_KEY}
    )

    assert response.status_code == 422
