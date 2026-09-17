import os
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "http://127.0.0.1:8000"
API_KEY = os.getenv("API_KEY", "")


def test_health():
    response = requests.get(f"{BASE_URL}/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_info():
    response = requests.get(f"{BASE_URL}/api/v1/model-info")
    assert response.status_code == 200

    data = response.json()
    assert "model_version" in data
    assert "training_date" in data


def test_prediction():
    headers = {"X-API-Key": API_KEY}

    payload = {
        "features": [5.1, 3.5, 1.4, 0.2]
    }

    response = requests.post(
        f"{BASE_URL}/api/v1/predict",
        json=payload,
        headers=headers
    )

    assert response.status_code == 200

    data = response.json()
    assert "prediction" in data


def test_batch_prediction():
    headers = {"X-API-Key": API_KEY}

    payload = {
        "inputs": [
            {
                "features": [5.1, 3.5, 1.4, 0.2]
            },
            {
                "features": [6.2, 3.4, 5.4, 2.3]
            }
        ]
    }

    response = requests.post(
        f"{BASE_URL}/api/v1/predict-batch",
        json=payload,
        headers=headers
    )

    assert response.status_code == 200

    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 2
