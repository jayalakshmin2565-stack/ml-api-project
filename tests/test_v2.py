import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY", "")


def test_v1_and_v2_have_different_response_shapes(client):
    payload = {
        "features": [0, 0, 0, 0]
    }

    headers = {
        "X-API-Key": API_KEY
    }

    v1_response = client.post(
        "/api/v1/predict",
        json=payload,
        headers=headers
    )

    v2_response = client.post(
        "/api/v2/predict",
        json=payload,
        headers=headers
    )

    assert v1_response.status_code == 200
    assert v2_response.status_code == 200

    v1_data = v1_response.json()
    v2_data = v2_response.json()

    assert v1_data != v2_data
