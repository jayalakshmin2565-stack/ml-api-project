def test_predict(client):
    response = client.post(
        "/api/v1/predict",
        json={
            "features": [0, 0, 0, 0]
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "confidence" in data
    assert "request_id" in data
    assert "model_version" in data

    assert isinstance(data["prediction"], int)
    assert 0 <= data["confidence"] <= 1


def test_predict_invalid_features(client):
    response = client.post(
        "/api/v1/predict",
        json={
            "features": [0, 0]
        }
    )

    assert response.status_code == 422


def test_predict_missing_features(client):
    response = client.post(
        "/api/v1/predict",
        json={}
    )

    assert response.status_code == 422
