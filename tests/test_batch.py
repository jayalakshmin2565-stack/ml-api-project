def test_predict_batch(client):
    response = client.post(
        "/api/v1/predict-batch",
        json={
            "inputs": [
                {
                    "features": [0, 0, 0, 0]
                },
                {
                    "features": [1, 1, 1, 1]
                }
            ]
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert "predictions" in data
    assert len(data["predictions"]) == 2


def test_predict_batch_too_large(client):
    inputs = [
        {
            "features": [0, 0, 0, 0]
        }
        for _ in range(101)
    ]

    response = client.post(
        "/api/v1/predict-batch",
        json={
            "inputs": inputs
        }
    )

    assert response.status_code == 422

