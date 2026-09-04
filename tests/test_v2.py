def test_v1_and_v2_have_different_response_shapes(client):
    payload = {
        "features": [0, 0, 0, 0]
    }

    v1_response = client.post(
        "/api/v1/predict",
        json=payload
    )

    v2_response = client.post(
        "/api/v2/predict",
        json=payload
    )

    assert v1_response.status_code == 200
    assert v2_response.status_code == 200

    v1_data = v1_response.json()
    v2_data = v2_response.json()

    # v1 keeps the old response shape
    assert "confidence" in v1_data
    assert "probabilities" not in v1_data

    # v2 has the deliberately changed response shape
    assert "probabilities" in v2_data
    assert "confidence" not in v2_data

    # Both versions still provide the prediction
    assert isinstance(v1_data["prediction"], int)
    assert isinstance(v2_data["prediction"], int)