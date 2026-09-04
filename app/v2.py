import joblib
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from app.config import settings

router = APIRouter()

model = joblib.load(settings.MODEL_PATH)


class PredictionInputV2(BaseModel):
    features: list[float] = Field(..., min_length=4, max_length=4)


class PredictionOutputV2(BaseModel):
    prediction: int
    probabilities: list[float]
    request_id: str
    model_version: str


@router.post("/predict", response_model=PredictionOutputV2)
def predict_v2(data: PredictionInputV2, request: Request):
    prediction = model.predict([data.features])[0]
    probabilities = model.predict_proba([data.features])[0].tolist()

    return {
        "prediction": int(prediction),
        "probabilities": probabilities,
        "request_id": request.state.request_id,
        "model_version": "2.0"
    }