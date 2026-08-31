from fastapi import APIRouter, HTTPException, Request
import joblib

from app.models.schemas import PredictionInput, PredictionOutput
from app.logging_config import logger


router = APIRouter(prefix="/api/v1")


# Load model once
model = joblib.load("ml/saved_model/model.joblib")


@router.get("/")
def root():
    return {"message": "ML API is alive"}


@router.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }


@router.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput, request: Request):
    request_id = request.state.request_id

    try:
        prediction = model.predict([data.features])

        confidence = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([data.features])
            confidence = float(max(probabilities[0]))

        logger.info(
            f"prediction_success "
            f"request_id={request_id} "
            f"prediction={int(prediction[0])} "
            f"confidence={confidence}"
        )

        return {
            "prediction": int(prediction[0]),
            "confidence": confidence,
            "request_id": request_id,
            "model_version": "1.0"
        }

    except Exception as exc:
        logger.error(
            f"prediction_failed "
            f"request_id={request_id} "
            f"error={exc}"
        )

        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        )