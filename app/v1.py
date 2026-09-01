from fastapi import APIRouter, HTTPException, Request
import joblib
import time
import json

from app.models.schemas import (
    PredictionInput,
    PredictionOutput,
    PredictionBatchInput,
    PredictionBatchOutput
)
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


@router.get("/model-info")
def model_info():
    try:
        with open("ml/saved_model/model_metadata.json", "r") as file:
            metadata = json.load(file)

        return metadata

    except Exception as exc:
        logger.error(
            f"model_info_failed "
            f"error={exc}"
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to load model metadata"
        )


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


# Batch Prediction
@router.post("/predict-batch", response_model=PredictionBatchOutput)
def predict_batch(data: PredictionBatchInput, request: Request):
    request_id = request.state.request_id

    start_time = time.time()

    try:
        # Convert input objects into a list of feature lists
        features = [item.features for item in data.inputs]

        # Predict the entire batch at once
        predictions = model.predict(features)

        # Calculate probabilities for the entire batch once
        probabilities = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)

        results = []

        for i, prediction in enumerate(predictions):

            confidence = None

            if probabilities is not None:
                confidence = float(max(probabilities[i]))

            results.append(
                {
                    "prediction": int(prediction),
                    "confidence": confidence,
                    "request_id": request_id,
                    "model_version": "1.0"
                }
            )

        duration = time.time() - start_time

        logger.info(
            f"prediction_batch_success "
            f"request_id={request_id} "
            f"batch_size={len(data.inputs)} "
            f"duration={duration:.4f}s"
        )

        return {
            "predictions": results
        }

    except Exception as exc:
        duration = time.time() - start_time

        logger.error(
            f"prediction_batch_failed "
            f"request_id={request_id} "
            f"batch_size={len(data.inputs)} "
            f"duration={duration:.4f}s "
            f"error={exc}"
        )

        raise HTTPException(
            status_code=500,
            detail="Batch prediction failed"
        )