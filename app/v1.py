from fastapi import APIRouter, HTTPException, Request, Depends
import joblib
import time
import json

from app.config import settings
from app.models.schemas import (
    PredictionInput,
    PredictionOutput,
    PredictionBatchInput,
    PredictionBatchOutput
)
from app.logging_config import logger
from app.security import verify_api_key


router = APIRouter()


# Load model once using configuration
model = joblib.load(settings.MODEL_PATH)


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
        with open(settings.MODEL_METADATA_PATH, "r") as file:
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


@router.post(
    "/predict",
    response_model=PredictionOutput,
    dependencies=[Depends(verify_api_key)]
)
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
            "model_version": settings.MODEL_VERSION
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


@router.post(
    "/predict-batch",
    response_model=PredictionBatchOutput,
    dependencies=[Depends(verify_api_key)]
)
def predict_batch(data: PredictionBatchInput, request: Request):
    request_id = request.state.request_id

    if len(data.inputs) > settings.MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size cannot exceed {settings.MAX_BATCH_SIZE}"
        )

    start_time = time.time()

    try:
        features = [item.features for item in data.inputs]

        predictions = model.predict(features)

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
                    "model_version": settings.MODEL_VERSION
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
