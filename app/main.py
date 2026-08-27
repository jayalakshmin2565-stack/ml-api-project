from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import joblib
import uuid
import time

from app.models.schemas import PredictionInput, PredictionOutput
from app.logging_config import logger

app = FastAPI()

# Load model once when the application starts
model = joblib.load("ml/saved_model/model.joblib")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    start_time = time.time()

    try:
        response = await call_next(request)

        duration = time.time() - start_time

        logger.info(
            f"request_id={request_id} "
            f"method={request.method} "
            f"path={request.url.path} "
            f"status_code={response.status_code} "
            f"duration={duration:.4f}s"
        )

        response.headers["X-Request-ID"] = request_id

        return response

    except Exception as exc:
        duration = time.time() - start_time

        logger.error(
            f"request_id={request_id} "
            f"method={request.method} "
            f"path={request.url.path} "
            f"duration={duration:.4f}s "
            f"error={exc}"
        )

        raise


@app.get("/")
def root():
    return {"message": "ML API is alive"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }


@app.post("/predict", response_model=PredictionOutput)
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


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    request_id = getattr(request.state, "request_id", "unknown")

    logger.error(
        f"value_error "
        f"request_id={request_id} "
        f"error={exc}"
    )

    return JSONResponse(
        status_code=500,
        content={"detail": "Invalid prediction data"}
    )