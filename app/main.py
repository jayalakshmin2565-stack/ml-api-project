from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import joblib
import uuid

from app.models.schemas import PredictionInput, PredictionOutput

app = FastAPI()

model = joblib.load("ml/saved_model/model.joblib")


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
def predict(data: PredictionInput):
    try:
        prediction = model.predict([data.features])

        confidence = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([data.features])
            confidence = float(max(probabilities[0]))

        request_id = str(uuid.uuid4())

        return {
            "prediction": int(prediction[0]),
            "confidence": confidence,
            "request_id": request_id,
            "model_version": "1.0"
        }

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=500,
        content={"detail": "Invalid prediction data"}
    )