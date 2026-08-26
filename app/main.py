from fastapi import FastAPI
import joblib
import uuid

from app.models.schemas import PredictionInput

app = FastAPI()

# Load model once when the application starts
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


@app.post("/predict")
def predict(data: PredictionInput):
    prediction = model.predict([data.features])

    confidence = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([data.features])
        confidence = float(max(probabilities[0]))

    request_id = str(uuid.uuid4())

    return {
        "prediction": int(prediction[0]),
        "confidence": confidence,
        "request_id": request_id
    }