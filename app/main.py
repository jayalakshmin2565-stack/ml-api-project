from fastapi import FastAPI
import joblib

from app.models.schemas import PredictionInput

app = FastAPI()

model = joblib.load("ml/saved_model/model.joblib")


@app.get("/")
def root():
    return {"message": "ML API is alive"}


@app.post("/predict")
def predict(data: PredictionInput):
    prediction = model.predict([data.features])

    return {
        "prediction": int(prediction[0])
    }