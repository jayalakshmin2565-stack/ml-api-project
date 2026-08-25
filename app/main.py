from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

model = joblib.load("ml/saved_model/model.joblib")


class PredictionInput(BaseModel):
    features: list[float]


@app.get("/")
def root():
    return {"message": "ML API is alive"}


@app.post("/predict")
def predict(data: PredictionInput):
    prediction = model.predict([data.features])

    return {
        "prediction": int(prediction[0])
    }
