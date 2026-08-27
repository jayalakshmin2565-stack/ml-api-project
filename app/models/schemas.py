from pydantic import BaseModel, Field


class PredictionInput(BaseModel):
    features: list[float] = Field(..., min_length=4, max_length=4)


class PredictionOutput(BaseModel):
    prediction: int
    confidence: float | None
    request_id: str
    model_version: str