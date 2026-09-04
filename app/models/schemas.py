from pydantic import BaseModel, Field

from app.config import settings


class PredictionInput(BaseModel):

    features: list[float] = Field(..., min_length=4, max_length=4)

class PredictionOutput(BaseModel):

    prediction: int

    confidence: float | None

    request_id: str

    model_version: str


class PredictionBatchInput(BaseModel):

    inputs: list[PredictionInput] = Field(
        ...,
        min_length=1,
        max_length=settings.MAX_BATCH_SIZE
    )


class PredictionBatchOutput(BaseModel):

    predictions: list[PredictionOutput]