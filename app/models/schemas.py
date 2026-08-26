from pydantic import BaseModel, Field


class PredictionInput(BaseModel):
    features: list[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Exactly 4 numeric features are required"
    )