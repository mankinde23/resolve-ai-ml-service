
from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10_000)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Ticket text cannot be empty.")

        return value


class PredictionResponse(BaseModel):
    predicted_category: str
    model_version: str
    confidence: float | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
