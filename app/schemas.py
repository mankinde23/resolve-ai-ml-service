
# from pydantic import BaseModel, Field, field_validator


# class PredictionRequest(BaseModel):
#     text: str = Field(min_length=1, max_length=10_000)

#     @field_validator("text")
#     @classmethod
#     def validate_text(cls, value: str) -> str:
#         value = value.strip()

#         if not value:
#             raise ValueError("Ticket text cannot be empty.")

#         return value


# class PredictionResponse(BaseModel):
#     predicted_category: str
#     model_version: str
#     confidence: float | None = None


# class HealthResponse(BaseModel):
#     status: str
#     model_loaded: bool
#     model_version: str

from typing import Literal

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

    # Triage decision
    action: Literal["auto_route", "human_review", "escalate"]
    priority: Literal["normal", "high"]
    requires_human_review: bool
    reason: str

    # Severity assessment
    severity: Literal[
        "critical",
        "high",
        "medium",
        "low",
        "undetermined",
    ]
    severity_reason: str
    needs_more_information: bool


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
