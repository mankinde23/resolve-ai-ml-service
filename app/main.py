# import streamlit as st
# import pickle

# # Load saved model
# model = pickle.load(open("model.pkl", "rb"))
# vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# st.title("IT Support Ticket Classification System")
# st.write("Enter a ticket description below to predict category.")

# user_input = st.text_area("Ticket Text")

# if st.button("Predict Category"):
#     if user_input:
#         transformed = vectorizer.transform([user_input])
#         prediction = model.predict(transformed)
#         st.success(f"Predicted Category: {prediction[0]}")
#     else:
#         st.warning("Please enter ticket text.")




# import os
# from contextlib import asynccontextmanager

# from fastapi import FastAPI, HTTPException
# from fastapi.responses import JSONResponse

# from app.model import classifier
# from app.schemas import (
#     HealthResponse,
#     PredictionRequest,
#     PredictionResponse,
# )


# MODEL_VERSION = os.getenv("MODEL_VERSION", "baseline-v1")


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     classifier.load()
#     yield


# app = FastAPI(
#     title="Resolve AI ML Service",
#     version="1.0.0",
#     description="ML inference API for support ticket classification.",
#     lifespan=lifespan,
# )


# @app.get("/health", response_model=HealthResponse)
# def health():
#     loaded = classifier.model is not None

#     return {
#         "status": "ok" if loaded else "not_ready",
#         "model_loaded": loaded,
#         "model_version": MODEL_VERSION,
#     }


# @app.post("/predict", response_model=PredictionResponse)
# def predict(request: PredictionRequest):
#     try:
#         result = classifier.predict(request.text)

#         return {
#             **result,
#             "model_version": MODEL_VERSION,
#         }

#     except RuntimeError:
#         raise HTTPException(
#             status_code=503,
#             detail="Prediction service is not ready.",
#         )

#     except Exception:
#         # Log the exception internally in the production version.
#         return JSONResponse(
#             status_code=500,
#             content={"detail": "Prediction failed."},
#         )

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from app.model import classifier
from app.schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)
from app.triage import decide_triage

from app.severity import assess_severity

MODEL_VERSION = os.getenv("MODEL_VERSION", "baseline-v1")
TRIAGE_CONFIDENCE_THRESHOLD = float(
    os.getenv("TRIAGE_CONFIDENCE_THRESHOLD", "0.75")
)

if not 0.0 <= TRIAGE_CONFIDENCE_THRESHOLD <= 1.0:
    raise ValueError(
        "TRIAGE_CONFIDENCE_THRESHOLD must be between 0 and 1."
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    classifier.load()
    yield


app = FastAPI(
    title="Resolve AI ML Service",
    version="1.0.0",
    description=(
        "ML inference API for support ticket classification "
        "and intelligent triage."
    ),
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    loaded = classifier.model is not None

    return {
        "status": "ok" if loaded else "not_ready",
        "model_loaded": loaded,
        "model_version": MODEL_VERSION,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        # prediction = classifier.predict(request.text)

        # triage = decide_triage(
        #     prediction=prediction,
        #     confidence_threshold=TRIAGE_CONFIDENCE_THRESHOLD,
        # )

        # severity_result = assess_severity(request.text)
        prediction = classifier.predict(request.text)
        severity_result = assess_severity(request.text)
        triage = decide_triage(
         prediction,
         TRIAGE_CONFIDENCE_THRESHOLD,
         severity_result,
)
        return {
            **triage,
            "model_version": MODEL_VERSION,
            "severity": severity_result["severity"],
            "severity_reason": severity_result["reason"],
            "needs_more_information": severity_result["needs_more_information"],
        }

    except RuntimeError:
        raise HTTPException(
            status_code=503,
            detail="Prediction service is not ready.",
        )

    except Exception:
        # Replace with structured internal logging in production.
        return JSONResponse(
            status_code=500,
            content={"detail": "Prediction failed."},
        )
