# Resolve AI — ML Service

Resolve AI is an AI-powered support-ticket classification and triage application. This repository contains the Python inference service that loads a trained scikit-learn model and vectorizer and exposes predictions through FastAPI.

## Phase One capabilities

- Serves a trained text-classification pipeline through HTTP.
- Accepts ticket text and returns a predicted category.
- Returns model-version metadata and a confidence score when available.
- Exposes a health endpoint.
- Validates input and handles missing model artifacts.
- Includes tests for prediction behavior.

## Architecture

```text
Node.js + Express backend
          |
          | HTTP / JSON
          v
Python + FastAPI ML service
          |
          v
TF-IDF vectorizer + trained scikit-learn classifier
```

The Node backend calls this service. The React frontend does not call it directly.

## Technology

- Python
- FastAPI and Pydantic
- scikit-learn
- TF-IDF text vectorization
- Logistic Regression baseline classifier
- pytest
- `uv` for Python project and dependency management

## Requirements

- A supported Python version compatible with the project dependencies
- `uv`
- The trained model and vectorizer artifacts required by the service

## Local setup

From this repository:

```bash
uv sync
```

Start the service:

```bash
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The service is expected to be available at `http://localhost:8000`.

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Check service health |
| `POST` | `/predict` | Predict a ticket category |

Interactive API documentation is normally available at `http://localhost:8000/docs` while the local service is running.

## Example prediction request

```json
{
  "text": "I cannot reset my account password."
}
```

The response includes the predicted category, model version, and confidence score where available. The exact response schema is defined in the service's Pydantic response model.

## Test and verify

```bash
uv run pytest
```

Run the health check and submit a sample ticket through `/docs` or the backend API to verify that the service can load the model artifacts and return a prediction.

## Model notes

- The current model is a baseline text classifier using TF-IDF features and Logistic Regression.
- The confidence value is a model output score, not a calibrated probability or a guarantee of correctness unless calibration has been separately evaluated.
- Model quality should be reported only after evaluating against a held-out dataset with appropriate metrics, such as precision, recall, F1-score, and a confusion matrix.
- Pickle-based model artifacts should only be loaded from trusted sources.

## Design decisions

- **Separate inference service:** isolates Python ML dependencies from the Node API.
- **Persisted artifacts:** inference loads a previously trained model rather than retraining on each request.
- **API contract:** allows the backend to consume predictions without depending on Python internals.

## Phase One scope

Phase One establishes a working inference service and its integration with the application. Future work may include more rigorous evaluation, dataset quality checks, error analysis, model comparisons, model versioning, monitoring for data drift, and research into advanced NLP or LLM-based approaches.
