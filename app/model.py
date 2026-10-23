
import os
import pickle
from pathlib import Path

from sklearn.linear_model import LogisticRegression


BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = Path(
    os.getenv("MODEL_ARTIFACTS_DIR", str(BASE_DIR / "artifacts"))
)

MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
VECTORIZER_PATH = ARTIFACTS_DIR / "vectorizer.pkl"


class TicketClassifier:
    def __init__(self) -> None:
        self.model = None
        self.vectorizer = None

    def load(self) -> None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Model artifact not found: {MODEL_PATH}"
            )

        if not VECTORIZER_PATH.is_file():
            raise FileNotFoundError(
                f"Vectorizer artifact not found: {VECTORIZER_PATH}"
            )

        # Only load trusted artifacts produced by your own
        # training process. Pickle can execute code on loading.
        with MODEL_PATH.open("rb") as file:
            model = pickle.load(file)

        with VECTORIZER_PATH.open("rb") as file:
            vectorizer = pickle.load(file)

        if not callable(getattr(model, "predict", None)):
            raise TypeError("Loaded model does not support predict().")

        if not callable(getattr(vectorizer, "transform", None)):
            raise TypeError(
                "Loaded vectorizer does not support transform()."
            )

        self.model = model
        self.vectorizer = vectorizer

    def predict(self, text: str) -> dict:
        if self.model is None or self.vectorizer is None:
            raise RuntimeError("The model has not been loaded.")

        features = self.vectorizer.transform([text])
        category = str(self.model.predict(features)[0])

        result = {
            "predicted_category": category,
            "confidence": None,
        }

        if callable(getattr(self.model, "predict_proba", None)):
            probabilities = self.model.predict_proba(features)[0]
            result["confidence"] = float(probabilities.max())

        return result


classifier = TicketClassifier()
