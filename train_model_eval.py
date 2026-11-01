
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "aa_dataset-tickets-multi-lang-5-2-50-version.csv"
RESULTS_DIR = BASE_DIR / "results"

TEST_SIZE = 0.20
RANDOM_STATE = 42
MIN_CATEGORY_SUPPORT = 5
MAX_FEATURES = 1000


def build_logistic_regression_pipeline():
    """Match the Logistic Regression configuration used in production."""
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                stop_words="english",
                max_features=MAX_FEATURES,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                solver="lbfgs",
                max_iter=300,
                C=1.0,
                class_weight="balanced",
            ),
        ),
    ])


def build_naive_bayes_pipeline():
    """Keep Naive Bayes as a baseline for comparison."""
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                stop_words="english",
                max_features=MAX_FEATURES,
            ),
        ),
        ("classifier", MultinomialNB()),
    ])


def evaluate_model(name, pipeline, X_train, X_test, y_train, y_test):
    start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    training_seconds = time.perf_counter() - start

    start = time.perf_counter()
    predictions = pipeline.predict(X_test)
    inference_ms = (
        time.perf_counter() - start
    ) / len(X_test) * 1000

    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "train_time_sec": round(training_seconds, 3),
        "inference_ms_per_ticket": round(inference_ms, 4),
    }

    print(f"\n===== {name} =====")
    for metric, value in metrics.items():
        if metric != "model":
            print(f"{metric}: {value:.4f}")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    return pipeline, predictions, metrics


def evaluate_confidence(pipeline, X_test, y_test, predictions):
    """
    Examine whether higher model confidence is associated
    with more correct predictions.

    These are diagnostic measurements, not proof that
    probabilities are calibrated.
    """
    probabilities = pipeline.predict_proba(X_test)
    confidence = probabilities.max(axis=1)

    results = pd.DataFrame({
        "text": X_test.to_numpy(),
        "actual_category": y_test.to_numpy(),
        "predicted_category": predictions,
        "confidence": confidence,
    })

    results["correct"] = (
        results["actual_category"] == results["predicted_category"]
    )

    results.to_csv(
        RESULTS_DIR / "confidence_predictions.csv",
        index=False,
    )

    results.loc[~results["correct"]].to_csv(
        RESULTS_DIR / "misclassified_examples.csv",
        index=False,
    )

    # Group predictions into confidence ranges.
    bins = np.linspace(0, 1, 11)

    results["confidence_range"] = pd.cut(
        results["confidence"],
        bins=bins,
        include_lowest=True,
    )

    confidence_report = (
        results.groupby(
            "confidence_range",
            observed=False,
        )
        .agg(
            ticket_count=("correct", "size"),
            average_confidence=("confidence", "mean"),
            accuracy=("correct", "mean"),
        )
        .reset_index()
    )

    confidence_report.to_csv(
        RESULTS_DIR / "confidence_analysis.csv",
        index=False,
    )

    # Expected calibration error (ECE), using 10 equal-width bins.
    populated_bins = confidence_report[
        confidence_report["ticket_count"] > 0
    ]

    ece = (
        (
            populated_bins["ticket_count"]
            / len(results)
        )
        * (
            populated_bins["accuracy"]
            - populated_bins["average_confidence"]
        ).abs()
    ).sum()

    print("\n===== CONFIDENCE ANALYSIS =====")
    print(f"Average confidence: {confidence.mean():.4f}")
    print(f"Accuracy: {results['correct'].mean():.4f}")
    print(f"Expected calibration error (ECE): {ece:.4f}")

    print("\nConfidence ranges:")
    print(confidence_report.to_string(index=False))

    return results, confidence_report, ece


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    required_columns = {"language", "body", "tag_1"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing columns: {sorted(missing_columns)}"
        )

    # Keep English tickets and exclude multi-label records.
    df = df.loc[
        df["language"].eq("en")
        & ~df["tag_1"].astype("string").str.contains(",", na=False),
        ["body", "tag_1"],
    ].dropna().copy()

    # Normalize whitespace and remove empty records.
    df["body"] = df["body"].astype(str).str.strip()
    df["tag_1"] = df["tag_1"].astype(str).str.strip()

    df = df[
        df["body"].ne("") & df["tag_1"].ne("")
    ].copy()

    # Remove exact duplicate text-label pairs.
    df = df.drop_duplicates(subset=["body", "tag_1"])

    # Exclude categories with insufficient examples.
    category_counts = df["tag_1"].value_counts()

    valid_categories = category_counts[
        category_counts >= MIN_CATEGORY_SUPPORT
    ].index

    excluded_categories = category_counts[
        category_counts < MIN_CATEGORY_SUPPORT
    ]

    df = df[df["tag_1"].isin(valid_categories)].copy()

    if df["tag_1"].nunique() < 2:
        raise ValueError("At least two categories are required.")

    if df["tag_1"].value_counts().min() < 2:
        raise ValueError(
            "Each retained category needs at least two examples."
        )

    X = df["body"]
    y = df["tag_1"]

    # Split before fitting TF-IDF to prevent test-set leakage.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\n===== DATASET SUMMARY =====")
    print(f"Retained records: {len(df)}")
    print(f"Retained categories: {y.nunique()}")
    print(f"Excluded rare categories: {len(excluded_categories)}")
    print(f"Training examples: {len(X_train)}")
    print(f"Test examples: {len(X_test)}")

    category_summary = pd.DataFrame({
        "category": category_counts.index,
        "total_records": category_counts.values,
    })

    category_summary["included"] = (
        category_summary["total_records"] >= MIN_CATEGORY_SUPPORT
    )

    category_summary.to_csv(
        RESULTS_DIR / "category_distribution.csv",
        index=False,
    )

    # Evaluate both models on the same split.
    models = [
        ("Logistic Regression", build_logistic_regression_pipeline()),
        ("Naive Bayes", build_naive_bayes_pipeline()),
    ]

    comparisons = []
    reports = []
    logistic_predictions = None
    logistic_pipeline = None

    for name, pipeline in models:
        fitted, predictions, metrics = evaluate_model(
            name,
            pipeline,
            X_train,
            X_test,
            y_train,
            y_test,
        )

        comparisons.append(metrics)

        report = classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        )

        pd.DataFrame(report).transpose().to_csv(
            RESULTS_DIR / f"{name.lower().replace(' ', '_')}_report.csv"
        )

        reports.append(
            f"===== {name} =====\n"
            + classification_report(
                y_test,
                predictions,
                zero_division=0,
            )
        )

        if name == "Logistic Regression":
            logistic_pipeline = fitted
            logistic_predictions = predictions

    # Detailed confidence analysis for Logistic Regression.
    evaluate_confidence(
        logistic_pipeline,
        X_test,
        y_test,
        logistic_predictions,
    )

    # Save model comparison.
    pd.DataFrame(comparisons).to_csv(
        RESULTS_DIR / "model_comparison.csv",
        index=False,
    )

    (RESULTS_DIR / "classification_report.txt").write_text(
        "\n\n".join(reports),
        encoding="utf-8",
    )

    # Save confusion matrix.
    labels = sorted(y.unique())

    matrix = confusion_matrix(
        y_test,
        logistic_predictions,
        labels=labels,
    )

    plt.figure(figsize=(16, 13))

    sns.heatmap(
        matrix,
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )

    plt.title("Production Logistic Regression - Held-out Test Set")
    plt.xlabel("Predicted category")
    plt.ylabel("Actual category")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "confusion_matrix.png",
        dpi=150,
    )

    plt.close()

    print("\nEvaluation completed.")
    print(f"Results saved in: {RESULTS_DIR}")
    print("Production artifacts were not modified.")


if __name__ == "__main__":
    main()




















# from pathlib import Path
# import time

# import matplotlib
# matplotlib.use("Agg")

# import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sns

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import (
#     accuracy_score,
#     classification_report,
#     confusion_matrix,
#     precision_recall_fscore_support,
# )
# from sklearn.model_selection import train_test_split
# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.pipeline import Pipeline


# BASE_DIR = Path(__file__).resolve().parent
# DATA_PATH = BASE_DIR / "aa_dataset-tickets-multi-lang-5-2-50-version.csv"
# RESULTS_DIR = BASE_DIR / "results"

# TEST_SIZE = 0.20
# RANDOM_STATE = 42
# MIN_CATEGORY_SUPPORT = 5
# MAX_FEATURES = 1000


# def build_logistic_regression_pipeline():
#     return Pipeline([
#         (
#             "tfidf",
#             TfidfVectorizer(
#                 stop_words="english",
#                 max_features=MAX_FEATURES,
#             ),
#         ),
#         (
#             "classifier",
#             OneVsRestClassifier(
#                 LogisticRegression(
#                     solver="liblinear",
#                     max_iter=300,
#                     C=1.0,
#                     class_weight="balanced",
#                 )
#             ),
#         ),
#     ])


# def build_naive_bayes_pipeline():
#     return Pipeline([
#         (
#             "tfidf",
#             TfidfVectorizer(
#                 stop_words="english",
#                 max_features=MAX_FEATURES,
#             ),
#         ),
#         ("classifier", MultinomialNB()),
#     ])


# def evaluate_model(name, pipeline, X_train, X_test, y_train, y_test):
#     start = time.perf_counter()
#     pipeline.fit(X_train, y_train)
#     training_seconds = time.perf_counter() - start

#     start = time.perf_counter()
#     predictions = pipeline.predict(X_test)
#     inference_ms = (
#         time.perf_counter() - start
#     ) / len(X_test) * 1000

#     precision, recall, f1, _ = precision_recall_fscore_support(
#         y_test,
#         predictions,
#         average="macro",
#         zero_division=0,
#     )

#     metrics = {
#         "model": name,
#         "accuracy": accuracy_score(y_test, predictions),
#         "macro_precision": precision,
#         "macro_recall": recall,
#         "macro_f1": f1,
#         "train_time_sec": round(training_seconds, 3),
#         "inference_ms_per_ticket": round(inference_ms, 4),
#     }

#     print(f"\n===== {name} =====")
#     for metric, value in metrics.items():
#         if metric != "model":
#             print(f"{metric}: {value:.4f}")

#     print(
#         classification_report(
#             y_test,
#             predictions,
#             zero_division=0,
#         )
#     )

#     return pipeline, predictions, metrics


# def main():
#     RESULTS_DIR.mkdir(parents=True, exist_ok=True)

#     print("Loading dataset...")
#     df = pd.read_csv(DATA_PATH)

#     required_columns = {"language", "body", "tag_1"}
#     missing_columns = required_columns - set(df.columns)

#     if missing_columns:
#         raise ValueError(
#             f"Dataset is missing columns: {sorted(missing_columns)}"
#         )

#     # Keep English tickets and the text/label columns.
#     # df = df.loc[
#     #     df["language"].eq("en"),
#     #     ["body", "tag_1"],
#     # ].dropna().copy()
#     df = df.loc[
#     df["language"].eq("en")
#     & ~df["tag_1"].str.contains(",", na=False),
#     ["body", "tag_1"],
# ].dropna().copy()

#     # Normalize whitespace and remove empty text or labels.
#     df["body"] = df["body"].astype(str).str.strip()
#     df["tag_1"] = df["tag_1"].astype(str).str.strip()

#     df = df[
#         df["body"].ne("") & df["tag_1"].ne("")
#     ].copy()

#     # Exact duplicate text-label pairs do not add new examples.
#     df = df.drop_duplicates(subset=["body", "tag_1"])

#     # Keep the existing exploratory policy. Categories below this
#     # threshold are excluded, and the exclusions are documented.
#     category_counts = df["tag_1"].value_counts()
#     valid_categories = category_counts[
#         category_counts >= MIN_CATEGORY_SUPPORT
#     ].index

#     excluded_categories = category_counts[
#         category_counts < MIN_CATEGORY_SUPPORT
#     ]

#     df = df[df["tag_1"].isin(valid_categories)].copy()

#     if df["tag_1"].nunique() < 2:
#         raise ValueError("At least two categories are required.")

#     if df["tag_1"].value_counts().min() < 2:
#         raise ValueError(
#             "Each retained category needs at least two examples "
#             "for a stratified split."
#         )

#     X = df["body"]
#     y = df["tag_1"]

#     # Split raw text FIRST. Each pipeline learns its own vocabulary
#     # exclusively from the training examples.
#     X_train, X_test, y_train, y_test = train_test_split(
#         X,
#         y,
#         test_size=TEST_SIZE,
#         random_state=RANDOM_STATE,
#         stratify=y,
#     )

#     print(f"Retained records: {len(df)}")
#     print(f"Retained categories: {y.nunique()}")
#     print(f"Excluded rare categories: {len(excluded_categories)}")
#     print(f"Training examples: {len(X_train)}")
#     print(f"Test examples: {len(X_test)}")
#     print(f"Training/test split: {1 - TEST_SIZE:.0%}/{TEST_SIZE:.0%}")

#     category_summary = pd.DataFrame({
#         "category": category_counts.index,
#         "total_records": category_counts.values,
#     })
#     category_summary["included"] = (
#         category_summary["total_records"] >= MIN_CATEGORY_SUPPORT
#     )
#     category_summary.to_csv(
#         RESULTS_DIR / "category_distribution.csv",
#         index=False,
#     )

#     models = [
#         ("Logistic Regression", build_logistic_regression_pipeline()),
#         ("Naive Bayes", build_naive_bayes_pipeline()),
#     ]

#     comparisons = []
#     reports = []
#     logistic_predictions = None

#     for name, pipeline in models:
#         fitted, predictions, metrics = evaluate_model(
#             name,
#             pipeline,
#             X_train,
#             X_test,
#             y_train,
#             y_test,
#         )

#         comparisons.append(metrics)

#         reports.append(
#             f"===== {name} =====\n"
#             + classification_report(
#                 y_test,
#                 predictions,
#                 zero_division=0,
#             )
#         )

#         if name == "Logistic Regression":
#             logistic_predictions = predictions

#             # Save a sample of mistakes for manual error analysis.
#             mistakes = pd.DataFrame({
#                 "text": X_test.to_numpy(),
#                 "actual_category": y_test.to_numpy(),
#                 "predicted_category": predictions,
#             })

#             mistakes = mistakes[
#                 mistakes["actual_category"]
#                 != mistakes["predicted_category"]
#             ]

#             mistakes.to_csv(
#                 RESULTS_DIR / "misclassified_examples.csv",
#                 index=False,
#             )

#     pd.DataFrame(comparisons).to_csv(
#         RESULTS_DIR / "model_comparison.csv",
#         index=False,
#     )

#     (RESULTS_DIR / "classification_report.txt").write_text(
#         "\n\n".join(reports),
#         encoding="utf-8",
#     )

#     # Save a labeled confusion matrix for Logistic Regression.
#     labels = sorted(y_test.unique())
#     matrix = confusion_matrix(
#         y_test,
#         logistic_predictions,
#         labels=labels,
#     )

#     plt.figure(figsize=(16, 13))
#     sns.heatmap(
#         matrix,
#         cmap="Blues",
#         xticklabels=labels,
#         yticklabels=labels,
#     )
#     plt.title("Logistic Regression - Held-out Test Set")
#     plt.xlabel("Predicted category")
#     plt.ylabel("Actual category")
#     plt.xticks(rotation=90)
#     plt.yticks(rotation=0)
#     plt.tight_layout()
#     plt.savefig(
#         RESULTS_DIR / "confusion_matrix.png",
#         dpi=150,
#     )
#     plt.close()

#     print("\nEvaluation completed.")
#     print(f"Results saved in: {RESULTS_DIR}")


# if __name__ == "__main__":
#     main()














# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import pickle
# import seaborn as sns
# import matplotlib.pyplot as plt
# import structlog

# logger = structlog.stdlib.get_logger()

# logger.info("Starting Evaluation.....")



# # Load dataset
# df = pd.read_csv("aa_dataset-tickets-multi-lang-5-2-50-version.csv")

# # Use English only
# df = df[df['language'] == 'en']
# df = df[['body', 'tag_1']].dropna()

# X = df['body']
# y = df['tag_1']

# # TF-IDF vectorization
# logger.info("Tokenizing dataset........")
# vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
# X_tfidf = vectorizer.fit_transform(X)

# # Split dataset for testing
# X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# # Train Logistic Regression
# logger.info("Training model......")
# model = LogisticRegression(max_iter=300, solver='lbfgs')
# model.fit(X_train, y_train)

# # Save model and vectorizer
# pickle.dump(model, open("model.pkl", "wb"))
# pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

# # Predictions
# logger.info("Making predictions.....")
# y_pred = model.predict(X_test)

# # Accuracy
# logger.info("Evaluation completed...")
# accuracy = accuracy_score(y_test, y_pred)
# logger.info(f"Accuracy: {accuracy}")

# # Classification report
# logger.info(f"\nClassification Report: {classification_report(y_test, y_pred)}")

# # Confusion matrix
# cm = confusion_matrix(y_test, y_pred)
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
# plt.title("Confusion Matrix")
# plt.ylabel("Actual")
# plt.xlabel("Predicted")
# plt.show()

# # Subscription issues

# # I want to unsubscribe from the service


# # Expected output: Subscription

# # Payment issues

# # My payment did not go through and I need help


# # Expected output: Payment

# # Technical issues

# # The app keeps crashing when I try to login


# # Expected output: Technical

# # General inquiries

# # How can I check my subscription status?


# # Expected output: Info / Inquiry


"""
Primary-dataset evaluation script.

Reproduces:
  - Table 4.2 (LR vs Naive Bayes overall comparison, incl. training time)
  - Table 4.3 (per-class precision/recall/F1 for Logistic Regression)
  - The confusion matrix discussed in Chapter 4

Also measures per-ticket inference latency, since the report claims
"under 100 milliseconds" (Section 4.x) but no existing script measured it.

Logistic Regression is configured per Table 3.5 in the report:
solver='liblinear', max_iter=300, C=1.0, penalty='l2',
multi_class='ovr', class_weight='balanced'.

Run from the project root, with the dataset CSV in the working
directory (or update DATA_PATH below).
"""

# import os
# import time
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import (
#     accuracy_score,
#     precision_recall_fscore_support,
#     classification_report,
#     confusion_matrix,
# )

# DATA_PATH = "aa_dataset-tickets-multi-lang-5-2-50-version.csv"
# RESULTS_DIR = "results"

# # 1. Load and filter (see preprocessing.py for the documented pipeline)
# df = pd.read_csv(DATA_PATH)
# df = df[df["language"] == "en"]
# df = df[["body", "tag_1"]].dropna()

# X = df["body"]
# y = df["tag_1"]

# # 2. TF-IDF vectorization (Table 3.4 config: 1,000-feature vocabulary)
# vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
# X_tfidf = vectorizer.fit_transform(X)

# # 3. 80/20 train/test split (as reported in Section 4.3)
# X_train, X_test, y_train, y_test = train_test_split(
#     X_tfidf, y, test_size=0.2, random_state=42, stratify=y
# )

# results_summary = []

# # 4. Logistic Regression — config matches Table 3.5 exactly
# lr = LogisticRegression(
#     solver="liblinear",
#     max_iter=300,
#     C=1.0,
#     penalty="l2",
#     multi_class="ovr",
#     class_weight="balanced",
# )

# t0 = time.time()
# lr.fit(X_train, y_train)
# lr_train_time = time.time() - t0

# t0 = time.time()
# lr_pred = lr.predict(X_test)
# lr_infer_ms = ((time.time() - t0) / len(X_test)) * 1000

# lr_acc = accuracy_score(y_test, lr_pred)
# lr_p, lr_r, lr_f1, _ = precision_recall_fscore_support(
#     y_test, lr_pred, average="macro", zero_division=0
# )

# print("\n===== LOGISTIC REGRESSION =====")
# print(f"Accuracy: {lr_acc:.4f}")
# print(f"Macro Precision: {lr_p:.4f} | Macro Recall: {lr_r:.4f} | Macro F1: {lr_f1:.4f}")
# print(f"Training time: {lr_train_time:.2f}s")
# print(f"Inference time: {lr_infer_ms:.3f} ms/ticket")
# print(classification_report(y_test, lr_pred, zero_division=0))

# results_summary.append({
#     "model": "Logistic Regression",
#     "accuracy": lr_acc,
#     "macro_precision": lr_p,
#     "macro_recall": lr_r,
#     "macro_f1": lr_f1,
#     "train_time_sec": round(lr_train_time, 3),
#     "inference_ms_per_ticket": round(lr_infer_ms, 4),
# })

# # 5. Naive Bayes baseline
# nb = MultinomialNB()

# t0 = time.time()
# nb.fit(X_train, y_train)
# nb_train_time = time.time() - t0

# t0 = time.time()
# nb_pred = nb.predict(X_test)
# nb_infer_ms = ((time.time() - t0) / len(X_test)) * 1000

# nb_acc = accuracy_score(y_test, nb_pred)
# nb_p, nb_r, nb_f1, _ = precision_recall_fscore_support(
#     y_test, nb_pred, average="macro", zero_division=0
# )

# print("\n===== NAIVE BAYES (BASELINE) =====")
# print(f"Accuracy: {nb_acc:.4f}")
# print(f"Macro Precision: {nb_p:.4f} | Macro Recall: {nb_r:.4f} | Macro F1: {nb_f1:.4f}")
# print(f"Training time: {nb_train_time:.2f}s")
# print(f"Inference time: {nb_infer_ms:.3f} ms/ticket")
# print(classification_report(y_test, nb_pred, zero_division=0))

# results_summary.append({
#     "model": "Naive Bayes",
#     "accuracy": nb_acc,
#     "macro_precision": nb_p,
#     "macro_recall": nb_r,
#     "macro_f1": nb_f1,
#     "train_time_sec": round(nb_train_time, 3),
#     "inference_ms_per_ticket": round(nb_infer_ms, 4),
# })

# # 6. Save outputs matching your results/ folder structure
# os.makedirs(RESULTS_DIR, exist_ok=True)

# pd.DataFrame(results_summary).to_csv(f"{RESULTS_DIR}/model_comparison.csv", index=False)

# with open(f"{RESULTS_DIR}/classification_report.txt", "w") as f:
#     f.write("LOGISTIC REGRESSION\n")
#     f.write(classification_report(y_test, lr_pred, zero_division=0))
#     f.write("\n\nNAIVE BAYES\n")
#     f.write(classification_report(y_test, nb_pred, zero_division=0))

# cm = confusion_matrix(y_test, lr_pred)
# plt.figure(figsize=(10, 8))
# sns.heatmap(cm, cmap="Blues")
# plt.title("Confusion Matrix - Logistic Regression (Primary Dataset)")
# plt.ylabel("Actual")
# plt.xlabel("Predicted")
# plt.tight_layout()
# plt.savefig(f"{RESULTS_DIR}/confusion_matrix.png", dpi=150)
# plt.close()

# print(f"\nSaved: {RESULTS_DIR}/model_comparison.csv")
# print(f"Saved: {RESULTS_DIR}/classification_report.txt")
# print(f"Saved: {RESULTS_DIR}/confusion_matrix.png")

# import os
# import time
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import (
#     accuracy_score,
#     precision_recall_fscore_support,
#     classification_report,
#     confusion_matrix,
# )

# DATA_PATH = "aa_dataset-tickets-multi-lang-5-2-50-version.csv"
# RESULTS_DIR = "results"
# MIN_CATEGORY_SUPPORT = 5


# # ============================================================
# # 1. LOAD AND FILTER DATASET
# # ============================================================

# df = pd.read_csv(DATA_PATH)

# # Use English tickets only
# df = df[df["language"] == "en"]

# # Keep only required columns and remove missing values
# df = df[["body", "tag_1"]].dropna()

# # Remove categories with fewer than 5 records
# category_counts = df["tag_1"].value_counts()

# valid_categories = category_counts[
#     category_counts >= MIN_CATEGORY_SUPPORT
# ].index

# df = df[df["tag_1"].isin(valid_categories)]

# print(
#     f"Categories after filtering "
#     f"(min support={MIN_CATEGORY_SUPPORT}): "
#     f"{df['tag_1'].nunique()}"
# )

# print(f"Records after filtering: {len(df)}")


# X = df["body"]
# y = df["tag_1"]


# # ============================================================
# # 2. TF-IDF VECTORIZATION
# # ============================================================

# vectorizer = TfidfVectorizer(
#     stop_words="english",
#     max_features=1000
# )

# X_tfidf = vectorizer.fit_transform(X)


# # ============================================================
# # 3. STRATIFIED 80/20 TRAIN/TEST SPLIT
# # ============================================================

# X_train, X_test, y_train, y_test = train_test_split(
#     X_tfidf,
#     y,
#     test_size=0.2,
#     random_state=42,
#     stratify=y
# )

# print(f"Training records: {X_train.shape[0]}")
# print(f"Testing records: {X_test.shape[0]}")


# results_summary = []


# # ============================================================
# # 4. LOGISTIC REGRESSION
# # ============================================================

# # One-vs-Rest wrapper is used because newer versions
# # of scikit-learn no longer accept multi_class="ovr"
# # directly inside LogisticRegression.

# lr = OneVsRestClassifier(
#     LogisticRegression(
#         solver="liblinear",
#         max_iter=300,
#         C=1.0,
#         class_weight="balanced",
#     )
# )


# # Training time
# t0 = time.time()

# lr.fit(X_train, y_train)

# lr_train_time = time.time() - t0


# # Prediction / inference time
# t0 = time.time()

# lr_pred = lr.predict(X_test)

# lr_infer_ms = (
#     (time.time() - t0) / X_test.shape[0]
# ) * 1000


# # Metrics
# lr_acc = accuracy_score(
#     y_test,
#     lr_pred
# )

# lr_p, lr_r, lr_f1, _ = precision_recall_fscore_support(
#     y_test,
#     lr_pred,
#     average="macro",
#     zero_division=0
# )


# print("\n===== LOGISTIC REGRESSION =====")

# print(
#     f"Accuracy: {lr_acc:.4f}"
# )

# print(
#     f"Macro Precision: {lr_p:.4f} | "
#     f"Macro Recall: {lr_r:.4f} | "
#     f"Macro F1: {lr_f1:.4f}"
# )

# print(
#     f"Training time: "
#     f"{lr_train_time:.2f}s"
# )

# print(
#     f"Inference time: "
#     f"{lr_infer_ms:.3f} ms/ticket"
# )

# print("\nClassification Report:")
# print(
#     classification_report(
#         y_test,
#         lr_pred,
#         zero_division=0
#     )
# )


# results_summary.append({
#     "model": "Logistic Regression",
#     "accuracy": lr_acc,
#     "macro_precision": lr_p,
#     "macro_recall": lr_r,
#     "macro_f1": lr_f1,
#     "train_time_sec": round(
#         lr_train_time,
#         3
#     ),
#     "inference_ms_per_ticket": round(
#         lr_infer_ms,
#         4
#     ),
# })


# # ============================================================
# # 5. NAIVE BAYES BASELINE
# # ============================================================

# nb = MultinomialNB()


# # Training time
# t0 = time.time()

# nb.fit(X_train, y_train)

# nb_train_time = time.time() - t0


# # Prediction / inference time
# t0 = time.time()

# nb_pred = nb.predict(X_test)

# nb_infer_ms = (
#     (time.time() - t0) / X_test.shape[0]
# ) * 1000


# # Metrics
# nb_acc = accuracy_score(
#     y_test,
#     nb_pred
# )

# nb_p, nb_r, nb_f1, _ = precision_recall_fscore_support(
#     y_test,
#     nb_pred,
#     average="macro",
#     zero_division=0
# )


# print("\n===== NAIVE BAYES (BASELINE) =====")

# print(
#     f"Accuracy: {nb_acc:.4f}"
# )

# print(
#     f"Macro Precision: {nb_p:.4f} | "
#     f"Macro Recall: {nb_r:.4f} | "
#     f"Macro F1: {nb_f1:.4f}"
# )

# print(
#     f"Training time: "
#     f"{nb_train_time:.2f}s"
# )

# print(
#     f"Inference time: "
#     f"{nb_infer_ms:.3f} ms/ticket"
# )

# print("\nClassification Report:")
# print(
#     classification_report(
#         y_test,
#         nb_pred,
#         zero_division=0
#     )
# )


# results_summary.append({
#     "model": "Naive Bayes",
#     "accuracy": nb_acc,
#     "macro_precision": nb_p,
#     "macro_recall": nb_r,
#     "macro_f1": nb_f1,
#     "train_time_sec": round(
#         nb_train_time,
#         3
#     ),
#     "inference_ms_per_ticket": round(
#         nb_infer_ms,
#         4
#     ),
# })


# # ============================================================
# # 6. CREATE RESULTS DIRECTORY
# # ============================================================

# os.makedirs(
#     RESULTS_DIR,
#     exist_ok=True
# )


# # ============================================================
# # 7. SAVE MODEL COMPARISON
# # ============================================================

# comparison_df = pd.DataFrame(
#     results_summary
# )

# comparison_df.to_csv(
#     f"{RESULTS_DIR}/model_comparison.csv",
#     index=False
# )


# # ============================================================
# # 8. SAVE CLASSIFICATION REPORT
# # ============================================================

# with open(
#     f"{RESULTS_DIR}/classification_report.txt",
#     "w"
# ) as f:

#     f.write(
#         "LOGISTIC REGRESSION\n"
#     )

#     f.write(
#         classification_report(
#             y_test,
#             lr_pred,
#             zero_division=0
#         )
#     )

#     f.write(
#         "\n\nNAIVE BAYES\n"
#     )

#     f.write(
#         classification_report(
#             y_test,
#             nb_pred,
#             zero_division=0
#         )
#     )


# # ============================================================
# # 9. CONFUSION MATRIX
# # ============================================================

# cm = confusion_matrix(
#     y_test,
#     lr_pred
# )

# plt.figure(
#     figsize=(12, 10)
# )

# sns.heatmap(
#     cm,
#     cmap="Blues"
# )

# plt.title(
#     "Confusion Matrix - "
#     "Logistic Regression "
#     "(Primary Dataset)"
# )

# plt.ylabel(
#     "Actual"
# )

# plt.xlabel(
#     "Predicted"
# )

# plt.tight_layout()

# plt.savefig(
#     f"{RESULTS_DIR}/confusion_matrix.png",
#     dpi=150
# )

# plt.close()


# # ============================================================
# # 10. FINISHED
# # ============================================================

# print(
#     f"\nSaved: "
#     f"{RESULTS_DIR}/model_comparison.csv"
# )

# print(
#     f"Saved: "
#     f"{RESULTS_DIR}/classification_report.txt"
# )

# print(
#     f"Saved: "
#     f"{RESULTS_DIR}/confusion_matrix.png"
# )

# print("\nEvaluation completed successfully!")

