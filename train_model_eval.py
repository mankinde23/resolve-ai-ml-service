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

import os
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

DATA_PATH = "aa_dataset-tickets-multi-lang-5-2-50-version.csv"
RESULTS_DIR = "results"
MIN_CATEGORY_SUPPORT = 5


# ============================================================
# 1. LOAD AND FILTER DATASET
# ============================================================

df = pd.read_csv(DATA_PATH)

# Use English tickets only
df = df[df["language"] == "en"]

# Keep only required columns and remove missing values
df = df[["body", "tag_1"]].dropna()

# Remove categories with fewer than 5 records
category_counts = df["tag_1"].value_counts()

valid_categories = category_counts[
    category_counts >= MIN_CATEGORY_SUPPORT
].index

df = df[df["tag_1"].isin(valid_categories)]

print(
    f"Categories after filtering "
    f"(min support={MIN_CATEGORY_SUPPORT}): "
    f"{df['tag_1'].nunique()}"
)

print(f"Records after filtering: {len(df)}")


X = df["body"]
y = df["tag_1"]


# ============================================================
# 2. TF-IDF VECTORIZATION
# ============================================================

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=1000
)

X_tfidf = vectorizer.fit_transform(X)


# ============================================================
# 3. STRATIFIED 80/20 TRAIN/TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training records: {X_train.shape[0]}")
print(f"Testing records: {X_test.shape[0]}")


results_summary = []


# ============================================================
# 4. LOGISTIC REGRESSION
# ============================================================

# One-vs-Rest wrapper is used because newer versions
# of scikit-learn no longer accept multi_class="ovr"
# directly inside LogisticRegression.

lr = OneVsRestClassifier(
    LogisticRegression(
        solver="liblinear",
        max_iter=300,
        C=1.0,
        class_weight="balanced",
    )
)


# Training time
t0 = time.time()

lr.fit(X_train, y_train)

lr_train_time = time.time() - t0


# Prediction / inference time
t0 = time.time()

lr_pred = lr.predict(X_test)

lr_infer_ms = (
    (time.time() - t0) / X_test.shape[0]
) * 1000


# Metrics
lr_acc = accuracy_score(
    y_test,
    lr_pred
)

lr_p, lr_r, lr_f1, _ = precision_recall_fscore_support(
    y_test,
    lr_pred,
    average="macro",
    zero_division=0
)


print("\n===== LOGISTIC REGRESSION =====")

print(
    f"Accuracy: {lr_acc:.4f}"
)

print(
    f"Macro Precision: {lr_p:.4f} | "
    f"Macro Recall: {lr_r:.4f} | "
    f"Macro F1: {lr_f1:.4f}"
)

print(
    f"Training time: "
    f"{lr_train_time:.2f}s"
)

print(
    f"Inference time: "
    f"{lr_infer_ms:.3f} ms/ticket"
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        lr_pred,
        zero_division=0
    )
)


results_summary.append({
    "model": "Logistic Regression",
    "accuracy": lr_acc,
    "macro_precision": lr_p,
    "macro_recall": lr_r,
    "macro_f1": lr_f1,
    "train_time_sec": round(
        lr_train_time,
        3
    ),
    "inference_ms_per_ticket": round(
        lr_infer_ms,
        4
    ),
})


# ============================================================
# 5. NAIVE BAYES BASELINE
# ============================================================

nb = MultinomialNB()


# Training time
t0 = time.time()

nb.fit(X_train, y_train)

nb_train_time = time.time() - t0


# Prediction / inference time
t0 = time.time()

nb_pred = nb.predict(X_test)

nb_infer_ms = (
    (time.time() - t0) / X_test.shape[0]
) * 1000


# Metrics
nb_acc = accuracy_score(
    y_test,
    nb_pred
)

nb_p, nb_r, nb_f1, _ = precision_recall_fscore_support(
    y_test,
    nb_pred,
    average="macro",
    zero_division=0
)


print("\n===== NAIVE BAYES (BASELINE) =====")

print(
    f"Accuracy: {nb_acc:.4f}"
)

print(
    f"Macro Precision: {nb_p:.4f} | "
    f"Macro Recall: {nb_r:.4f} | "
    f"Macro F1: {nb_f1:.4f}"
)

print(
    f"Training time: "
    f"{nb_train_time:.2f}s"
)

print(
    f"Inference time: "
    f"{nb_infer_ms:.3f} ms/ticket"
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        nb_pred,
        zero_division=0
    )
)


results_summary.append({
    "model": "Naive Bayes",
    "accuracy": nb_acc,
    "macro_precision": nb_p,
    "macro_recall": nb_r,
    "macro_f1": nb_f1,
    "train_time_sec": round(
        nb_train_time,
        3
    ),
    "inference_ms_per_ticket": round(
        nb_infer_ms,
        4
    ),
})


# ============================================================
# 6. CREATE RESULTS DIRECTORY
# ============================================================

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# 7. SAVE MODEL COMPARISON
# ============================================================

comparison_df = pd.DataFrame(
    results_summary
)

comparison_df.to_csv(
    f"{RESULTS_DIR}/model_comparison.csv",
    index=False
)


# ============================================================
# 8. SAVE CLASSIFICATION REPORT
# ============================================================

with open(
    f"{RESULTS_DIR}/classification_report.txt",
    "w"
) as f:

    f.write(
        "LOGISTIC REGRESSION\n"
    )

    f.write(
        classification_report(
            y_test,
            lr_pred,
            zero_division=0
        )
    )

    f.write(
        "\n\nNAIVE BAYES\n"
    )

    f.write(
        classification_report(
            y_test,
            nb_pred,
            zero_division=0
        )
    )


# ============================================================
# 9. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    lr_pred
)

plt.figure(
    figsize=(12, 10)
)

sns.heatmap(
    cm,
    cmap="Blues"
)

plt.title(
    "Confusion Matrix - "
    "Logistic Regression "
    "(Primary Dataset)"
)

plt.ylabel(
    "Actual"
)

plt.xlabel(
    "Predicted"
)

plt.tight_layout()

plt.savefig(
    f"{RESULTS_DIR}/confusion_matrix.png",
    dpi=150
)

plt.close()


# ============================================================
# 10. FINISHED
# ============================================================

print(
    f"\nSaved: "
    f"{RESULTS_DIR}/model_comparison.csv"
)

print(
    f"Saved: "
    f"{RESULTS_DIR}/classification_report.txt"
)

print(
    f"Saved: "
    f"{RESULTS_DIR}/confusion_matrix.png"
)

print("\nEvaluation completed successfully!")

