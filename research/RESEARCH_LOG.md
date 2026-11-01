# Resolve AI — Research Log

## 1. Project Overview

**Project:** Resolve AI  
**Research Area:** Artificial Intelligence, Machine Learning, Natural Language Processing  
**Application Domain:** IT Support Ticket Classification and Intelligent Triage  
**Primary Model:** TF-IDF + Logistic Regression  
**Development Architecture:** React/TypeScript frontend, Node.js backend, Python ML service.

### Research Objective

To investigate and develop an intelligent IT support ticket classification and triage system capable of categorizing support requests, assessing prediction confidence, and supporting decisions about automated handling and human intervention.

The research will investigate the effectiveness and limitations of traditional machine learning approaches and explore methods for improving classification reliability.

---

## 2. Research Questions

The current research will investigate:

1. How effectively can traditional machine learning models classify IT support tickets into predefined categories?
2. How does Logistic Regression compare with Naive Bayes for this classification task?
3. How do category imbalance and insufficient training examples affect classification performance?
4. How does semantic overlap between ticket categories affect prediction quality?
5. How reliable are the confidence scores produced by the classification model?
6. Can confidence-aware decision policies improve the reliability of automated ticket triage?

---

## 3. Dataset Description

**Dataset:** `aa_dataset-tickets-multi-lang-5-2-50-version.csv`

Relevant columns:

- `language`: Ticket language.
- `body`: Ticket description.
- `tag_1`: Ticket category label.

### Dataset preparation

The current evaluation experiment:

- Retains English-language records.
- Excludes comma-separated multi-label records.
- Removes missing and empty ticket descriptions.
- Removes missing and empty labels.
- Removes exact duplicate text-label pairs.
- Excludes categories with fewer than five examples.

### Evaluation dataset statistics

| Measurement | Value |
|---|---:|
| Retained records | 16,259 |
| Retained categories | 32 |
| Excluded rare categories | 44 |
| Training examples | 13,007 |
| Test examples | 3,252 |
| Training/test split | 80/20 |
| Random state | 42 |

**Important:** These figures describe the current evaluation dataset, not necessarily the exact dataset used by the full production training script.

---

## 4. Experiment Log

### EXP-001: Baseline Model Comparison

**Objective:** Establish baseline performance for Logistic Regression and Naive Bayes on IT support ticket classification.

**Hypothesis:** Logistic Regression will provide stronger overall category discrimination than Naive Bayes, particularly when evaluating macro-averaged performance.

#### Experimental configuration

**TF-IDF vectorizer**

- Stop words: English
- Maximum features: 1,000

**Logistic Regression**

- Solver: `lbfgs`
- Maximum iterations: 300
- Regularization parameter: 1.0
- Class weighting: `balanced`

**Naive Bayes**

- Algorithm: Multinomial Naive Bayes

**Evaluation methodology**

- Stratified train/test split.
- 80% training data.
- 20% test data.
- Random state: 42.
- Evaluation metrics: Accuracy, macro precision, macro recall, macro F1, weighted F1, training time and inference time.

#### Results

| Metric | Logistic Regression | Naive Bayes |
|---|---:|---:|
| Accuracy | 70.97% | 71.49% |
| Macro Precision | 42.04% | 29.15% |
| Macro Recall | 54.44% | 21.08% |
| Macro F1 | 45.17% | 21.56% |
| Weighted F1 | 73.10% | 67.44% |
| Training time | 1.568 s | 0.398 s |
| Inference time/ticket | 0.0343 ms | 0.0335 ms |

#### Interpretation

Logistic Regression achieved a higher macro F1 score and weighted F1 score than Naive Bayes.

Naive Bayes achieved slightly higher accuracy, but its substantially lower macro F1 indicates weaker performance across the less frequent categories.

Logistic Regression is therefore retained as the primary baseline model.

#### Conclusion

The experiment supports the use of Logistic Regression as the current baseline for subsequent model improvement experiments.

---

## 5. EXP-002: Prediction Confidence Analysis

**Objective:** Investigate the relationship between predicted class confidence and actual classification accuracy.

### Results

| Measurement | Result |
|---|---:|
| Average confidence | 54.26% |
| Overall accuracy | 70.97% |
| Expected Calibration Error | 0.1672 |

### Confidence distribution

| Confidence range | Test tickets | Accuracy |
|---|---:|---:|
| 0.1–0.2 | 176 | 34.09% |
| 0.2–0.3 | 449 | 47.22% |
| 0.3–0.4 | 478 | 55.86% |
| 0.4–0.5 | 460 | 69.13% |
| 0.5–0.6 | 398 | 75.13% |
| 0.6–0.7 | 321 | 80.69% |
| 0.7–0.8 | 310 | 84.84% |
| 0.8–0.9 | 336 | 93.15% |
| 0.9–1.0 | 322 | 98.45% |

### Interpretation

The observed accuracy generally increases with predicted confidence.

However, the ECE of 0.1672 indicates a noticeable difference between predicted confidence and observed accuracy across confidence intervals.

This suggests that confidence calibration deserves further investigation before confidence scores are used to control automated triage decisions.

### Conclusion

The current confidence scores provide useful ranking information, but should not yet be treated as fully calibrated probabilities of correctness.

---

## 6. EXP-003: Initial Error Analysis

**Objective:** Identify recurring patterns in incorrect predictions and investigate potential causes.

### Observed patterns

| Actual category | Predicted category | Potential issue |
|---|---|---|
| Feedback | Marketing | Semantic overlap |
| Product | Documentation | Information request ambiguity |
| Performance | Bug | Technical symptom overlap |
| Feature | Product | Product inquiry versus feature request |
| Billing | Payment | Commercial category overlap |
| Sales | Pricing | Commercial intent overlap |
| Bug | Crash | Related technical categories |

### Additional observation

The ticket "Please address the issue" was labelled Security but predicted as Performance.

The ticket does not provide sufficient information to independently establish either category.

This highlights the importance of investigating insufficient-context tickets and annotation quality.

### Research implications

Further investigation should determine:

1. Whether category definitions are sufficiently distinct.
2. Whether the training dataset contains inconsistent labels.
3. Whether repeated ticket templates create misleading patterns.
4. Whether some examples require more information before classification.
5. Whether a hierarchical classification approach could improve category discrimination.

**Important:** These are preliminary observations, not confirmed causes of the classification errors.

---

## 7. Current Research Limitations

1. Some categories have very few examples.
2. The dataset contains potential semantic overlap between categories.
3. The current experiment excludes rare categories.
4. The evaluation preprocessing differs from the full production training script.
5. Confidence calibration has not yet been optimized.
6. Only one stratified train/test split has been evaluated.
7. No independent external dataset has yet been used to assess generalization.

---

## 8. Planned Experiments

| Experiment | Research focus | Status |
|---|---|---|
| EXP-001 | Baseline model comparison | Completed |
| EXP-002 | Confidence analysis | Completed |
| EXP-003 | Error analysis | Initial analysis completed |
| EXP-004 | Dataset quality and category overlap | Planned |
| EXP-005 | TF-IDF hyperparameter experiments | Planned |
| EXP-006 | Alternative model comparison | Planned |
| EXP-007 | Probability calibration | Planned |
| EXP-008 | Confidence-aware triage | Planned |
| EXP-009 | Independent validation and robustness | Planned |

---

## 9. Research Integrity and Reproducibility

Every experiment should record:

- Experiment identifier.
- Date of execution.
- Research question.
- Hypothesis.
- Dataset version.
- Data preprocessing steps.
- Model architecture and hyperparameters.
- Training and evaluation methodology.
- Evaluation metrics.
- Results.
- Interpretation.
- Limitations.
- Decisions and next steps.

Results should be distinguished from interpretations and hypotheses.

Evaluation scripts must not overwrite production model artifacts.

The held-out test set should not be repeatedly used to tune model configurations or decision thresholds.

