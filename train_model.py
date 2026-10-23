# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# import pickle

# # 1. Load dataset
# df = pd.read_csv("aa_dataset-tickets-multi-lang-5-2-50-version.csv")

# # 2. Use only English tickets
# df = df[df['language'] == 'en']

# # 3. Select relevant columns
# df = df[['body', 'tag_1']].dropna()

# X = df['body']
# y = df['tag_1']

# # 4. Convert text to numerical features
# vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
# X_tfidf = vectorizer.fit_transform(X)

# # 5. Train Logistic Regression (Multiclass)
# model = LogisticRegression(
#     max_iter=500,
#     solver='lbfgs',
#     # multi_class='multinomial'
# )

# model.fit(X_tfidf, y)

# # 6. Save model and vectorizer
# pickle.dump(model, open("model.pkl", "wb"))
# pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

# print("Model trained and saved successfully!")
"""
Production training script — fits the final model on the FULL primary
dataset (no train/test split), so the deployed Streamlit app (app.py)
has the maximum amount of training data available.

Do NOT read accuracy claims from this script — there is no held-out
test set here. For reported performance metrics (Table 4.2/4.3), run
evaluate_primary.py, which does an 80/20 split.

Hyperparameters match Table 3.5 in the report exactly.
"""

import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

DATA_PATH = "aa_dataset-tickets-multi-lang-5-2-50-version.csv"

# 1. Load dataset
df = pd.read_csv(DATA_PATH)

# 2. English tickets only
df = df[df["language"] == "en"]

# 3. Relevant columns, drop nulls
df = df[["body", "tag_1"]].dropna()

X = df["body"]
y = df["tag_1"]

# 4. TF-IDF vectorization (Table 3.4 config)
vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
X_tfidf = vectorizer.fit_transform(X)

# 5. Logistic Regression — config matches Table 3.5
model = LogisticRegression(
    solver='lbfgs',
    max_iter=300,
    C=1.0,
    # multi_class="ovr",
    class_weight="balanced",
)
model.fit(X_tfidf, y)

# 6. Save model and vectorizer for the Streamlit app
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model trained on full dataset and saved successfully!")
