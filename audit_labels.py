
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "aa_dataset-tickets-multi-lang-5-2-50-version.csv"
RESULTS_DIR = BASE_DIR / "results"

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH, usecols=["language", "body", "tag_1"])

    df = df.loc[df["language"].eq("en"), ["body", "tag_1"]].dropna().copy()

    df["body"] = df["body"].astype(str).str.strip()
    df["tag_1"] = df["tag_1"].astype(str).str.strip()

    df = df[(df["body"] != "") & (df["tag_1"] != "")].copy()

    # Normalize whitespace and capitalization to detect exact-text variants.
    df["normalized_body"] = (
        df["body"]
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # Find identical normalized ticket text with multiple labels.
    label_counts = df.groupby("normalized_body")["tag_1"].nunique()
    conflicting_texts = label_counts[label_counts > 1].index

    conflicts = (
        df[df["normalized_body"].isin(conflicting_texts)]
        .sort_values(["normalized_body", "tag_1"])
    )

    conflicts.to_csv(
        RESULTS_DIR / "conflicting_labels.csv",
        index=False,
    )

    print(f"\nEnglish records audited: {len(df)}")
    print(f"Unique normalized ticket texts: {df['normalized_body'].nunique()}")
    print(f"Texts assigned multiple labels: {len(conflicting_texts)}")
    print(f"Rows involved in conflicts: {len(conflicts)}")
    print(f"Saved conflicts to: {RESULTS_DIR / 'conflicting_labels.csv'}")

    # Show representative examples for categories involved in confusion.
    focus_categories = [
        "Bug", "Performance", "Crash",
        "Feedback", "Feature", "Documentation",
        "Product", "Support", "Customer Support",
    ]

    print("\n===== REPRESENTATIVE CATEGORY EXAMPLES =====")

    for category in focus_categories:
        examples = df[df["tag_1"].eq(category)].drop_duplicates(
            subset=["normalized_body"]
        )

        print(f"\n--- {category}: {len(examples)} unique texts ---")

        for _, row in examples.sample(
            n=min(3, len(examples)),
            random_state=42,
        ).iterrows():
            print(f"- {row['body'][:400]}")

    print("\nAudit complete.")

if __name__ == "__main__":
    main()
