"""
Diagnostic: inspect the actual tag_1 category distribution.
Run this BEFORE re-sending anything to your supervisor — we need to see
exactly how messy the raw label column is before deciding how to fix it.
"""
import pandas as pd

df = pd.read_csv("aa_dataset-tickets-multi-lang-5-2-50-version.csv")
df = df[df["language"] == "en"]
df = df[["body", "tag_1"]].dropna()

counts = df["tag_1"].value_counts()

print(f"Total records: {len(df)}")
print(f"Total distinct categories: {df['tag_1'].nunique()}")
print(f"\nCategories with fewer than 5 records:")
print(counts[counts < 5])
print(f"\nNumber of categories with fewer than 5 records: {(counts < 5).sum()}")
print(f"\nCategories containing a comma (possible multi-label leakage):")
print(counts[counts.index.str.contains(",")])
print(f"\nFull category counts:")
print(counts)

counts.to_csv("results/category_distribution.csv")
print("\nSaved full breakdown to results/category_distribution.csv")