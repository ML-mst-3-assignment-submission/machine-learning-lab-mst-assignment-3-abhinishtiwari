"""
eda_preprocessing.py  (FIXED)
------------------------------
- Uses 'targets' column for better_label (includes half-true as Real)
- Fixes extreme class imbalance with SMOTE-compatible oversampling
- Skips stopword removal for short political statements
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk
from collections import Counter

nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords
from sklearn.utils import resample

STOP_WORDS = set(stopwords.words("english"))


def load_data(path: str = "data/politifact_data.csv") -> pd.DataFrame:
    data = pd.read_csv(path)
    print(f"Loaded {data.shape[0]} rows, {data.shape[1]} columns")
    return data


def better_label(x: str):
    """
    FIXED label mapping – includes half-true as Real and barely-true as Fake
    to increase the minority (Real) class from 5% → ~10%.
    """
    x = str(x).lower().strip()
    if x in ["true", "mostly-true", "half-true"]:
        return 1   # Real
    elif x in ["false", "pants-fire", "barely-true"]:
        return 0   # Fake
    return None    # drop ambiguous labels


def clean_text(text: str) -> str:
    """Light cleaning – keep most words since political statements are short."""
    words = str(text).split()
    # only remove very common stopwords, preserve informative words
    keep = [w for w in words if w.lower() not in STOP_WORDS or len(words) < 5]
    return " ".join(keep) if keep else " ".join(words)


def run_eda(data: pd.DataFrame) -> None:
    import os
    os.makedirs("outputs", exist_ok=True)
    print("\n=== Shape ===", data.shape)
    print("\n=== Columns ===", data.columns.tolist())
    print("\n=== Missing ===\n", data.isnull().sum())
    print("\n=== Target Distribution ===")
    col = "targets" if "targets" in data.columns else data.columns[-1]
    print(data[col].value_counts())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    data[col].value_counts().plot(kind="bar", ax=axes[0, 0], color="steelblue")
    axes[0, 0].set_title("Class Distribution")

    data["text_length"] = data["statements"].apply(len)
    axes[0, 1].hist(data["text_length"], bins=30, color="coral")
    axes[0, 1].set_title("Text Length Distribution")

    data["word_count"] = data["statements"].apply(lambda x: len(str(x).split()))
    axes[1, 0].hist(data["word_count"], bins=30, color="mediumseagreen")
    axes[1, 0].set_title("Word Count Distribution")

    axes[1, 1].boxplot(data["text_length"].dropna())
    axes[1, 1].set_title("Text Length Boxplot")

    plt.tight_layout()
    plt.savefig("outputs/eda_plots.png", dpi=100)
    plt.close()
    print("EDA plots saved → outputs/eda_plots.png")


def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    """Full clean + label pipeline with fixed label mapping."""
    print("\n=== Preprocessing ===")
    data = data.dropna(subset=["statements"])
    data = data.drop_duplicates(subset=["statements"])

    # Use 'targets' for label (raw PolitiFact verdicts)
    target_col = "targets" if "targets" in data.columns else "BianryNumTarget"

    if target_col == "targets":
        data["label"] = data["targets"].apply(better_label)
    else:
        data["label"] = data[target_col]

    data = data.dropna(subset=["label"])
    data["label"] = data["label"].astype(int)
    data["statements"] = data["statements"].apply(clean_text)

    # Remove very short statements (< 5 chars)
    data = data[data["statements"].str.len() >= 5].reset_index(drop=True)

    print(f"After preprocessing: {len(data)} rows")
    print("Label distribution:\n", data["label"].value_counts())
    return data


def balance_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    FIXED: Use SMOTE-friendly oversampling at text level.
    Oversample Real (minority) to 40% of total instead of 50%
    to keep some natural class distribution.
    """
    majority = data[data["label"] == 0]
    minority = data[data["label"] == 1]

    target_n = int(len(majority) * 0.5)   # oversample Real to 50% of Fake count
    target_n = max(target_n, len(minority))

    minority_up = resample(minority, replace=True, n_samples=target_n, random_state=42)
    balanced = pd.concat([majority, minority_up]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Balanced dataset: {len(balanced)} rows")
    print(balanced["label"].value_counts())
    return balanced


if __name__ == "__main__":
    data = load_data()
    run_eda(data)
    data = preprocess(data)
    data = balance_data(data)
    data.to_csv("data/politifact_clean.csv", index=False)
    print("Clean data saved → data/politifact_clean.csv")