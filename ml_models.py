"""
ml_models.py  (FIXED)
----------------------
Key fixes:
1. Uses SMOTE oversampling on TF-IDF vectors to fix 16:1 class imbalance
2. All 5 models trained on balanced data
3. Saves tfidf + each model separately so app.py can show ALL model results
4. Returns dict of all results for the web app
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import joblib


TFIDF_PARAMS = dict(max_features=8000, ngram_range=(1, 2),
                    stop_words="english", min_df=2)


def build_classifiers() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Decision Tree":       DecisionTreeClassifier(class_weight="balanced", max_depth=20),
        "Random Forest":       RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                                      random_state=42, n_jobs=-1),
        "Naive Bayes":         MultinomialNB(),
        "SVM":                 SVC(class_weight="balanced", probability=True, kernel="linear"),
    }


def train_and_evaluate(data: pd.DataFrame):
    """
    Train all 5 models with SMOTE-balanced data.
    Returns best pipeline + results DataFrame + all trained models dict.
    """
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    X = data["statements"].values
    y = data["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Step 1: Vectorize
    tfidf = TfidfVectorizer(**TFIDF_PARAMS)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf  = tfidf.transform(X_test)

    # Step 2: SMOTE to balance
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_tfidf, y_train)
    print(f"After SMOTE: {X_train_res.shape}, labels={np.bincount(y_train_res)}")

    classifiers = build_classifiers()
    results = {}
    trained = {}

    for name, clf in classifiers.items():
        print(f"  Training {name}...", end=" ", flush=True)
        try:
            if name == "Naive Bayes":
                # NB needs non-negative; SMOTE can produce tiny negatives
                from scipy.sparse import csr_matrix
                X_nb = X_train_res.copy()
                X_nb.data = np.abs(X_nb.data)
                clf.fit(X_nb, y_train_res)
                X_te_nb = X_test_tfidf.copy()
                X_te_nb.data = np.abs(X_te_nb.data)
                y_pred = clf.predict(X_te_nb)
            else:
                clf.fit(X_train_res, y_train_res)
                y_pred = clf.predict(X_test_tfidf)

            acc = accuracy_score(y_test, y_pred)
            results[name] = acc
            trained[name] = clf
            print(f"Acc={acc:.4f}")
        except Exception as e:
            print(f"FAILED: {e}")

    # Save tfidf + all models
    joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")
    joblib.dump(trained, "models/all_ml_models.pkl")
    joblib.dump({"results": results, "X_test": X_test, "y_test": y_test,
                 "X_test_tfidf": X_test_tfidf}, "models/eval_data.pkl")

    results_df = (
        pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"])
        .sort_values("Accuracy", ascending=False).reset_index(drop=True)
    )
    print("\n  Model Comparison:")
    print(results_df.to_string(index=False))

    # Plot
    plt.figure(figsize=(9, 5))
    colors = ["#2ecc71" if i == 0 else "steelblue" for i in range(len(results_df))]
    plt.barh(results_df["Model"], results_df["Accuracy"], color=colors)
    plt.xlabel("Accuracy")
    plt.title("ML Model Comparison (SMOTE balanced)")
    plt.xlim(0, 1)
    for i, (_, row) in enumerate(results_df.iterrows()):
        plt.text(row["Accuracy"] + 0.005, i, f'{row["Accuracy"]:.3f}', va="center", fontsize=10)
    plt.tight_layout()
    plt.savefig("outputs/model_comparison.png", dpi=100)
    plt.close()

    best_name  = results_df.iloc[0]["Model"]
    best_clf   = trained[best_name]

    # Build & save the best pipeline (takes raw text → prediction)
    best_pipe = Pipeline([("tfidf", tfidf), ("model", best_clf)])
    joblib.dump(best_pipe, "models/best_ml_model.pkl")
    print(f"\n  Best model: {best_name} → saved to models/best_ml_model.pkl")

    return best_pipe, results_df, trained, tfidf


def predict_all(text: str, tfidf, trained_models: dict) -> dict:
    """Return predictions from ALL ML models for the web app."""
    x_vec = tfidf.transform([text])
    out = {}
    for name, clf in trained_models.items():
        try:
            if name == "Naive Bayes":
                import scipy.sparse as sp
                x_nb = x_vec.copy()
                x_nb.data = np.abs(x_nb.data)
                pred  = int(clf.predict(x_nb)[0])
                proba = clf.predict_proba(x_nb)[0]
            else:
                pred  = int(clf.predict(x_vec)[0])
                proba = clf.predict_proba(x_vec)[0]
            out[name] = {
                "label":      "Real" if pred == 1 else "Fake",
                "confidence": round(float(max(proba)) * 100, 1),
            }
        except Exception as e:
            out[name] = {"label": "Error", "confidence": 0.0}
    return out


def predict(model, text: str) -> str:
    pred = model.predict([text])[0]
    return "Real News" if pred == 1 else "Fake News"


if __name__ == "__main__":
    data = pd.read_csv("data/politifact_clean.csv")
    best_model, results_df, trained, tfidf = train_and_evaluate(data)

    samples = [
        "Vaccines contain microchips tracked by the government",
        "The government announced a new education policy",
        "The earth is flat",
    ]
    print("\n=== Demo Predictions (All Models) ===")
    for s in samples:
        preds = predict_all(s, tfidf, trained)
        print(f"\n  '{s}'")
        for m, r in preds.items():
            print(f"    {m}: {r['label']} ({r['confidence']}%)")