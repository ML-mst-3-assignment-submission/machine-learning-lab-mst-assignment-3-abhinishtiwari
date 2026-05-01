"""
Train.py  (FIXED)
------------------
Run this FIRST before starting the web app.
Fixes:
  1. Reads the pre-scraped CSV directly (with_target__1_.csv or politifact_data.csv)
  2. Uses improved label mapping (half-true → Real, barely-true → Fake)
  3. Applies SMOTE to fix 16:1 class imbalance
  4. Saves all 5 ML models + tfidf so app.py can show each one

Usage:
    python Train.py

Output files:
    models/best_ml_model.pkl        ← best sklearn pipeline (raw text input)
    models/tfidf_vectorizer.pkl     ← shared TF-IDF vectorizer
    models/all_ml_models.pkl        ← dict of all 5 trained classifiers
    models/tokenizer.pkl            ← Keras tokenizer for DL models
    models/ann_model.keras          ← ANN
    models/rnn_model.keras          ← RNN/LSTM
"""

import os, sys, warnings
warnings.filterwarnings("ignore")
os.makedirs("data",    exist_ok=True)
os.makedirs("models",  exist_ok=True)
os.makedirs("outputs", exist_ok=True)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
from sklearn.model_selection import train_test_split
import joblib

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print(" STEP 1 — LOAD & PREPROCESS DATA")
print("="*55)
# ─────────────────────────────────────────────────────────────────────────────

CLEAN_CSV = "data/politifact_clean.csv"

# Check for existing scraped files
CANDIDATE_RAWS = [
    "data/with_target__1_.csv",
    "with_target__1_.csv",
    "data/politifact_data.csv",
]

if os.path.exists(CLEAN_CSV):
    print(f"  Found clean CSV: {CLEAN_CSV}")
    data = pd.read_csv(CLEAN_CSV)
else:
    raw_path = next((p for p in CANDIDATE_RAWS if os.path.exists(p)), None)

    if raw_path:
        print(f"  Found raw CSV: {raw_path} — preprocessing...")
        raw_data = pd.read_csv(raw_path)
    else:
        print("  No CSV found — scraping PolitiFact (this takes ~5 min)...")
        from scraper import scrape_all
        raw_data = scrape_all(num_pages=300)

    from eda_preprocessing import preprocess, balance_data
    data = preprocess(raw_data)
    data = balance_data(data)
    data.to_csv(CLEAN_CSV, index=False)
    print(f"  Saved → {CLEAN_CSV}")

print(f"  Dataset: {len(data)} rows | label dist: {dict(data['label'].value_counts())}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print(" STEP 2 — TRAIN ALL 5 ML MODELS (with SMOTE)")
print("="*55)
# ─────────────────────────────────────────────────────────────────────────────

from ml_models import train_and_evaluate, predict_all

best_model, results_df, trained_models, tfidf = train_and_evaluate(data)

print("\n  Demo predictions (all ML models):")
demo_stmts = [
    "Vaccines contain microchips",
    "The government launched a new education policy",
]
for s in demo_stmts:
    preds = predict_all(s, tfidf, trained_models)
    print(f"\n  '{s}'")
    for m, r in preds.items():
        icon = "Real" if r["label"]=="Real" else "Fake"
        print(f"    {m:20s}: {icon}  ({r['confidence']}%)")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print(" STEP 3 — DEEP LEARNING TOKENIZATION")
print("="*55)
# ─────────────────────────────────────────────────────────────────────────────

try:
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    TF_OK = True
except ImportError:
    try:
        from keras.preprocessing.text import Tokenizer
        from keras.preprocessing.sequence import pad_sequences
        TF_OK = True
    except ImportError:
        TF_OK = False
        print("  WARNING: TensorFlow/Keras not found. Skipping DL training.")
        print("  Install with: pip install tensorflow")

if TF_OK:
    from imblearn.over_sampling import RandomOverSampler

    X = data["statements"].values
    y = data["label"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Oversample at text level for DL
    ros = RandomOverSampler(random_state=42)
    X_train_r, y_train_r = ros.fit_resample(X_train.reshape(-1,1), y_train)
    X_train_r = X_train_r.ravel()
    print(f"  After oversample: {np.bincount(y_train_r)}")

    VOCAB_SIZE = 5000
    MAX_LEN    = 100

    tok = Tokenizer(num_words=VOCAB_SIZE)
    tok.fit_on_texts(X_train_r)
    X_tr = pad_sequences(tok.texts_to_sequences(X_train_r), maxlen=MAX_LEN)
    X_te = pad_sequences(tok.texts_to_sequences(X_test),    maxlen=MAX_LEN)
    joblib.dump(tok, "models/tokenizer.pkl")
    print("  Tokenizer saved → models/tokenizer.pkl")

    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print(" STEP 4 — TRAIN ANN")
    print("="*55)
    # ─────────────────────────────────────────────────────────────────────
    from deep_learning_models import build_ann, build_rnn, train_model, plot_history

    ann = build_ann()
    ann, ann_hist = train_model(ann, X_tr, y_train_r, X_te, y_test, "ANN")
    plot_history(ann_hist, "ANN", save_path="outputs/ann_training.png")
    ann.save("models/ann_model.keras")
    print("  Saved → models/ann_model.keras")

    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print(" STEP 5 — TRAIN RNN (LSTM)")
    print("="*55)
    # ─────────────────────────────────────────────────────────────────────
    rnn = build_rnn()
    rnn, rnn_hist = train_model(rnn, X_tr, y_train_r, X_te, y_test, "RNN (LSTM)")
    plot_history(rnn_hist, "RNN (LSTM)", save_path="outputs/rnn_training.png")
    rnn.save("models/rnn_model.keras")
    print("  Saved → models/rnn_model.keras")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print(" TRAINING COMPLETE")
print("="*55)
print("  models/best_ml_model.pkl")
print("  models/tfidf_vectorizer.pkl")
print("  models/all_ml_models.pkl")
if TF_OK:
    print("  models/tokenizer.pkl")
    print("  models/ann_model.keras")
    print("  models/rnn_model.keras")
print()
print("  Next → python app.py")
print("="*55)