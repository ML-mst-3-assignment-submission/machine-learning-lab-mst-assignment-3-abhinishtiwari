"""
main.py
--------
Master pipeline: runs all 5 parts of the MST-3 ML Assignment end-to-end.

Run:
    python main.py
"""

import os

# ── ensure output/model directories exist ────────────────────────────────────
os.makedirs("data",    exist_ok=True)
os.makedirs("models",  exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ── imports ───────────────────────────────────────────────────────────────────
from eda_preprocessing      import load_data, run_eda, preprocess, balance_data
from ml_models              import train_and_evaluate, predict
from fact_check_api         import FactChecker, combined_check
from activation_functions   import demo_values, plot_activations, plot_derivatives
from deep_learning_models   import (tokenize, build_ann, build_rnn,
                                    train_model, plot_history, dl_predict)
import joblib


# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print(" PART 1 – DATA SCRAPING")
print("="*60)
# ═════════════════════════════════════════════════════════════════════════════

CLEAN_CSV = "data/politifact_clean.csv"

if not os.path.exists(CLEAN_CSV):
    from scraper import scrape_all
    raw_data = scrape_all(num_pages=300)
else:
    print(f"Using existing data at '{CLEAN_CSV}' – skipping scrape.")
    raw_data = load_data(CLEAN_CSV)


# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print(" PART 1b – EDA + PREPROCESSING")
print("="*60)
# ═════════════════════════════════════════════════════════════════════════════

if not os.path.exists(CLEAN_CSV):
    raw_data = load_data("data/politifact_data.csv")
    run_eda(raw_data)
    data = preprocess(raw_data)
    data = balance_data(data)
    data.to_csv(CLEAN_CSV, index=False)
else:
    import pandas as pd
    data = pd.read_csv(CLEAN_CSV)
    print(f"Loaded clean data: {len(data)} rows")


# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print(" PART 2 – SUPERVISED ML MODELS")
print("="*60)
# ═════════════════════════════════════════════════════════════════════════════

best_ml_model, results_df = train_and_evaluate(data)
print("\nDemo predictions (ML):")
for stmt in ["Vaccines contain microchips",
             "Government announced new education policy"]:
    print(f"  '{stmt}' → {predict(best_ml_model, stmt)}")


# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print(" PART 3 – GOOGLE FACT CHECK API")
print("="*60)
# ═════════════════════════════════════════════════════════════════════════════

checker = FactChecker()       # uses API_KEY from fact_check_api.py
test_stmts = [
    "Vaccines contain microchips",
    "The earth is flat",
    "India won the cricket world cup",
]
for stmt in test_stmts:
    combined_check(stmt, best_ml_model, checker)


# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print(" PART 4 – ACTIVATION FUNCTIONS")
print("="*60)
# ═════════════════════════════════════════════════════════════════════════════

demo_values()
plot_activations(save_path="outputs/activation_functions.png")
plot_derivatives(save_path="outputs/activation_derivatives.png")


# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print(" PART 5 – DEEP LEARNING (ANN + RNN)")
print("="*60)
# ═════════════════════════════════════════════════════════════════════════════

from sklearn.model_selection import train_test_split

X = data["statements"].values
y = data["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_seq, X_test_seq, tokenizer = tokenize(X_train, X_test)
joblib.dump(tokenizer, "models/tokenizer.pkl")

# ANN
ann = build_ann()
ann, ann_hist = train_model(ann, X_train_seq, y_train,
                            X_test_seq, y_test, "ANN")
plot_history(ann_hist, "ANN", save_path="outputs/ann_training.png")
ann.save("models/ann_model.keras")

# RNN
rnn = build_rnn()
rnn, rnn_hist = train_model(rnn, X_train_seq, y_train,
                            X_test_seq, y_test, "RNN (LSTM)")
plot_history(rnn_hist, "RNN (LSTM)", save_path="outputs/rnn_training.png")
rnn.save("models/rnn_model.keras")

# Final demo predictions
print("\n=== Final Combined Predictions (All Models) ===")
for stmt in test_stmts:
    dl_res = dl_predict(stmt, tokenizer, ann, rnn)
    ml_res = predict(best_ml_model, stmt)
    api_res = checker.get_verdict(stmt)
    print(f"\n  📰 '{stmt}'")
    print(f"     ML  : {ml_res}")
    for m, (verdict, conf) in dl_res.items():
        print(f"     {m}  : {verdict} ({conf}%)")
    print(f"     API : {api_res}")


print("\n" + "="*60)
print(" ✅ ALL PARTS COMPLETE")
print("   Run 'python app.py' to start the Flask web app.")
print("="*60)