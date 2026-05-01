"""
deep_learning_models.py
------------------------
ANN (Artificial Neural Network) and RNN/LSTM models on PolitiFact data.
Requires TensorFlow / Keras.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
try:
    from tensorflow.keras.preprocessing.text import Tokenizer
except ImportError:
    from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os

# ── Constants ───────────────────────────────────────────────────────────────
VOCAB_SIZE  = 5000
MAX_LEN     = 100
EMBED_DIM   = 64
EPOCHS      = 10
BATCH_SIZE  = 32


# ─────────────────────────────────────────────
# 1. TOKENISATION
# ─────────────────────────────────────────────

def tokenize(X_train, X_test):
    """Fit tokenizer on training text, return padded sequences + tokenizer."""
    tok = Tokenizer(num_words=VOCAB_SIZE)
    tok.fit_on_texts(X_train)

    X_train_seq = pad_sequences(tok.texts_to_sequences(X_train), maxlen=MAX_LEN)
    X_test_seq  = pad_sequences(tok.texts_to_sequences(X_test),  maxlen=MAX_LEN)

    return X_train_seq, X_test_seq, tok


# ─────────────────────────────────────────────
# 2. ANN MODEL
# ─────────────────────────────────────────────

def build_ann() -> Sequential:
    """
    ANN using Flatten + Dense layers.
    Demonstrates ReLU, Tanh, and Sigmoid activation functions.
    """
    model = Sequential([
        Flatten(input_shape=(MAX_LEN,)),
        Dense(128, activation="relu"),    # ReLU hidden layer
        Dropout(0.3),
        Dense(64,  activation="relu"),    # ReLU hidden layer
        Dense(32,  activation="tanh"),    # Tanh hidden layer
        Dense(1,   activation="sigmoid") # Sigmoid output (binary classification)
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ─────────────────────────────────────────────
# 3. RNN / LSTM MODEL
# ─────────────────────────────────────────────

def build_rnn() -> Sequential:
    """
    RNN using Embedding + LSTM layers.
    Good for sequential / text data.
    """
    model = Sequential([
        Embedding(input_dim=VOCAB_SIZE, output_dim=EMBED_DIM, input_length=MAX_LEN),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(1,  activation="sigmoid")
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ─────────────────────────────────────────────
# 4. TRAIN & EVALUATE
# ─────────────────────────────────────────────

def train_model(model: Sequential, X_train, y_train, X_test, y_test,
                model_name: str = "model"):
    """Train a Keras model with early stopping and return the trained model + history."""
    print(f"\n{'='*50}")
    print(f"  Training: {model_name}")
    print(f"{'='*50}")
    model.summary()

    early_stop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=1
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n{model_name} Test Accuracy: {acc:.4f}")

    return model, history


def plot_history(history, model_name: str, save_path: str = None) -> None:
    """Plot training and validation accuracy/loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"],     label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Val")
    axes[0].set_title(f"{model_name} – Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"],     label="Train")
    axes[1].plot(history.history["val_loss"], label="Val")
    axes[1].set_title(f"{model_name} – Loss")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100)
    plt.show()


# ─────────────────────────────────────────────
# 5. FINAL PREDICTION
# ─────────────────────────────────────────────

def dl_predict(text: str, tokenizer: Tokenizer,
               ann_model: Sequential, rnn_model: Sequential) -> dict:
    """Run both DL models on a single text statement."""
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=MAX_LEN)

    ann_prob = ann_model.predict(pad, verbose=0)[0][0]
    rnn_prob = rnn_model.predict(pad, verbose=0)[0][0]

    return {
        "ANN": ("✅ Real" if ann_prob > 0.5 else "❌ Fake", round(ann_prob * 100, 2)),
        "RNN": ("✅ Real" if rnn_prob > 0.5 else "❌ Fake", round(rnn_prob * 100, 2)),
    }


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # load data
    data = pd.read_csv("data/politifact_clean.csv")
    X = data["statements"].values
    y = data["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # tokenise
    X_train_seq, X_test_seq, tokenizer = tokenize(X_train, X_test)
    joblib.dump(tokenizer, "models/tokenizer.pkl")

    # ── ANN ──────────────────────────────────
    ann = build_ann()
    ann, ann_hist = train_model(ann, X_train_seq, y_train,
                                X_test_seq, y_test, "ANN")
    plot_history(ann_hist, "ANN", save_path="outputs/ann_training.png")
    ann.save("models/ann_model.keras")

    # ── RNN ──────────────────────────────────
    rnn = build_rnn()
    rnn, rnn_hist = train_model(rnn, X_train_seq, y_train,
                                X_test_seq, y_test, "RNN (LSTM)")
    plot_history(rnn_hist, "RNN (LSTM)", save_path="outputs/rnn_training.png")
    rnn.save("models/rnn_model.keras")

    # ── Demo predictions ─────────────────────
    test_statements = [
        "Vaccines contain microchips",
        "Government launched a new education policy",
        "The earth is flat",
    ]
    print("\n=== Deep Learning Predictions ===")
    for stmt in test_statements:
        results = dl_predict(stmt, tokenizer, ann, rnn)
        print(f"\n  '{stmt}'")
        for model_name, (verdict, conf) in results.items():
            print(f"    {model_name}: {verdict}  (confidence: {conf}%)")