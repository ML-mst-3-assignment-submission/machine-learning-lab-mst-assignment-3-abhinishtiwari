# Fake News Detection System

A complete machine learning + deep learning pipeline for fake news detection using PolitiFact data, Google Fact Check API verification, and a Flask web UI.

---

## UI Preview (Light Theme)

Place the provided screenshot at `docs/ui-preview.png` and it will render here:

![Fake News Detection System UI](docs/ui-preview.png)

---

## Project Structure

```
ML_Pipeline_project/
│
├── scraper.py               # Scrape PolitiFact list pages
├── eda_preprocessing.py     # EDA + clean + balance dataset
├── ml_models.py             # ML models (LR, RF, SVM, etc.)
├── fact_check_api.py        # Google Fact Check API integration
├── activation_functions.py  # Activation function plots
├── deep_learning_models.py  # ANN + RNN (LSTM)
│
├── app.py                   # Flask web UI
├── main.py                  # Master pipeline
├── Train.py                 # End-to-end training script
├── requirements.txt
│
├── data/                    # Scraped + cleaned CSVs (created at runtime)
├── models/                  # Saved models (created at runtime)
└── outputs/                 # Plots and charts (created at runtime)
```

---

## Quick Start

### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Configure Google Fact Check API key
Edit `app.py` or `fact_check_api.py` and set:

```python
API_KEY = "YOUR_GOOGLE_API_KEY"
```

Enable the "Fact Check Tools API" at https://console.cloud.google.com/

### 3) Train models
```bash
python Train.py
```

### 4) Run the web app
```bash
python app.py
```
Open: http://127.0.0.1:5000

---

## What This Project Includes

- PolitiFact scraping and aligned dataset creation
- Text preprocessing, label mapping, and optional class balancing
- ML models: Logistic Regression, Decision Tree, Random Forest, Naive Bayes, SVM
- Deep learning models: ANN and RNN (LSTM)
- Google Fact Check API verification
- Flask UI with light theme dashboard

---

## Outputs

After running `main.py` or `Train.py`, the `outputs/` folder will contain:

- `eda_plots.png`
- `wordcloud.png`
- `correlation_heatmap.png`
- `model_comparison.png`
- `activation_functions.png`
- `activation_derivatives.png`
- `ann_training.png`
- `rnn_training.png`
