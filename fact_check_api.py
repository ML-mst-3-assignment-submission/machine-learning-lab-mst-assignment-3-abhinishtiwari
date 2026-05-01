"""
fact_check_api.py
------------------
Verify statements using the Google Fact Check Tools API.

Usage:
    Set your API key in the API_KEY variable or pass it to FactChecker().
    Get a free key at: https://console.cloud.google.com/
"""

import requests


API_KEY = "AIzaSyBV_AR3NID-5Lo861qZXI5clMG-qCy2Twg"   # <-- replace with your key


class FactChecker:
    """Wrapper around the Google Fact Check Tools API."""

    BASE_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

    def __init__(self, api_key: str = API_KEY):
        self.api_key = api_key

    def search(self, query: str) -> dict:
        """Raw API call – returns full JSON response."""
        params = {"query": query, "key": self.api_key}
        response = requests.get(self.BASE_URL, params=params)
        response.raise_for_status()
        return response.json()

    def get_verdict(self, query: str) -> str:
        """
        Return a human-readable verdict string.

        Returns:
            'Publisher Name: rating'  or  a descriptive fallback string.
        """
        try:
            data = self.search(query)
            claims = data.get("claims", [])
            if not claims:
                return "No fact-check found for this statement"
            review = claims[0]["claimReview"][0]
            source = review["publisher"]["name"]
            rating = review["textualRating"]
            return f"{source}: {rating}"
        except requests.HTTPError as e:
            return f"HTTP error: {e}"
        except Exception:
            return "Error parsing API response"


# ─────────────────────────────────────────────
# COMBINED CHECK FUNCTION
# ─────────────────────────────────────────────

def combined_check(statement: str, ml_model, fact_checker: FactChecker) -> None:
    """
    Run ML model + Fact Check API on a single statement and print results.

    Args:
        statement:    News/claim text to evaluate.
        ml_model:     Fitted sklearn pipeline (from ml_models.py).
        fact_checker: FactChecker instance.
    """
    pred = ml_model.predict([statement])[0]
    prob = ml_model.predict_proba([statement])[0]

    ml_result = "✅ Real News" if pred == 1 else "❌ Fake News"
    confidence = round(max(prob) * 100, 2)
    api_result = fact_checker.get_verdict(statement)

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"📰 Statement : {statement}")
    print(f"🧠 ML Result : {ml_result}")
    print(f"📊 Confidence: {confidence}%")
    print(f"🌐 API Check : {api_result}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


if __name__ == "__main__":
    import joblib

    # load pre-trained model
    try:
        ml_model = joblib.load("models/best_ml_model.pkl")
    except FileNotFoundError:
        print("Run ml_models.py first to train and save the model.")
        exit(1)

    checker = FactChecker()

    test_statements = [
        "Vaccines contain microchips",
        "The government launched a new education policy",
        "The earth is flat",
        "India won the cricket world cup",
    ]

    for stmt in test_statements:
        combined_check(stmt, ml_model, checker)
