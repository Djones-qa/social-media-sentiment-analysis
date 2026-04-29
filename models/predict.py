"""
predict.py — Load saved model and classify new text.
"""

import pandas as pd
import numpy as np
import joblib
from glob import glob


def load_latest_model(model_dir="models/saved_models"):
    model_files = sorted(glob(f"{model_dir}/*.joblib"))
    if not model_files:
        raise FileNotFoundError(f"No saved models in {model_dir}/")
    latest = model_files[-1]
    pipeline = joblib.load(latest)
    print(f"Loaded: {latest}")
    return pipeline, latest


def predict_single(pipeline, text):
    """Classify a single text string."""
    prediction = pipeline.predict([text])[0]
    probas = None
    if hasattr(pipeline, "predict_proba"):
        probas = dict(zip(pipeline.classes_, pipeline.predict_proba([text])[0].round(4)))
    return {"text": text, "prediction": prediction, "probabilities": probas}


def predict_batch(pipeline, df, text_col="text_processed", output_col="predicted_sentiment"):
    """Classify all texts in a DataFrame."""
    df = df.copy()
    texts = df[text_col].fillna("")
    df[output_col] = pipeline.predict(texts)
    print(f"Classified {len(df):,} posts")
    print(df[output_col].value_counts().to_string())
    return df
