"""
train.py — Train sentiment classifiers using scikit-learn.
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.utils import load_config


def get_models():
    return {
        "NaiveBayes": MultinomialNB(alpha=1.0),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1),
        "LinearSVM": LinearSVC(max_iter=2000, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=150, max_depth=5, random_state=42),
    }


def train_and_compare(df, text_col="text_processed", label_col="sentiment",
                      test_size=0.2, cv_folds=5, save_best=True):
    """Train all classifiers and return comparison DataFrame."""
    config = load_config()
    X = df[text_col].fillna("")
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"Classes: {y.value_counts().to_dict()}\n")

    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    models = get_models()
    results = []
    best_acc, best_name, best_pipe = 0, None, None

    for name, model in models.items():
        print(f"Training {name}...", end=" ")
        pipe = Pipeline([("tfidf", tfidf), ("clf", model)])
        cv = cross_val_score(pipe, X_train, y_train, cv=cv_folds, scoring="accuracy", n_jobs=-1)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results.append({
            "model": name, "cv_accuracy": round(cv.mean(), 4),
            "cv_std": round(cv.std(), 4), "test_accuracy": round(acc, 4),
        })
        print(f"Accuracy={acc:.4f}")
        if acc > best_acc:
            best_acc, best_name, best_pipe = acc, name, pipe

    results_df = pd.DataFrame(results).sort_values("test_accuracy", ascending=False)
    print(f"\nBest: {best_name} ({best_acc:.4f})")

    if save_best and best_pipe:
        save_dir = "models/saved_models"
        os.makedirs(save_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"{save_dir}/{best_name}_{ts}.joblib"
        joblib.dump(best_pipe, path)
        print(f"Saved to {path}")

    return results_df, best_pipe, (X_test, y_test)
