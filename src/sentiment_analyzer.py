"""
sentiment_analyzer.py — VADER + TextBlob sentiment classification.
"""

import pandas as pd
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def get_vader_sentiment(text):
    """Get VADER compound sentiment score."""
    analyzer = SentimentIntensityAnalyzer()
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {"compound": 0.0, "pos": 0.0, "neu": 1.0, "neg": 0.0}
    return analyzer.polarity_scores(text)


def get_textblob_sentiment(text):
    """Get TextBlob polarity and subjectivity scores."""
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {"polarity": 0.0, "subjectivity": 0.0}
    blob = TextBlob(text)
    return {"polarity": blob.sentiment.polarity, "subjectivity": blob.sentiment.subjectivity}


def classify_sentiment(score, pos_threshold=0.05, neg_threshold=-0.05):
    """Classify a numeric score into Positive/Negative/Neutral."""
    if score > pos_threshold:
        return "Positive"
    elif score < neg_threshold:
        return "Negative"
    return "Neutral"


def analyze_dataframe(df, text_col="text_original"):
    """Apply both VADER and TextBlob sentiment analysis to a DataFrame."""
    df = df.copy()
    print(f"Analyzing sentiment for {len(df):,} posts...")

    # VADER scores
    vader_scores = df[text_col].apply(get_vader_sentiment)
    df["vader_compound"] = vader_scores.apply(lambda x: x["compound"])
    df["vader_pos"] = vader_scores.apply(lambda x: x["pos"])
    df["vader_neu"] = vader_scores.apply(lambda x: x["neu"])
    df["vader_neg"] = vader_scores.apply(lambda x: x["neg"])
    df["vader_sentiment"] = df["vader_compound"].apply(
        lambda x: classify_sentiment(x, 0.05, -0.05)
    )

    # TextBlob scores
    tb_scores = df[text_col].apply(get_textblob_sentiment)
    df["textblob_polarity"] = tb_scores.apply(lambda x: x["polarity"])
    df["textblob_subjectivity"] = tb_scores.apply(lambda x: x["subjectivity"])
    df["textblob_sentiment"] = df["textblob_polarity"].apply(
        lambda x: classify_sentiment(x, 0.1, -0.1)
    )

    # Ensemble — majority vote
    def ensemble_vote(row):
        votes = [row["vader_sentiment"], row["textblob_sentiment"]]
        for label in ["Positive", "Negative", "Neutral"]:
            if votes.count(label) >= 1 and row["vader_sentiment"] == label:
                return label
        return row["vader_sentiment"]

    df["ensemble_sentiment"] = df.apply(ensemble_vote, axis=1)

    # Summary
    dist = df["ensemble_sentiment"].value_counts()
    print("Sentiment Distribution:")
    for label in ["Positive", "Negative", "Neutral"]:
        count = dist.get(label, 0)
        pct = count / len(df) * 100
        print(f"  {label}: {count:,} ({pct:.1f}%)")

    return df


def sentiment_summary(df, sentiment_col="ensemble_sentiment"):
    """Generate sentiment summary statistics."""
    summary = {
        "total_posts": len(df),
        "positive": len(df[df[sentiment_col] == "Positive"]),
        "negative": len(df[df[sentiment_col] == "Negative"]),
        "neutral": len(df[df[sentiment_col] == "Neutral"]),
        "avg_vader_compound": round(df["vader_compound"].mean(), 4),
        "avg_textblob_polarity": round(df["textblob_polarity"].mean(), 4),
        "avg_subjectivity": round(df["textblob_subjectivity"].mean(), 4),
    }
    summary["positive_pct"] = round(summary["positive"] / summary["total_posts"] * 100, 1)
    summary["negative_pct"] = round(summary["negative"] / summary["total_posts"] * 100, 1)
    summary["neutral_pct"] = round(summary["neutral"] / summary["total_posts"] * 100, 1)
    return summary
