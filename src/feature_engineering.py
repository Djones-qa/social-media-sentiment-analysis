"""
feature_engineering.py — Engagement metrics, platform features, influencer scoring.
"""

import pandas as pd
import numpy as np


def add_engagement_metrics(df):
    """Calculate engagement rate and interaction metrics."""
    df = df.copy()
    for col in ["likes", "comments", "shares", "followers"]:
        if col not in df.columns:
            df[col] = 0

    df["total_interactions"] = df["likes"] + df["comments"] + df["shares"]
    df["engagement_rate"] = (
        df["total_interactions"] / df["followers"].replace(0, np.nan)
    ).fillna(0).round(6)

    bins = [0, 0.01, 0.03, 0.06, 0.1, float("inf")]
    labels = ["Very Low", "Low", "Medium", "High", "Viral"]
    df["engagement_tier"] = pd.cut(df["engagement_rate"], bins=bins, labels=labels)

    df["like_share_ratio"] = (df["likes"] / df["shares"].replace(0, np.nan)).fillna(0).round(2)
    df["comment_rate"] = (df["comments"] / df["total_interactions"].replace(0, np.nan)).fillna(0).round(4)
    return df


def add_temporal_features(df, date_col="timestamp"):
    """Extract time-based features from timestamps."""
    df = df.copy()
    if date_col in df.columns:
        dt = pd.to_datetime(df[date_col])
        df["post_hour"] = dt.dt.hour
        df["post_day_of_week"] = dt.dt.dayofweek
        df["post_month"] = dt.dt.month
        df["post_year"] = dt.dt.year
        df["is_weekend"] = dt.dt.dayofweek.ge(5).astype(int)
        df["time_of_day"] = pd.cut(
            dt.dt.hour, bins=[0, 6, 12, 18, 24],
            labels=["Night", "Morning", "Afternoon", "Evening"], right=False
        )
    return df


def add_influencer_score(df):
    """Score authors by follower count and verification status."""
    df = df.copy()
    if "followers" not in df.columns:
        df["followers"] = 0
    if "is_verified" not in df.columns:
        df["is_verified"] = False

    bins = [0, 1000, 10000, 100000, 1000000, float("inf")]
    labels = ["Nano", "Micro", "Mid-Tier", "Macro", "Mega"]
    df["influencer_tier"] = pd.cut(df["followers"], bins=bins, labels=labels)

    df["influencer_score"] = np.log1p(df["followers"]) * (1 + df["is_verified"].astype(int) * 0.5)
    df["influencer_score"] = df["influencer_score"].round(2)
    return df


def add_text_complexity(df, text_col="text_processed"):
    """Add readability and complexity features."""
    df = df.copy()
    df["word_count"] = df[text_col].apply(lambda x: len(str(x).split()))
    df["avg_word_length"] = df[text_col].apply(
        lambda x: np.mean([len(w) for w in str(x).split()]) if len(str(x).split()) > 0 else 0
    ).round(2)
    df["unique_word_ratio"] = df[text_col].apply(
        lambda x: len(set(str(x).split())) / max(len(str(x).split()), 1)
    ).round(4)
    return df


def run_feature_pipeline(df):
    """Execute full feature engineering pipeline."""
    print("Starting feature engineering...")
    initial = len(df.columns)
    df = add_engagement_metrics(df)
    df = add_temporal_features(df)
    df = add_influencer_score(df)
    df = add_text_complexity(df)
    print(f"  Added {len(df.columns) - initial} features ({initial} -> {len(df.columns)} total)")
    return df
