"""
exploratory_analysis.py — Automated EDA for social media sentiment data.
"""

import pandas as pd


def sentiment_distribution(df, sentiment_col="ensemble_sentiment"):
    """Count and percentage by sentiment label."""
    dist = df[sentiment_col].value_counts()
    pct = df[sentiment_col].value_counts(normalize=True) * 100
    return pd.DataFrame({"count": dist, "pct": pct.round(1)})


def platform_comparison(df, sentiment_col="ensemble_sentiment"):
    """Compare sentiment distribution across platforms."""
    if "platform" not in df.columns:
        return pd.DataFrame()
    ct = pd.crosstab(df["platform"], df[sentiment_col], normalize="index") * 100
    return ct.round(1)


def engagement_by_sentiment(df, sentiment_col="ensemble_sentiment"):
    """Average engagement metrics grouped by sentiment."""
    metrics = ["likes", "comments", "shares", "engagement_rate"]
    cols = [c for c in metrics if c in df.columns]
    if not cols:
        return pd.DataFrame()
    return df.groupby(sentiment_col)[cols].mean().round(2)


def temporal_sentiment_trend(df, sentiment_col="ensemble_sentiment", date_col="timestamp"):
    """Weekly sentiment trend over time."""
    df = df.copy()
    df["week"] = pd.to_datetime(df[date_col]).dt.to_period("W").dt.to_timestamp()
    weekly = pd.crosstab(df["week"], df[sentiment_col], normalize="index") * 100
    return weekly.round(1)


def top_hashtags(df, n=20):
    """Most frequent hashtags across all posts."""
    if "hashtags_list" not in df.columns:
        return pd.Series(dtype=str)
    all_tags = df["hashtags_list"].explode().dropna()
    all_tags = all_tags[all_tags.str.len() > 0]
    return all_tags.value_counts().head(n)


def influencer_impact(df, sentiment_col="ensemble_sentiment"):
    """Sentiment breakdown by influencer tier."""
    if "influencer_tier" not in df.columns:
        return pd.DataFrame()
    ct = pd.crosstab(df["influencer_tier"], df[sentiment_col], normalize="index") * 100
    return ct.round(1)


def posting_patterns(df):
    """Analyze posting frequency by hour and day of week."""
    result = {}
    if "post_hour" in df.columns:
        result["by_hour"] = df["post_hour"].value_counts().sort_index()
    if "post_day_of_week" in df.columns:
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        dow = df["post_day_of_week"].value_counts().sort_index()
        dow.index = [day_names[i] for i in dow.index]
        result["by_day"] = dow
    return result


def run_full_eda(df, output_dir="visualizations/output"):
    """Execute complete EDA pipeline and print results."""
    print("\n=== Sentiment Distribution ===")
    print(sentiment_distribution(df).to_string())
    print("\n=== Platform Comparison ===")
    pc = platform_comparison(df)
    if not pc.empty:
        print(pc.to_string())
    print("\n=== Engagement by Sentiment ===")
    ebs = engagement_by_sentiment(df)
    if not ebs.empty:
        print(ebs.to_string())
    print("\n=== Top 15 Hashtags ===")
    th = top_hashtags(df, 15)
    if not th.empty:
        print(th.to_string())
    print("\n=== Influencer Impact ===")
    ii = influencer_impact(df)
    if not ii.empty:
        print(ii.to_string())
    print("\nEDA complete.")
