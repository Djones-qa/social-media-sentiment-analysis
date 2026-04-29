"""
plots.py — 8 professional visualizations for social media sentiment analytics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

COLORS = {"Positive": "#4CAF50", "Negative": "#F44336", "Neutral": "#2196F3"}
OUTPUT_DIR = "visualizations/output"


def _save(fig, name, output_dir=None):
    out = output_dir or OUTPUT_DIR
    Path(out).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out}/{name}", dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}/{name}")
    plt.close(fig)


def plot_sentiment_distribution(df, col="ensemble_sentiment", output_dir=None):
    """1. Pie + bar chart of sentiment distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    counts = df[col].value_counts()
    colors = [COLORS.get(l, "#999") for l in counts.index]
    axes[0].pie(counts, labels=counts.index, autopct="%1.1f%%", colors=colors, startangle=90)
    axes[0].set_title("Sentiment Distribution", fontweight="bold")
    axes[1].bar(counts.index, counts.values, color=colors, edgecolor="white")
    axes[1].set_title("Sentiment Counts", fontweight="bold")
    axes[1].set_ylabel("Posts")
    plt.tight_layout()
    _save(fig, "01_sentiment_distribution.png", output_dir)


def plot_platform_comparison(df, col="ensemble_sentiment", output_dir=None):
    """2. Stacked bar — sentiment by platform."""
    if "platform" not in df.columns:
        return
    ct = pd.crosstab(df["platform"], df[col], normalize="index") * 100
    fig, ax = plt.subplots(figsize=(10, 6))
    order = [c for c in ["Positive","Neutral","Negative"] if c in ct.columns]
    ct[order].plot(kind="barh", stacked=True, ax=ax,
        color=[COLORS.get(c, "#999") for c in order])
    ax.set_title("Sentiment by Platform", fontweight="bold", fontsize=14)
    ax.set_xlabel("Percentage (%)")
    ax.legend(title="Sentiment")
    plt.tight_layout()
    _save(fig, "02_platform_comparison.png", output_dir)


def plot_sentiment_over_time(df, date_col="timestamp", col="ensemble_sentiment", output_dir=None):
    """3. Weekly sentiment trend line."""
    df = df.copy()
    df["week"] = pd.to_datetime(df[date_col]).dt.to_period("W").dt.to_timestamp()
    weekly = pd.crosstab(df["week"], df[col], normalize="index") * 100
    fig, ax = plt.subplots(figsize=(14, 6))
    for label, color in COLORS.items():
        if label in weekly.columns:
            ax.plot(weekly.index, weekly[label], color=color, linewidth=2,
                    marker="o", markersize=4, label=label)
    ax.set_title("Sentiment Trend Over Time", fontweight="bold", fontsize=14)
    ax.set_ylabel("Percentage (%)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, "03_sentiment_trend.png", output_dir)


def plot_engagement_by_sentiment(df, col="ensemble_sentiment", output_dir=None):
    """4. Grouped bar — avg engagement by sentiment."""
    metrics = [c for c in ["likes", "comments", "shares"] if c in df.columns]
    if not metrics:
        return
    agg = df.groupby(col)[metrics].mean()
    fig, ax = plt.subplots(figsize=(10, 6))
    agg.plot(kind="bar", ax=ax, colormap="Set2", edgecolor="white")
    ax.set_title("Avg Engagement by Sentiment", fontweight="bold", fontsize=14)
    ax.set_ylabel("Average Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    _save(fig, "04_engagement_by_sentiment.png", output_dir)


def plot_vader_score_distribution(df, output_dir=None):
    """5. Histogram of VADER compound scores."""
    if "vader_compound" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(df["vader_compound"], bins=60, color="#673AB7", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Neutral (0)")
    ax.axvline(df["vader_compound"].mean(), color="orange", linestyle="--",
               label=f"Mean: {df['vader_compound'].mean():.3f}")
    ax.set_title("VADER Compound Score Distribution", fontweight="bold", fontsize=14)
    ax.set_xlabel("Compound Score")
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.tight_layout()
    _save(fig, "05_vader_distribution.png", output_dir)


def plot_top_hashtags(df, n=15, output_dir=None):
    """6. Horizontal bar — top hashtags by frequency."""
    if "hashtags_list" not in df.columns:
        return
    all_tags = df["hashtags_list"].explode().dropna()
    all_tags = all_tags[all_tags.str.len() > 0]
    top = all_tags.value_counts().head(n)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top.index[::-1], top.values[::-1], color="#FF9800", edgecolor="white")
    ax.set_title(f"Top {n} Hashtags", fontweight="bold", fontsize=14)
    ax.set_xlabel("Post Count")
    plt.tight_layout()
    _save(fig, "06_top_hashtags.png", output_dir)


def plot_posting_heatmap(df, output_dir=None):
    """7. Heatmap — posting activity by hour and day."""
    if "post_hour" not in df.columns or "post_day_of_week" not in df.columns:
        return
    pivot = df.groupby(["post_day_of_week", "post_hour"]).size().unstack(fill_value=0)
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    pivot.index = [days[i] if i < len(days) else str(i) for i in pivot.index]
    fig, ax = plt.subplots(figsize=(16, 5))
    sns.heatmap(pivot, cmap="YlOrRd", ax=ax, linewidths=0.5, cbar_kws={"label": "Posts"})
    ax.set_title("Posting Activity Heatmap", fontweight="bold", fontsize=14)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("")
    plt.tight_layout()
    _save(fig, "07_posting_heatmap.png", output_dir)


def plot_influencer_sentiment(df, col="ensemble_sentiment", output_dir=None):
    """8. Grouped bar — sentiment by influencer tier."""
    if "influencer_tier" not in df.columns:
        return
    ct = pd.crosstab(df["influencer_tier"], df[col], normalize="index") * 100
    fig, ax = plt.subplots(figsize=(10, 6))
    order = [c for c in ["Positive","Neutral","Negative"] if c in ct.columns]
    ct[order].plot(kind="bar", ax=ax, color=[COLORS.get(c, "#999") for c in order])
    ax.set_title("Sentiment by Influencer Tier", fontweight="bold", fontsize=14)
    ax.set_ylabel("Percentage (%)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(title="Sentiment")
    plt.tight_layout()
    _save(fig, "08_influencer_sentiment.png", output_dir)


def generate_all_plots(df, output_dir=None):
    """Generate all 8 visualizations."""
    out = output_dir or OUTPUT_DIR
    print(f"Generating all plots to {out}/...")
    plot_sentiment_distribution(df, output_dir=out)
    plot_platform_comparison(df, output_dir=out)
    plot_sentiment_over_time(df, output_dir=out)
    plot_engagement_by_sentiment(df, output_dir=out)
    plot_vader_score_distribution(df, output_dir=out)
    plot_top_hashtags(df, output_dir=out)
    plot_posting_heatmap(df, output_dir=out)
    plot_influencer_sentiment(df, output_dir=out)
    print("All 8 plots generated.")
