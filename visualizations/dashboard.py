"""
dashboard.py — Interactive Plotly dashboards exported as HTML.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path


def create_sentiment_dashboard(df, sentiment_col="ensemble_sentiment"):
    """Multi-panel sentiment overview dashboard."""
    fig = make_subplots(rows=2, cols=2,
        subplot_titles=("Sentiment Distribution", "VADER Score Distribution",
                        "Subjectivity Distribution", "Engagement by Sentiment"),
        specs=[[{"type": "pie"}, {"type": "histogram"}],
               [{"type": "histogram"}, {"type": "bar"}]])

    counts = df[sentiment_col].value_counts()
    colors = {"Positive": "#4CAF50", "Negative": "#F44336", "Neutral": "#2196F3"}
    fig.add_trace(go.Pie(labels=counts.index, values=counts.values,
        marker_colors=[colors.get(l, "#999") for l in counts.index]), row=1, col=1)

    if "vader_compound" in df.columns:
        fig.add_trace(go.Histogram(x=df["vader_compound"], nbinsx=50,
            marker_color="#673AB7", name="VADER"), row=1, col=2)

    if "textblob_subjectivity" in df.columns:
        fig.add_trace(go.Histogram(x=df["textblob_subjectivity"], nbinsx=40,
            marker_color="#FF9800", name="Subjectivity"), row=2, col=1)

    if "likes" in df.columns:
        eng = df.groupby(sentiment_col)["likes"].mean()
        fig.add_trace(go.Bar(x=eng.index, y=eng.values,
            marker_color=[colors.get(l, "#999") for l in eng.index], name="Avg Likes"), row=2, col=2)

    fig.update_layout(title="Sentiment Analysis Dashboard", height=700,
                      showlegend=False, template="plotly_white")
    return fig


def create_trend_dashboard(df, date_col="timestamp", sentiment_col="ensemble_sentiment"):
    """Time-series sentiment trend dashboard."""
    df = df.copy()
    df["week"] = pd.to_datetime(df[date_col]).dt.to_period("W").dt.to_timestamp()
    weekly = df.groupby("week").agg(
        total=("post_id", "count") if "post_id" in df.columns else (sentiment_col, "count"),
        avg_vader=("vader_compound", "mean") if "vader_compound" in df.columns else (sentiment_col, "count"),
    ).reset_index()

    fig = make_subplots(rows=2, cols=1,
        subplot_titles=("Weekly Post Volume", "Weekly Avg VADER Score"))
    fig.add_trace(go.Bar(x=weekly["week"], y=weekly["total"],
        marker_color="#2196F3", name="Posts"), row=1, col=1)
    fig.add_trace(go.Scatter(x=weekly["week"], y=weekly["avg_vader"],
        mode="lines+markers", line=dict(color="#4CAF50", width=2), name="VADER"), row=2, col=1)
    fig.update_layout(title="Trend Dashboard", height=600, showlegend=False, template="plotly_white")
    return fig


def create_platform_dashboard(df, sentiment_col="ensemble_sentiment"):
    """Platform comparison dashboard."""
    if "platform" not in df.columns:
        return None
    fig = px.sunburst(df, path=["platform", sentiment_col],
        color=sentiment_col, color_discrete_map={"Positive":"#4CAF50","Negative":"#F44336","Neutral":"#2196F3"},
        title="Platform Sentiment Sunburst")
    fig.update_layout(height=600, template="plotly_white")
    return fig


def export_dashboards(df, output_dir="visualizations/output"):
    """Generate and save all interactive dashboards as HTML."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print("Generating interactive dashboards...")

    fig1 = create_sentiment_dashboard(df)
    fig1.write_html(str(out / "dashboard_sentiment.html"))
    print(f"  Saved: {out}/dashboard_sentiment.html")

    fig2 = create_trend_dashboard(df)
    fig2.write_html(str(out / "dashboard_trends.html"))
    print(f"  Saved: {out}/dashboard_trends.html")

    fig3 = create_platform_dashboard(df)
    if fig3:
        fig3.write_html(str(out / "dashboard_platform.html"))
        print(f"  Saved: {out}/dashboard_platform.html")

    print("Dashboards complete.")
