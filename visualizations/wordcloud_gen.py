"""
wordcloud_gen.py — Generate word clouds by sentiment, platform, and hashtag.
"""

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from pathlib import Path

OUTPUT_DIR = "visualizations/output"
SENTIMENT_COLORS = {
    "Positive": "Greens",
    "Negative": "Reds",
    "Neutral": "Blues",
}


def generate_wordcloud(text, title="Word Cloud", colormap="viridis", output_path=None):
    """Generate a single word cloud from a text string."""
    wc = WordCloud(width=1200, height=600, background_color="white",
                   colormap=colormap, max_words=150, collocations=False)
    wc.generate(text)
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    plt.close(fig)


def wordcloud_by_sentiment(df, text_col="text_processed", sentiment_col="ensemble_sentiment",
                           output_dir=None):
    """Generate separate word clouds for each sentiment category."""
    out = output_dir or OUTPUT_DIR
    for label in ["Positive", "Negative", "Neutral"]:
        subset = df[df[sentiment_col] == label]
        if len(subset) < 5:
            continue
        text = " ".join(subset[text_col].dropna())
        if len(text.strip()) < 10:
            continue
        cmap = SENTIMENT_COLORS.get(label, "viridis")
        generate_wordcloud(text, title=f"{label} Sentiment — Word Cloud",
                          colormap=cmap, output_path=f"{out}/wc_{label.lower()}.png")


def wordcloud_by_platform(df, text_col="text_processed", output_dir=None):
    """Generate word clouds per platform."""
    out = output_dir or OUTPUT_DIR
    if "platform" not in df.columns:
        return
    cmaps = {"Twitter": "Blues", "Instagram": "Purples", "LinkedIn": "Greens"}
    for platform in df["platform"].unique():
        subset = df[df["platform"] == platform]
        if len(subset) < 5:
            continue
        text = " ".join(subset[text_col].dropna())
        if len(text.strip()) < 10:
            continue
        cmap = cmaps.get(platform, "viridis")
        generate_wordcloud(text, title=f"{platform} — Word Cloud",
                          colormap=cmap, output_path=f"{out}/wc_{platform.lower()}.png")


def wordcloud_hashtags(df, output_dir=None):
    """Generate word cloud from hashtags."""
    out = output_dir or OUTPUT_DIR
    if "hashtags_list" not in df.columns:
        return
    all_tags = df["hashtags_list"].explode().dropna()
    all_tags = all_tags[all_tags.str.len() > 0]
    if len(all_tags) < 5:
        return
    text = " ".join(all_tags)
    generate_wordcloud(text, title="Hashtag Cloud", colormap="plasma",
                      output_path=f"{out}/wc_hashtags.png")


def generate_all_wordclouds(df, output_dir=None):
    """Generate all word clouds."""
    out = output_dir or OUTPUT_DIR
    print(f"Generating word clouds to {out}/...")
    text = " ".join(df["text_processed"].dropna())
    generate_wordcloud(text, "All Posts — Word Cloud", output_path=f"{out}/wc_all.png")
    wordcloud_by_sentiment(df, output_dir=out)
    wordcloud_by_platform(df, output_dir=out)
    wordcloud_hashtags(df, output_dir=out)
    print("Word clouds generated.")
