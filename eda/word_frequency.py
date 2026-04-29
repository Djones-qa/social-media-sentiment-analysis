"""
word_frequency.py — Word counts, n-grams, and term frequency analysis.
"""

import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def word_frequency(df, text_col="text_processed", n=30):
    """Get top N most frequent words."""
    all_words = " ".join(df[text_col].dropna()).split()
    return pd.Series(Counter(all_words)).sort_values(ascending=False).head(n)


def word_frequency_by_sentiment(df, text_col="text_processed", sentiment_col="ensemble_sentiment", n=15):
    """Top words per sentiment category."""
    results = {}
    for label in ["Positive", "Negative", "Neutral"]:
        subset = df[df[sentiment_col] == label]
        if len(subset) > 0:
            results[label] = word_frequency(subset, text_col, n)
    return results


def get_ngrams(df, text_col="text_processed", n=2, top_k=20):
    """Extract top-k n-grams from text."""
    vec = CountVectorizer(ngram_range=(n, n), max_features=top_k)
    X = vec.fit_transform(df[text_col].dropna())
    freqs = X.sum(axis=0).A1
    ngrams = vec.get_feature_names_out()
    return pd.Series(freqs, index=ngrams).sort_values(ascending=False)


def tfidf_top_terms(df, text_col="text_processed", max_features=50):
    """Get top terms by TF-IDF score."""
    vec = TfidfVectorizer(max_features=max_features)
    X = vec.fit_transform(df[text_col].dropna())
    scores = X.mean(axis=0).A1
    terms = vec.get_feature_names_out()
    return pd.Series(scores, index=terms).sort_values(ascending=False)


def hashtag_frequency(df, n=25):
    """Top hashtags by frequency."""
    if "hashtags_list" not in df.columns:
        return pd.Series(dtype=str)
    all_tags = df["hashtags_list"].explode().dropna()
    all_tags = all_tags[all_tags.str.len() > 0]
    return all_tags.value_counts().head(n)


def hashtag_sentiment(df, sentiment_col="ensemble_sentiment", top_n=10):
    """Sentiment breakdown for top hashtags."""
    if "hashtags_list" not in df.columns:
        return pd.DataFrame()
    exploded = df.explode("hashtags_list").dropna(subset=["hashtags_list"])
    exploded = exploded[exploded["hashtags_list"].str.len() > 0]
    top_tags = exploded["hashtags_list"].value_counts().head(top_n).index
    subset = exploded[exploded["hashtags_list"].isin(top_tags)]
    ct = pd.crosstab(subset["hashtags_list"], subset[sentiment_col], normalize="index") * 100
    return ct.round(1)
