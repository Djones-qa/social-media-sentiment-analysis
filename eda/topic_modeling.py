"""
topic_modeling.py — LDA topic extraction from social media text.
"""

import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


def fit_lda_topics(df, text_col="text_processed", num_topics=5, max_features=3000, words_per_topic=10):
    """Fit LDA model and return topics with top words."""
    vec = CountVectorizer(max_features=max_features, max_df=0.95, min_df=2)
    dtm = vec.fit_transform(df[text_col].dropna())
    feature_names = vec.get_feature_names_out()

    lda = LatentDirichletAllocation(
        n_components=num_topics, random_state=42, max_iter=20, learning_method="online"
    )
    lda.fit(dtm)

    topics = {}
    for idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[-words_per_topic:][::-1]]
        topics[f"Topic {idx + 1}"] = top_words

    return lda, vec, topics


def assign_dominant_topic(df, lda, vec, text_col="text_processed"):
    """Assign the dominant topic to each document."""
    df = df.copy()
    dtm = vec.transform(df[text_col].fillna(""))
    topic_dist = lda.transform(dtm)
    df["dominant_topic"] = topic_dist.argmax(axis=1) + 1
    df["topic_confidence"] = topic_dist.max(axis=1).round(4)
    return df


def topic_sentiment_breakdown(df, sentiment_col="ensemble_sentiment"):
    """Sentiment distribution within each topic."""
    if "dominant_topic" not in df.columns:
        return pd.DataFrame()
    ct = pd.crosstab(df["dominant_topic"], df[sentiment_col], normalize="index") * 100
    return ct.round(1)


def print_topics(topics):
    """Print discovered topics and their top words."""
    print("\n=== Discovered Topics ===")
    for name, words in topics.items():
        print(f"  {name}: {', '.join(words)}")
