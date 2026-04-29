"""
text_preprocessing.py — Tokenization, stopwords, stemming, lemmatization, cleaning.
"""

import re
import string
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download required NLTK data
for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "averaged_perceptron_tagger"]:
    nltk.download(pkg, quiet=True)


def clean_text(text):
    """Remove URLs, mentions, special characters, extra whitespace."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def tokenize(text):
    """Split text into word tokens."""
    return word_tokenize(text)


def remove_stopwords(tokens, extra_stopwords=None):
    """Remove English stopwords from token list."""
    stop_words = set(stopwords.words("english"))
    if extra_stopwords:
        stop_words.update(extra_stopwords)
    return [t for t in tokens if t not in stop_words and len(t) > 1]


def stem_tokens(tokens):
    """Apply Porter stemming to tokens."""
    stemmer = PorterStemmer()
    return [stemmer.stem(t) for t in tokens]


def lemmatize_tokens(tokens):
    """Apply WordNet lemmatization to tokens."""
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(t) for t in tokens]


def extract_hashtags(text):
    """Extract hashtags from raw text."""
    if not isinstance(text, str):
        return []
    return re.findall(r"#(\w+)", text.lower())


def extract_mentions(text):
    """Extract @mentions from raw text."""
    if not isinstance(text, str):
        return []
    return re.findall(r"@(\w+)", text.lower())


def count_emojis(text):
    """Count emoji characters in text."""
    if not isinstance(text, str):
        return 0
    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0\U000024C2-\U0001F251]+",
        flags=re.UNICODE
    )
    return len(emoji_pattern.findall(text))


def preprocess_text(text, use_lemmatize=True):
    """Full preprocessing pipeline for a single text string."""
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    tokens = remove_stopwords(tokens)
    if use_lemmatize:
        tokens = lemmatize_tokens(tokens)
    else:
        tokens = stem_tokens(tokens)
    return " ".join(tokens)


def preprocess_dataframe(df, text_col="text", use_lemmatize=True):
    """Apply full preprocessing to a DataFrame text column."""
    df = df.copy()
    print(f"Preprocessing {len(df):,} posts...")
    df["text_original"] = df[text_col]
    df["text_clean"] = df[text_col].apply(clean_text)
    df["text_processed"] = df[text_col].apply(lambda x: preprocess_text(x, use_lemmatize))
    df["tokens"] = df["text_processed"].apply(lambda x: x.split())
    df["token_count"] = df["tokens"].apply(len)
    df["char_count"] = df["text_original"].apply(lambda x: len(str(x)))
    df["hashtags_list"] = df["text_original"].apply(extract_hashtags)
    df["hashtag_count"] = df["hashtags_list"].apply(len)
    df["mention_count"] = df["text_original"].apply(lambda x: len(extract_mentions(x)))
    df["emoji_count"] = df["text_original"].apply(count_emojis)
    print(f"Preprocessing complete. Added {9} feature columns.")
    return df
