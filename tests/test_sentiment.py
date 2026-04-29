"""Unit tests for sentiment analysis functions."""
import pytest
import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.sentiment_analyzer import get_vader_sentiment, get_textblob_sentiment, classify_sentiment


class TestVaderSentiment:
    def test_positive(self):
        result = get_vader_sentiment("I absolutely love this product, it is amazing!")
        assert result["compound"] > 0

    def test_negative(self):
        result = get_vader_sentiment("This is terrible and I hate it so much")
        assert result["compound"] < 0

    def test_neutral(self):
        result = get_vader_sentiment("The meeting is at 3pm today")
        assert -0.3 < result["compound"] < 0.3

    def test_empty_string(self):
        result = get_vader_sentiment("")
        assert result["compound"] == 0.0

    def test_none_input(self):
        result = get_vader_sentiment(None)
        assert result["compound"] == 0.0

    def test_returns_all_keys(self):
        result = get_vader_sentiment("test text")
        assert all(k in result for k in ["compound", "pos", "neu", "neg"])


class TestTextBlobSentiment:
    def test_positive(self):
        result = get_textblob_sentiment("This is wonderful and fantastic!")
        assert result["polarity"] > 0

    def test_negative(self):
        result = get_textblob_sentiment("This is awful and horrible")
        assert result["polarity"] < 0

    def test_returns_subjectivity(self):
        result = get_textblob_sentiment("I think this is great")
        assert 0 <= result["subjectivity"] <= 1


class TestClassifySentiment:
    def test_positive(self):
        assert classify_sentiment(0.5) == "Positive"

    def test_negative(self):
        assert classify_sentiment(-0.5) == "Negative"

    def test_neutral(self):
        assert classify_sentiment(0.0) == "Neutral"

    def test_threshold_boundary_pos(self):
        assert classify_sentiment(0.05) == "Neutral"
        assert classify_sentiment(0.06) == "Positive"

    def test_threshold_boundary_neg(self):
        assert classify_sentiment(-0.05) == "Neutral"
        assert classify_sentiment(-0.06) == "Negative"

    def test_custom_thresholds(self):
        assert classify_sentiment(0.15, pos_threshold=0.2) == "Neutral"
        assert classify_sentiment(0.25, pos_threshold=0.2) == "Positive"
