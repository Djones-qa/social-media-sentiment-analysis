"""Unit tests for text preprocessing pipeline."""
import pytest
import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.text_preprocessing import clean_text, tokenize, remove_stopwords, extract_hashtags, extract_mentions, preprocess_text


class TestCleanText:
    def test_removes_urls(self):
        assert "http" not in clean_text("Check this https://example.com out")

    def test_removes_mentions(self):
        assert "@" not in clean_text("Hello @user123 how are you")

    def test_removes_hashtag_symbol(self):
        result = clean_text("#python is great")
        assert "#" not in result
        assert "python" in result

    def test_lowercase(self):
        assert clean_text("HELLO WORLD") == "hello world"

    def test_handles_empty(self):
        assert clean_text("") == ""
        assert clean_text(None) == ""

    def test_removes_numbers(self):
        result = clean_text("I have 3 dogs and 2 cats")
        assert "3" not in result
        assert "2" not in result


class TestTokenize:
    def test_returns_list(self):
        assert isinstance(tokenize("hello world"), list)

    def test_correct_count(self):
        assert len(tokenize("one two three")) == 3


class TestRemoveStopwords:
    def test_removes_common_words(self):
        tokens = ["this", "is", "a", "great", "product"]
        result = remove_stopwords(tokens)
        assert "is" not in result
        assert "great" in result

    def test_custom_stopwords(self):
        tokens = ["hello", "world", "test"]
        result = remove_stopwords(tokens, extra_stopwords=["test"])
        assert "test" not in result


class TestExtractHashtags:
    def test_extracts_tags(self):
        assert extract_hashtags("Love #python and #data") == ["python", "data"]

    def test_no_tags(self):
        assert extract_hashtags("No hashtags here") == []

    def test_handles_none(self):
        assert extract_hashtags(None) == []


class TestExtractMentions:
    def test_extracts_mentions(self):
        assert extract_mentions("Thanks @alice and @bob") == ["alice", "bob"]


class TestPreprocessText:
    def test_full_pipeline(self):
        text = "Check out https://link.com @user #NLP is amazing!!!"
        result = preprocess_text(text)
        assert isinstance(result, str)
        assert "http" not in result
        assert "@" not in result
        assert len(result) > 0
