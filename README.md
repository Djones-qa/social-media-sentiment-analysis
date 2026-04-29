# Social Media Sentiment Analysis

**NLP-powered sentiment classification and trend analytics across Twitter, Instagram, and LinkedIn.**

[![CI](https://github.com/Djones-qa/social-media-sentiment-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/Djones-qa/social-media-sentiment-analysis/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Project Overview

This project builds a complete NLP pipeline for analyzing social media sentiment at scale. It classifies posts as **Positive**, **Negative**, or **Neutral** using both lexicon-based (VADER, TextBlob) and machine learning approaches, then surfaces actionable insights through engagement analysis, hashtag trends, topic modeling, and influencer impact scoring.

### Key Objectives

- **Sentiment Classification** - VADER, TextBlob, and trained ML classifiers (NB, LR, RF, SVM, GB)
- **Text Preprocessing** - Tokenization, stopword removal, lemmatization, URL/mention stripping
- **Platform Comparison** - Sentiment patterns across Twitter, Instagram, and LinkedIn
- **Engagement Analytics** - Correlate sentiment with likes, comments, shares
- **Topic Modeling** - LDA topic extraction with per-topic sentiment breakdown
- **Hashtag Trend Analysis** - Trending hashtags and sentiment associations
- **Influencer Impact** - Tiered scoring (Nano to Mega) with sentiment influence
- **8 Professional Visualizations** - Charts, word clouds, and interactive Plotly dashboards

---

## Repository Structure

    social-media-sentiment-analysis/
    .github/workflows/ci.yml
    config/config.yaml
    data/
        raw/                        Original social media CSVs
        processed/                  Cleaned, sentiment-scored data
        external/                   Lexicons, stopword lists
        README.md
    eda/
        exploratory_analysis.py     Sentiment distribution, platform comparison
        word_frequency.py           Word counts, n-grams, TF-IDF
        topic_modeling.py           LDA topic extraction
    models/
        train.py                    5-model classifier comparison
        predict.py                  Single and batch prediction
        evaluate.py                 Metrics, confusion matrix, error analysis
        saved_models/
    notebooks/
        01_data_exploration.ipynb
        02_preprocessing_sentiment.ipynb
        03_model_training.ipynb
        04_visualization.ipynb
    sql/
        create_tables.sql           posts, sentiment_scores, hashtag_trends, influencers
        sentiment_queries.sql       Distribution, VADER vs TextBlob agreement
        engagement_analytics.sql    Engagement by sentiment, time, media type
        trend_analysis.sql          Daily/weekly trends, platform growth
    src/
        __init__.py
        data_loader.py
        text_preprocessing.py       Tokenization, stopwords, lemmatization
        sentiment_analyzer.py       VADER + TextBlob + ensemble
        feature_engineering.py      Engagement, influencer scoring, temporal
        utils.py
    tests/
        test_preprocessing.py       14 text cleaning tests
        test_sentiment.py           14 sentiment scoring tests
        test_models.py              5 evaluation metric tests
    visualizations/
        plots.py                    8 matplotlib/seaborn charts
        wordcloud_gen.py            Word clouds by sentiment/platform/hashtag
        dashboard.py                Interactive Plotly dashboards
        output/
    .gitignore
    requirements.txt
    README.md

---

## NLP Pipeline

    Raw Text -> Clean -> Tokenize -> Stopwords -> Lemmatize -> Score -> Classify -> Analyze

### Text Preprocessing (src/text_preprocessing.py)
- URL, mention, and hashtag symbol removal
- Emoji counting and special character stripping
- NLTK tokenization with stopword removal
- WordNet lemmatization (configurable stemming)

### Sentiment Analysis (src/sentiment_analyzer.py)
- **VADER**: Rule-based, optimized for social media (handles slang, emojis, caps)
- **TextBlob**: Pattern-based polarity and subjectivity scoring
- **Ensemble**: Majority vote combining both for robust classification

### ML Classifiers (models/train.py)

| Model | Approach |
|---|---|
| Naive Bayes | Probabilistic baseline |
| Logistic Regression | Linear with TF-IDF |
| Random Forest | 200 decision trees |
| Linear SVM | Maximum margin |
| Gradient Boosting | 150 sequential estimators |

All models use TF-IDF (5000 features, uni+bigrams) with 5-fold stratified CV.

---

## Visualizations

### Static Plots (8 charts)
1. Sentiment pie + bar distribution
2. Platform comparison (stacked bar)
3. Sentiment trend over time (line)
4. Engagement by sentiment (grouped bar)
5. VADER score distribution (histogram)
6. Top hashtags (horizontal bar)
7. Posting activity heatmap (hour x day)
8. Influencer tier sentiment breakdown

### Word Clouds
- Overall, per-sentiment, per-platform, and hashtag clouds

### Interactive Dashboards (Plotly HTML)
- Sentiment overview (4 panels)
- Time-series trend
- Platform sunburst

---

## Quick Start

    git clone https://github.com/Djones-qa/social-media-sentiment-analysis.git
    cd social-media-sentiment-analysis
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    pytest tests/ -v --cov=src

Place your CSV in data/raw/ then run notebooks 01 through 04.

---

## Tech Stack

| Layer | Tools |
|---|---|
| NLP | NLTK, TextBlob, VADER |
| ML | scikit-learn (NB, LR, RF, SVM, GB) |
| Topics | LDA via scikit-learn |
| Data | pandas, NumPy, SciPy |
| Database | SQLite, SQLAlchemy |
| Viz | matplotlib, seaborn, WordCloud, Plotly |
| Testing | pytest, pytest-cov |
| CI/CD | GitHub Actions |

---

## Author

**Darrius Jones**
QA Automation Specialist | Backend Engineering | NLP and Data Analytics
