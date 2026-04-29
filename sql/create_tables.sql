CREATE TABLE IF NOT EXISTS posts (
    post_id         TEXT PRIMARY KEY,
    text            TEXT NOT NULL,
    text_clean      TEXT,
    text_processed  TEXT,
    timestamp       DATETIME NOT NULL,
    platform        TEXT CHECK (platform IN ('Twitter','Instagram','LinkedIn')),
    username        TEXT,
    followers       INTEGER DEFAULT 0,
    likes           INTEGER DEFAULT 0,
    comments        INTEGER DEFAULT 0,
    shares          INTEGER DEFAULT 0,
    hashtags        TEXT,
    mentions        TEXT,
    media_type      TEXT,
    language        TEXT DEFAULT 'en',
    is_verified     BOOLEAN DEFAULT 0,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sentiment_scores (
    score_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    post_id             TEXT REFERENCES posts(post_id),
    vader_compound      REAL,
    vader_pos           REAL,
    vader_neu           REAL,
    vader_neg           REAL,
    vader_sentiment     TEXT CHECK (vader_sentiment IN ('Positive','Negative','Neutral')),
    textblob_polarity   REAL,
    textblob_subjectivity REAL,
    textblob_sentiment  TEXT CHECK (textblob_sentiment IN ('Positive','Negative','Neutral')),
    ensemble_sentiment  TEXT CHECK (ensemble_sentiment IN ('Positive','Negative','Neutral')),
    scored_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS hashtag_trends (
    trend_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    hashtag         TEXT NOT NULL,
    date            DATE NOT NULL,
    post_count      INTEGER,
    avg_sentiment   REAL,
    avg_engagement  REAL,
    UNIQUE(hashtag, date)
);

CREATE TABLE IF NOT EXISTS influencers (
    influencer_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    username        TEXT UNIQUE NOT NULL,
    platform        TEXT,
    followers       INTEGER,
    is_verified     BOOLEAN,
    total_posts     INTEGER,
    avg_sentiment   REAL,
    avg_engagement  REAL,
    influencer_tier TEXT,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_posts_platform ON posts(platform);
CREATE INDEX IF NOT EXISTS idx_posts_timestamp ON posts(timestamp);
CREATE INDEX IF NOT EXISTS idx_posts_username ON posts(username);
CREATE INDEX IF NOT EXISTS idx_sentiment_post ON sentiment_scores(post_id);
CREATE INDEX IF NOT EXISTS idx_sentiment_label ON sentiment_scores(ensemble_sentiment);
CREATE INDEX IF NOT EXISTS idx_trends_hashtag ON hashtag_trends(hashtag);
