-- Overall sentiment distribution
SELECT ensemble_sentiment, COUNT(*) AS post_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM sentiment_scores), 1) AS pct
FROM sentiment_scores GROUP BY ensemble_sentiment ORDER BY post_count DESC;

-- Sentiment by platform
SELECT p.platform, s.ensemble_sentiment, COUNT(*) AS count,
    ROUND(AVG(s.vader_compound), 4) AS avg_vader
FROM posts p JOIN sentiment_scores s ON p.post_id = s.post_id
GROUP BY p.platform, s.ensemble_sentiment ORDER BY p.platform, count DESC;

-- Most positive posts
SELECT p.text, s.vader_compound, s.ensemble_sentiment, p.likes, p.shares
FROM posts p JOIN sentiment_scores s ON p.post_id = s.post_id
WHERE s.ensemble_sentiment = 'Positive'
ORDER BY s.vader_compound DESC LIMIT 20;

-- Most negative posts
SELECT p.text, s.vader_compound, s.ensemble_sentiment, p.likes, p.shares
FROM posts p JOIN sentiment_scores s ON p.post_id = s.post_id
WHERE s.ensemble_sentiment = 'Negative'
ORDER BY s.vader_compound ASC LIMIT 20;

-- VADER vs TextBlob agreement rate
SELECT
    ROUND(SUM(CASE WHEN vader_sentiment = textblob_sentiment THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS agreement_pct,
    COUNT(*) AS total
FROM sentiment_scores;

-- Average subjectivity by sentiment
SELECT ensemble_sentiment, ROUND(AVG(textblob_subjectivity), 4) AS avg_subjectivity,
    COUNT(*) AS count
FROM sentiment_scores GROUP BY ensemble_sentiment;
