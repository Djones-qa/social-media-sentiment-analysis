-- Engagement by sentiment
SELECT s.ensemble_sentiment,
    ROUND(AVG(p.likes), 0) AS avg_likes,
    ROUND(AVG(p.comments), 0) AS avg_comments,
    ROUND(AVG(p.shares), 0) AS avg_shares,
    ROUND(AVG(CAST(p.likes + p.comments + p.shares AS REAL) / NULLIF(p.followers, 0)) * 100, 4) AS avg_engagement_pct
FROM posts p JOIN sentiment_scores s ON p.post_id = s.post_id
GROUP BY s.ensemble_sentiment ORDER BY avg_engagement_pct DESC;

-- Top engaged posts
SELECT p.text, p.platform, p.likes, p.comments, p.shares,
    (p.likes + p.comments + p.shares) AS total_engagement,
    s.ensemble_sentiment
FROM posts p JOIN sentiment_scores s ON p.post_id = s.post_id
ORDER BY total_engagement DESC LIMIT 25;

-- Engagement by time of day
SELECT CASE
    WHEN CAST(strftime('%H', p.timestamp) AS INT) BETWEEN 0 AND 5 THEN 'Night'
    WHEN CAST(strftime('%H', p.timestamp) AS INT) BETWEEN 6 AND 11 THEN 'Morning'
    WHEN CAST(strftime('%H', p.timestamp) AS INT) BETWEEN 12 AND 17 THEN 'Afternoon'
    ELSE 'Evening' END AS time_of_day,
    COUNT(*) AS posts, ROUND(AVG(p.likes + p.comments + p.shares), 0) AS avg_engagement
FROM posts p GROUP BY time_of_day ORDER BY avg_engagement DESC;

-- Verified vs non-verified engagement
SELECT p.is_verified, COUNT(*) AS posts,
    ROUND(AVG(p.likes), 0) AS avg_likes,
    ROUND(AVG(p.comments), 0) AS avg_comments,
    ROUND(AVG(p.shares), 0) AS avg_shares
FROM posts p GROUP BY p.is_verified;

-- Media type performance
SELECT p.media_type, COUNT(*) AS posts,
    ROUND(AVG(p.likes + p.comments + p.shares), 0) AS avg_engagement,
    s.ensemble_sentiment, COUNT(*) AS sentiment_count
FROM posts p JOIN sentiment_scores s ON p.post_id = s.post_id
GROUP BY p.media_type, s.ensemble_sentiment ORDER BY p.media_type;
