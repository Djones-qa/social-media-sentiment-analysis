-- Daily sentiment trend
SELECT DATE(p.timestamp) AS post_date, s.ensemble_sentiment, COUNT(*) AS count,
    ROUND(AVG(s.vader_compound), 4) AS avg_compound
FROM posts p JOIN sentiment_scores s ON p.post_id = s.post_id
GROUP BY post_date, s.ensemble_sentiment ORDER BY post_date;

-- Weekly volume and sentiment
SELECT strftime('%Y-W%W', p.timestamp) AS week,
    COUNT(*) AS total_posts,
    SUM(CASE WHEN s.ensemble_sentiment = 'Positive' THEN 1 ELSE 0 END) AS positive,
    SUM(CASE WHEN s.ensemble_sentiment = 'Negative' THEN 1 ELSE 0 END) AS negative,
    ROUND(AVG(s.vader_compound), 4) AS avg_sentiment
FROM posts p JOIN sentiment_scores s ON p.post_id = s.post_id
GROUP BY week ORDER BY week;

-- Trending hashtags (last 7 days)
SELECT hashtag, post_count, ROUND(avg_sentiment, 3) AS avg_sentiment,
    ROUND(avg_engagement, 0) AS avg_engagement
FROM hashtag_trends
WHERE date >= DATE('now', '-7 days')
ORDER BY post_count DESC LIMIT 20;

-- Influencer sentiment leaders
SELECT username, platform, followers, total_posts,
    ROUND(avg_sentiment, 3) AS avg_sentiment, influencer_tier
FROM influencers WHERE total_posts >= 5
ORDER BY avg_sentiment DESC LIMIT 15;

-- Platform growth over time
SELECT strftime('%Y-%m', p.timestamp) AS month, p.platform, COUNT(*) AS posts,
    ROUND(AVG(s.vader_compound), 4) AS avg_sentiment
FROM posts p JOIN sentiment_scores s ON p.post_id = s.post_id
GROUP BY month, p.platform ORDER BY month, p.platform;
