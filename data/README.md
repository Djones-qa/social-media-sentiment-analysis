# Data Directory

## Structure

| Folder | Purpose |
|---|---|
| `raw/` | Original social media datasets (CSV/JSON) |
| `processed/` | Cleaned, tokenized, sentiment-labeled data |
| `external/` | Lexicons, stopword lists, emoji mappings |

## Expected Schema

| Column | Type | Description |
|---|---|---|
| `post_id` | string | Unique post identifier |
| `text` | string | Raw post text content |
| `timestamp` | datetime | Post publication time |
| `platform` | string | Twitter, Instagram, or LinkedIn |
| `username` | string | Author handle |
| `followers` | int | Author follower count |
| `likes` | int | Like/favorite count |
| `comments` | int | Comment/reply count |
| `shares` | int | Retweet/share count |
| `hashtags` | string | Comma-separated hashtags |
| `mentions` | string | Comma-separated mentions |
| `sentiment` | string | Positive, Negative, Neutral (labeled data) |
| `is_verified` | bool | Verified/influencer account flag |
| `media_type` | string | text, image, video, link |
| `language` | string | ISO language code |
| `engagement_rate` | float | (likes+comments+shares)/followers |

> Raw data files are excluded from version control via `.gitignore`.
