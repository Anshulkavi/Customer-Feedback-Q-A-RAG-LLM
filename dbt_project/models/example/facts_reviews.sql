SELECT
    review_id,
    product_id,
    user_id,
    rating,
    sentiment_score,
    review_time,
    review_text,
    CASE
        WHEN sentiment_score > 0.2 THEN 'Positive'
        WHEN sentiment_score < -0.2 THEN 'Negative'
        ELSE 'Neutral'
    END AS sentiment_label
FROM {{ ref('staging_reviews') }}  -- no semicolon
