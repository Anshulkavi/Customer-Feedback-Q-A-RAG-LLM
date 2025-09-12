SELECT
    "Id" AS review_id,
    "ProductId" AS product_id,
    "UserId" AS user_id,
    "Score" AS rating,
    "sentiment_score",
    "Time" AS review_time,
    "Text" AS review_text
FROM public.reviews_raw  -- no semicolon here
