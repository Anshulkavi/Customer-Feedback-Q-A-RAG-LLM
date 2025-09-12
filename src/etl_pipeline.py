import pandas as pd
from sqlalchemy import create_engine
from textblob import TextBlob
from dotenv import load_dotenv
import os

load_dotenv()

engine = create_engine(
    f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "Reviews.csv")

# Load CSV
df = pd.read_csv(data_path, nrows=10000)  # use subset for speed

# Sentiment Analysis
def get_sentiment(text):
    try:
        return TextBlob(str(text)).sentiment.polarity
    except:
        return 0

df["sentiment_score"] = df["Text"].apply(get_sentiment)

# Save to DB
df.to_sql("reviews_raw", engine, if_exists="replace", index=False)

print("✅ Data loaded into PostgreSQL with sentiment_score")
