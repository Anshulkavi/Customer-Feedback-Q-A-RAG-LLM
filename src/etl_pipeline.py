# import pandas as pd
# from sqlalchemy import create_engine
# from textblob import TextBlob
# from dotenv import load_dotenv
# import os

# load_dotenv()

# engine = create_engine(
#     f"postgresql+pg8000://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
# )

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# data_path = os.path.join(BASE_DIR, "data", "Reviews.csv")

# # Load CSV
# df = pd.read_csv(data_path, nrows=10000)  # use subset for speed

# # Sentiment Analysis
# def get_sentiment(text):
#     try:
#         return TextBlob(str(text)).sentiment.polarity
#     except:
#         return 0

# df["sentiment_score"] = df["Text"].apply(get_sentiment)

# # Save to DB
# df.to_sql("reviews_raw", engine, if_exists="replace", index=False)

# print("✅ Data loaded into PostgreSQL with sentiment_score")


import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from textblob import TextBlob
from dotenv import load_dotenv
import os
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

def setup_database_connection():
    """Setup the main database connection."""
    try:
        engine = create_engine(
            f"postgresql+pg8000://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
        )
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("✅ Database connection established")
        return engine
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        raise

# --- NEW FUNCTION ---
def cleanup_database_objects(engine):
    """Drop all related tables and views to ensure a clean slate."""
    with engine.connect() as conn:
        logger.info("🧹 Cleaning up old database objects...")
        
        # This order will now work correctly on a clean database
        conn.execute(text("DROP TABLE IF EXISTS facts_reviews CASCADE;"))
        conn.execute(text("DROP VIEW IF EXISTS staging_reviews CASCADE;"))
        conn.execute(text("DROP TABLE IF EXISTS reviews_raw CASCADE;"))
        
        conn.commit()
        logger.info("✅ Cleanup complete.")

def find_data_file():
    """Find the Reviews.csv file or create a sample if not found."""
    # (This function remains the same as your version)
    possible_paths = [ "data/Reviews.csv", "../data/Reviews.csv"]
    for path in possible_paths:
        if os.path.exists(path):
            logger.info(f"✅ Found data file at: {path}")
            return path
    logger.warning("❌ Reviews.csv not found. Creating sample data...")
    return create_sample_data()

def create_sample_data():
    """Create sample review data for demonstration."""
    # (This function remains the same as your version)
    sample_data = {
        'Id': range(1, 101), 'ProductId': [f'B00{i%10:03d}' for i in range(1, 101)],
        'UserId': [f'U{i%20:04d}' for i in range(1, 101)], 'ProfileName': [f'User{i%20}' for i in range(1, 101)],
        'HelpfulnessNumerator': np.random.randint(0, 20, 100), 'HelpfulnessDenominator': np.random.randint(0, 30, 100),
        'Score': np.random.choice([1, 2, 3, 4, 5], 100, p=[0.1, 0.1, 0.2, 0.3, 0.3]),
        'Time': np.random.randint(1000000000, 1600000000, 100),
        'Summary': [f'{"Great" if i%3==0 else "Good" if i%3==1 else "Poor"} product experience' for i in range(100)],
        'Text': [f"""{'This product exceeded my expectations! The quality is outstanding and delivery was fast.' if i%4==0 else 'Decent product but could be better.' if i%4==1 else 'Not satisfied with this purchase.' if i%4==2 else 'Average product, does the job.'} Review number {i}.""" for i in range(1, 101)]
    }
    df = pd.DataFrame(sample_data)
    os.makedirs('data', exist_ok=True)
    data_path = 'data/Reviews.csv'
    df.to_csv(data_path, index=False)
    logger.info(f"✅ Sample data created at: {data_path}")
    return data_path

def get_sentiment(text):
    """Calculate sentiment score for text."""
    try:
        if pd.isna(text) or text == '': return 0.0
        return TextBlob(str(text)).sentiment.polarity
    except:
        return 0.0

def clean_and_process_data(df):
    """Clean and process the review data."""
    # (This function remains the same as your version)
    logger.info("🔄 Cleaning and processing data...")
    df['Text'] = df['Text'].fillna('')
    df['Summary'] = df['Summary'].fillna('')
    logger.info("🔄 Calculating sentiment scores...")
    df["sentiment_score"] = df["Text"].apply(get_sentiment)
    df["sentiment_label"] = df["sentiment_score"].apply(lambda x: "Positive" if x > 0.1 else "Negative" if x < -0.1 else "Neutral")
    df['Score'] = pd.to_numeric(df['Score'], errors='coerce').fillna(3)
    df['Text'] = df['Text'].str.slice(0, 2000)
    logger.info(f"✅ Data processed: {len(df)} reviews")
    return df

# --- MODIFIED FUNCTION ---
def load_data_to_database(engine, df):
    """Load processed data to PostgreSQL. Now much simpler."""
    try:
        logger.info("🔄 Loading data into reviews_raw table...")
        df.to_sql("reviews_raw", engine, if_exists="replace", index=False, method='multi', chunksize=1000)
        with engine.connect() as conn:
            count = conn.execute(text("SELECT COUNT(*) FROM reviews_raw")).scalar()
        logger.info(f"✅ Data loaded successfully: {count} records in reviews_raw table")
    except Exception as e:
        logger.error(f"❌ Failed to load data to database: {e}")
        raise

def run_dbt_models(engine):
    """Create the transformed tables directly in PostgreSQL."""
    # (This function remains the same as your version)
    try:
        logger.info("🔄 Creating transformed views and tables...")
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE VIEW staging_reviews AS
                SELECT
                    "Id" AS review_id, "ProductId" AS product_id, "UserId" AS user_id,
                    "Score" AS rating, sentiment_score, "Time" AS review_time,
                    "Text" AS review_text
                FROM reviews_raw;
            """))
            conn.execute(text("""
                CREATE TABLE facts_reviews AS
                SELECT
                    review_id, product_id, user_id, rating, sentiment_score,
                    review_time, review_text,
                    CASE
                        WHEN sentiment_score > 0.1 THEN 'Positive'
                        WHEN sentiment_score < -0.1 THEN 'Negative'
                        ELSE 'Neutral'
                    END AS sentiment_label
                FROM staging_reviews;
            """))
            conn.commit()
        logger.info("✅ Transformed tables created successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to create transformed tables: {e}")
        raise

def main():
    """Main ETL pipeline execution."""
    engine = None
    try:
        logger.info("🚀 Starting ETL Pipeline...")
        engine = setup_database_connection()
        
        # --- NEW STEP 1 ---
        cleanup_database_objects(engine)

        data_path = find_data_file()
        df = pd.read_csv(data_path, nrows=10000)
        logger.info(f"📊 Loaded {len(df)} records")
        
        df = clean_and_process_data(df)
        load_data_to_database(engine, df)
        run_dbt_models(engine)
        
        logger.info("🎉 ETL Pipeline completed successfully!")
        logger.info("Next steps: Run -> streamlit run streamlit_app/app.py")
        return True
    except Exception as e:
        logger.error(f"❌ ETL Pipeline failed: {e}")
        return False
    finally:
        if engine:
            engine.dispose()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)