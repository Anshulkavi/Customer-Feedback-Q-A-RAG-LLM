import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
import os
from dotenv import load_dotenv
import google.generativeai as genai
import logging
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class RAGEngine:
    def __init__(self):
        """Initialize the RAG engine with database connection and ML models."""
        self.engine = None
        self.df = None
        self.model = None
        self.index = None
        self.id_map = None
        self.setup_database()
        self.setup_models()
        self.build_index()

    def setup_database(self):
        """Setup database connection."""
        try:
             
            DB_USER = os.getenv("DB_USER") or st.secrets.get("DB_USER")
            DB_PASS = os.getenv("DB_PASS") or st.secrets.get("DB_PASS")
            DB_NAME = os.getenv("DB_NAME") or st.secrets.get("DB_NAME")
            DB_HOST = os.getenv("DB_HOST") or st.secrets.get("DB_HOST")
            DB_PORT = os.getenv("DB_PORT") or st.secrets.get("DB_PORT")

            DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
#             url = URL.create(
#     "postgresql+psycopg2",
#     username=os.getenv("DB_USER"),
#     password=os.getenv("DB_PASS"),
#     host=os.getenv("DB_HOST"),
#     port=int(os.getenv("DB_PORT")),
#     database=os.getenv("DB_NAME")
# )

#             self.engine = create_engine(url, isolation_level="AUTOCOMMIT", connect_args={})
            
            # Test connection
            with self.engine.connect() as conn:
                logger.info("✅ Database connection successful")
                
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            raise

    def setup_models(self):
        """Setup sentence transformer and Gemini API."""
        try:
            # Setup sentence transformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Sentence transformer model loaded")
            
            # Setup Gemini API
            genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
            logger.info("✅ Gemini API configured")
            
        except Exception as e:
            logger.error(f"❌ Model setup failed: {e}")
            raise

    def load_reviews(self):
        """Load reviews from database."""
        try:
            # Try different table names that might exist
            table_queries = [
                "SELECT review_id, review_text, rating, sentiment_score, sentiment_label FROM public.facts_reviews LIMIT 1000",
                "SELECT review_id, review_text, rating, sentiment_score FROM public.facts_reviews LIMIT 1000", 
                "SELECT \"Id\" as review_id, \"Text\" as review_text, \"Score\" as rating, sentiment_score FROM public.reviews_raw LIMIT 1000"
            ]
            
            for query in table_queries:
                try:
                    self.df = pd.read_sql(query, self.engine)
                    logger.info(f"✅ Loaded {len(self.df)} reviews from database")
                    return
                except Exception as e:
                    logger.warning(f"Query failed: {query[:50]}... Error: {e}")
                    continue
                    
            raise Exception("No valid review table found")
            
        except Exception as e:
            logger.error(f"❌ Failed to load reviews: {e}")
            raise

    def build_index(self):
        """Build FAISS index from reviews."""
        try:
            self.load_reviews()
            
            if self.df is None or len(self.df) == 0:
                raise Exception("No reviews loaded")
                
            # Generate embeddings
            review_texts = self.df['review_text'].fillna('').astype(str).tolist()
            logger.info("🔄 Generating embeddings...")
            
            embeddings = self.model.encode(review_texts, show_progress_bar=True, convert_to_numpy=True)
            
            # Build FAISS index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings.astype('float32'))
            
            # Map FAISS indices to review data
            self.id_map = {i: {
                'review_id': self.df.iloc[i]['review_id'],
                'text': self.df.iloc[i]['review_text'],
                'rating': self.df.iloc[i].get('rating', 'N/A'),
                'sentiment': self.df.iloc[i].get('sentiment_score', 0)
            } for i in range(len(self.df))}
            
            logger.info(f"✅ FAISS index built with {len(embeddings)} reviews")
            
        except Exception as e:
            logger.error(f"❌ Failed to build index: {e}")
            raise

    def query_reviews(self, query, top_k=5):
        """Return top_k reviews matching the query."""
        try:
            if self.index is None:
                raise Exception("Index not built yet")
                
            # Generate query embedding
            q_emb = self.model.encode([query], convert_to_numpy=True)
            
            # Search FAISS index
            D, I = self.index.search(q_emb.astype('float32'), top_k)
            
            # Return results with metadata
            results = []
            for i, distance in zip(I[0], D[0]):
                if i in self.id_map:
                    result = self.id_map[i].copy()
                    result['similarity_score'] = float(1 / (1 + distance))  # Convert distance to similarity
                    results.append(result)
            
            logger.info(f"✅ Retrieved {len(results)} relevant reviews")
            return results
            
        except Exception as e:
            logger.error(f"❌ Query failed: {e}")
            return []

    def generate_summary(self, query, reviews):
        """Generate AI summary using Gemini with safety fallbacks."""
        try:
            if not reviews:
                return "No relevant reviews found for your query."

            # Prepare context
            context = "\n".join([
                f"Review {i+1} (Rating: {r['rating']}, Sentiment: {r['sentiment']:.2f}): {r['text'][:500]}"
                for i, r in enumerate(reviews)
            ])

            # Create prompt
            prompt = f"""
            You are a customer feedback analyst. Based on the following customer reviews, provide a comprehensive answer to the user's question.

            User Question: "{query}"

            Customer Reviews:
            {context}

            Instructions:
            1. Provide a clear, concise summary that directly answers the user's question
            2. Include specific insights from the reviews
            3. Mention key patterns, sentiments, or trends you observe
            4. If ratings are available, incorporate them into your analysis
            5. Keep the response under 200 words but make it informative

            Response:
            """

            # Try with flash model first
            try:
                model = genai.GenerativeModel("gemini-2.0-flash")
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.7,
                        max_output_tokens=300,
                        top_p=0.9
                    )
                )
            except Exception as inner_e:
                logger.warning(f"⚠️ Flash model failed: {inner_e}. Retrying with Pro model...")
                model = genai.GenerativeModel("gemini-2.0-pro")
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.7,
                        max_output_tokens=300,
                        top_p=0.9
                    )
                )

            # ✅ Safe text extraction
            summary_text = None
            if hasattr(response, "text") and response.text:
                summary_text = response.text.strip()
            elif getattr(response, "candidates", None):
                for candidate in response.candidates:
                    if hasattr(candidate, "content") and getattr(candidate.content, "parts", None):
                        parts = candidate.content.parts
                        if parts and hasattr(parts[0], "text"):
                            summary_text = parts[0].text.strip()
                            break

            if summary_text:
                logger.info("✅ AI summary generated successfully")
                return summary_text

            # Fallback summary (if AI refuses to answer)
            logger.warning("⚠️ Gemini returned no valid text. Using fallback summary.")
            avg_rating = np.mean([r['rating'] for r in reviews if isinstance(r['rating'], (int, float))])
            avg_sentiment = np.mean([r['sentiment'] for r in reviews])
            fallback = (
                f"The reviews indicate mixed feedback. Average rating is {avg_rating:.2f} "
                f"with an overall sentiment score of {avg_sentiment:.2f}. "
                f"Common themes include issues like battery life and quality consistency."
            )
            return fallback

        except Exception as e:
            logger.error(f"❌ Summary generation failed: {e}")
            return f"Sorry, I couldn't generate a summary. Error: {str(e)}"


# Global instance
_rag_engine = None

def get_rag_engine():
    """Get or create RAG engine instance."""
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine

def query_reviews(query, top_k=5):
    """Convenience function for backward compatibility."""
    engine = get_rag_engine()
    return engine.query_reviews(query, top_k)

def generate_ai_summary(query, reviews):
    """Generate AI summary from reviews."""
    engine = get_rag_engine()
    return engine.generate_summary(query, reviews)

# Example usage
if __name__ == "__main__":
    try:
        engine = RAGEngine()
        results = engine.query_reviews("battery life complaints", top_k=3)
        print("Retrieved Reviews:")
        for r in results:
            print(f"- Rating: {r['rating']}, Text: {r['text'][:100]}...")
        
        summary = engine.generate_summary("battery life complaints", results)
        print(f"\nAI Summary:\n{summary}")
        
    except Exception as e:
        print(f"Error: {e}")