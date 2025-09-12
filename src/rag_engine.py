# import faiss
# import openai
# from sentence_transformers import SentenceTransformer
# import pandas as pd
# from sqlalchemy import create_engine
# from dotenv import load_dotenv
# import os

# load_dotenv()

# # DB + Model Setup
# engine = create_engine(
#     f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
# )

# df = pd.read_sql("SELECT review_id, review_text FROM public.facts_reviews", engine)

# model = SentenceTransformer('all-MiniLM-L6-v2')
# embeddings = model.encode(df['review_text'].tolist())

# # Build FAISS index
# dimension = embeddings.shape[1]
# index = faiss.IndexFlatL2(dimension)
# index.add(embeddings)

# # Store mapping
# id_map = {i: rid for i, rid in enumerate(df['review_id'])}

# # Query Function
# def query_reviews(query, top_k=3):
#     q_emb = model.encode([query])
#     D, I = index.search(q_emb, top_k)
#     results = [df.iloc[i].review_text for i in I[0]]
#     return results

# # Example
# print(query_reviews("complaints about delivery"))

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

load_dotenv()

# --- DB Setup ---
engine = create_engine(
    f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@"
    f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

df = pd.read_sql("SELECT review_id, review_text FROM public.facts_reviews", engine)

# --- Model Setup ---
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df['review_text'].tolist(), show_progress_bar=True)

# --- Build FAISS index ---
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Map FAISS indices to review IDs
id_map = {i: rid for i, rid in enumerate(df['review_id'])}

# --- Query Function ---
def query_reviews(query, top_k=3):
    """Return top_k reviews matching the query."""
    q_emb = model.encode([query])
    D, I = index.search(q_emb, top_k)
    results = [{"id": id_map[i], "text": df.iloc[i].review_text} for i in I[0]]
    return results

# --- Example ---
if __name__ == "__main__":
    print(query_reviews("complaints about delivery"))
