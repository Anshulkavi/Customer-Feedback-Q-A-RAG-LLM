import streamlit as st
import sys
import os
from google.cloud import aiplatform
from google.oauth2 import service_account

# Add src folder for rag_engine
# Make sure this path is correct for your project structure
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from rag_engine import query_reviews

# --- Configuration ---
# 1. UPDATE THIS with the path to your NEW service account key file
SERVICE_ACCOUNT_FILE = r"D:\project\customer-feedback-analytics\gen-lang-client-0056236150-53d07f9e018d.json"

# 2. UPDATE THIS with your Google Cloud project ID
PROJECT_ID = "gen-lang-client-0056236150"
REGION = "us-central1"

# --- Initialize Vertex AI ---
# The SDK will use your service account file to authenticate
try:
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)
    aiplatform.init(project=PROJECT_ID, location=REGION, credentials=credentials)
except Exception as e:
    st.error(f"Failed to initialize Vertex AI. Check your service account path and permissions. Error: {e}")
    st.stop()

# --- Streamlit App UI ---
st.set_page_config(page_title="Customer Feedback Q&A", layout="wide")
st.title("Customer Feedback Q&A (RAG + Gemini)")

query = st.text_input("Ask a question about customer reviews:")

if st.button("Get Answer"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        try:
            # Step 1: Retrieve relevant reviews (No change here)
            with st.spinner("Searching for relevant reviews..."):
                results = query_reviews(query, top_k=5)
                st.write("📌 **Retrieved Reviews for Context:**")
                for r in results:
                    st.info(f"ID: {r['id']} - {r['text']}")

            # Step 2: Generate a summary using the Gemini model
            with st.spinner("🤖 Generating AI summary..."):
                # Prepare the prompt for the LLM
                review_texts = "\n".join([f"- {r['text']}" for r in results])
                prompt = f"""
                Based on the following customer reviews, answer the user's question.
                The user's question is: "{query}"

                Here are the relevant reviews:
                {review_texts}

                Provide a concise answer based only on the information in these reviews.
                """

                # --- CODE CHANGE IS HERE ---
                # OLD: model = aiplatform.from_pretrained("gemini-1.0-pro")
                # NEW: Use the GenerativeModel class directly
                model = aiplatform.GenerativeModel("gemini-1.0-pro")

                # OLD: response = model.predict(...)
                # NEW: Use the generate_content method
                response = model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": 0.7,
                        "max_output_tokens": 300
                    }
                )
                # --- END OF CODE CHANGE ---
                
                summary = response.text
                st.subheader("🤖 AI Answer")
                st.success(summary)

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")