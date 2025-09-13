# import streamlit as st
# import sys
# import os
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# from datetime import datetime
# import time

# # Add src folder for rag_engine
# # Ensure this path is correct for your project structure
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# try:
#     from src.rag_engine import query_reviews, generate_ai_summary, get_rag_engine
# except ImportError as e:
#     st.error(f"Failed to import RAG engine: {e}")
#     st.stop()

# # --- Page Configuration ---
# st.set_page_config(
#     page_title="🤖 AI Customer Feedback Analytics",
#     page_icon="🤖",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # --- Custom CSS for Modern Look ---
# st.markdown("""
# <style>
#     .main-header {
#         text-align: center;
#         color: #1f77b4;
#         font-size: 3rem;
#         font-weight: bold;
#         margin-bottom: 1rem;
#         background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#     }
    
#     .sub-header {
#         text-align: center;
#         color: #666;
#         font-size: 1.2rem;
#         margin-bottom: 2rem;
#     }
    
#     .metric-card {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         padding: 1rem;
#         border-radius: 10px;
#         color: white;
#         text-align: center;
#         margin: 0.5rem 0;
#     }
    
#     .review-card {
#         background: #f8f9fa;
#         padding: 1rem;
#         border-radius: 8px;
#         border-left: 4px solid #667eea;
#         margin: 0.5rem 0;
#         box-shadow: 0 2px 4px rgba(0,0,0,0.1);
#     }
    
#     .ai-response {
#         background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
#         padding: 1.5rem;
#         border-radius: 10px;
#         color: white;
#         margin: 1rem 0;
#         box-shadow: 0 4px 8px rgba(0,0,0,0.1);
#     }
    
#     .stButton button {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         color: white;
#         border: none;
#         padding: 0.5rem 2rem;
#         border-radius: 25px;
#         font-weight: bold;
#         transition: transform 0.3s ease;
#     }
    
#     .stButton button:hover {
#         transform: translateY(-2px);
#         box-shadow: 0 4px 8px rgba(0,0,0,0.2);
#     }
# </style>
# """, unsafe_allow_html=True)

# # --- App Header ---
# st.markdown('<div class="main-header">🤖 AI Customer Feedback Analytics</div>', unsafe_allow_html=True)
# st.markdown('<div class="sub-header">Powered by RAG + Gemini AI • Transform Reviews into Insights</div>', unsafe_allow_html=True)

# # --- Initialize Session State ---
# if 'chat_history' not in st.session_state:
#     st.session_state.chat_history = []
# # Initialize the key for our text input
# if 'query_input' not in st.session_state:
#     st.session_state.query_input = ""

# # --- Sidebar ---
# with st.sidebar:
#     st.header("🎛️ Control Panel")
    
#     # Model settings
#     st.subheader("⚙️ Settings")
#     top_k = st.slider("Number of reviews to analyze", 3, 10, 5)
    
#     # Example queries
#     st.subheader("💡 Example Queries")
#     example_queries = [
#         "What are the main complaints about battery life?",
#         "How do customers feel about delivery speed?", 
#         "Summarize positive feedback about product quality",
#         "What issues do customers mention about customer service?",
#         "Are there any complaints about packaging?",
#         "What do customers love most about this product?"
#     ]
    
#     for query_text in example_queries:
#         # --- CHANGE 1: Update st.session_state.query_input directly ---
#         if st.button(f"💬 {query_text}", key=f"example_{query_text[:20]}"):
#             st.session_state.query_input = query_text

# # --- Main Interface ---
# col1, col2 = st.columns([2, 1])

# with col1:
#     st.subheader("🔍 Ask About Customer Feedback")
    
#     # --- CHANGE 2: Use the key to manage the text input's state ---
#     query = st.text_input(
#         "Enter your question:",
#         key="query_input", # Assign the key here
#         placeholder="e.g., What are customers saying about the product quality?",
#         help="Ask natural language questions about customer reviews"
#     )

# with col2:
#     st.subheader("📊 Quick Stats")
#     try:
#         engine = get_rag_engine()
#         if hasattr(engine, 'df') and engine.df is not None:
#             total_reviews = len(engine.df)
#             avg_rating = engine.df.get('rating', pd.Series([0])).mean()
#             avg_sentiment = engine.df.get('sentiment_score', pd.Series([0])).mean()
            
#             st.markdown(f'<div class="metric-card"><h3>{total_reviews:,}</h3><p>Total Reviews Analyzed</p></div>', unsafe_allow_html=True)
#             st.markdown(f'<div class="metric-card"><h3>{avg_rating:.1f}/5</h3><p>Average Rating</p></div>', unsafe_allow_html=True)
            
#             sentiment_emoji = "😊" if avg_sentiment > 0.1 else "😐" if avg_sentiment > -0.1 else "😞"
#             st.markdown(f'<div class="metric-card"><h3>{sentiment_emoji} {avg_sentiment:.2f}</h3><p>Average Sentiment</p></div>', unsafe_allow_html=True)
#     except Exception as e:
#         st.error(f"Could not load stats: {e}")

# # --- Query Processing ---
# # --- CHANGE 3: Read from st.session_state.query_input ---
# if st.button("🚀 Get AI Insights", type="primary"):
#     if not st.session_state.query_input.strip():
#         st.warning("⚠️ Please enter a question.")
#     else:
#         # Use the query from the session state
#         process_query = st.session_state.query_input
        
#         timestamp = datetime.now().strftime("%H:%M:%S")
#         st.session_state.chat_history.append({
#             'timestamp': timestamp,
#             'query': process_query,
#             'results': None,
#             'summary': None
#         })
        
#         with st.spinner("🔍 Searching through customer reviews..."):
#             try:
#                 results = query_reviews(process_query, top_k=top_k)
                
#                 if not results:
#                     st.error("❌ No relevant reviews found. Try a different query.")
#                 else:
#                     with st.spinner("🤖 Generating AI insights..."):
#                         summary = generate_ai_summary(process_query, results)
                        
#                         st.session_state.chat_history[-1]['results'] = results
#                         st.session_state.chat_history[-1]['summary'] = summary
            
#             except Exception as e:
#                 st.error(f"An error occurred while processing your request: {e}")
#                 st.session_state.chat_history.pop()

# # --- Display Results ---
# if st.session_state.chat_history:
#     latest_chat = st.session_state.chat_history[-1]
    
#     if latest_chat['results'] and latest_chat['summary']:
#         st.markdown(f"""
#         <div class="ai-response">
#             <h3>🤖 AI Analysis</h3>
#             <p style="font-size: 1.1rem; line-height: 1.6;">{latest_chat['summary']}</p>
#         </div>
#         """, unsafe_allow_html=True)
        
#         tab1, tab2, tab3 = st.tabs(["📄 Source Reviews", "📈 Analysis", "💬 Chat History"])
        
#         with tab1:
#             st.subheader("📄 Source Reviews Used for Analysis")
#             for i, review in enumerate(latest_chat['results'], 1):
#                 rating_stars = "⭐" * int(float(review.get('rating', 0))) if str(review.get('rating', 'N/A')).replace('.', '', 1).isdigit() else ""
#                 sentiment_color = "#28a745" if review['sentiment'] > 0.1 else "#dc3545" if review['sentiment'] < -0.1 else "#ffc107"
#                 st.markdown(f"""
#                 <div class="review-card">
#                     <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
#                         <strong>Review #{i}</strong>
#                         <div>
#                             <span style="color: {sentiment_color}; font-weight: bold;">Sentiment: {review['sentiment']:.2f}</span>
#                             {f"<span style='margin-left: 1rem;'>{rating_stars}</span>" if rating_stars else ""}
#                         </div>
#                     </div>
#                     <p style="margin: 0; color: #444;">{review['text'][:500]}{"..." if len(review['text']) > 500 else ""}</p>
#                 </div>
#                 """, unsafe_allow_html=True)
        
#         with tab2:
#             # ... (rest of the tab2 code is fine)
#             if latest_chat['results']:
#                 sentiments = [r['sentiment'] for r in latest_chat['results']]
#                 ratings = [float(r['rating']) for r in latest_chat['results'] if str(r.get('rating', 'N/A')).replace('.', '', 1).isdigit()]
                
#                 col1, col2 = st.columns(2)
                
#                 with col1:
#                     if sentiments:
#                         fig_sentiment = px.histogram(x=sentiments, nbins=10, title="Sentiment Distribution", labels={'x': 'Sentiment Score', 'y': 'Count'}, color_discrete_sequence=['#667eea'])
#                         fig_sentiment.update_layout(showlegend=False)
#                         st.plotly_chart(fig_sentiment, use_container_width=True)
                
#                 with col2:
#                     if ratings:
#                         fig_rating = px.bar(x=list(range(1, 6)), y=[ratings.count(i) for i in range(1, 6)], title="Rating Distribution", labels={'x': 'Rating', 'y': 'Count'}, color_discrete_sequence=['#764ba2'])
#                         st.plotly_chart(fig_rating, use_container_width=True)
        
#         with tab3:
#             st.subheader("💬 Query History")
#             for i, chat in enumerate(reversed(st.session_state.chat_history), 1):
#                 with st.expander(f"{chat['timestamp']} - {chat['query'][:50]}..."):
#                     st.write(f"**Query:** {chat['query']}")
#                     if chat['summary']:
#                         st.write(f"**AI Response:** {chat['summary']}")
#                     if chat['results']:
#                         st.write(f"**Reviews Analyzed:** {len(chat['results'])}")

# # --- Footer ---
# st.markdown("---")
# st.markdown("""
# <div style="text-align: center; color: #666; padding: 1rem;">
#     <p>🚀 Built with Streamlit • Powered by RAG + Gemini AI • 
#     <a href="https://github.com/Anshulkavi/Customer-Feedback-Analysis-and-AI-insights" style="color: #667eea;">View on GitHub</a></p>
# </div>
# """, unsafe_allow_html=True)

# # --- Error Handling & Initialization ---
# # This part is fine and doesn't need changes for this bug
# try:
#     if 'rag_initialized' not in st.session_state:
#         with st.spinner("🔄 Initializing AI models... This may take a moment."):
#             get_rag_engine()
#             st.session_state.rag_initialized = True
#             st.success("✅ AI system ready!")
#             time.sleep(1)
#             st.rerun()
# except Exception as e:
#     st.error(f"""
#     ❌ **System Initialization Failed**
    
#     Error: {str(e)}
    
#     **Troubleshooting Steps:**
#     1. Check your `.env` file for correct API keys and DB credentials.
#     2. Ensure your PostgreSQL database is running.
#     3. Make sure you have run the `src/etl_pipeline.py` script first.
#     """)
#     if st.button("🔄 Retry Initialization"):
#         st.rerun()

import streamlit as st
import sys
import os
import pandas as pd
import plotly.express as px
from datetime import datetime
import time

# Add src folder for rag_engine
# Ensure this path is correct for your project structure
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.rag_engine import query_reviews, generate_ai_summary, get_rag_engine
except ImportError as e:
    st.error(f"Failed to import RAG engine: {e}")
    st.stop()

# --- NEW HELPER FUNCTION to translate scores ---
def get_sentiment_descriptor(score):
    """Translates a sentiment score into a descriptive string and emoji."""
    if score > 0.6:
        return "Very Positive", "😁"
    elif score > 0.1:
        return "Positive", "😊"
    elif score < -0.6:
        return "Very Negative", "😠"
    elif score < -0.1:
        return "Negative", "😞"
    else:
        return "Neutral", "😐"
# -----------------------------------------------

# --- Page Configuration ---
st.set_page_config(
    page_title="🤖 AI Customer Feedback Analytics",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Modern Look ---
st.markdown("""
<style>
    /* (Your existing CSS is perfect, no changes needed here) */
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header { text-align: center; color: #666; font-size: 1.2rem; margin-bottom: 2rem; }
    .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; color: white; text-align: center; margin: 0.5rem 0; }
    .review-card { background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #667eea; margin: 0.5rem 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .ai-response { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); padding: 1.5rem; border-radius: 10px; color: white; margin: 1rem 0; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
    .stButton button { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; padding: 0.5rem 2rem; border-radius: 25px; font-weight: bold; transition: transform 0.3s ease; }
    .stButton button:hover { transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.2); }
</style>
""", unsafe_allow_html=True)

# --- App Header ---
st.markdown('<div class="main-header">🤖 AI Customer Feedback Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Powered by RAG + Gemini AI • Transform Reviews into Insights</div>', unsafe_allow_html=True)

# --- Initialize Session State ---
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'query_input' not in st.session_state:
    st.session_state.query_input = ""

# --- Sidebar ---
with st.sidebar:
    st.header("🎛️ Control Panel")
    st.subheader("⚙️ Settings")
    top_k = st.slider("Number of reviews to analyze", 3, 10, 5)
    st.subheader("💡 Example Queries")
    example_queries = [
        "What are the main complaints about battery life?", "How do customers feel about delivery speed?",
        "Summarize positive feedback about product quality", "What issues do customers mention about customer service?",
        "Are there any complaints about packaging?", "What do customers love most about this product?"
    ]
    for query_text in example_queries:
        if st.button(f"💬 {query_text}", key=f"example_{query_text[:20]}"):
            st.session_state.query_input = query_text

# --- Main Interface ---
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("🔍 Ask About Customer Feedback")
    query = st.text_input(
        "Enter your question:", key="query_input",
        placeholder="e.g., What are customers saying about the product quality?",
        help="Ask natural language questions about customer reviews"
    )
with col2:
    st.subheader("📊 Quick Stats")
    try:
        engine = get_rag_engine()
        if hasattr(engine, 'df') and engine.df is not None:
            total_reviews = len(engine.df)
            avg_rating = engine.df.get('rating', pd.Series([0])).mean()
            avg_sentiment = engine.df.get('sentiment_score', pd.Series([0])).mean()
            
            st.markdown(f'<div class="metric-card"><h3>{total_reviews:,}</h3><p>Total Reviews Analyzed</p></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-card"><h3>{avg_rating:.1f}/5</h3><p>Average Rating</p></div>', unsafe_allow_html=True)
            
            # --- REFINEMENT 1: Emoji is now included in the metric card ---
            sentiment_descriptor, sentiment_emoji = get_sentiment_descriptor(avg_sentiment)
            st.markdown(f'<div class="metric-card"><h3>{sentiment_emoji} {avg_sentiment:.2f}</h3><p>Average Sentiment ({sentiment_descriptor})</p></div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Could not load stats: {e}")

# --- Query Processing ---
if st.button("🚀 Get AI Insights", type="primary"):
    if not st.session_state.query_input.strip():
        st.warning("⚠️ Please enter a question.")
    else:
        process_query = st.session_state.query_input
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.chat_history.append({'timestamp': timestamp, 'query': process_query, 'results': None, 'summary': None})
        
        with st.spinner("🔍 Searching through customer reviews..."):
            try:
                results = query_reviews(process_query, top_k=top_k)
                if not results:
                    st.error("❌ No relevant reviews found. Try a different query.")
                else:
                    with st.spinner("🤖 Generating AI insights..."):
                        summary = generate_ai_summary(process_query, results)
                        st.session_state.chat_history[-1]['results'] = results
                        st.session_state.chat_history[-1]['summary'] = summary
            except Exception as e:
                st.error(f"An error occurred while processing your request: {e}")
                st.session_state.chat_history.pop()

# --- Display Results ---
if st.session_state.chat_history:
    latest_chat = st.session_state.chat_history[-1]
    
    if latest_chat['results'] and latest_chat['summary']:
        st.markdown(f"""<div class="ai-response"><h3>🤖 AI Analysis</h3><p style="font-size: 1.1rem; line-height: 1.6;">{latest_chat['summary']}</p></div>""", unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["📄 Source Reviews", "📈 Analysis", "💬 Chat History"])
        
        with tab1:
            st.subheader("📄 Source Reviews Used for Analysis")
            for i, review in enumerate(latest_chat['results'], 1):
                rating_stars = "⭐" * int(float(review.get('rating', 0))) if str(review.get('rating', 'N/A')).replace('.', '', 1).isdigit() else ""
                
                # --- REFINEMENT 2: Using the new descriptor function in review cards ---
                sentiment_descriptor, sentiment_emoji = get_sentiment_descriptor(review['sentiment'])
                sentiment_color = "#28a745" if review['sentiment'] > 0.1 else "#dc3545" if review['sentiment'] < -0.1 else "#ffc107"
                
                st.markdown(f"""
                <div class="review-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <strong>Review #{i}</strong>
                        <div>
                            <span style="color: {sentiment_color}; font-weight: bold;">
                                {sentiment_descriptor} {sentiment_emoji}
                            </span>
                            {f"<span style='margin-left: 1rem;'>{rating_stars}</span>" if rating_stars else ""}
                        </div>
                    </div>
                    <p style="margin: 0; color: #444;">{review['text'][:500]}{"..." if len(review['text']) > 500 else ""}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with tab2:
            if latest_chat['results']:
                sentiments = [r['sentiment'] for r in latest_chat['results']]
                ratings = [float(r['rating']) for r in latest_chat['results'] if str(r.get('rating', 'N/A')).replace('.', '', 1).isdigit()]
                col1, col2 = st.columns(2)
                with col1:
                    if sentiments:
                        fig_sentiment = px.histogram(x=sentiments, nbins=10, title="Sentiment Distribution", labels={'x': 'Sentiment Score', 'y': 'Count'}, color_discrete_sequence=['#667eea'])
                        fig_sentiment.update_layout(showlegend=False)
                        st.plotly_chart(fig_sentiment, use_container_width=True)
                with col2:
                    if ratings:
                        fig_rating = px.bar(x=list(range(1, 6)), y=[ratings.count(i) for i in range(1, 6)], title="Rating Distribution", labels={'x': 'Rating', 'y': 'Count'}, color_discrete_sequence=['#764ba2'])
                        st.plotly_chart(fig_rating, use_container_width=True)
        
        with tab3:
            st.subheader("💬 Query History")
            for i, chat in enumerate(reversed(st.session_state.chat_history), 1):
                with st.expander(f"{chat['timestamp']} - {chat['query'][:50]}..."):
                    st.write(f"**Query:** {chat['query']}")
                    if chat['summary']: st.write(f"**AI Response:** {chat['summary']}")
                    if chat['results']: st.write(f"**Reviews Analyzed:** {len(chat['results'])}")

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🚀 Built with Streamlit • Powered by RAG + Gemini AI • 
    <a href="https://github.com/Anshulkavi/Customer-Feedback-Q-A-RAG-LLM" style="color: #667eea;">View on GitHub</a></p>
</div>
""", unsafe_allow_html=True)

# --- Error Handling & Initialization ---
try:
    if 'rag_initialized' not in st.session_state:
        with st.spinner("🔄 Initializing AI models... This may take a moment."):
            get_rag_engine()
            st.session_state.rag_initialized = True
            st.success("✅ AI system ready!")
            time.sleep(1)
            st.rerun()
except Exception as e:
    st.error(f"""
    ❌ **System Initialization Failed**
    
    Error: {str(e)}
    
    **Troubleshooting Steps:**
    1. Check your `.env` file for correct API keys and DB credentials.
    2. Ensure your PostgreSQL database is running.
    3. Make sure you have run the `src/etl_pipeline.py` script first.
    """)
    if st.button("🔄 Retry Initialization"):
        st.rerun()