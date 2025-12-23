"""
TechCorp RAG Assistant - Streamlit Web Interface
=================================================

This is the main entry point for the Streamlit web application.
Run with: streamlit run app.py

The app has multiple pages (see the sidebar):
1. Home (this page) - Welcome and instructions
2. Data Explorer - View and explore the raw documents
3. Chat - Ask questions using the RAG pipeline
"""

import streamlit as st

# Page configuration - must be the first Streamlit command
st.set_page_config(
    page_title="TechCorp RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Main content
st.markdown(
    '<p class="main-header">🤖 TechCorp RAG Assistant</p>', unsafe_allow_html=True
)
st.markdown(
    '<p class="sub-header">Your AI-powered knowledge base for TechCorp documents</p>',
    unsafe_allow_html=True,
)

st.divider()

# Welcome message
st.markdown("""
## Welcome! 👋

This web application demonstrates a **RAG (Retrieval-Augmented Generation)** system built with:
- 🐍 Python backend
- 🧠 Google Gemini API for LLM and embeddings  
- 🎈 Streamlit for the web interface

### How RAG Works

```
📝 Your Question
      ↓
🔍 Find relevant documents (semantic search)
      ↓
📚 Retrieved context (most similar chunks)
      ↓
🤖 LLM generates answer using context
      ↓
💬 Your Answer
```
""")

st.divider()

# Features section
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 📊 Data Explorer
    
    Explore the raw documents that power this assistant:
    - View original document contents
    - See how documents are split into chunks
    - Understand the data preprocessing step
    
    **Navigate to:** `1_Data_Explorer` in the sidebar
    """)

with col2:
    st.markdown("""
    ### 💬 Chat Interface
    
    Ask questions about TechCorp:
    - Interactive chat experience
    - See which documents were retrieved
    - Get AI-powered answers
    
    **Navigate to:** `2_Chat` in the sidebar
    """)

st.divider()

# Getting started
st.markdown("""
## 🚀 Getting Started

1. **Explore the Data** - Start with the Data Explorer to understand your knowledge base
2. **Ask Questions** - Go to the Chat page and try these example questions:
   - "What products does TechCorp offer?"
   - "What is the refund policy?"
   - "How can I contact customer support?"

## 📚 Learning Resources

This app was built following the RAG Tutorial. The key components are:
- `src/rag/data_loader.py` - Loads and chunks documents
- `src/rag/embeddings.py` - Converts text to vectors
- `src/rag/vector_store.py` - Stores and searches vectors
- `src/rag/pipeline.py` - Orchestrates the RAG flow
""")

# Sidebar info
st.sidebar.success("Select a page above to get started!")
st.sidebar.markdown("---")
st.sidebar.markdown("""
### About

This is a tutorial project for learning:
- Python programming
- LLM integration (Gemini)
- RAG systems
- Streamlit web apps
""")
