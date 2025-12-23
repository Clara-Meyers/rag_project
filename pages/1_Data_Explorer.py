"""
Data Explorer Page
==================

This page displays the raw documents and shows how they are chunked.
"""

import streamlit as st
from pathlib import Path
from src.rag.data_loader import load_documents, split_into_chunks

st.title("📊 Data Explorer")
st.markdown("*Explore the documents that power the RAG assistant*")

st.divider()

# Get document files
doc_dir = Path("data/documents")
doc_files = list(doc_dir.glob("*.txt"))
doc_names = [f.name for f in doc_files]

# Sidebar controls
st.sidebar.header("⚙️ Settings")
selected_doc = st.sidebar.selectbox("Select Document", doc_names)
chunk_size = st.sidebar.slider("Chunk Size", 200, 1000, 500, 50)
overlap = st.sidebar.slider("Overlap", 0, 100, 50, 10)

# Read selected document
doc_path = doc_dir / selected_doc
with open(doc_path, "r", encoding="utf-8") as f:
    content = f.read()

# Create chunks
chunks = split_into_chunks(content, chunk_size=chunk_size, overlap=overlap)

# Statistics row
st.subheader("📈 Statistics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Documents", len(doc_names))
col2.metric("Words", len(content.split()))
col3.metric("Characters", len(content))
col4.metric("Chunks", len(chunks))

st.divider()

# Two columns layout
left_col, right_col = st.columns(2)

with left_col:
    st.subheader(f"📄 {selected_doc}")
    st.text_area(
        "Original Document Content",
        content,
        height=500,
        disabled=True,
        label_visibility="collapsed",
    )

with right_col:
    st.subheader(f"✂️ Chunks ({len(chunks)} total)")

    for i, chunk in enumerate(chunks):
        with st.expander(f"Chunk {i + 1} — {len(chunk)} characters"):
            st.code(chunk, language=None)

st.divider()

# Educational section
with st.expander("💡 How Chunking Works"):
    st.markdown("""
    ### Why do we chunk documents?
    
    1. **LLM Context Limits**: LLMs can only process a limited amount of text at once
    2. **Better Retrieval**: Smaller chunks allow more precise matching
    3. **Reduced Noise**: Relevant information isn't diluted by unrelated content
    
    ### Chunking Parameters
    
    - **Chunk Size**: Maximum number of characters per chunk
    - **Overlap**: Characters shared between consecutive chunks (helps maintain context)
    
    ### Try it!
    
    Use the sliders in the sidebar to see how different settings affect chunking.
    """)
