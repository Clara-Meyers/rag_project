# RAG Tutorial Project

Welcome to your first RAG (Retrieval-Augmented Generation) project! This tutorial will teach you how to build a simple question-answering system that uses real documents to provide accurate responses.

## What is RAG?

RAG combines two powerful concepts:
1. **Retrieval**: Finding relevant documents from a knowledge base
2. **Generation**: Using an LLM to generate answers based on those documents

Instead of the AI making things up, it answers questions using YOUR data!

```
┌─────────────────────────────────────────────────────────────────┐
│                         RAG Flow                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   User Question                                                 │
│        │                                                        │
│        ▼                                                        │
│   ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐   │
│   │  Embedding  │───▶│ Vector Store │───▶│ Similar Docs    │   │
│   │  (Question) │    │   Search     │    │ (Top 3)         │   │
│   └─────────────┘    └──────────────┘    └────────┬────────┘   │
│                                                    │            │
│                                                    ▼            │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │              LLM (Gemini)                                │  │
│   │  "Answer this question using ONLY the provided context" │  │
│   └─────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│                         Answer                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
rag_sample_project/
├── README.md                 # You are here!
├── requirements.txt          # Python dependencies
├── env.example               # Template for API keys
├── main.py                   # Entry point (implement last)
├── data/
│   └── documents/
│       ├── company_faq.txt   # TechCorp FAQ
│       ├── product_info.txt  # Product descriptions
│       └── policies.txt      # Company policies
└── src/
    ├── __init__.py
    ├── llm_example.py        # ✅ Working example to start with
    └── rag/
        ├── __init__.py
        ├── data_loader.py    # 📝 TODO: Load documents
        ├── embeddings.py     # 📝 TODO: Generate embeddings
        ├── vector_store.py   # 📝 TODO: Store & search vectors
        └── pipeline.py       # 📝 TODO: Combine everything
```

---

## Getting Started

### Step 1: Set Up Your Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate it
# On Linux/Mac:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Get Your Gemini API Key
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Click "Create API Key"
3. Copy your API key

### Step 3: Configure Your API Key

```bash
# Copy the example file
cp env.example .env

# Edit .env and add your API key
# GEMINI_API_KEY=your_actual_api_key_here
```

### Step 4: Run the LLM Example

```bash
python -m src.llm_example


```

This demonstrates basic Gemini API usage. Make sure this works before continuing!

---

## Your TODO List

Complete these tasks in order to build your RAG system:

### Task 1: Data Loader (`src/rag/data_loader.py`)

**Goal**: Load text files and split them into chunks.

**What to implement**:
```python
def load_documents(directory: str) -> list[str]:
    """
    Load all .txt files from a directory.
    Returns a list of document contents.
    """
    # Hint: Use os.listdir() or pathlib.Path.glob()
    pass

def split_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into smaller chunks with optional overlap.
    Returns a list of text chunks.
    """
    # Hint: Split by paragraphs first, then by size if needed
    pass
```

**Test your implementation**:
```python
# In a Python shell or test script:
from src.rag.data_loader import load_documents, split_into_chunks

docs = load_documents("data/documents")
print(f"Loaded {len(docs)} documents")

chunks = split_into_chunks(docs[0], chunk_size=300)
print(f"Created {len(chunks)} chunks from first document")
print(f"First chunk: {chunks[0][:100]}...")
```

**Concepts to learn**:
- File I/O in Python
- String manipulation
- Working with paths

---

### Task 2: Embeddings (`src/rag/embeddings.py`)

**Goal**: Convert text into numerical vectors using Gemini.

**What to implement**:
```python
def get_embedding(text: str, task_type: str = "retrieval_document") -> list[float]:
    """
    Generate an embedding vector for a text.
    
    task_type: 
        - "retrieval_document" for documents you're storing
        - "retrieval_query" for user questions
    """
    # Hint: Use genai.embed_content()
    pass
```

**Test your implementation**:
```python
from src.rag.embeddings import get_embedding

# Get embedding for a sample text
embedding = get_embedding("TechCorp sells software products")
print(f"Embedding length: {len(embedding)}")  # Should be 768
print(f"First 5 values: {embedding[:5]}")
```

**Concepts to learn**:
- What embeddings are and why they matter
- Using external APIs
- Vector representations of text

---

### Task 3: Vector Store (`src/rag/vector_store.py`)

**Goal**: Store embeddings and find similar documents.

**What to implement**:
```python
import numpy as np

def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    # Hint: dot(A,B) / (norm(A) * norm(B))
    pass

class SimpleVectorStore:
    def __init__(self):
        self.texts = []
        self.embeddings = []
    
    def add(self, text: str, embedding: list[float]):
        """Add a text and its embedding to the store."""
        pass
    
    def search(self, query_embedding: list[float], top_k: int = 3) -> list[str]:
        """Find the top_k most similar texts."""
        # Hint: Calculate similarity with all stored embeddings
        # Return the texts with highest similarity scores
        pass
```

**Test your implementation**:
```python
from src.rag.vector_store import SimpleVectorStore, cosine_similarity
from src.rag.embeddings import get_embedding

# Create store and add some texts
store = SimpleVectorStore()
texts = [
    "DataFlow Pro is a data analytics platform",
    "SecureVault provides cloud storage",
    "TaskMaster is for project management"
]

for text in texts:
    emb = get_embedding(text)
    store.add(text, emb)

# Search for similar texts
query = "I need help analyzing data"
query_emb = get_embedding(query, task_type="retrieval_query")
results = store.search(query_emb, top_k=2)
print("Most relevant texts:", results)
```

**Concepts to learn**:
- Cosine similarity
- Basic linear algebra with numpy
- Building simple data structures

---

### Task 4: RAG Pipeline (`src/rag/pipeline.py`)

**Goal**: Connect all components into a working system.

**What to implement**:
```python
class RAGPipeline:
    def __init__(self, vector_store, embedding_func):
        self.vector_store = vector_store
        self.get_embedding = embedding_func
    
    def query(self, question: str, top_k: int = 3) -> str:
        """
        Answer a question using retrieved context.
        
        Steps:
        1. Embed the question
        2. Search for relevant documents
        3. Build prompt with context
        4. Generate answer
        """
        pass
    
    def build_prompt(self, contexts: list[str], question: str) -> str:
        """Create a prompt with context and question."""
        pass
```

**Test your implementation**:
```python
from src.rag.pipeline import RAGPipeline

# Assuming you've set up the vector store from Task 3
rag = RAGPipeline(store, get_embedding)

# Ask questions!
answer = rag.query("What is DataFlow Pro?")
print(answer)

answer = rag.query("How can I contact support?")
print(answer)
```

**Concepts to learn**:
- Combining multiple modules
- Prompt engineering
- System architecture

---

### Task 5: Main Application (`main.py`)

**Goal**: Create an interactive Q&A application.

**What to implement**:
```python
def main():
    # 1. Load all documents
    # 2. Split into chunks
    # 3. Generate embeddings and store them
    # 4. Create RAG pipeline
    # 5. Interactive question loop
    
    while True:
        question = input("\nAsk a question (or 'quit'): ")
        if question.lower() == 'quit':
            break
        answer = rag.query(question)
        print(f"\nAnswer: {answer}")
```

---

### Task 6 (Bonus): Improvements

Once the basic system works, try these enhancements:

1. **Better chunking**: Experiment with different chunk sizes and overlap
2. **Caching**: Save embeddings to disk so you don't regenerate them
3. **Better prompts**: Improve the prompt template for better answers
4. **Source citation**: Show which documents the answer came from
5. **Error handling**: Handle API errors gracefully

---

## Testing Questions

Once your RAG system is complete, try these questions:

**Questions that SHOULD be answered from the documents**:
- "What products does TechCorp offer?"
- "How much does DataFlow Pro cost?"
- "What is the refund policy?"
- "How can I contact customer support?"
- "Is my data encrypted?"

**Questions that should NOT be answerable** (test the "I don't know" response):
- "What is the weather like today?"
- "Who is the CEO of Apple?"
- "What programming languages does TechCorp support?"

---

## Helpful Resources

- [Gemini API Documentation](https://ai.google.dev/docs)
- [Python File I/O Tutorial](https://docs.python.org/3/tutorial/inputoutput.html#reading-and-writing-files)
- [NumPy Documentation](https://numpy.org/doc/stable/)
- [Understanding Embeddings](https://www.pinecone.io/learn/vector-embeddings/)

---

## Need Help?

If you get stuck:
1. Read the docstrings in each module - they have hints!
2. Check the `src/llm_example.py` for API usage examples
3. Print intermediate values to debug
4. Ask your tutor for guidance

Good luck and have fun learning! 🚀

---

# Part 2: Streamlit Web Interface

Congratulations on completing the RAG backend! 🎉 

Now let's add a beautiful web interface using **Streamlit** - a Python library that makes it easy to create web apps.

## New Project Structure

```
rag_sample_project/
├── ... (existing files unchanged)
├── app.py                      # 🆕 Streamlit entry point
└── pages/
    ├── 1_Data_Explorer.py      # 🆕 Page 1: Visualize documents
    └── 2_Chat.py               # 🆕 Page 2: Chat interface
```

## Getting Started with Streamlit

### Step 1: Install Streamlit

```bash
# Make sure your virtual environment is activated
source venv/bin/activate

# Install the new dependency
pip install -r requirements.txt
# Or just: pip install streamlit
```

### Step 2: Run the App

```bash
streamlit run app.py
```

This will open your browser at `http://localhost:8501` with a beautiful web interface!

### Step 3: Explore the Home Page

The home page (`app.py`) is already complete. It explains the app and provides navigation.

---

## Streamlit TODO List

Now it's your turn to build the interactive pages!

### Task 7: Data Explorer (`pages/1_Data_Explorer.py`)

**Goal**: Create a page to visualize the raw documents and chunks.

**What to implement**:
- Display a list of available documents
- Show the content of a selected document
- Demonstrate how documents are split into chunks
- Show statistics (word count, chunk count)

**Streamlit components to use**:
```python
import streamlit as st

# Page title
st.title("📊 Data Explorer")

# Selectbox for choosing options
selected = st.selectbox("Choose document", ["doc1.txt", "doc2.txt"])

# Display text in a read-only area
st.text_area("Content", value="Hello world", height=300, disabled=True)

# Show metrics
st.metric(label="Chunks", value=42)

# Columns for side-by-side layout
col1, col2 = st.columns(2)
with col1:
    st.write("Left side")
with col2:
    st.write("Right side")

# Expandable section
with st.expander("Click to see details"):
    st.write("Hidden content")
```

**Your implementation steps**:
1. Import your `load_documents` and `split_into_chunks` functions
2. Load documents and list them in a selectbox
3. Display the selected document content
4. Show the chunks in expandable sections
5. Add statistics like word count and chunk count

**Test it**: Run `streamlit run app.py` and navigate to "Data Explorer"

---

### Task 8: Chat Interface (`pages/2_Chat.py`)

**Goal**: Create a chat-like interface for the RAG assistant.

**Key Concepts**:

1. **Session State** - Remember data between interactions:
```python
# Initialize chat history (only runs once)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Add a message
st.session_state.messages.append({"role": "user", "content": "Hello"})
```

2. **Chat Components**:
```python
# Display a message bubble
with st.chat_message("user"):
    st.write("Hello!")

with st.chat_message("assistant"):
    st.write("Hi there!")

# Chat input at bottom of page
prompt = st.chat_input("Ask a question...")
```

3. **Caching** - Avoid reloading the RAG pipeline:
```python
@st.cache_resource
def initialize_rag():
    """This only runs once, then the result is cached."""
    # Load documents, create embeddings, build pipeline...
    return rag_pipeline
```

**Your implementation steps**:
1. Create a cached function to initialize the RAG pipeline (copy from `main.py`)
2. Set up session state for chat history
3. Display previous messages in chat bubbles
4. Handle new user input with `st.chat_input()`
5. Generate and display the AI response
6. (Bonus) Add a "Clear Chat" button

**Example chat pattern**:
```python
# Initialize
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Handle new input
if prompt := st.chat_input("Ask something"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generate response using your RAG pipeline
    response = rag.query(prompt)
    
    # Add assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Rerun to display new messages
    st.rerun()
```

**Test it**: Run `streamlit run app.py` and navigate to "Chat"

---

### Task 9 (Bonus): Enhancements

Once the basic pages work, try these improvements:

1. **Data Explorer**:
   - Add a slider to adjust chunk size dynamically
   - Highlight search terms in documents
   - Show a bar chart of chunk sizes

2. **Chat Interface**:
   - Show the retrieved context chunks alongside the answer
   - Display response time
   - Add example question buttons
   - Style the chat with custom CSS

---

## Streamlit Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Streamlit Cheat Sheet](https://docs.streamlit.io/library/cheatsheet)
- [Session State Guide](https://docs.streamlit.io/library/api-reference/session-state)
- [Chat Elements](https://docs.streamlit.io/library/api-reference/chat)

---

## Quick Reference: Running the App

```bash
# Terminal version (original)
python main.py

# Web interface (new!)
streamlit run app.py
```

Both use the same backend code - you're just adding a visual layer!

