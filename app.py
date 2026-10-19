"""
Stage 9: Minimalist Streamlit Interface (app.py)
------------------------------------------------
Minimalist, clean, single-page UI for academic PDF semantic search.
Uses Streamlit session state to avoid re-indexing documents on every interaction.
"""

import streamlit as st
from loader import load_multiple_pdfs
from chunker import chunk_pages
from search_engine import SearchEngine
from search import execute_search

# 1. Page Configuration (Centered, Neutral, No Sidebar)
st.set_page_config(
    page_title="Semantic PDF Search",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 2. Header and Subtitle
st.title("Semantic Search Engine for Academic Documents")
st.caption("Retrieve relevant passages from research papers and lecture notes using TF-IDF and Cosine Similarity.")
st.divider()

# 3. Session State Initialization
if "engine" not in st.session_state:
    st.session_state["engine"] = None
if "indexed_summary" not in st.session_state:
    st.session_state["indexed_summary"] = ""
if "warnings" not in st.session_state:
    st.session_state["warnings"] = []

# 4. Document Upload & Indexing Section
uploaded_files = st.file_uploader(
    "Upload Academic Documents (PDF format)",
    type=["pdf"],
    accept_multiple_files=True,
    help="Upload one or multiple lecture notes or research papers."
)

col1, col2 = st.columns([3, 1])
with col1:
    index_button = st.button("Index documents", type="primary", use_container_width=True)
with col2:
    top_k = st.selectbox("Top results", options=[3, 5, 10], index=1)

if index_button:
    if not uploaded_files:
        st.warning("Please select at least one PDF file before indexing.")
    else:
        with st.spinner("Extracting text and building TF-IDF index..."):
            # Load pages and extract text
            pages, warnings = load_multiple_pdfs(uploaded_files)
            st.session_state["warnings"] = warnings

            if not pages:
                st.error("No extractable text found in uploaded document(s). They may be scanned images.")
                st.session_state["engine"] = None
                st.session_state["indexed_summary"] = ""
            else:
                # Chunk text into 150-word passages with 30-word overlap
                chunks = chunk_pages(pages, chunk_size=150, overlap=30)

                # Build TF-IDF Search Engine
                engine = SearchEngine()
                engine.build_index(chunks)

                st.session_state["engine"] = engine
                num_files = len(uploaded_files)
                num_chunks = len(chunks)
                st.session_state["indexed_summary"] = f"{num_files} file(s), {num_chunks} passage(s) indexed."

# Display indexing status & any parser warnings
if st.session_state["indexed_summary"]:
    st.success(st.session_state["indexed_summary"])

if st.session_state["warnings"]:
    for warn_msg in st.session_state["warnings"]:
        st.warning(warn_msg)

st.divider()

# 5. Search Query Section
query = st.text_input("Enter search query", placeholder="e.g. gradient descent optimization, attention mechanism...")

if query:
    if st.session_state["engine"] is None:
        st.info("Please upload and index documents before searching.")
    else:
        results, status_msg = execute_search(
            engine=st.session_state["engine"],
            query=query,
            top_k=top_k,
            min_score=0.01
        )

        st.caption(status_msg)

        if not results:
            st.info("No matching passages found. Try different keywords or check spelling.")
        else:
            for item in results:
                title = f"Rank {item['rank']} | Score: {item['score']:.4f} | {item['file_name']} (Page {item['page_number']})"
                with st.expander(title, expanded=(item["rank"] == 1)):
                    st.markdown(item["highlighted_text"])
                    st.caption(f"Source: {item['file_name']}, Page {item['page_number']} | Chunk ID: {item['chunk_id']}")
