# Semantic Search Engine for Academic Documents

A lightweight, interpretable, end-to-end semantic search engine for academic PDFs (research papers, lecture notes) built with **TF-IDF (Term Frequency-Inverse Document Frequency)** and **Cosine Similarity**, featuring a clean **Streamlit** user interface.

---

## 1. Project Pipeline Overview

```
PDF Document(s)
       │
       ▼
 [1] loader.py              -> Extract text per page, keep track of file name and page number
       │
       ▼
 [2] chunker.py             -> Split into 150-word overlapping passages
       │
       ▼
 [3] preprocess.py          -> Lowercase, tokenize, remove stopwords, lemmatize
       │
       ├────────────────────────────────────────┐
       ▼ (Passages)                             ▼ (User Query)
 [4] TF-IDF Matrix (search_engine.py)    [5] Preprocess Query & Vectorize (transform)
       │                                        │
       └────────────────────┬───────────────────┘
                            │
                            ▼
 [6] Cosine Similarity   -> Compute dot product / (norm1 * norm2)
                            │
                            ▼
 [7] search.py / app.py  -> Rank descending, filter thresholds, highlight keywords
```

---

## 2. Directory Structure

```
semantic-pdf-search/
├── requirements.txt         # Project dependencies
├── stage1_setup.py          # Stage 1: Environment verification & architecture
├── loader.py                # Stage 2: PDF loading & page-level text extraction
├── chunker.py               # Stage 3: Sliding window passage chunking
├── preprocess.py            # Stage 4: Text normalization, tokenization, lemmatization
├── tfidf_scratch.py         # Stage 5: TF-IDF matrix built from scratch (NumPy)
├── cosine_scratch.py        # Stage 6: Cosine similarity built from scratch (NumPy)
├── search_engine.py         # Stage 7: Production engine with scikit-learn
├── search.py                # Stage 8: Query processing, edge cases & CLI
├── app.py                   # Stage 9: Minimalist Streamlit web interface
├── evaluate.py              # Stage 10: Evaluation (Precision@k, Recall@k, MRR)
├── extensions_analysis.py   # Stage 11: Limitations, BM25, and RAG comparison
└── sample_docs/             # Sample PDF documents for demonstration
```

---

## 3. Quick Start & Execution

1. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run any stage demo independently**:
   ```bash
   python3 stage1_setup.py
   python3 loader.py
   python3 chunker.py
   python3 preprocess.py
   python3 tfidf_scratch.py
   python3 cosine_scratch.py
   python3 search_engine.py
   python3 search.py --query "gradient descent"
   python3 evaluate.py
   python3 extensions_analysis.py
   ```

3. **Launch the Streamlit Web Application**:
   ```bash
   streamlit run app.py
   ```
