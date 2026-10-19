"""
Stage 10: Information Retrieval Evaluation (evaluate.py)
--------------------------------------------------------
Evaluates the search engine using classic Information Retrieval (IR) metrics:
  - Precision@k: (Relevant items in top-k) / k
  - Recall@k: (Relevant items in top-k) / (Total relevant items)
  - Mean Reciprocal Rank (MRR): (1 / |Q|) * sum(1 / rank_of_first_relevant_item)

Ablation Study Variants:
  1. Baseline: Stopwords removed + Lemmatization + Unigrams
  2. Variant A (No Stopword Removal): Stopwords retained
  3. Variant B (Unigram + Bigram): ngram_range=(1, 2)
  4. Variant C (No Lemmatization): Raw tokens without lemmatizer
"""

import re
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from loader import load_multiple_pdfs
from chunker import chunk_pages
from preprocess import STOP_WORDS, clean_and_tokenize


# -------------------------------------------------------------
# 1. Ground Truth Benchmark Dataset (12 Queries)
# -------------------------------------------------------------
BENCHMARK_QUERIES = [
    {
        "query": "gradient descent optimization loss function",
        "target_file": "machine_learning_basics.pdf",
        "target_page": 1
    },
    {
        "query": "supervised learning labeled data algorithms",
        "target_file": "machine_learning_basics.pdf",
        "target_page": 1
    },
    {
        "query": "overfitting regularization lasso ridge",
        "target_file": "machine_learning_basics.pdf",
        "target_page": 2
    },
    {
        "query": "cross validation generalization unseen data",
        "target_file": "machine_learning_basics.pdf",
        "target_page": 2
    },
    {
        "query": "unsupervised clustering pca unlabeled",
        "target_file": "machine_learning_basics.pdf",
        "target_page": 2
    },
    {
        "query": "tf-idf term weighting document frequency",
        "target_file": "nlp_and_transformers.pdf",
        "target_page": 1
    },
    {
        "query": "cosine similarity vector space orientation angle",
        "target_file": "nlp_and_transformers.pdf",
        "target_page": 1
    },
    {
        "query": "lemmatization linguistic base root forms",
        "target_file": "nlp_and_transformers.pdf",
        "target_page": 1
    },
    {
        "query": "transformer self attention mechanism parallel",
        "target_file": "nlp_and_transformers.pdf",
        "target_page": 2
    },
    {
        "query": "bert bidirectional encoder representation contextual",
        "target_file": "nlp_and_transformers.pdf",
        "target_page": 2
    },
    {
        "query": "dense embeddings sentence transformers vector",
        "target_file": "nlp_and_transformers.pdf",
        "target_page": 2
    },
    {
        "query": "retrieval augmented generation rag language models",
        "target_file": "nlp_and_transformers.pdf",
        "target_page": 2
    },
]


# -------------------------------------------------------------
# 2. Configurable Preprocessing for Ablation Testing
# -------------------------------------------------------------
def preprocess_variant(text: str, remove_stopwords: bool = True, apply_lemma: bool = True) -> str:
    """Configurable preprocessor to isolate the effect of each NLP component."""
    if not text:
        return ""
    text_lower = text.lower()
    words = re.findall(r"\b[a-z]{2,}\b", text_lower)

    tokens = []
    from preprocess import LEMMATIZER
    for w in words:
        if remove_stopwords and w in STOP_WORDS:
            continue
        if apply_lemma:
            w = LEMMATIZER.lemmatize(w, pos="v")
            w = LEMMATIZER.lemmatize(w, pos="n")
        tokens.append(w)

    return " ".join(tokens)


# -------------------------------------------------------------
# 3. Metric Calculations: Precision@k, Recall@k, MRR
# -------------------------------------------------------------
def evaluate_pipeline(
    chunks: List[Dict[str, Any]],
    test_set: List[Dict[str, Any]],
    remove_stopwords: bool = True,
    apply_lemma: bool = True,
    ngram_range: Tuple[int, int] = (1, 1),
    top_k: int = 3
) -> Dict[str, float]:
    """
    Indexes chunks under specific preprocessing/ngram settings and calculates
    Precision@k, Recall@k, and MRR across all benchmark queries.
    """
    # 1. Preprocess all chunks under this configuration
    cleaned_corpus = [
        preprocess_variant(c["text"], remove_stopwords=remove_stopwords, apply_lemma=apply_lemma)
        for c in chunks
    ]

    vectorizer = TfidfVectorizer(ngram_range=ngram_range, norm="l2", smooth_idf=True)
    tfidf_matrix = vectorizer.fit_transform(cleaned_corpus)

    precisions = []
    recalls = []
    reciprocal_ranks = []

    for test_item in test_set:
        query_str = test_item["query"]
        target_file = test_item["target_file"]
        target_page = test_item["target_page"]

        cleaned_query = preprocess_variant(query_str, remove_stopwords=remove_stopwords, apply_lemma=apply_lemma)
        if not cleaned_query.strip():
            precisions.append(0.0)
            recalls.append(0.0)
            reciprocal_ranks.append(0.0)
            continue

        q_vec = vectorizer.transform([cleaned_query])
        sim_scores = cosine_similarity(q_vec, tfidf_matrix)[0]
        top_indices = np.argsort(sim_scores)[::-1][:top_k]

        # Total relevant chunks in the whole corpus for this target
        total_relevant = sum(
            1 for c in chunks if c["file_name"] == target_file and c["page_number"] == target_page
        )
        if total_relevant == 0:
            total_relevant = 1

        # Check hits in top-k
        relevant_retrieved = 0
        first_relevant_rank = None

        for rank_idx, doc_idx in enumerate(top_indices, start=1):
            chunk = chunks[doc_idx]
            is_match = (chunk["file_name"] == target_file and chunk["page_number"] == target_page)
            if is_match:
                relevant_retrieved += 1
                if first_relevant_rank is None:
                    first_relevant_rank = rank_idx

        # Calculate metrics for this query
        p_at_k = relevant_retrieved / top_k
        r_at_k = relevant_retrieved / total_relevant
        rr = (1.0 / first_relevant_rank) if first_relevant_rank is not None else 0.0

        precisions.append(p_at_k)
        recalls.append(r_at_k)
        reciprocal_ranks.append(rr)

    return {
        f"Precision@{top_k}": float(np.mean(precisions)),
        f"Recall@{top_k}": float(np.mean(recalls)),
        "MRR": float(np.mean(reciprocal_ranks))
    }


if __name__ == "__main__":
    print("=" * 75)
    print(" DEMO: evaluate.py (IR Metrics & Ablation Study Comparison)")
    print("=" * 75)

    # Load corpus
    pdf_paths = [
        "sample_docs/machine_learning_basics.pdf",
        "sample_docs/nlp_and_transformers.pdf"
    ]
    pages, _ = load_multiple_pdfs(pdf_paths)
    chunks = chunk_pages(pages, chunk_size=40, overlap=10)

    print(f"Dataset: {len(BENCHMARK_QUERIES)} Test Queries evaluated across {len(chunks)} indexed passages.\n")

    # Define experiment variants
    experiments = [
        ("1. Baseline (Stopwords + Lemma + Unigrams)", {"remove_stopwords": True, "apply_lemma": True, "ngram_range": (1, 1)}),
        ("2. No Stopword Removal",                     {"remove_stopwords": False, "apply_lemma": True, "ngram_range": (1, 1)}),
        ("3. Unigrams + Bigrams",                      {"remove_stopwords": True, "apply_lemma": True, "ngram_range": (1, 2)}),
        ("4. No Lemmatization",                        {"remove_stopwords": True, "apply_lemma": False, "ngram_range": (1, 1)}),
    ]

    print(f"{'Variant Configuration':<44} | {'Precision@3':<11} | {'Recall@3':<9} | {'MRR':<6}")
    print("-" * 75)

    for name, params in experiments:
        metrics = evaluate_pipeline(chunks, BENCHMARK_QUERIES, top_k=3, **params)
        print(f"{name:<44} | {metrics['Precision@3']:<11.4f} | {metrics['Recall@3']:<9.4f} | {metrics['MRR']:<6.4f}")

    print("=" * 75)
    print("\nInterpretation of Evaluation Results:")
    print(" • MRR (Mean Reciprocal Rank) near 1.0 indicates the primary relevant passage is ranked at #1.")
    print(" • Removing stop words prevents common functional words from polluting cosine similarity scores.")
    print(" • Lemmatization enhances recall by bridging verb/noun forms ('updates' -> 'update').")
    print(" • Bigrams help preserve phrase context (e.g. 'gradient descent', 'loss function').")
    print("=" * 75)
