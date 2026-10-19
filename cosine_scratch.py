"""
Stage 6: Cosine Similarity From Scratch (cosine_scratch.py)
-----------------------------------------------------------
Calculates the geometric cosine angle between a query vector and document vectors.

The Math:
                    u · v               sum(u_i * v_i)
Cosine(u, v) = ─────────────── = ───────────────────────────
               ||u||2 * ||v||2    sqrt(sum(u_i^2)) * sqrt(sum(v_i^2))

Why Cosine Similarity over Euclidean Distance:
- Euclidean distance penalizes long documents simply because they have more words.
- Cosine similarity normalizes vector lengths to unit length, measuring only the
  ORIENTATION (topical similarity), not document length.
- Scores range from 0.0 (orthogonal, completely unrelated) to 1.0 (identical direction).
"""

from typing import List, Tuple
import numpy as np
from tfidf_scratch import build_vocabulary, compute_tfidf


def vector_norm(v: np.ndarray) -> float:
    """
    Computes Euclidean (L2) norm of a 1D vector: sqrt(sum(v_i^2))
    """
    return float(np.sqrt(np.sum(v ** 2)))


def cosine_similarity_scratch(u: np.ndarray, v: np.ndarray) -> float:
    """
    Computes cosine similarity between two 1D vectors u and v.
    Returns 0.0 if either vector has zero length.
    """
    norm_u = vector_norm(u)
    norm_v = vector_norm(v)

    # Edge case: If either vector is all zeros (no matched terms), return 0.0
    if norm_u == 0.0 or norm_v == 0.0:
        return 0.0

    dot_product = float(np.dot(u, v))
    return dot_product / (norm_u * norm_v)


def rank_documents_scratch(
    query_vector: np.ndarray,
    doc_matrix: np.ndarray
) -> List[Tuple[int, float]]:
    """
    Ranks documents against a query vector using manual cosine similarity.

    Returns:
    --------
    ranked_indices : list of (doc_index, similarity_score) sorted descending.
    """
    scores = []
    for idx, doc_vec in enumerate(doc_matrix):
        sim = cosine_similarity_scratch(query_vector, doc_vec)
        scores.append((idx, sim))

    # Sort descending by score
    scores.sort(key=lambda item: item[1], reverse=True)
    return scores


if __name__ == "__main__":
    print("=" * 70)
    print(" DEMO: cosine_scratch.py (Mathematical Worked Example For Viva)")
    print("=" * 70)

    # 1. Documents and Vocabulary from Stage 5
    raw_docs = [
        "supervised learning model",
        "unsupervised learning algorithm",
        "deep learning neural model"
    ]
    tokenized_docs = [d.split() for d in raw_docs]
    vocab = build_vocabulary(tokenized_docs)
    _, idf_vec, doc_matrix = compute_tfidf(tokenized_docs, vocab)

    # 2. Query
    query_str = "supervised model"
    query_tokens = query_str.split()
    print(f"\nCorpus Documents:")
    for i, d in enumerate(raw_docs, 1):
        print(f" D{i}: '{d}'")
    print(f"\nUser Query: '{query_str}'")

    # 3. Vectorize Query using same vocabulary and IDF
    query_tf = np.zeros(len(vocab), dtype=float)
    for term in query_tokens:
        if term in vocab:
            idx = vocab.index(term)
            query_tf[idx] = query_tokens.count(term) / len(query_tokens)
    query_vector = query_tf * idf_vec

    print(f"\nVocabulary: {vocab}")
    print("\n--- Vector Representation ---")
    print(" Query Vector:    ", np.round(query_vector, 4))
    print(" Query L2 Norm:   ", round(vector_norm(query_vector), 4))

    # 4. Step-by-Step Similarity Calculation
    print("\n--- Step-by-Step Cosine Similarity Math ---")
    print(f" {'Doc':<5} | {'Dot Product':<12} | {'Doc Norm':<10} | {'Cosine Score':<12} | Formula: Dot / (Norm_Q * Norm_D)")
    print("-" * 75)

    norm_q = vector_norm(query_vector)
    for doc_idx, doc_vec in enumerate(doc_matrix, 1):
        dot = float(np.dot(query_vector, doc_vec))
        norm_d = vector_norm(doc_vec)
        score = dot / (norm_q * norm_d) if (norm_q > 0 and norm_d > 0) else 0.0
        print(f" D{doc_idx:<4} | {dot:<12.4f} | {norm_d:<10.4f} | {score:<12.4f} | {dot:.4f} / ({norm_q:.4f} * {norm_d:.4f})")

    # 5. Final Ranked Results
    ranked = rank_documents_scratch(query_vector, doc_matrix)
    print("\n--- Ranked Results ---")
    for rank, (doc_idx, score) in enumerate(ranked, 1):
        print(f" Rank {rank}: Document D{doc_idx + 1} ('{raw_docs[doc_idx]}') -> Score: {score:.4f}")
    print("=" * 70)
