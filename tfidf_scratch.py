"""
Stage 5: TF-IDF From Scratch (tfidf_scratch.py)
------------------------------------------------
Manual implementation of Term Frequency (TF), Inverse Document Frequency (IDF),
and the Document-Term Matrix using pure Python and NumPy.

The Math:
1. Term Frequency (TF):
   Measures how frequently term 't' appears in document 'd'.
   TF(t, d) = (count of t in d) / (total words in d)

2. Document Frequency (DF):
   Number of documents in the collection that contain term 't'.

3. Inverse Document Frequency (IDF):
   Penalizes terms that appear everywhere (e.g. 'learning') and boosts rare,
   informative terms (e.g. 'gradient', 'unsupervised').
   IDF(t) = log( (N + 1) / (DF(t) + 1) ) + 1
   (Where N is total number of documents, and +1 prevents division by zero / negative logs)

4. TF-IDF Score:
   TF-IDF(t, d) = TF(t, d) * IDF(t)
"""

import math
from typing import List, Dict, Tuple
import numpy as np


def build_vocabulary(corpus_tokens: List[List[str]]) -> List[str]:
    """
    Extracts sorted unique words across all documents.
    """
    vocab_set = set()
    for doc in corpus_tokens:
        for token in doc:
            vocab_set.add(token)
    return sorted(list(vocab_set))


def compute_tf(corpus_tokens: List[List[str]], vocab: List[str]) -> np.ndarray:
    """
    Computes Term Frequency matrix of shape (num_docs, vocab_size).
    TF(t, d) = count(t, d) / len(d)
    """
    num_docs = len(corpus_tokens)
    vocab_size = len(vocab)
    tf_matrix = np.zeros((num_docs, vocab_size), dtype=float)

    for doc_idx, doc in enumerate(corpus_tokens):
        doc_len = len(doc)
        if doc_len == 0:
            continue
        for term_idx, term in enumerate(vocab):
            count = doc.count(term)
            tf_matrix[doc_idx, term_idx] = count / doc_len

    return tf_matrix


def compute_idf(corpus_tokens: List[List[str]], vocab: List[str]) -> np.ndarray:
    """
    Computes smooth Inverse Document Frequency vector of shape (vocab_size,).
    IDF(t) = ln((N + 1) / (DF(t) + 1)) + 1
    """
    num_docs = len(corpus_tokens)
    vocab_size = len(vocab)
    idf_vector = np.zeros(vocab_size, dtype=float)

    for term_idx, term in enumerate(vocab):
        # Count documents containing this term
        df = sum(1 for doc in corpus_tokens if term in doc)
        # Smooth IDF formula (matches standard information retrieval & sklearn default)
        idf_vector[term_idx] = math.log((num_docs + 1) / (df + 1)) + 1.0

    return idf_vector


def compute_tfidf(corpus_tokens: List[List[str]], vocab: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates TF matrix, IDF vector, and the final TF-IDF matrix.
    TF-IDF = TF * IDF
    """
    tf_matrix = compute_tf(corpus_tokens, vocab)
    idf_vector = compute_idf(corpus_tokens, vocab)
    tfidf_matrix = tf_matrix * idf_vector  # Element-wise broadcast multiplication
    return tf_matrix, idf_vector, tfidf_matrix


if __name__ == "__main__":
    print("=" * 70)
    print(" DEMO: tfidf_scratch.py (3-Document Worked Example For Viva/Slides)")
    print("=" * 70)

    # 3 Tiny Sample Documents
    raw_docs = [
        "supervised learning model",
        "unsupervised learning algorithm",
        "deep learning neural model"
    ]
    
    print("\nCorpus Documents:")
    for i, d in enumerate(raw_docs, 1):
        print(f" D{i}: '{d}'")

    tokenized_docs = [d.split() for d in raw_docs]
    vocab = build_vocabulary(tokenized_docs)

    print(f"\nVocabulary ({len(vocab)} unique terms):")
    print(" ", vocab)

    tf_mat, idf_vec, tfidf_mat = compute_tfidf(tokenized_docs, vocab)

    # Display DF and IDF
    print("\n--- Document Frequency (DF) & IDF Weights ---")
    print(f" {'Term':<15} | {'DF':<4} | {'IDF Weight':<10}")
    print("-" * 38)
    for idx, term in enumerate(vocab):
        df_val = sum(1 for doc in tokenized_docs if term in doc)
        print(f" {term:<15} | {df_val:<4} | {idf_vec[idx]:.4f}")

    # Display TF-IDF Matrix
    print("\n--- Final TF-IDF Matrix (Documents x Terms) ---")
    header = "Doc   " + " ".join([f"{term:>12}" for term in vocab])
    print(header)
    print("-" * len(header))
    for doc_idx, row in enumerate(tfidf_mat, 1):
        row_str = f"D{doc_idx:<4}" + " ".join([f"{val:12.4f}" for val in row])
        print(row_str)

    print("\nNotice:")
    print(" • 'learning' appears in all 3 documents (DF=3), so its IDF weight is low (1.0000).")
    print(" • 'supervised' appears in only 1 document (DF=1), so its IDF weight is high (1.6931).")
    print(" • This allows unique keywords to drive retrieval over generic common words.")
    print("=" * 70)
