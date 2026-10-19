"""
Stage 11: Limitations, Extensions, and Next Steps (extensions_analysis.py)
--------------------------------------------------------------------------
Demonstrates the fundamental theoretical limitations of lexical TF-IDF search,
and analyzes why modern search architectures evolved towards BM25 and
Dense Embeddings (Sentence Transformers / RAG).

1. Lexical Failure Case (The Synonymy Problem):
   TF-IDF requires exact or morphological word overlap. If a document discusses
   "cardiovascular myocardial infarction" and a user searches for "heart attack",
   TF-IDF cosine similarity is strictly 0.0!

2. Lexical Failure Case (The Polysemy Problem):
   The word "bank" can mean a financial institution or a riverbank. TF-IDF assigns
   identical vector weights to both usages.

3. Why BM25 is an Improvement over TF-IDF:
   - Term Saturation: In TF-IDF, repeating a term 100 times scales linearly or logarithmically.
     In BM25, term frequency asymptotically saturates (diminishing returns).
   - Document Length Normalization: Penalizes verbose documents using parameter 'b'.

4. Dense Semantic Embeddings & RAG:
   - Deep bi-encoders (e.g. all-MiniLM-L6-v2) project text into 384-dimensional continuous
     vectors where "heart attack" and "myocardial infarction" have cosine similarity > 0.85.
   - In Retrieval-Augmented Generation (RAG), dense retrieval locates passages and passes
     them into an LLM context window to generate factual, hallucination-free answers.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def demonstrate_synonym_failure():
    """
    Proves that TF-IDF cannot connect synonymous terms without shared lexical tokens.
    """
    corpus = [
        "The patient suffered a severe myocardial infarction requiring cardiac surgery.",
        "Deep neural networks require backpropagation and stochastic gradient descent.",
    ]

    queries = [
        ("Lexical Match", "myocardial infarction"),
        ("Synonym Query", "heart attack"),
    ]

    vectorizer = TfidfVectorizer()
    corpus_matrix = vectorizer.fit_transform(corpus)

    print("=" * 70)
    print(" DEMO 1: TF-IDF Lexical Limitation (The Synonymy Problem)")
    print("=" * 70)
    print("Document 1: 'The patient suffered a severe myocardial infarction requiring cardiac surgery.'")
    print("-" * 70)

    for q_type, q_text in queries:
        q_vec = vectorizer.transform([q_text])
        score = cosine_similarity(q_vec, corpus_matrix)[0][0]
        print(f" Query Type: {q_type:<15} | Query: '{q_text}'")
        print(f" -> TF-IDF Cosine Similarity Score: {score:.4f}")
        if score == 0.0:
            print("    [!] FAILURE: Even though 'heart attack' is a direct synonym for 'myocardial")
            print("        infarction', TF-IDF yields a score of 0.0 because there is no character overlap.")
        print("-" * 70)


def display_architectural_comparison():
    """
    Prints a comparison table contrasting TF-IDF, BM25, Dense Embeddings, and RAG.
    """
    print("\n" + "=" * 70)
    print(" COMPARISON: Information Retrieval Paradigms")
    print("=" * 70)
    
    comparisons = [
        ("Feature", "TF-IDF (Ours)", "BM25", "Dense Embeddings (Bi-Encoder)", "RAG (Retrieval + LLM)"),
        ("-" * 15, "-" * 15, "-" * 15, "-" * 25, "-" * 22),
        ("Matching Type", "Exact lexical", "Probabilistic lexical", "Dense latent semantics", "Semantic retrieval + Gen AI"),
        ("Synonym Support", "No (score = 0.0)", "No", "Yes (high cosine)", "Yes (high cosine)"),
        ("Term Saturation", "No (linear TF)", "Yes (k1 parameter)", "N/A (continuous embeddings)", "N/A"),
        ("Length Penalty", "L2 Norm", "Yes (b parameter)", "Fixed sequence length", "Chunk window size"),
        ("Compute Cost", "Extremely Low (CPU)", "Extremely Low (CPU)", "Medium (GPU/CPU)", "High (GPU inference)"),
        ("Interpretability", "100% Explainable", "High", "Low (Black-box vectors)", "Medium (Prompt citations)"),
        ("Hallucination Risk", "Zero", "Zero", "Zero", "Low (grounded by context)"),
    ]

    for row in comparisons:
        print(f"{row[0]:<18} | {row[1]:<16} | {row[2]:<16} | {row[3]:<26} | {row[4]}")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_synonym_failure()
    display_architectural_comparison()
