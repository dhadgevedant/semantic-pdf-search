"""
Stage 8: Search Logic, Edge Cases, and CLI (search.py)
------------------------------------------------------
A robust interface over the core SearchEngine. Handles edge cases:
  - Empty or whitespace-only queries
  - Queries with zero matching terms in vocabulary (out-of-vocabulary)
  - Score thresholding (filters out irrelevant 0.0 scores)
  - Keyword highlighting in result passages

Includes an optional Command Line Interface (CLI) for terminal testing.
"""

import argparse
import re
from typing import List, Dict, Any, Tuple
from search_engine import SearchEngine
from preprocess import clean_and_tokenize


def highlight_query_terms(text: str, query: str) -> str:
    """
    Highlights words in the snippet that match the query terms.
    Uses markdown bold syntax (**word**).
    """
    query_tokens = clean_and_tokenize(query)
    if not query_tokens:
        return text

    highlighted = text
    # Match query words case-insensitively using regex word boundaries
    for token in set(query_tokens):
        # We highlight word stems/prefixes matching token
        pattern = re.compile(rf"\b({re.escape(token)}\w*)\b", re.IGNORECASE)
        highlighted = pattern.sub(r"**\1**", highlighted)

    return highlighted


def execute_search(
    engine: SearchEngine,
    query: str,
    top_k: int = 5,
    min_score: float = 0.01
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Safely executes a query on the search engine with validation and thresholding.

    Parameters:
    -----------
    engine : SearchEngine
        An indexed SearchEngine instance.
    query : str
        User input query.
    top_k : int
        Maximum results to return.
    min_score : float
        Minimum cosine similarity score required (default 0.01).

    Returns:
    --------
    results : list of dict
        Validated and filtered search results with highlighted text snippets.
    status_message : str
        Descriptive user-facing status message.
    """
    # Edge Case 1: Check if index is built
    if not engine.is_indexed:
        return [], "Error: Search index is empty. Please upload and index documents first."

    # Edge Case 2: Empty or whitespace query
    stripped_query = query.strip()
    if not stripped_query:
        return [], "Please enter a valid search query."

    # Edge Case 3: Query tokens out-of-vocabulary
    query_tokens = clean_and_tokenize(stripped_query)
    if not query_tokens:
        return [], "Query contains only stop words or symbols with no searchable terms."

    vocab = engine.vectorizer.vocabulary_
    matched_terms = [t for t in query_tokens if t in vocab]
    if not matched_terms:
        return [], f"No matching terms found in indexed documents for: '{stripped_query}'."

    # Perform search
    raw_results = engine.search(stripped_query, top_k=top_k)

    # Edge Case 4: Filter by minimum score threshold
    filtered_results = []
    for r in raw_results:
        if r["score"] >= min_score:
            r_copy = dict(r)
            r_copy["highlighted_text"] = highlight_query_terms(r["text"], stripped_query)
            filtered_results.append(r_copy)

    if not filtered_results:
        return [], "No relevant passages met the minimum similarity threshold."

    status_message = f"Found {len(filtered_results)} matching passage(s)."
    return filtered_results, status_message


def run_cli():
    """
    Optional Command-Line Interface for quick local testing without launching Streamlit.
    """
    parser = argparse.ArgumentParser(description="Semantic PDF Search CLI")
    parser.add_argument("--query", type=str, default="gradient descent optimization", help="Search query")
    parser.add_argument("--top_k", type=int, default=3, help="Number of results")
    args = parser.parse_args()

    from loader import load_multiple_pdfs
    from chunker import chunk_pages

    docs = [
        "sample_docs/machine_learning_basics.pdf",
        "sample_docs/nlp_and_transformers.pdf"
    ]
    pages, _ = load_multiple_pdfs(docs)
    chunks = chunk_pages(pages, chunk_size=40, overlap=10)

    engine = SearchEngine()
    engine.build_index(chunks)

    results, status = execute_search(engine, args.query, top_k=args.top_k)
    print("\nStatus:", status)
    for r in results:
        print(f"\n[Rank {r['rank']}] Score: {r['score']:.4f} | {r['file_name']} (p. {r['page_number']})")
        print(f"Highlighted: {r['highlighted_text']}")


if __name__ == "__main__":
    print("=" * 70)
    print(" DEMO: search.py (Edge Case Validation & Highlighted Retrieval)")
    print("=" * 70)

    from loader import load_multiple_pdfs
    from chunker import chunk_pages

    docs = [
        "sample_docs/machine_learning_basics.pdf",
        "sample_docs/nlp_and_transformers.pdf"
    ]
    pages, _ = load_multiple_pdfs(docs)
    chunks = chunk_pages(pages, chunk_size=40, overlap=10)

    engine = SearchEngine()
    engine.build_index(chunks)

    # Test Edge Cases
    test_queries = [
        "",                                          # Edge Case: Empty
        "   ",                                       # Edge Case: Whitespace
        "quantum entanglement teleportation galaxy", # Edge Case: Out of vocabulary
        "gradient descent",                          # Valid query
    ]

    for q in test_queries:
        print(f"\n--- Testing Query: '{q}' ---")
        res, msg = execute_search(engine, q, top_k=2)
        print(f"Status Message: {msg}")
        for r in res:
            print(f" -> Rank {r['rank']} ({r['score']}): {r['highlighted_text'][:120]}...")
    print("=" * 70)
