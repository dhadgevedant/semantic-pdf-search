"""
Stage 3: Passage Chunking (chunker.py)
--------------------------------------
Splits page-level document text into smaller, overlapping passages (chunks).
Carries forward document metadata (file name, page number) and assigns
a unique chunk ID to each passage.

Why Chunking Matters:
1. Granularity: A full 10-page paper covers many topics. Chunking localizes
   information to specific paragraphs.
2. Density: In long documents, term frequency is diluted across thousands of
   words. Chunks concentrate keyword density.
3. User Experience: Showing the user a concise 150-word answer with its
   exact page citation is far more useful than pointing them to a 50-page file.
"""

from typing import List, Dict, Any


def chunk_text(
    text: str,
    file_name: str,
    page_number: int,
    chunk_size: int = 150,
    overlap: int = 30
) -> List[Dict[str, Any]]:
    """
    Splits text from a single page into overlapping word-based chunks.

    Parameters:
    -----------
    text : str
        The raw text of the page.
    file_name : str
        Name of the source document.
    page_number : int
        Page number where this text originated.
    chunk_size : int, default=150
        Target number of words per chunk.
    overlap : int, default=30
        Number of overlapping words between consecutive chunks to preserve
        boundary context.

    Returns:
    --------
    chunks : list of dict
        Each dict has 'chunk_id', 'file_name', 'page_number', 'text', and 'word_count'.
    """
    words = text.split()
    total_words = len(words)

    # Edge case: If the text is empty, return no chunks
    if total_words == 0:
        return []

    # Edge case: If the text is shorter than chunk_size, keep it as one chunk
    if total_words <= chunk_size:
        return [{
            "chunk_id": f"{file_name}_p{page_number}_c1",
            "file_name": file_name,
            "page_number": page_number,
            "text": " ".join(words),
            "word_count": total_words
        }]

    chunks = []
    step = chunk_size - overlap
    chunk_index = 1

    start_idx = 0
    while start_idx < total_words:
        end_idx = min(start_idx + chunk_size, total_words)
        chunk_words = words[start_idx:end_idx]
        chunk_text_str = " ".join(chunk_words)

        chunks.append({
            "chunk_id": f"{file_name}_p{page_number}_c{chunk_index}",
            "file_name": file_name,
            "page_number": page_number,
            "text": chunk_text_str,
            "word_count": len(chunk_words)
        })

        chunk_index += 1
        start_idx += step

    return chunks


def chunk_pages(
    pages: List[Dict[str, Any]],
    chunk_size: int = 150,
    overlap: int = 30
) -> List[Dict[str, Any]]:
    """
    Chunks a list of extracted pages from multiple documents.

    Parameters:
    -----------
    pages : list of dict
        Output from loader.py containing 'file_name', 'page_number', and 'text'.
    chunk_size : int
        Words per chunk.
    overlap : int
        Overlapping words between chunks.

    Returns:
    --------
    all_chunks : list of dict
        Flat list of all generated chunks with metadata.
    """
    all_chunks = []
    for page in pages:
        chunks = chunk_text(
            text=page["text"],
            file_name=page["file_name"],
            page_number=page["page_number"],
            chunk_size=chunk_size,
            overlap=overlap
        )
        all_chunks.extend(chunks)

    return all_chunks


if __name__ == "__main__":
    print("=" * 65)
    print(" DEMO: chunker.py (Splitting extracted pages into passages)")
    print("=" * 65)

    from loader import load_multiple_pdfs

    sample_files = [
        "sample_docs/machine_learning_basics.pdf",
        "sample_docs/nlp_and_transformers.pdf",
    ]

    pages, _ = load_multiple_pdfs(sample_files)
    # Using a small chunk_size=35 and overlap=10 for demo clarity
    chunks = chunk_pages(pages, chunk_size=35, overlap=10)

    print(f"\nTotal chunks created: {len(chunks)} from {len(pages)} pages.\n")
    for ch in chunks[:4]:
        print(f" ID: {ch['chunk_id']:<35} | Words: {ch['word_count']}")
        print(f" File: {ch['file_name']} (Page {ch['page_number']})")
        print(f" Snippet: {ch['text'][:90]}...")
        print("-" * 65)
