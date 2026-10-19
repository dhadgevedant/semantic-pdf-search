"""
Stage 2: PDF Loader (loader.py)
--------------------------------
Loads single or multiple PDF documents from local file paths or Streamlit
file upload objects. Extracts text page-by-page while tracking metadata
(filename, page number) and gracefully handles empty or scanned PDFs.
"""

import io
from pathlib import Path
from typing import List, Dict, Any, Tuple, Union
import pypdf


def extract_pages_from_pdf(source: Union[str, Path, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Extracts text page-by-page from a single PDF.

    Parameters:
    -----------
    source : str, Path, or Streamlit UploadedFile
        The PDF source. Can be a filesystem path or a file-like stream.

    Returns:
    --------
    pages : list of dict
        Each dict contains:
          - "file_name": Name of the source PDF
          - "page_number": 1-based page index
          - "text": Cleaned extracted text string
    warnings : list of str
        Human-readable warnings for empty pages or potential scanned PDFs.
    """
    pages = []
    warnings = []

    # 1. Determine file name and stream object
    if isinstance(source, (str, Path)):
        file_path = Path(source)
        file_name = file_path.name
        if not file_path.exists():
            warnings.append(f"File not found: {source}")
            return pages, warnings
        stream = open(file_path, "rb")
        should_close = True
    else:
        # Streamlit UploadedFile or BytesIO object
        file_name = getattr(source, "name", "uploaded_document.pdf")
        if hasattr(source, "getvalue"):
            stream = io.BytesIO(source.getvalue())
        elif hasattr(source, "read"):
            stream = io.BytesIO(source.read())
        else:
            stream = source
        should_close = False

    # 2. Parse PDF with pypdf
    try:
        reader = pypdf.PdfReader(stream)
        total_pages = len(reader.pages)
        total_chars_extracted = 0

        for page_idx in range(total_pages):
            page_obj = reader.pages[page_idx]
            extracted_text = page_obj.extract_text() or ""
            cleaned_text = extracted_text.strip()
            page_number = page_idx + 1

            if len(cleaned_text) == 0:
                warnings.append(
                    f"Warning: Page {page_number} in '{file_name}' contains no extractable text."
                )
            else:
                total_chars_extracted += len(cleaned_text)
                pages.append({
                    "file_name": file_name,
                    "page_number": page_number,
                    "text": cleaned_text
                })

        # Check if the entire document had zero text (likely a scanned image)
        if total_pages > 0 and total_chars_extracted == 0:
            warnings.append(
                f"Warning: '{file_name}' has {total_pages} page(s) but 0 extractable words. "
                "This document appears to be a scanned image or photo PDF lacking a digital text layer."
            )

    except Exception as exc:
        warnings.append(f"Error reading '{file_name}': {str(exc)}")
    finally:
        if should_close:
            stream.close()

    return pages, warnings


def load_multiple_pdfs(sources: List[Union[str, Path, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Loads and extracts text from multiple PDF sources.

    Parameters:
    -----------
    sources : list of (str, Path, or Streamlit UploadedFile)
        A list of PDF paths or uploaded file objects.

    Returns:
    --------
    all_pages : list of dict
        Combined list of page dictionaries from all PDFs.
    all_warnings : list of str
        Combined list of warnings across all files.
    """
    all_pages = []
    all_warnings = []

    for src in sources:
        pages, warnings = extract_pages_from_pdf(src)
        all_pages.extend(pages)
        all_warnings.extend(warnings)

    return all_pages, all_warnings


if __name__ == "__main__":
    print("=" * 65)
    print(" DEMO: loader.py (Extracting text from sample academic PDFs)")
    print("=" * 65)

    sample_files = [
        "sample_docs/machine_learning_basics.pdf",
        "sample_docs/nlp_and_transformers.pdf",
        "sample_docs/empty_scanned_paper.pdf",
    ]

    extracted_pages, warnings = load_multiple_pdfs(sample_files)

    print(f"\nTotal pages successfully extracted: {len(extracted_pages)}")
    for i, p in enumerate(extracted_pages, 1):
        preview = p["text"].replace("\n", " ")[:75]
        print(f" [{i}] File: {p['file_name']:<30} | Page: {p['page_number']} | Preview: {preview}...")

    if warnings:
        print("\nWarnings encountered during loading:")
        for w in warnings:
            print(f" [!] {w}")
    print("=" * 65)
