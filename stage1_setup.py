"""
Stage 1: Project Setup and Pipeline Overview
---------------------------------------------
This script verifies all required libraries are installed
and prints a visual ASCII overview of the search pipeline.
"""

import sys

def verify_environment():
    """
    Check if required packages are installed and print their versions.
    """
    packages = [
        ("numpy", "NumPy"),
        ("sklearn", "Scikit-Learn"),
        ("pypdf", "pypdf"),
        ("nltk", "NLTK"),
        ("joblib", "joblib"),
        ("streamlit", "Streamlit"),
    ]
    
    print("=" * 60)
    print(" 1. Verifying Python Environment & Dependencies")
    print("=" * 60)
    print(f" Python version: {sys.version.split()[0]}")
    print("-" * 60)
    
    all_ok = True
    for module_name, display_name in packages:
        try:
            mod = __import__(module_name)
            version = getattr(mod, "__version__", "installed")
            print(f" [OK] {display_name:<15} version: {version}")
        except ImportError:
            print(f" [MISSING] {display_name:<15} (Please install via requirements.txt)")
            all_ok = False
            
    print("-" * 60)
    if all_ok:
        print(" All core dependencies are successfully installed!")
    else:
        print(" Some dependencies are missing. Run: pip install -r requirements.txt")
    print("=" * 60)
    print()


def display_pipeline_diagram():
    """
    Prints the text-based architecture diagram of the search engine.
    """
    diagram = """
========================================================================
             SEMANTIC PDF SEARCH ENGINE PIPELINE ARCHITECTURE
========================================================================

 [ User PDFs ]
      │
      ▼
┌──────────────┐
│  loader.py   │  Extract raw text page-by-page + metadata (filename, page)
└──────┬───────┘
      │
      ▼
┌──────────────┐
│  chunker.py  │  Break text into 150-250 word overlapping passages
└──────┬───────┘
      │
      ▼
┌──────────────┐
│preprocess.py │  Normalize: lowercase, remove noise/stopwords, lemmatize
└──────┬───────┘
      │
      ├────────────────────────────────────────┐
      ▼ (Indexed Documents)                    ▼ (User Query)
┌──────────────────────┐             ┌───────────────────┐
│   TF-IDF Matrix      │             │ Preprocess Query  │
│ (Chunks x Vocabulary)│             │  & TF-IDF Vector  │
└──────────┬───────────┘             └─────────┬─────────┘
           │                                   │
           └─────────────────┬─────────────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │   Cosine Similarity   │  Calculate dot-product / norms
                 └───────────┬───────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │ Ranked Search Results │  (Score, File, Page, Snippet)
                 └───────────────────────┘
========================================================================
"""
    print(diagram)


def display_pipeline_steps():
    """
    Explains each stage in plain language.
    """
    steps = [
        ("Step 1: PDF Loading", "Extract text from uploaded PDFs page-by-page while tracking page numbers."),
        ("Step 2: Chunking", "Cut continuous text into smaller overlapping chunks (150-250 words) for targeted retrieval."),
        ("Step 3: Preprocessing", "Clean text (lowercase, strip symbols, remove stop words, lemmatize) for both chunks and queries."),
        ("Step 4: TF-IDF Matrix", "Convert text chunks into mathematical vectors based on term importance across all chunks."),
        ("Step 5: Query Vectorization", "Transform the user's search query into the identical vector space."),
        ("Step 6: Cosine Similarity", "Measure the angle between query vector and every chunk vector (score from 0.0 to 1.0)."),
        ("Step 7: Ranking & Display", "Sort scores descending and display the top-matching passages in the Streamlit UI."),
    ]
    
    print("Pipeline Step Breakdown:")
    print("-" * 60)
    for title, desc in steps:
        print(f" • {title}:")
        print(f"   {desc}")
    print("=" * 60)


if __name__ == "__main__":
    verify_environment()
    display_pipeline_diagram()
    display_pipeline_steps()
