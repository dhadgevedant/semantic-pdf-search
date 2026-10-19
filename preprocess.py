"""
Stage 4: Text Preprocessing (preprocess.py)
--------------------------------------------
Performs linguistic normalization on raw text:
  1. Lowercasing
  2. Removing punctuation, numbers, and special symbols
  3. Tokenization
  4. Stop-word removal (e.g., 'the', 'is', 'at')
  5. Lemmatization (e.g., 'optimization' -> 'optimization', 'networks' -> 'network', 'learning' -> 'learn')

CRITICAL RULE:
Both document chunks and user search queries MUST pass through the exact same
preprocessing pipeline. If a query searches for "learning models" but chunks contain
"learn model", vocabulary mismatch will cause zero matches unless both are normalized
identically.
"""

import re
from typing import List
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure required NLTK corpora are downloaded
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)

# Initialize reusable stop words set and lemmatizer
STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def clean_and_tokenize(text: str) -> List[str]:
    """
    Cleans raw text and splits it into a list of normalized lemma tokens.

    Parameters:
    -----------
    text : str
        Input string (either a document chunk or a user query).

    Returns:
    --------
    tokens : list of str
        List of cleaned, non-stopword, lemmatized tokens.
    """
    if not text or not isinstance(text, str):
        return []

    # 1. Lowercase the text
    text_lower = text.lower()

    # 2. Keep only alphabetical words (remove punctuation, digits, symbols)
    words = re.findall(r"\b[a-z]{2,}\b", text_lower)

    # 3. Filter stop words and apply lemmatization
    cleaned_tokens = []
    for word in words:
        if word not in STOP_WORDS:
            # Lemmatize as noun (default) and verb
            lemma = LEMMATIZER.lemmatize(word, pos="v")
            lemma = LEMMATIZER.lemmatize(lemma, pos="n")
            cleaned_tokens.append(lemma)

    return cleaned_tokens


def preprocess_text(text: str) -> str:
    """
    Preprocesses text and returns a single normalized string with space-separated tokens.
    This format is directly consumable by TF-IDF vectorizers (both custom and scikit-learn).

    Parameters:
    -----------
    text : str
        Input passage or search query.

    Returns:
    --------
    normalized_string : str
        Space-separated lemmatized tokens.
    """
    tokens = clean_and_tokenize(text)
    return " ".join(tokens)


if __name__ == "__main__":
    print("=" * 65)
    print(" DEMO: preprocess.py (Normalizing chunks and queries)")
    print("=" * 65)

    sample_chunk = (
        "Gradient descent iteratively updates model weights in the direction "
        "opposite to the gradients! It achieves 99.5% accuracy."
    )
    sample_query = "How are model weights updated during gradient descent?"

    cleaned_chunk = preprocess_text(sample_chunk)
    cleaned_query = preprocess_text(sample_query)

    print("\n--- Original Chunk ---")
    print(sample_chunk)
    print("\n--- Preprocessed Chunk ---")
    print(cleaned_chunk)

    print("\n--- Original User Query ---")
    print(sample_query)
    print("\n--- Preprocessed User Query ---")
    print(cleaned_query)

    print("\n--- Overlapping Vocabulary Tokens ---")
    chunk_tokens = set(clean_and_tokenize(sample_chunk))
    query_tokens = set(clean_and_tokenize(sample_query))
    overlap = chunk_tokens.intersection(query_tokens)
    print("Shared tokens ready for TF-IDF matching:", sorted(list(overlap)))
    print("=" * 65)
