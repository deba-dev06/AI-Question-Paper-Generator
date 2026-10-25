"""
metrics.py — Quality metrics for the AI Question Paper Generator.

Provides three measurable metrics:
  1. Generation speed  — measured in generator.py via time.perf_counter()
  2. Question validity — well-formedness, length compliance, uniqueness
  3. Source relevance  — cosine similarity between questions and source chunks
"""

import re
import numpy as np

# ---------------------------------------------------------------------------
# Metric 2 — Question Validity
# ---------------------------------------------------------------------------

# Questions must start with one of these keywords OR end with "?"
VALID_STARTERS = (
    "define", "explain", "discuss", "compare",
    "describe", "what", "how", "why", "list",
    "state", "differentiate", "illustrate", "analyse",
    "analyze", "evaluate", "outline", "justify",
)

# Expected word-count range per mark category
MARK_LENGTH_RANGE = {
    2: (3, 20),   # short recall questions
    3: (8, 40),   # medium questions
    5: (15, 80),  # descriptive questions
}


def check_validity(questions: list, marks: int) -> dict:
    """
    Check the validity of a list of generated questions.

    Rules checked per question:
      - well_formed  : starts with a recognised question word OR ends with '?'
      - right_length : word count falls within the expected range for `marks`
      - unique       : not a duplicate of a previous question (exact match)

    Returns a summary dict with counts and percentages.
    """
    well_formed_count = 0
    right_length_count = 0
    seen = set()
    unique_count = 0

    lo, hi = MARK_LENGTH_RANGE.get(marks, (0, 9999))

    for q in questions:
        q_stripped = q.strip()
        q_lower = q_stripped.lower()
        word_count = len(q_stripped.split())

        # Rule 1: well-formed
        starts_ok = q_lower.startswith(VALID_STARTERS)
        ends_ok = q_stripped.endswith("?")
        if starts_ok or ends_ok:
            well_formed_count += 1

        # Rule 2: length appropriate for mark category
        if lo <= word_count <= hi:
            right_length_count += 1

        # Rule 3: unique (case-insensitive exact dedup)
        if q_lower not in seen:
            unique_count += 1
            seen.add(q_lower)

    n = max(len(questions), 1)  # avoid division by zero

    return {
        "total_questions": n,
        "well_formed": {
            "count": well_formed_count,
            "percent": round(100 * well_formed_count / n, 1),
        },
        "right_length": {
            "count": right_length_count,
            "percent": round(100 * right_length_count / n, 1),
        },
        "unique": {
            "count": unique_count,
            "percent": round(100 * unique_count / n, 1),
        },
    }


# ---------------------------------------------------------------------------
# Metric 3 — Source Relevance (sentence-transformers cosine similarity)
# ---------------------------------------------------------------------------

_embedding_model = None  # lazy-loaded to avoid slowing down app startup


def _get_embedding_model():
    """Lazy-load the sentence-transformer model (cached after first call)."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def chunk_text(text: str, chunk_size: int = 100, overlap: int = 50) -> list:
    """
    Split source text into overlapping word-level chunks for finer-grained
    similarity matching.
    """
    words = text.split()
    step = max(chunk_size - overlap, 1)
    chunks = []
    for i in range(0, len(words), step):
        chunk = " ".join(words[i: i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks if chunks else [text]


def relevance_score(questions: list, source_text: str,
                    threshold: float = 0.5) -> dict:
    """
    Compute cosine similarity between each question and the closest chunk
    of the source text.

    Returns:
      - avg_cosine_similarity : mean of per-question max similarities
      - above_threshold       : count/percent of questions above `threshold`
      - per_question_scores   : list of floats for bar chart display
    """
    from sentence_transformers import util as st_util

    model = _get_embedding_model()
    chunks = chunk_text(source_text)

    # Encode questions and source chunks
    q_embeddings = model.encode(questions, convert_to_tensor=True)
    c_embeddings = model.encode(chunks, convert_to_tensor=True)

    # Cosine similarity matrix: shape (n_questions, n_chunks)
    sim_matrix = st_util.cos_sim(q_embeddings, c_embeddings)

    # Best-matching chunk score per question
    max_sims = sim_matrix.max(dim=1).values.cpu().numpy()

    avg_sim = float(np.mean(max_sims))
    n = len(questions)
    above = int(np.sum(max_sims >= threshold))

    return {
        "avg_cosine_similarity": round(avg_sim, 3),
        "threshold": threshold,
        "above_threshold": {
            "count": above,
            "percent": round(100 * above / max(n, 1), 1),
        },
        "per_question_scores": [round(float(s), 3) for s in max_sims],
    }
