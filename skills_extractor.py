"""Semantic similarity scoring between a resume and a job description.

Primary method: TF-IDF + cosine similarity (fast, fully offline, no model
downloads). If `sentence-transformers` is installed and the caller opts
in via use_embeddings=True, a transformer embedding model is used instead
for higher-quality semantic matching.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def tfidf_similarity(resume_text: str, jd_text: str) -> float:
    """Return a 0-100 similarity score using TF-IDF cosine similarity."""
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(float(score) * 100, 2)


def embedding_similarity(resume_text: str, jd_text: str) -> float:
    """Return a 0-100 similarity score using sentence-transformer embeddings.

    Falls back to TF-IDF automatically if the library/model isn't available
    (e.g. no internet access to download the model).
    """
    try:
        from sentence_transformers import SentenceTransformer, util

        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode([resume_text, jd_text], convert_to_tensor=True)
        score = util.cos_sim(embeddings[0], embeddings[1]).item()
        return round(score * 100, 2)
    except Exception:
        return tfidf_similarity(resume_text, jd_text)


def compute_similarity(resume_text: str, jd_text: str, use_embeddings: bool = False) -> float:
    if use_embeddings:
        return embedding_similarity(resume_text, jd_text)
    return tfidf_similarity(resume_text, jd_text)
