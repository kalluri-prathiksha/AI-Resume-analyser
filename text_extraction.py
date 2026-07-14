"""NLP-based technical skill extraction using a curated skills taxonomy.

Uses regex word-boundary matching plus simple normalization (handles
aliases like "js" -> "javascript") so it works fully offline with no
model downloads required.
"""

import re
from collections import defaultdict

# Curated skills taxonomy, grouped by category. Extend freely.
SKILLS_DB = {
    "languages": [
        "python", "java", "javascript", "typescript", "c++", "c#", "go",
        "rust", "sql", "r", "scala", "kotlin", "swift", "php", "ruby",
        "matlab", "bash",
    ],
    "ml_ai": [
        "machine learning", "deep learning", "natural language processing",
        "computer vision", "llm", "large language models", "transformers",
        "pytorch", "tensorflow", "keras", "scikit-learn", "sklearn",
        "hugging face", "huggingface", "openai", "langchain", "spacy", "nltk",
        "bert", "gpt", "rag", "prompt engineering", "generative ai",
    ],
    "data": [
        "pandas", "numpy", "matplotlib", "seaborn", "power bi", "tableau",
        "spark", "hadoop", "airflow", "etl", "data warehousing", "big data",
    ],
    "web_backend": [
        "django", "flask", "fastapi", "react", "angular", "vue", "node.js",
        "nodejs", "express", "rest api", "graphql", "microservices",
    ],
    "cloud_devops": [
        "aws", "azure", "gcp", "docker", "kubernetes", "ci/cd", "jenkins",
        "terraform", "git", "github actions", "linux",
    ],
    "databases": [
        "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
        "dynamodb", "oracle", "cassandra",
    ],
    "soft_skills": [
        "leadership", "communication", "teamwork", "problem solving",
        "project management", "agile", "scrum", "stakeholder management",
        "mentoring", "cross-functional collaboration",
    ],
}

# Aliases mapped to a canonical skill name
ALIASES = {
    "js": "javascript",
    "ts": "typescript",
    "nlp": "natural language processing",
    "sklearn": "scikit-learn",
    "huggingface": "hugging face",
    "nodejs": "node.js",
    "ml": "machine learning",
    "dl": "deep learning",
    "cv": "computer vision",
}

_ALL_SKILLS = {skill for group in SKILLS_DB.values() for skill in group}
_SKILL_TO_CATEGORY = {
    skill: category for category, group in SKILLS_DB.items() for skill in group
}


def _build_pattern(skill: str) -> re.Pattern:
    # Escape regex special chars (e.g. "c++"), then apply word boundaries.
    # For skills containing non-word chars like "c++" or "node.js",
    # boundaries are relaxed on that side.
    escaped = re.escape(skill)
    return re.compile(rf"(?<![\w+#.]){escaped}(?![\w+#.])", re.IGNORECASE)


_PATTERNS = {skill: _build_pattern(skill) for skill in _ALL_SKILLS}


def extract_skills(text: str) -> dict:
    """Extract technical/soft skills found in `text`.

    Returns:
        dict with keys:
          - "flat": sorted list of canonical skill names found
          - "by_category": dict[category] -> sorted list of skills
    """
    text_lower = text.lower()
    found = set()

    for skill, pattern in _PATTERNS.items():
        if pattern.search(text_lower):
            found.add(skill)

    # Resolve short aliases separately (e.g. standalone "js", "ml")
    for alias, canonical in ALIASES.items():
        pattern = re.compile(rf"(?<![\w+#.])" + re.escape(alias) + r"(?![\w+#.])", re.IGNORECASE)
        if pattern.search(text_lower):
            found.add(canonical)

    by_category = defaultdict(list)
    for skill in found:
        category = _SKILL_TO_CATEGORY.get(skill, "other")
        by_category[category].append(skill)

    for category in by_category:
        by_category[category].sort()

    return {
        "flat": sorted(found),
        "by_category": dict(by_category),
    }


def skill_gap(resume_skills: set, jd_skills: set) -> list:
    """Skills required by the JD but missing from the resume."""
    return sorted(jd_skills - resume_skills)
