"""High-level ResumeAnalyzer that ties extraction, NLP, similarity and
LLM-based suggestions together into a single easy-to-use API.
"""

from .text_extraction import extract_text
from .skills_extractor import extract_skills, skill_gap
from .similarity import compute_similarity
from .ats_suggestions import rule_based_checks, llm_suggestions
from . import report as report_mod


class ResumeAnalyzer:
    def __init__(self, use_embeddings: bool = False, llm_provider: str = "auto"):
        """
        Args:
            use_embeddings: use sentence-transformer embeddings for semantic
                similarity instead of TF-IDF (requires internet access to
                download the model on first use).
            llm_provider: "auto" | "openai" | "huggingface" | "none".
                Controls which LLM backend generates personalized
                suggestions. "auto" uses OPENAI_API_KEY / HF_API_TOKEN env
                vars if present, otherwise falls back to rule-based
                suggestions automatically.
        """
        self.use_embeddings = use_embeddings
        self.llm_provider = llm_provider

    def analyze(self, resume_path: str = None, jd_path: str = None,
                resume_text: str = None, jd_text: str = None) -> dict:
        """Run the full analysis pipeline.

        Provide either a file path (resume_path/jd_path) or raw text
        (resume_text/jd_text) for each document.
        """
        resume_text = resume_text or extract_text(resume_path)
        jd_text = jd_text or extract_text(jd_path)

        if not resume_text.strip() or not jd_text.strip():
            raise ValueError("Resume and job description text must not be empty.")

        similarity_score = compute_similarity(
            resume_text, jd_text, use_embeddings=self.use_embeddings
        )

        resume_skills = extract_skills(resume_text)
        jd_skills = extract_skills(jd_text)

        missing = skill_gap(set(resume_skills["flat"]), set(jd_skills["flat"]))

        ats_findings = rule_based_checks(resume_text)

        suggestions = llm_suggestions(
            resume_text, jd_text, missing, similarity_score, provider=self.llm_provider
        )

        return {
            "similarity_score": similarity_score,
            "resume_skills": resume_skills["flat"],
            "resume_skills_by_category": resume_skills["by_category"],
            "jd_skills": jd_skills["flat"],
            "missing_skills": missing,
            "ats_findings": ats_findings,
            "suggestions": suggestions,
        }

    @staticmethod
    def to_text(result: dict) -> str:
        return report_mod.to_text(result)

    @staticmethod
    def to_html(result: dict) -> str:
        return report_mod.to_html(result)
