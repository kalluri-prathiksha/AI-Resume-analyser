"""ATS optimization checks and personalized recommendations.

Two layers of suggestions are produced:
  1. Rule-based ATS checks (always run, fully offline).
  2. LLM-generated personalized suggestions (OpenAI or Hugging Face),
     used automatically when an API key is configured; otherwise a
     rule-based fallback is used so the tool always works offline.
"""

import os
import re

ACTION_VERBS = {
    "built", "led", "managed", "designed", "developed", "created",
    "implemented", "optimized", "launched", "improved", "automated",
    "architected", "reduced", "increased", "delivered", "spearheaded",
    "engineered", "analyzed", "streamlined",
}

SECTION_HEADERS = ["experience", "education", "skills", "projects", "summary"]


def rule_based_checks(resume_text: str) -> list:
    """Run offline heuristic ATS checks and return a list of findings."""
    findings = []
    text_lower = resume_text.lower()
    word_count = len(resume_text.split())

    # Length check
    if word_count < 150:
        findings.append(
            "Resume looks short (%d words). Add more detail on projects and impact." % word_count
        )
    elif word_count > 1000:
        findings.append(
            "Resume is long (%d words). Consider trimming to 1-2 pages for ATS and recruiter readability."
            % word_count
        )

    # Contact info
    has_email = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", resume_text)
    has_phone = re.search(r"(\+?\d[\d\-\s()]{8,}\d)", resume_text)
    if not has_email:
        findings.append("No email address detected — make sure contact info is machine-readable text, not an image.")
    if not has_phone:
        findings.append("No phone number detected — add one so ATS systems can parse your contact details.")

    # Section headers
    missing_sections = [s for s in SECTION_HEADERS if s not in text_lower]
    if missing_sections:
        findings.append(
            "Consider adding clear section headers for: " + ", ".join(missing_sections) + "."
        )

    # Quantifiable achievements
    numbers = re.findall(r"\b\d+%?\b", resume_text)
    if len(numbers) < 3:
        findings.append(
            "Few quantifiable results found. Add metrics (%, $, time saved, users, scale) to strengthen impact statements."
        )

    # Action verbs at start of bullets
    bullet_lines = [l.strip("•-* \t") for l in resume_text.splitlines() if l.strip()]
    strong_starts = sum(
        1 for line in bullet_lines
        if line.split(" ")[0].lower().strip(".,") in ACTION_VERBS
    )
    if bullet_lines and strong_starts < max(1, len(bullet_lines) // 5):
        findings.append(
            "Start more bullet points with strong action verbs (e.g. Built, Led, Optimized, Automated)."
        )

    # File-format style checks (tables/images break many ATS parsers)
    if "\t" in resume_text and resume_text.count("\t") > 20:
        findings.append(
            "Heavy use of tabs/tables detected — some ATS parsers mis-read tables; prefer simple linear formatting."
        )

    if not findings:
        findings.append("Resume passes basic ATS formatting checks.")

    return findings


def _fallback_llm_suggestions(missing_skills: list, similarity_score: float) -> list:
    """Rule-based stand-in used when no LLM API key is configured."""
    suggestions = []
    if missing_skills:
        top_missing = missing_skills[:8]
        suggestions.append(
            "Job description emphasizes skills not clearly present in your resume: "
            + ", ".join(top_missing)
            + ". If you have experience with these, add them explicitly using the same terminology as the JD."
        )
    if similarity_score < 40:
        suggestions.append(
            "Overall semantic match with this job description is low (%.1f%%). Mirror key phrases and priorities "
            "from the job description in your summary and bullet points." % similarity_score
        )
    elif similarity_score < 70:
        suggestions.append(
            "Match is moderate (%.1f%%). Tailor 2-3 bullet points to more directly reflect the responsibilities "
            "in this job description." % similarity_score
        )
    else:
        suggestions.append(
            "Strong semantic match (%.1f%%) with this job description — keep this level of alignment." % similarity_score
        )
    suggestions.append(
        "Quantify achievements wherever possible (e.g. 'Reduced processing time by 30%' rather than 'Improved processing time')."
    )
    return suggestions


def llm_suggestions(
    resume_text: str,
    jd_text: str,
    missing_skills: list,
    similarity_score: float,
    provider: str = "auto",
) -> list:
    """Generate personalized suggestions using an LLM when available.

    provider: "openai" | "huggingface" | "auto" (auto-detects based on
              env vars OPENAI_API_KEY / HF_API_TOKEN) | "none" (force
              rule-based fallback).

    Falls back to a rule-based summary if no API key is set or the call
    fails for any reason (e.g. no network access), so this function is
    always safe to call.
    """
    if provider == "none":
        return _fallback_llm_suggestions(missing_skills, similarity_score)

    if provider in ("auto", "openai") and os.environ.get("OPENAI_API_KEY"):
        try:
            return _openai_suggestions(resume_text, jd_text, missing_skills)
        except Exception:
            pass

    if provider in ("auto", "huggingface") and os.environ.get("HF_API_TOKEN"):
        try:
            return _huggingface_suggestions(resume_text, jd_text, missing_skills)
        except Exception:
            pass

    return _fallback_llm_suggestions(missing_skills, similarity_score)


def _openai_suggestions(resume_text: str, jd_text: str, missing_skills: list) -> list:
    from openai import OpenAI

    client = OpenAI()
    prompt = f"""You are an expert resume coach and ATS specialist.

Job Description:
{jd_text[:3000]}

Resume:
{resume_text[:3000]}

Skills present in the job description but not clearly found in the resume: {', '.join(missing_skills) or 'none'}

Give 4-6 short, specific, actionable bullet-point suggestions to improve this
resume's ATS score and alignment with the job description. Be concrete
(mention exact skills/phrasing to add) and concise (one sentence each)."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=500,
    )
    content = response.choices[0].message.content
    return [line.strip("-• \t") for line in content.splitlines() if line.strip()]


def _huggingface_suggestions(resume_text: str, jd_text: str, missing_skills: list) -> list:
    import requests

    token = os.environ["HF_API_TOKEN"]
    model = os.environ.get("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
    prompt = f"""[INST] You are an ATS resume coach. Job description: {jd_text[:2000]}
Resume: {resume_text[:2000]}
Missing skills: {', '.join(missing_skills) or 'none'}
Give 4-6 short actionable bullet-point suggestions to improve ATS alignment. [/INST]"""

    resp = requests.post(
        f"https://api-inference.huggingface.co/models/{model}",
        headers={"Authorization": f"Bearer {token}"},
        json={"inputs": prompt, "parameters": {"max_new_tokens": 400}},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    text = data[0]["generated_text"] if isinstance(data, list) else str(data)
    return [line.strip("-• \t") for line in text.splitlines() if line.strip()]
