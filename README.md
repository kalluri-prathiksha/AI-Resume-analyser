# AI Resume Analyzer

An intelligent resume analyzer built with **Python, NLP, and LLMs**. It
compares a resume against a job description using semantic similarity,
extracts technical skills with NLP techniques, and generates ATS
optimization suggestions and personalized recommendations using
OpenAI / Hugging Face models.

## Features

- **Multi-format parsing** — reads `.pdf`, `.docx`, and `.txt` resumes/JDs.
- **Semantic similarity scoring** — TF-IDF + cosine similarity by default
  (fully offline), or optional sentence-transformer embeddings for
  higher-fidelity matching.
- **NLP skill extraction** — regex/taxonomy-based extraction across 7
  categories (languages, ML/AI, data, web/backend, cloud/devops,
  databases, soft skills), with alias resolution (e.g. `js` → `javascript`).
- **Skill gap analysis** — skills the job description wants that your
  resume doesn't mention.
- **ATS formatting checks** — length, contact info, section headers,
  quantified achievements, action verbs, table/tab usage.
- **LLM-powered personalized suggestions** — via OpenAI or Hugging Face
  Inference API when a key is configured; otherwise a solid rule-based
  fallback runs automatically so the tool always works offline.
- **Reports** — plain text (console) and a self-contained HTML report.

## Project Structure

```
ai-resume-analyzer/
├── resume_analyzer/
│   ├── __init__.py
│   ├── analyzer.py          # ResumeAnalyzer orchestration class
│   ├── text_extraction.py   # PDF/DOCX/TXT parsing
│   ├── skills_extractor.py  # Skills taxonomy + NLP extraction
│   ├── similarity.py        # TF-IDF / embedding similarity scoring
│   ├── ats_suggestions.py   # Rule-based checks + LLM suggestions
│   └── report.py            # Text / HTML report rendering
├── cli.py                   # Command-line interface
├── sample_data/             # Example resume + job description
├── reports/                 # Generated HTML reports land here
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

Core dependencies (`scikit-learn`, `pdfplumber`, `python-docx`) are
required. `sentence-transformers`, `openai`, and `requests` are optional
and only needed if you use `--embeddings` or LLM-powered suggestions.

## Usage

### CLI

```bash
python cli.py --resume sample_data/resume_sample.txt \
              --jd sample_data/job_description_sample.txt \
              --html reports/my_report.html
```

Options:

| Flag           | Description                                                        |
|----------------|----------------------------------------------------------------------|
| `--resume`     | Path to resume (`.pdf`, `.docx`, `.txt`)                             |
| `--jd`         | Path to job description                                              |
| `--html`       | Write an HTML report to this path                                    |
| `--llm`        | `auto` (default) \| `openai` \| `huggingface` \| `none`              |
| `--embeddings` | Use sentence-transformer embeddings instead of TF-IDF                |

### Enable LLM-powered suggestions

```bash
export OPENAI_API_KEY=sk-...       # for OpenAI
# or
export HF_API_TOKEN=hf_...         # for Hugging Face Inference API
```

With no key set, the tool automatically falls back to rule-based
suggestions — no configuration needed to get started.

### As a library

```python
from resume_analyzer import ResumeAnalyzer

analyzer = ResumeAnalyzer()
result = analyzer.analyze(
    resume_path="resume.pdf",
    jd_path="job_description.txt",
)

print(result["similarity_score"])   # e.g. 62.4
print(result["missing_skills"])     # e.g. ["docker", "aws"]
print(result["suggestions"])        # personalized recommendations

# Reports
open("report.html", "w").write(analyzer.to_html(result))
```

## How it works

1. **Extraction** — pulls raw text out of the resume/JD regardless of format.
2. **Similarity** — vectorizes both documents (TF-IDF by default) and
   computes cosine similarity as a 0–100% match score.
3. **Skill extraction** — scans both texts against a curated skills
   taxonomy using word-boundary regex matching (handles multi-word
   skills like "machine learning" and symbol-heavy ones like "c++" /
   "node.js").
4. **Gap analysis** — set difference between JD skills and resume skills.
5. **ATS checks** — heuristic rules covering common ATS pitfalls
   (missing contact info, no quantifiable metrics, weak bullet phrasing,
   table-heavy formatting).
6. **Suggestions** — an LLM (OpenAI `gpt-4o-mini` or a Hugging Face
   Inference model) is prompted with the resume, JD, and skill gaps to
   produce concrete, actionable suggestions; falls back to a rule-based
   summary when no LLM is configured or reachable.

## Extending

- Add more skills/aliases in `resume_analyzer/skills_extractor.py`.
- Add more ATS heuristics in `resume_analyzer/ats_suggestions.py`.
- Swap in a different LLM provider by adding a new `_<provider>_suggestions`
  function in `ats_suggestions.py`.
