"""AI Resume Analyzer
A lightweight NLP + LLM powered tool that scores a resume against a job
description, extracts technical skills, finds gaps, and produces ATS
optimization suggestions.
"""

from .analyzer import ResumeAnalyzer

__all__ = ["ResumeAnalyzer"]
__version__ = "1.0.0"
