"""Render an analysis result dict as plain text or a self-contained HTML report."""

from datetime import datetime


def to_text(result: dict) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("AI RESUME ANALYZER — REPORT")
    lines.append("=" * 60)
    lines.append(f"Match Score: {result['similarity_score']}%")
    lines.append("")
    lines.append("Matched Skills:")
    lines.append("  " + (", ".join(result["resume_skills"]) or "None detected"))
    lines.append("")
    lines.append("Missing Skills (present in JD, not in resume):")
    lines.append("  " + (", ".join(result["missing_skills"]) or "None — great coverage!"))
    lines.append("")
    lines.append("ATS Formatting Checks:")
    for f in result["ats_findings"]:
        lines.append(f"  • {f}")
    lines.append("")
    lines.append("Personalized Suggestions:")
    for s in result["suggestions"]:
        lines.append(f"  • {s}")
    lines.append("=" * 60)
    return "\n".join(lines)


def to_html(result: dict) -> str:
    score = result["similarity_score"]
    color = "#16a34a" if score >= 70 else "#d97706" if score >= 40 else "#dc2626"

    def li(items):
        return "\n".join(f"<li>{item}</li>" for item in items) or "<li>None</li>"

    skill_chips = "".join(
        f'<span class="chip chip-match">{s}</span>' for s in result["resume_skills"]
    ) or '<span class="muted">None detected</span>'
    missing_chips = "".join(
        f'<span class="chip chip-missing">{s}</span>' for s in result["missing_skills"]
    ) or '<span class="muted">None — great coverage!</span>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AI Resume Analyzer Report</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 760px;
         margin: 40px auto; padding: 0 20px; color: #1f2937; background: #f9fafb; }}
  h1 {{ font-size: 1.5rem; }}
  .score-box {{ background: white; border-radius: 12px; padding: 24px; margin: 16px 0;
                box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  .score {{ font-size: 2.5rem; font-weight: 700; color: {color}; }}
  .chip {{ display: inline-block; padding: 4px 10px; margin: 3px; border-radius: 999px;
           font-size: 0.85rem; }}
  .chip-match {{ background: #dcfce7; color: #166534; }}
  .chip-missing {{ background: #fee2e2; color: #991b1b; }}
  .muted {{ color: #6b7280; }}
  ul {{ padding-left: 20px; }}
  li {{ margin-bottom: 6px; }}
  .footer {{ color: #9ca3af; font-size: 0.8rem; margin-top: 30px; text-align: center; }}
</style>
</head>
<body>
  <h1>🧠 AI Resume Analyzer Report</h1>

  <div class="score-box">
    <div>Match Score vs. Job Description</div>
    <div class="score">{score}%</div>
  </div>

  <div class="score-box">
    <h3>✅ Matched Skills</h3>
    {skill_chips}
  </div>

  <div class="score-box">
    <h3>⚠️ Missing Skills</h3>
    {missing_chips}
  </div>

  <div class="score-box">
    <h3>📋 ATS Formatting Checks</h3>
    <ul>{li(result['ats_findings'])}</ul>
  </div>

  <div class="score-box">
    <h3>💡 Personalized Suggestions</h3>
    <ul>{li(result['suggestions'])}</ul>
  </div>

  <div class="footer">Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} · AI Resume Analyzer</div>
</body>
</html>
"""
