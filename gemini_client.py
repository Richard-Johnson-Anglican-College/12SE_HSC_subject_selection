"""
gemini_client.py
----------------
Phase 4: Gemini API Integration

Generates a personalised, student-friendly natural language summary
explaining why the ML model recommended a particular subject cluster.

Falls back to a clean template string if the API is unavailable or the key
is missing — so the app keeps working without internet access.
"""

import google.generativeai as genai

# ---------------------------------------------------------------------------
# Load API key
# ---------------------------------------------------------------------------
_CONFIGURED = False

def _configure():
    global _CONFIGURED
    if _CONFIGURED:
        return True
    try:
        from config import GEMINI_API_KEY
        if GEMINI_API_KEY and not GEMINI_API_KEY.startswith("YOUR_"):
            genai.configure(api_key=GEMINI_API_KEY)
            _CONFIGURED = True
            return True
    except (ImportError, Exception):
        pass
    return False


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def generate_summary(
    cluster: str,
    match_pct: int,
    subjects: list[dict],
    answers: dict,
) -> str:
    """
    Call Gemini to generate a 2-3 sentence, student-friendly explanation
    of the recommendation. Returns a fallback string if the API fails.

    Parameters
    ----------
    cluster    : e.g. "Maths"
    match_pct  : e.g. 82
    subjects   : list of {name, description} dicts
    answers    : raw survey answers {q1..q12}
    """
    if not _configure():
        return _fallback(cluster, match_pct, subjects)

    subject_names = ", ".join(s["name"] for s in subjects[:4])
    
    # Map question numbers to descriptive labels for the prompt
    q_labels = {
        "q1":  f"logic puzzle interest {answers.get('q1', 5)}/10",
        "q2":  f"love of writing/literature {answers.get('q2', 3)}/5",
        "q3":  f"enjoys hands-on making {'yes' if str(answers.get('q3', 0)) == '1' else 'no'}",
        "q4":  f"performance/spotlight comfort {answers.get('q4', 5)}/10",
        "q5":  f"interest in biology/body {answers.get('q5', 3)}/5",
        "q6":  f"interested in business {'yes' if str(answers.get('q6', 0)) == '1' else 'no'}",
        "q7":  f"coding/tech interest {answers.get('q7', 5)}/10",
        "q8":  f"visual arts preference {answers.get('q8', 3)}/5",
        "q9":  f"interest in history/social issues {answers.get('q9', 5)}/10",
        "q10": f"enjoys science experiments {'yes' if str(answers.get('q10', 0)) == '1' else 'no'}",
        "q11": f"culinary/hospitality interest {answers.get('q11', 3)}/5",
        "q12": f"systems thinking {answers.get('q12', 5)}/10",
    }
    profile_summary = "; ".join(q_labels.values())

    prompt = f"""You are a friendly HSC subject advisor writing to an Australian Year 10 student.

A machine learning model has recommended the '{cluster}' subject cluster for this student (match score: {match_pct}%).
The top subjects in this cluster include: {subject_names}.

The student's interest profile: {profile_summary}.

Write a warm, encouraging 2-3 sentence explanation (in plain Australian English, no jargon) of:
1. Why their specific interests suggest this cluster suits them.
2. What kind of person tends to thrive in this area.

Do NOT use the word "algorithm" or "machine learning". Write directly to the student (use "you").
Keep it under 60 words."""

    try:
        model = genai.GenerativeModel("gemini-flash-latest")
        response = model.generate_content(prompt)
        text = response.text.strip()
        # Safety: ensure we got something reasonable back
        if len(text) > 20:
            return text
    except Exception:
        pass

    return _fallback(cluster, match_pct, subjects)


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------

def _fallback(cluster: str, match_pct: int, subjects: list[dict]) -> str:
    top = subjects[0]["name"] if subjects else cluster
    return (
        f"Your interest profile closely matches students who have chosen {cluster} subjects. "
        f"Students with a similar profile to yours tend to do well in areas like {top}. "
        f"This recommendation reflects patterns from real student data collected at your school."
    )
