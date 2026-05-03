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
    cluster_scores: list[dict] | None = None,
) -> str:
    """
    Call Gemini to generate a 2-3 sentence, student-friendly explanation
    of the recommendation. Returns a fallback string if the API fails.

    Parameters
    ----------
    cluster        : e.g. "Science"
    match_pct      : e.g. 82
    subjects       : full list of recommended subjects across ALL clusters
    answers        : raw survey answers {q1..q12}
    cluster_scores : list of {cluster, pct} dicts for all shown clusters
    """
    if not _configure():
        return _fallback(cluster, match_pct, subjects)

    # Group all recommended subjects by cluster for the prompt
    clusters_seen = []
    subjects_by_cluster: dict[str, list[str]] = {}
    for s in subjects:
        c = s.get("cluster", "")
        if c not in subjects_by_cluster:
            subjects_by_cluster[c] = []
            clusters_seen.append(c)
        subjects_by_cluster[c].append(s["name"])

    subject_block = "\n".join(
        f"  • {c}: {', '.join(names)}"
        for c, names in subjects_by_cluster.items()
        if c != "English"
    )

    # Build a readable affinity summary
    affinity_lines = ""
    if cluster_scores:
        affinity_lines = ", ".join(
            f"{cs['cluster']} {cs['pct']}%"
            for cs in cluster_scores
            if cs.get("pct", 0) > 0
        )

    # Only highlight Q values that are notably high (top third of their scale)
    highlights = []
    q_map = [
        ("q1",  int(answers.get("q1",  5)), 10, "logic and maths"),
        ("q2",  int(answers.get("q2",  5)), 10, "writing and literature"),
        ("q3",  int(answers.get("q3",  0)),  1, "hands-on building and making"),
        ("q4",  int(answers.get("q4",  5)), 10, "performance and creative expression"),
        ("q5",  int(answers.get("q5",  5)), 10, "biology and the human body"),
        ("q7",  int(answers.get("q7",  5)), 10, "coding and technology"),
        ("q8",  int(answers.get("q8",  5)), 10, "visual arts and design"),
        ("q9",  int(answers.get("q9",  5)), 10, "society, history, and law"),
        ("q10", int(answers.get("q10", 0)),  1, "science experiments and lab work"),
        ("q12", int(answers.get("q12", 5)), 10, "systems thinking and engineering"),
    ]
    for key, val, scale, label in q_map:
        if scale == 1 and val == 1:
            highlights.append(label)
        elif scale > 1 and val / scale >= 0.7:
            highlights.append(f"{label} ({val}/{scale})")

    strengths = "; ".join(highlights) if highlights else "a broad range of interests"

    prompt = f"""You are a friendly HSC subject advisor writing to an Australian Year 10 student.

### WHAT THE MODEL FOUND ###
Primary strength area: {cluster} ({match_pct}% match)
NOTE: {match_pct}% means {match_pct} out of 100 of this student's interest answers pointed toward {cluster} subjects.
Subject area affinities: {affinity_lines}

Recommended subjects across ALL areas:
{subject_block}

### STUDENT'S KEY INTERESTS ###
{strengths}

### YOUR TASK ###
Write a warm, encouraging 4-5 sentence explanation directly to the student (use "you") that covers ALL of the following:

1. WHY these specific subjects suit them — connect at least 2 of their high-scoring interests to the subjects recommended (e.g. "your love of systems thinking is exactly what Engineering Studies is built around").
2. WHY the combination makes sense — explain how subjects from different areas complement each other (e.g. Physics gives the theory, Engineering Studies applies it, Software Engineering automates it).
3. WHAT KIND OF STUDENT thrives here — paint a picture of the learner type, not just the subjects (e.g. "students who like to understand why something works, not just that it works").
4. A FORWARD-LOOKING sentence — one concrete example of where this combination could lead (a career, a uni degree, or a project type) without being prescriptive.
5. In one sentence, mention that {match_pct}% of their answers pointed toward {cluster} — phrase it like "around X in every 10 answers" so the number feels human, not like a test score. Make clear it reflects interest alignment, not ability.

Rules:
- Plain Australian English, no jargon.
- Do NOT say "algorithm" or "machine learning".
- Do NOT invent subjects not listed above.
- Use "you" and "your" throughout — speak directly to the student.
- Maximum 120 words."""

    try:
        model = genai.GenerativeModel("gemma-3n-e4b-it")
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
