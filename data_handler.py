"""
data_handler.py
---------------
Phase 2: Data Engineering

Responsibilities:
  - Define the authoritative subject → cluster mapping
  - Load and validate training_data.csv
  - Append new student rows submitted via survey_train
  - Preprocess raw data into features (X) and labels (y) for the ML engine
"""

import csv
import os
import uuid
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Subject → Cluster Mapping
#    Single source of truth — mirrors SPEC.md Section 7.
# ---------------------------------------------------------------------------
SUBJECT_CLUSTER_MAP: dict[str, str] = {
    # English
    "English Advanced":         "English",
    "English Extension":        "English",
    "English Standard":         "English",

    # HSIE
    "Ancient History":          "HSIE",
    "Biblical Studies":         "HSIE",
    "Business Studies":         "HSIE",
    "Commerce":                 "HSIE",
    "Community and Family Studies": "HSIE",
    "Economics":                "HSIE",
    "Geography":                "HSIE",
    "Legal Studies":            "HSIE",
    "School of Languages":      "HSIE",
    "Society and Culture":      "HSIE",

    # Maths
    "Mathematics Advanced":     "Maths",
    "Mathematics Extension 1":  "Maths",
    "Mathematics Extension 2":  "Maths",
    "Mathematics Standard":     "Maths",

    # PDHPE
    "Health and Movement Science": "PDHPE",

    # Science
    "Biology":                  "Science",
    "Chemistry":                "Science",
    "Physics":                  "Science",

    # TAS
    "Software Engineering":     "TAS",
    "Design and Technology":    "TAS",
    "Engineering Studies":      "TAS",
    "Food Technology":          "TAS",
    "Hospitality":              "TAS",
    "Industrial Technology Multimedia": "TAS",
    "Industrial Technology Timber":     "TAS",

    # Visual and Performing Arts
    "Drama":                    "Visual and Performing Arts",
    "Music":                    "Visual and Performing Arts",
    "Visual Arts":              "Visual and Performing Arts",
}

ALL_CLUSTERS: list[str] = sorted(set(SUBJECT_CLUSTER_MAP.values()))

CSV_PATH = os.path.join(os.path.dirname(__file__), "training_data.csv")

CSV_COLUMNS = [
    "student_id",
    "q1", "q2", "q3", "q4", "q5", "q6",
    "q7", "q8", "q9", "q10", "q11", "q12",
    "target_subject_1", "target_subject_2", "target_subject_3",
    "target_subject_4", "target_subject_5", "target_subject_6",
    "target_subject_7",
    "satisfaction_score",
]


# ---------------------------------------------------------------------------
# 2. Cluster Mapping Helpers
# ---------------------------------------------------------------------------

def subject_to_cluster(subject: str) -> str | None:
    """Return the cluster name for a given subject, or None if unknown."""
    return SUBJECT_CLUSTER_MAP.get(subject.strip(), None)


def subjects_to_clusters(subjects: list[str]) -> list[str]:
    """
    Map a list of subject names to their clusters.
    Unknown subjects are silently skipped.
    Returns the most-frequent cluster (mode) as the primary label.
    """
    return [SUBJECT_CLUSTER_MAP[s] for s in subjects if s in SUBJECT_CLUSTER_MAP]


# When cluster counts tie, more distinctive/technical clusters take priority
# over background subjects (English, Maths) that almost every student takes.
_CLUSTER_PRIORITY: dict[str, int] = {
    "TAS":                      7,
    "Science":                  6,
    "PDHPE":                    5,
    "Visual and Performing Arts": 4,
    "HSIE":                     3,
    "Maths":                    2,
    "English":                  1,
}


def majority_cluster(subjects: list[str]) -> str | None:
    """
    Determine the dominant cluster from a student's subject list.
    Uses frequency count; ties broken by _CLUSTER_PRIORITY so that
    technical/specialist clusters (TAS, Science) beat background ones.
    Returns None if no subjects can be mapped.
    """
    clusters = subjects_to_clusters(subjects)
    if not clusters:
        return None
    max_count = max(clusters.count(c) for c in set(clusters))
    candidates = [c for c in set(clusters) if clusters.count(c) == max_count]
    return max(candidates, key=lambda c: _CLUSTER_PRIORITY.get(c, 0))


# ---------------------------------------------------------------------------
# 3. CSV I/O
# ---------------------------------------------------------------------------

def load_training_data() -> pd.DataFrame:
    """
    Load training_data.csv and return a clean DataFrame.
    Raises FileNotFoundError if the file is missing.
    """
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"training_data.csv not found at {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, dtype=str)
    # Strip whitespace from all string columns
    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)
    return df


def append_training_row(
    answers: dict,
    target_subjects: list[str],
    satisfaction: int,
) -> str:
    """
    Append one student's training submission to training_data.csv.

    Parameters
    ----------
    answers       : dict  — keys q1..q12, values are the raw form strings
    target_subjects: list — list of selected subject names (6-7 items)
    satisfaction  : int   — satisfaction score 1-10

    Returns
    -------
    str — the generated student_id for this record
    """
    # Pad subjects list to 7 slots (last slot is optional)
    subjects_padded = (target_subjects + [""] * 7)[:7]

    # Generate a unique student ID
    student_id = "S" + uuid.uuid4().hex[:6].upper()

    row = {
        "student_id": student_id,
        **{f"q{i}": answers.get(f"q{i}", "") for i in range(1, 13)},
        "target_subject_1": subjects_padded[0],
        "target_subject_2": subjects_padded[1],
        "target_subject_3": subjects_padded[2],
        "target_subject_4": subjects_padded[3],
        "target_subject_5": subjects_padded[4],
        "target_subject_6": subjects_padded[5],
        "target_subject_7": subjects_padded[6],
        "satisfaction_score": str(satisfaction),
    }

    file_exists = os.path.exists(CSV_PATH)
    
    # Safely check if the file is missing a trailing newline before appending
    if file_exists and os.path.getsize(CSV_PATH) > 0:
        with open(CSV_PATH, "rb") as read_f:
            read_f.seek(-1, os.SEEK_END)
            needs_newline = read_f.read(1) != b'\n'
    else:
        needs_newline = False

    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        if needs_newline:
            f.write("\n")
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return student_id


# ---------------------------------------------------------------------------
# 4. Feature Preprocessing
# ---------------------------------------------------------------------------

def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Convert raw training CSV into numeric feature matrix X and label vector y.

    Steps:
      1. Cast q1-q12 to int (fill missing with median)
      2. Derive label y = majority_cluster from target_subject_1..7
      3. Drop rows where y cannot be determined

    Returns
    -------
    X : DataFrame  — shape (n_samples, 12), columns q1..q12
    y : Series     — cluster labels, e.g. "Maths", "HSIE", ...
    """
    feature_cols = [f"q{i}" for i in range(1, 13)]
    subject_cols = [f"target_subject_{i}" for i in range(1, 8)]

    df = df.copy()

    # Cast features to numeric
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    # Derive cluster label for each row
    def _row_cluster(row):
        subjects = [row[c] for c in subject_cols if pd.notna(row[c]) and str(row[c]).strip()]
        return majority_cluster(subjects)

    df["cluster"] = df.apply(_row_cluster, axis=1)

    # Drop rows with no determinable cluster
    df = df.dropna(subset=["cluster"])

    X = df[feature_cols].astype(float)
    y = df["cluster"]

    return X, y


def get_record_count() -> int:
    """Return the number of student records currently in training_data.csv."""
    try:
        df = load_training_data()
        return len(df)
    except FileNotFoundError:
        return 0
