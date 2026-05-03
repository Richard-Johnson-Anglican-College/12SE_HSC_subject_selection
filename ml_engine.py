"""
ml_engine.py
------------
Phase 3: Machine Learning Engine

Pipeline:
  1. MinMaxScaler    — normalise q1-q12 to [0, 1]
  2. RandomForestClassifier — predict Subject Cluster (label)
  3. PolynomialFeatures (degree=2) + LinearRegression — predict satisfaction
  4. Serialise all artefacts to models/

Public API:
  train()           -> dict  (accuracy, r2, record_count, timestamp)
  predict(answers)  -> dict  (cluster, match_pct, subjects)
  is_trained()      -> bool
  get_model_info()  -> dict
  retrain()         -> dict  (alias for train, used by admin route)
"""

import os
import json
import pickle
import datetime
import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')           # Non-interactive backend — no GUI needed
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

import data_handler as dh

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(__file__)
MODEL_DIR  = os.path.join(BASE_DIR, "models")
CHART_DIR  = os.path.join(BASE_DIR, "static", "charts")

SCALER_PATH     = os.path.join(MODEL_DIR, "scaler.pkl")
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "classifier.pkl")
POLY_PATH       = os.path.join(MODEL_DIR, "poly.pkl")
REGRESSOR_PATH  = os.path.join(MODEL_DIR, "regressor.pkl")
INFO_PATH       = os.path.join(MODEL_DIR, "model_info.json")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)

FEATURE_NAMES = [f"q{i}" for i in range(1, 13)]

# ---------------------------------------------------------------------------
# In-memory model cache — avoids 4 disk reads on every prediction request
# ---------------------------------------------------------------------------
_CACHE: dict = {}


def _get_models() -> dict:
    """Return cached models, loading from disk only on first call after (re)train."""
    global _CACHE
    if not _CACHE:
        _CACHE = {
            'scaler': _load(SCALER_PATH),
            'clf':    _load(CLASSIFIER_PATH),
            'poly':   _load(POLY_PATH),
            'reg':    _load(REGRESSOR_PATH),
        }
    return _CACHE


def _clear_cache() -> None:
    """Invalidate cache after retraining so next predict() reloads fresh models."""
    global _CACHE
    _CACHE = {}


# ---------------------------------------------------------------------------
# 1. Train
# ---------------------------------------------------------------------------

def train() -> dict:
    """
    Load training_data.csv, fit all models, save to disk,
    generate visualisation charts, and return a summary dict.
    """
    df = dh.load_training_data()
    X, y = dh.prepare_features(df)

    if len(X) < 10:
        raise ValueError("Not enough training records (need at least 10).")

    # --- Scaler ---
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Classifier: 80/20 split ---
    can_stratify = y.value_counts().min() >= 2
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y if can_stratify else None
    )
    clf = RandomForestClassifier(n_estimators=50, max_depth=8, min_samples_leaf=2, max_features='sqrt', random_state=42)
    clf.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, clf.predict(X_test))

    # --- Regressor: satisfaction score ---
    sat_col = "satisfaction_score"
    df_reg = df.copy()
    df_reg[sat_col] = pd.to_numeric(df_reg[sat_col], errors="coerce")
    df_reg = df_reg.dropna(subset=[sat_col])

    X_reg_raw = df_reg[FEATURE_NAMES].apply(pd.to_numeric, errors='coerce').fillna(5)
    X_reg_scaled = scaler.transform(X_reg_raw)
    y_reg = df_reg[sat_col].astype(float)

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X_reg_scaled)
    reg = Ridge(alpha=10)   # Regularised — prevents wild extrapolation on sparse polynomial features
    reg.fit(X_poly, y_reg)
    y_pred_reg = reg.predict(X_poly)
    r2 = r2_score(y_reg, y_pred_reg)

    # --- Persist ---
    _save(scaler,     SCALER_PATH)
    _save(clf,        CLASSIFIER_PATH)
    _save(poly,       POLY_PATH)
    _save(reg,        REGRESSOR_PATH)
    _clear_cache()

    version = _next_version()

    # Training log string (returned to admin page)
    log = (
        f"Loading training_data.csv...  {len(df)} records found.\n"
        f"Performing 80/20 train-test split...\n"
        f"Training Random Forest Classifier (trees=50)...  Accuracy: {accuracy:.2%}\n"
        f"Training Polynomial Regressor (degree=2)...  R²: {r2:.4f}\n"
        f"Saving models to models/  ✓\n"
        f"Generating visualisation charts...  ✓\n"
        f"Complete. Model v{version} saved."
    )
    
    info = {
        "version":      version,
        "trained_at":   datetime.datetime.now().isoformat(timespec="seconds"),
        "record_count": int(len(X)),
        "accuracy":     round(float(accuracy), 4),
        "r2":           round(float(r2), 4),
        "clusters":     clf.classes_.tolist(),
        "log":          log,
    }
    with open(INFO_PATH, "w") as f:
        json.dump(info, f, indent=2)

    # --- Visualisations ---
    _plot_decision_tree(clf.estimators_[0], y.unique())  # Plot one tree from the forest
    _plot_feature_importance(clf)
    _plot_regression_curve(y_reg, y_pred_reg)

    return info


def retrain() -> dict:
    """Alias for train() — called by the admin route."""
    return train()


# ---------------------------------------------------------------------------
# 2. Predict
# ---------------------------------------------------------------------------

def predict(answers: dict) -> dict:
    """
    Given a student's survey answers, return a prediction dict:
      {
        "cluster":   str,         e.g. "Maths"
        "match_pct": int,         e.g. 78  (0-100)
        "subjects":  list[dict]   e.g. [{"name": "Software Engineering", "description": "..."}]
      }

    Raises RuntimeError if models are not trained yet.
    """
    if not is_trained():
        raise RuntimeError("Models are not trained yet. Visit /admin to train.")

    models = _get_models()
    scaler = models['scaler']
    clf    = models['clf']
    poly   = models['poly']
    reg    = models['reg']

    # Build feature vector (named DataFrame to avoid sklearn warning)
    X_raw = pd.DataFrame(
        [[float(answers.get(f"q{i}", 5)) for i in range(1, 13)]],
        columns=FEATURE_NAMES
    )
    X_scaled = scaler.transform(X_raw)

    # Classify
    cluster = clf.predict(X_scaled)[0]
    proba   = clf.predict_proba(X_scaled)[0]
    cluster_idx = list(clf.classes_).index(cluster)
    confidence  = proba[cluster_idx]   # 0.0–1.0

    # Regress satisfaction
    X_poly = poly.transform(X_scaled)
    sat_raw = reg.predict(X_poly)[0]
    # Clamp to 1-10, then blend with classifier confidence to get match %
    # Satisfaction (80%) is the primary signal; training data clusters at 7-10 so
    # it reliably drives the score. Confidence (20%) acts as a secondary modifier.
    sat_clamped = float(np.clip(sat_raw, 5, 10))  # Floor at 5 — regression can extrapolate below training range (7-10)
    match_pct = int(round((sat_clamped / 10) * 80 + confidence * 20))
    match_pct = max(5, min(99, match_pct))   # Sensible display range

    # Build cross-cluster subject recommendations
    # Always show top 3 clusters; also include any beyond top 3 that score > 10%
    proba_pairs  = sorted(zip(clf.classes_, proba), key=lambda x: x[1], reverse=True)
    SECONDARY_THRESHOLD = 0.10
    MIN_CLUSTERS = 3

    selected_clusters = [
        (c, p) for i, (c, p) in enumerate(proba_pairs)
        if i < MIN_CLUSTERS or p >= SECONDARY_THRESHOLD
    ]

    cluster_scores = [
        {"cluster": c, "pct": int(round(p * 100))}
        for c, p in selected_clusters
    ]

    MIN_RELEVANCE = 0.55   # subjects scoring below this are not shown even if a slot is free
    subjects = []
    for i, (c, p) in enumerate(selected_clusters):
        limit = 4 if c == cluster else (2 if i == 1 else 1)
        candidates = [s for s in _subjects_for_cluster(c, answers)
                      if _subject_relevance(s["name"], answers) >= MIN_RELEVANCE]
        for s in candidates[:limit]:
            subjects.append({
                "name":        s["name"],
                "description": s["description"],
                "cluster":     c,
                "is_primary":  c == cluster,
                "is_compulsory": False,
            })

    # Maths gate — if Q1 >= 7 and no Maths cluster shown, inject top Maths subject
    q1 = float(answers.get("q1", 5))
    if q1 >= 7 and not any(s["cluster"] == "Maths" for s in subjects):
        maths_candidates = [s for s in _subjects_for_cluster("Maths", answers)
                            if _subject_relevance(s["name"], answers) >= MIN_RELEVANCE]
        if maths_candidates:
            top_maths = maths_candidates[0]
            subjects.append({
                "name":          top_maths["name"],
                "description":   top_maths["description"],
                "cluster":       "Maths",
                "is_primary":    False,
                "is_compulsory": False,
            })
            cluster_scores.append({"cluster": "Maths", "pct": 0})

    # English is compulsory — always include it if not already recommended
    if not any(s["cluster"] == "English" for s in subjects):
        q2 = float(answers.get("q2", 3))
        english_all = _subjects_for_cluster("English", answers)
        english_map = {s["name"]: s for s in english_all}
        if q2 >= 4:
            suggested = [english_map.get("English Advanced"), english_map.get("English Extension")]
        elif q2 >= 2:
            suggested = [english_map.get("English Advanced")]
        else:
            suggested = [english_map.get("English Standard")]
        for s in suggested:
            if s:
                subjects.append({
                    "name":          s["name"],
                    "description":   s["description"],
                    "cluster":       "English",
                    "is_primary":    False,
                    "is_compulsory": True,
                })
        cluster_scores.append({"cluster": "English", "pct": 0, "compulsory": True})

    # Generate per-student profile chart (base64)
    chart_b64 = _generate_student_chart(answers)

    return {
        "cluster":       cluster,
        "match_pct":     match_pct,
        "subjects":      subjects,
        "cluster_scores": cluster_scores,
        "chart_b64":     chart_b64,
    }


# ---------------------------------------------------------------------------
# 3. Model State Helpers
# ---------------------------------------------------------------------------

def is_trained() -> bool:
    """Return True if all model files exist on disk."""
    return all(os.path.exists(p) for p in [SCALER_PATH, CLASSIFIER_PATH, POLY_PATH, REGRESSOR_PATH])


def get_model_info() -> dict:
    """Return the last training run metadata, or sensible defaults."""
    if os.path.exists(INFO_PATH):
        with open(INFO_PATH) as f:
            return json.load(f)
    return {
        "version": "—",
        "trained_at": "Not yet trained",
        "record_count": dh.get_record_count(),
        "accuracy": None,
        "r2": None,
        "clusters": [],
        "log": "No training run yet. Click Retrain Model to start.",
    }


# ---------------------------------------------------------------------------
# 4. Visualisations
# ---------------------------------------------------------------------------

def _plot_decision_tree(clf: DecisionTreeClassifier, class_names) -> str:
    fig, ax = plt.subplots(figsize=(18, 8))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0f172a')
    plot_tree(
        clf,
        feature_names=FEATURE_NAMES,
        class_names=sorted(class_names),
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax,
    )
    ax.set_title("Decision Tree — Subject Cluster Classifier", color="white", pad=12)
    path = os.path.join(CHART_DIR, "tree.png")
    plt.savefig(path, bbox_inches="tight", dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def _plot_feature_importance(clf: RandomForestClassifier) -> str:
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')
    bars = ax.bar(
        range(len(importances)),
        importances[indices],
        color='#2563eb',
        edgecolor='#3b82f6',
        linewidth=0.5,
    )
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([FEATURE_NAMES[i] for i in indices], rotation=45, ha='right', color='white')
    ax.set_ylabel("Importance", color='white')
    ax.set_title("Feature Importance — Which questions matter most?", color='white', pad=10)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')
    path = os.path.join(CHART_DIR, "importance.png")
    plt.tight_layout()
    plt.savefig(path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def _plot_regression_curve(y_true: pd.Series, y_pred: np.ndarray) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')
    
    # Sort by predicted values to create a smooth, sweeping curve (Lift Chart)
    sorted_indices = np.argsort(y_pred)
    y_pred_sorted = y_pred[sorted_indices]
    y_true_sorted = y_true.iloc[sorted_indices].values
    
    x_axis = np.arange(len(y_pred))
    
    # Scatter actual student reported satisfaction
    ax.scatter(x_axis, y_true_sorted, color='#2563eb', alpha=0.5, edgecolors='#93c5fd', linewidth=0.5, label="Actual Student Data")
    
    # Plot the model's non-linear regression curve through the data
    ax.plot(x_axis, y_pred_sorted, 'r-', linewidth=2.5, label='Model Regression Curve', alpha=0.9)
    
    ax.set_xlabel("Student Cohort (Sorted from Low to High Profile Match)", color='white')
    ax.set_ylabel("Satisfaction Score (1-10)", color='white')
    ax.set_title("Regression — How Profile Match Drives Satisfaction", color='white', pad=10)
    ax.tick_params(colors='white')
    
    ax.set_xticks([]) # Hide arbitrary index numbers
    ax.set_ylim(0, 11)
    ax.legend(facecolor='#1e293b', labelcolor='white', loc='upper left')
    
    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')
        
    path = os.path.join(CHART_DIR, "regression.png")
    plt.tight_layout()
    plt.savefig(path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def _generate_student_chart(answers: dict) -> str:
    """Generate a base64-encoded bar chart of the student's normalised profile."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')
    
    # Extract raw values
    vals = [float(answers.get(f"q{i}", 5)) for i in range(1, 13)]
    
    # Normalise all values visually to a 1-10 scale for the chart
    display_vals = []
    for i, v in enumerate(vals):
        q_num = i + 1
        if q_num in [2, 5, 8, 11]:      # 1-5 scale -> multiply by 2
            display_vals.append(v * 2)
        elif q_num in [3, 6, 10]:       # Yes/No (1/0) -> multiply by 10
            display_vals.append(v * 10)
        else:                           # Already 1-10 scale
            display_vals.append(v)
            
    labels = [
        "1. Logic/Maths", "2. Literature", "3. Craft/Build", "4. Performance", 
        "5. Biology/Body", "6. Business", "7. Coding/Tech", "8. Visual Arts", 
        "9. Society/Law", "10. Sci Experiments", "11. Culinary", "12. Systems"
    ]
    
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, display_vals, color='#22c55e', edgecolor='#4ade80', height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, color='white', fontsize=9)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel("Normalised Interest Intensity", color='white', labelpad=10)
    ax.set_title("Your Personal Interest Profile", color='white', pad=12, fontweight='bold')
    ax.tick_params(colors='white')
    ax.set_xlim(0, 10.5)
    
    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')
        
    plt.tight_layout()
    
    # Save to base64 string instead of disk (ephemeral per-student chart)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), dpi=100)
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{b64}"


def get_chart_urls() -> dict:
    """Return URL paths for charts if they exist, else None."""
    charts = {"tree_img": None, "importance_img": None, "regression_img": None}
    mapping = {
        "tree_img":        "charts/tree.png",
        "importance_img":  "charts/importance.png",
        "regression_img":  "charts/regression.png",
    }
    for key, rel_path in mapping.items():
        full = os.path.join(BASE_DIR, "static", rel_path)
        if os.path.exists(full):
            charts[key] = "/static/" + rel_path.replace("\\", "/")
    return charts


# ---------------------------------------------------------------------------
# 5. Private Helpers
# ---------------------------------------------------------------------------

def _save(obj, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _load(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _next_version() -> str:
    info = get_model_info()
    try:
        major, minor = info["version"].split(".")
        return f"{major}.{int(minor) + 1}"
    except Exception:
        return "1.0"


# Primary question signals for each subject.
# Each entry: list of (question_key, weight) tuples.
# Questions are normalised to [0,1] before scoring: /10 for 1-10, /5 for 1-5, as-is for binary.
_SUBJECT_SIGNALS: dict[str, list[tuple[str, float]]] = {
    "Drama":                          [("q4", 1.0)],
    "Music":                          [("q4", 0.7), ("q8", 0.3)],
    "Visual Arts":                    [("q8", 1.0)],
    "Software Engineering":           [("q7", 0.7), ("q1", 0.25), ("q12", 0.15)],
    "Engineering Studies":            [("q12", 0.7), ("q3", 0.25)],
    "Design and Technology":          [("q3", 0.5), ("q8", 0.5)],
    "Industrial Technology Multimedia": [("q7", 0.6), ("q3", 0.4)],
    "Industrial Technology Timber":   [("q3", 0.6), ("q11", 0.4)],
    "Food Technology":                [("q11", 1.0)],
    "Hospitality":                    [("q11", 0.7), ("q6", 0.3)],
    "Biology":                        [("q5", 0.8), ("q10", 0.2)],
    "Chemistry":                      [("q1", 0.45), ("q10", 0.4), ("q5", 0.15)],
    "Physics":                        [("q1", 0.6), ("q12", 0.25), ("q10", 0.15)],
    "Mathematics Advanced":           [("q1", 0.7), ("q12", 0.3)],
    "Mathematics Extension 1":        [("q1", 0.8), ("q12", 0.2)],
    "Mathematics Extension 2":        [("q1", 1.0)],
    "Mathematics Standard":           [("q1", 0.5)],
    "English Advanced":               [("q2", 0.8), ("q9", 0.2)],
    "English Extension":              [("q2", 1.0)],
    "English Standard":               [("q2", 0.4)],
    "Health and Movement Science":    [("q5", 0.5), ("q4", 0.5)],
    "Legal Studies":                  [("q9", 1.0)],
    "Economics":                      [("q6", 0.6), ("q9", 0.4)],
    "Business Studies":               [("q6", 0.8), ("q9", 0.2)],
    "Geography":                      [("q9", 0.8), ("q10", 0.2)],
    "Ancient History":                [("q9", 1.0)],
    "Society and Culture":            [("q9", 0.7), ("q4", 0.3)],
    "Community and Family Studies":   [("q5", 0.6), ("q9", 0.4)],
    "Commerce":                       [("q6", 0.6), ("q9", 0.4)],
    "Biblical Studies":               [("q9", 1.0)],
    "School of Languages":            [("q2", 0.6), ("q9", 0.4)],
}

_Q_MAX = {"q1":10,"q2":5,"q3":1,"q4":10,"q5":5,"q6":1,"q7":10,"q8":5,"q9":10,"q10":1,"q11":5,"q12":10}


def _subject_relevance(name: str, answers: dict) -> float:
    """Score a subject 0-1 based on how well the student's answers match its signals."""
    signals = _SUBJECT_SIGNALS.get(name, [])
    if not signals:
        return 0.0
    return sum((float(answers.get(q, 0)) / _Q_MAX.get(q, 10)) * w for q, w in signals)


def _subjects_for_cluster(cluster: str, answers: dict | None = None) -> list[dict]:
    """Return subjects in a cluster sorted by relevance to the student's answers."""
    subjects = [s for s, c in dh.SUBJECT_CLUSTER_MAP.items() if c == cluster]
    if answers:
        subjects.sort(key=lambda s: _subject_relevance(s, answers), reverse=True)
    else:
        subjects.sort()
    # Brief descriptions for the results page modal
    descriptions = {
        "English Advanced":               "An in-depth study of complex texts, critical thinking, and sophisticated written expression. Suits students who love analysing language and ideas.",
        "English Extension":              "Extends English Advanced with challenging literary theory and extended independent research. Ideal for students who thrive on intellectual challenge.",
        "English Standard":               "Develops communication and literacy skills through a range of texts. Great for students who want a solid English foundation.",
        "Ancient History":                "Explore the societies, politics, and cultures of the ancient world — from Egypt to Rome. Perfect for students passionate about the past.",
        "Biblical Studies":               "Examines the Bible as a literary and historical document, exploring its cultural and theological impact.",
        "Business Studies":               "Learn how businesses operate, are managed, and respond to change. A practical subject with strong links to commerce and economics.",
        "Commerce":                       "Introduces students to the world of business, finance, and consumer decision-making. Excellent foundation for business and law.",
        "Community and Family Studies":   "Explores wellbeing, family structures, and community support. Suits students interested in health, education, or social work.",
        "Economics":                      "Understand how markets, governments, and global systems allocate resources. Great for analytically minded students.",
        "Geography":                      "Investigate natural and human environments — from ecosystems to urbanisation. Combines fieldwork with analysis.",
        "Legal Studies":                  "Explore the Australian legal system, rights, and justice. A strong pathway to law, politics, or public service.",
        "School of Languages":            "Study a language in depth, developing communicative competence in speaking, reading, and writing.",
        "Society and Culture":            "Examine how societies are shaped by culture, technology, and change through a cross-disciplinary lens.",
        "Mathematics Advanced":           "A rigorous course covering calculus, statistics, and algebra. Essential for STEM-oriented university pathways.",
        "Mathematics Extension 1":        "Extends Mathematics Advanced with deeper algebra, calculus, and proof. Recommended for students aiming for science or engineering degrees.",
        "Mathematics Extension 2":        "The highest level of HSC mathematics. Covers advanced proof, complex numbers, and mechanics. For top-performing maths students.",
        "Mathematics Standard":           "A practical mathematics course focused on real-world applications in finance, statistics, and measurement.",
        "Software Engineering":           "Design and build software systems using industry-standard practices. Strongly recommended for students interested in technology careers.",
        "Health and Movement Science":    "Study human movement, health, and physical activity. A great foundation for sport science, physiotherapy, or teaching.",
        "Biology":                        "Explore living systems — from cells and genetics to ecosystems. Ideal for students considering medicine, science, or environment.",
        "Chemistry":                      "Study matter, reactions, and molecular structures through rigorous quantitative analysis and lab work. Essential for medicine, pharmacy, and engineering pathways.",
        "Physics":                        "Understand the fundamental laws of the universe through motion, waves, and electricity. Core for engineering and physical science pathways.",
        "Design and Technology":          "Design and create products that solve real problems. Develops creative thinking, prototyping, and project management skills.",
        "Engineering Studies":            "Apply physics and maths to engineering challenges. Ideal preparation for engineering degrees.",
        "Food Technology":                "Examine food production, processing, and nutrition. Excellent for students interested in hospitality, nutrition, or food science.",
        "Hospitality":                    "Develop practical skills in food preparation, service, and hospitality management. Industry-relevant and hands-on.",
        "Industrial Technology Multimedia": "Produce multimedia products using industry tools. Great for students with a passion for digital content and design.",
        "Industrial Technology Timber":   "Design and construct timber projects while developing woodworking and project planning skills.",
        "Drama":                          "Explore storytelling through performance, directing, and scriptwriting. Builds confidence, empathy, and creative expression.",
        "Music":                          "Develop musicianship through performance, composition, and analysis. Suits students with a love of music in any style.",
        "Visual Arts":                    "Create and critically analyse artworks across a range of media. Ideal for students with a strong visual imagination.",
    }
    return [
        {"name": s, "description": descriptions.get(s, f"{s} is offered as part of the {cluster} cluster.")}
        for s in subjects
    ]
