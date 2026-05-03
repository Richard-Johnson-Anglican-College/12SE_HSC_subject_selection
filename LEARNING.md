# HSC Subject Recommendation Engine — Complete Learning Outline

---

## 1. What the Project Is

A **Flask web application** that uses real student interest data to recommend HSC (Year 11–12) subjects to Year 7–10 students. It is simultaneously:
- A **live product** used by students
- A **teaching artefact** demonstrating the full Software Development Life Cycle (SDLC) to Year 11–12 Software Engineering students

**Dual-path design:**
- **Year 11–12** submit their interests + actual subject choices → trains the model
- **Year 7–10** submit their interests → receive predictions from that model

This solves the **cold start problem** with ~207 pre-seeded synthetic records so predictions work on Day 1 before real data arrives.

---

## 2. Supervised Learning

### What it is
- You give the model **labelled examples** (input + correct answer)
- It learns the pattern and generalises to new inputs
- Analogous to a student learning from marked tests

### In this project
| Element | What it is |
|---------|-----------|
| **Input (features)** | 12 survey answers Q1–Q12 |
| **Label (target)** | Subject cluster the student chose |
| **Task** | Predict cluster for a new student |

### Weak Supervision
Labels weren't hand-annotated — they were **programmatically derived**:
- Student chose Biology + Chemistry + Maths Advanced → most common cluster = Science → label = `"Science"`
- This is still supervised learning because every record has a known output

### The Tie-Breaking Bug
Python's `max()` broke ties **alphabetically** — `"Visual and Performing Arts"` (V) always won. TAS students were mislabelled as VPA, contaminating training data and making TAS invisible to the model. Fixed with a **priority order**: `TAS > Science > PDHPE > VPA > HSIE > Maths > English`.

---

## 3. Machine Learning Models

### 3.1 Random Forest Classifier
- Trains **50 Decision Trees** on random data subsets, takes majority vote
- Outputs `predict_proba()` — a probability per cluster, not just a single answer
- **Why not a single Decision Tree?** Single trees overfit (memorise training data, fail on new inputs). Ensemble averaging reduces variance.
- **Why not KNN?** Admin dashboard requires `feature_importances_` and `plot_tree` — unavailable on KNN

### 3.2 Decision Trees (base learner)
- Asks yes/no questions to split data: `Q1 > 7? → Science vs Maths`
- Splits chosen by **Gini impurity** / **information gain**
- **Strengths:** interpretable, fast, handles mixed types
- **Weaknesses:** overfits easily, sensitive to small data changes

### 3.3 Polynomial Regression with Ridge Regularisation
- Predicts **satisfaction score** (continuous, 1–10) — a regression problem, not classification
- `PolynomialFeatures(degree=2)` creates ~91 features from 12 (adds Q1², Q2², Q1×Q2, etc.) to capture curved relationships
- `Ridge(alpha=10)` adds a penalty for large coefficients — prevents wild extrapolation from 91 features on ~200 records
- Satisfaction clamped to [5, 10] — realistic floor for students rating their own choices

### 3.4 Match Percentage Formula
```
match_pct = (sat_clamped / 10) × 80 + confidence × 20
```
Satisfaction weighted 80% because training scores cluster 7–10 (strong signal). Classifier confidence weighted 20% as a secondary modifier.

### 3.5 Feature Normalisation
- `MinMaxScaler` maps all inputs to [0, 1]
- All scored sliders are 1–10; binary questions (Q3, Q6, Q10) are 0/1
- Signal scoring also uses `_Q_MAX` per-question normalisation for the rule-based layer

### 3.6 Model Caching
Trained models stored in `_CACHE` dict after first call. Invalidated on retrain. Eliminates 4 disk reads per prediction request.

---

## 4. Rules vs ML — The Honest Split

| Component | Approach | Why |
|-----------|----------|-----|
| Cluster prediction | **ML** (RandomForest) | Complex patterns, non-linear |
| Confidence % | **ML** (`predict_proba`) | Probability from data |
| Subject ranking within cluster | **Rules** (`_SUBJECT_SIGNALS`) | Insufficient per-subject data |
| Minimum relevance threshold (0.55) | **Hard rule** | Prevents irrelevant subjects filling slots |
| English always included | **Hard rule** | Compulsory HSC requirement |
| Maths included when Q1 ≥ 7 | **Hard rule** | Domain knowledge gate |

**~35% ML, ~65% rules.** This is normal in production AI — each layer does what it's best at given the available data.

**When would it become fully ML?** With 50+ examples per subject, each subject gets its own binary classifier. Subject ordering becomes learned, not hand-coded.

---

## 5. Neural Networks & Generative AI

### Why no custom neural network
- Needs **thousands of records** — 207 is far too few
- Would overfit severely
- Random Forest achieves comparable accuracy and is interpretable

### Gemini (LLM = neural network)
- Google Gemini is a **Large Language Model** — a neural network with billions of parameters
- Used via API to generate the "Why this fits you" explanation
- The neural network IS present — just not custom-trained by us

| | Custom NN | Gemini LLM |
|--|-----------|------------|
| Trained by us? | Would need to be | No — Google pre-trained |
| Data needed | 1000s of records | Already trained |
| Task here | Not used | Natural language from ML output |
| Access | sklearn/tensorflow | REST API |

---

## 6. Prompt Engineering

### Structure (ROLE / CONTEXT / TASK / INPUT)
- **ROLE:** Anchors persona and audience (friendly HSC advisor, Year 10 student)
- **CONTEXT (WHAT THE MODEL FOUND):** Cluster, match %, affinities, recommended subjects, interest highlights
- **TASK:** 5-point numbered checklist — models follow numbered lists reliably
- **Rules block:** Constrains tone, banned words, word limit

### Key prompt techniques used
- **Pre-computing values:** `match_in_10 = round(match_pct / 10)` done in Python so Gemini never has to calculate it (prevents "84 in every 10" errors)
- **Explicit constraints:** "Do NOT invent subjects not listed above" prevents hallucination
- **Banned vocabulary:** "Do NOT say algorithm or machine learning" — keeps language student-friendly
- **Specificity over vagueness:** Each task point says exactly *what kind* of sentence to write

### RAG (Retrieval-Augmented Generation)
The ML output is passed as **grounded context** into the prompt. Gemini can only reference subjects it has been explicitly given — it cannot hallucinate subjects from outside the recommendation list. This is the same pattern used in industry-scale AI products.

---

## 7. The Three-Layer Hybrid System

```
Student fills survey
        ↓
LAYER 1: ML (scikit-learn)
  RandomForest → cluster + probabilities
  Ridge regression → satisfaction score
  Output: cluster, match %, affinities
        ↓
LAYER 2: Rules (_SUBJECT_SIGNALS)
  Signal weights → subject ranking
  Relevance threshold → filter low scorers
  Hard gates → English always, Maths if Q1≥7
  Output: ordered subject list
        ↓
LAYER 3: Generative AI (Gemini)
  ML output → structured prompt → explanation
  Output: personalised "why this fits you" paragraph
```

**Why combine all three?**

| Approach alone | Limitation |
|----------------|------------|
| Rules only | Can't learn from data |
| ML only | Can't explain itself in plain English; needs large data for subject-level decisions |
| GenAI only | Hallucinates; unreliable without grounding |
| **Hybrid** | Each layer does what it's best at |

---

## 8. System Limitations & Honest Assessment

### What the model does well
- **Clear profiles** (pure STEM, pure arts, pure trades) → correct cluster with high confidence
- **Cross-cluster profiles** → surfaces subjects from both Science and TAS when appropriate
- **Subject ranking** → signal weights reliably surface the most relevant subjects first
- **Relevance filtering** → 0.55 threshold suppresses low-scoring subjects (e.g. Biology hidden when Q5=2)
- **Guard rails** → English and Maths always surfaced via hard rules

### Current limitations
- 207 records (mostly synthetic) — borderline for 7 clusters
- Mixed profiles (equal Science + TAS interest) rely on whichever cluster the forest leans toward
- No per-subject ML — ordering is rules-based, not learned
- Fragile cluster boundaries — one strong Q shift can change top-3 ranking (expected ML behaviour)

### What would make it better
1. Real student data (100+ responses) → better cluster boundaries
2. Binary subject columns → per-subject classifiers, removes signal weight dependency
3. Cross-validation reporting on admin dashboard → true generalisation accuracy
4. Feedback loop — retrain on actual Year 11 enrolment data after students complete their courses
