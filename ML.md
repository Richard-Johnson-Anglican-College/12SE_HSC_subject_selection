# Machine Learning in the HSC Subject Recommendation Engine
### A Student Guide to ML Models, Supervised Learning, and Hybrid AI Systems

---

## 1. What is Supervised Learning?

Supervised learning is the most common type of machine learning. The idea is simple:

> **You show the model labelled examples, and it learns the pattern.**

Think of it like a student learning from a marked test paper. The student sees the question (input) AND the correct answer (label). Over many examples, they learn to answer new questions they have never seen before.

In this project:
- **Input (features):** A student's 12 survey answers (Q1–Q12) — interest in maths, coding, creative arts, biology, etc.
- **Label (target):** The subject cluster that real students ended up choosing — Maths, Science, TAS, HSIE, Visual and Performing Arts, etc.
- **Task:** Given a *new* student's answers, predict which cluster they belong to.

### Who uses supervised learning?
| Industry | Input | Label | Prediction |
|----------|-------|-------|------------|
| Medicine | Patient symptoms + test results | Diagnosed condition | Disease classification |
| Banking | Spending history | Fraud / Not Fraud | Fraud detection |
| Streaming | Watch history | What you rated highly | Movie recommendations |
| Education | Student interest survey | Subject group chosen | Subject cluster recommendation |

---

## 2. How the Labels Were Created (Weak Supervision)

No student ever said "I belong to the Science cluster." Instead, labels were **programmatically derived** from real subject choices.

```
Student chose: Biology, Chemistry, Maths Advanced, English Advanced
              ↓           ↓             ↓                ↓
Cluster:    Science    Science         Maths           English

Most common cluster = Science → Label = "Science"
```

This is called **weak supervision** — the labels are inferred, not hand-annotated. It is still supervised learning because every training record has a known output.

### The Tie-Breaking Problem
When a student took one subject from each cluster (e.g. Design and Technology + Visual Arts + Maths + English), a tie-break rule was needed. The system uses a **priority order** favouring technical/specialist clusters:

```
TAS (7) > Science (6) > PDHPE (5) > Visual Arts (4) > HSIE (3) > Maths (2) > English (1)
```

This reflects educational reality: English and Maths are taken by nearly everyone, so they are "background" subjects. The *distinctive* choice is the specialist cluster.

### Why the Tie-Breaking Rule Matters — A Real Bug
Before this priority order was implemented, Python's default `max()` broke ties **alphabetically**. The cluster name `"Visual and Performing Arts"` starts with V — the highest letter — so it *always* won ties. A student who chose Design and Technology + Visual Arts + Maths + English (one subject per cluster, all tied at count=1) was labelled **Visual and Performing Arts**, not TAS. This silently contaminated the training data: TAS students were being mislabelled as VPA, so the RandomForest never properly learned the TAS cluster boundary. Fixing the tie-breaking rule and retraining immediately surfaced TAS predictions that were previously invisible.

---

## 3. Rules vs True ML — What's the Difference?

A common question in AI: **when should you use data-driven ML, and when should you use hand-crafted rules?**

### In this project:

| Component | Approach | Why |
|-----------|----------|-----|
| Which cluster to predict | **ML (RandomForest)** | Complex patterns, many features, non-linear |
| Cluster confidence (%) | **ML (predict_proba)** | Probability learned from data |
| Match percentage | **ML + formula** | Regression output blended with confidence |
| Which subjects to show | **Rules (_SUBJECT_SIGNALS)** | Not enough data per subject for per-subject ML |
| Subject ranking within cluster | **Rules (signal weights)** | Placeholder until 50+ examples per subject exist |
| Minimum relevance threshold (0.55) | **Hard rule** | Prevents low-scoring subjects filling available slots (e.g. Biology when Q5=2) |
| English always included | **Hard rule** | Domain knowledge — compulsory HSC requirement |
| Maths always included when Q1 ≥ 7 | **Hard rule** | High maths interest should always surface a Maths subject, even if ML ranks another cluster 3rd |

### The Honest ML/Rules Split: ~35% ML, ~65% Rules

The ML model does one meaningful job: **predict the cluster**. Everything after that — how many subjects, which ones rank first, what the percentage means in words — is rule-based. This is normal in production AI systems. The presentation layer is rule-based *because we don't yet have enough data* to learn it automatically.

### When would it become fully ML-driven?
With 50+ real student examples per subject, each subject could have its own binary classifier:

```
"Will this student pick Physics?" → RandomForest(Q1–Q12) → Yes (0.87 probability)
"Will this student pick Chemistry?" → RandomForest(Q1–Q12) → Yes (0.81 probability)
```

The results page would rank subjects by probability — no signal weights needed at all. **The data teaches the ordering.**

---

## 4. ML Models Used in This Project

### 4.1 Random Forest Classifier (Cluster Prediction)

A **Random Forest** is an ensemble of Decision Trees. Instead of training one tree, it trains hundreds of trees on random subsets of the data, then takes a majority vote.

```
Survey Answers (Q1–Q12)
        ↓
  [Tree 1] → "Science"
  [Tree 2] → "Science"
  [Tree 3] → "TAS"
  [Tree 4] → "Science"
  ...
  [Tree 100] → "Science"
        ↓
  Majority Vote → "Science" (69% confidence)
```

**Why Random Forest over a single Decision Tree?**
- A single tree **overfits** — it memorises the training data and fails on new inputs
- Multiple trees with random variation **average out errors** — more robust
- Provides `predict_proba()` — a probability for each cluster, not just a single answer

**Effectiveness in this project:**
- Trained on 207 synthetic student records across 7 clusters
- Correctly identifies the dominant cluster for well-defined profiles (pure STEM, pure arts, pure trades)
- Less reliable on mixed profiles (student with equal interest in arts AND science) — this is expected and honest

---

### 4.2 Decision Tree (Conceptual Explanation)

A Decision Tree asks a series of yes/no questions about the features:

```
Is Q1 (maths interest) > 7?
├── YES → Is Q10 (lab work) = Yes?
│         ├── YES → Predict: Science
│         └── NO  → Predict: Maths
└── NO  → Is Q4 (performance) > 6?
          ├── YES → Predict: Visual and Performing Arts
          └── NO  → Is Q3 (craft/build) = Yes?
                    ├── YES → Predict: TAS
                    └── NO  → Predict: HSIE
```

Each split is chosen to maximally separate the classes (measured by **Gini impurity** or **information gain**). The tree keeps splitting until it reaches pure leaf nodes or a maximum depth.

**Strengths:** Interpretable, fast, handles mixed data types  
**Weaknesses:** Overfits easily, sensitive to small data changes  
**Used in this project:** As the base learner inside Random Forest

---

### 4.3 Polynomial Regression (Satisfaction Prediction)

The second ML task is predicting how satisfied a student will be with their recommended cluster (1–10 scale). This is a **regression** problem — the output is a continuous number, not a category.

**Why Polynomial, not Linear?**
Linear regression assumes a straight-line relationship:
```
satisfaction = a×Q1 + b×Q2 + ... + k
```
But satisfaction might peak for moderate values — very high or very low scores in all areas can both reduce satisfaction. Polynomial features capture curved relationships:
```
new features: Q1², Q2², Q1×Q2, Q1×Q3 ...
```

This project uses **degree 2** polynomial features + **Ridge regression** (a regularised variant). Ridge adds a penalty for large coefficients, preventing the model from overfitting on the expanded feature set.

```
Raw features (Q1–Q12)
        ↓
PolynomialFeatures(degree=2)   ← creates ~91 features from 12
        ↓
Ridge regression               ← learns weights, penalises extremes
        ↓
Predicted satisfaction (5–10)
```

**Why clip to 5–10?**  
Satisfaction scores in the training data cluster between 7–10 (students are generally positive about their own subject choices). Allowing predictions below 5 would be unrealistically negative, so a floor of 5 is applied.

---

### 4.4 Neural Networks — Used via Gemini, Not for Classification

A neural network passes inputs through layers of weighted nodes, learning non-linear patterns through backpropagation.

```
Input Layer     Hidden Layer     Output Layer
Q1  ──┐                          ┌── Science
Q2  ──┤──[neurons]──[neurons]────┼── TAS
...   │                          ├── Maths
Q12 ──┘                          └── ...
```

**This project does NOT train a custom neural network** for classification or regression — and for good reason:
- A custom neural network needs **thousands of records** to train reliably
- With 207 training records, it would overfit severely (memorise the data, fail on new students)
- Random Forest achieves comparable accuracy with far less data and is easier to interpret

**However, a neural network IS used in this project — Gemini.**

Google Gemini is a **Large Language Model (LLM)** — a neural network with billions of parameters trained on vast amounts of text. This project calls Gemini via API to generate the "Why this fits you" explanation on the results page.

```
ML output (cluster, subjects, Q scores)
              ↓
        Structured prompt
              ↓
    Gemini (neural network)        ← neural network IS used here
              ↓
  Personalised written explanation
```

The key distinction is:
| | Custom neural network | Gemini (LLM) |
|--|----------------------|--------------|
| **Trained by us?** | Would need to be | No — pre-trained by Google |
| **Data needed** | Thousands of records | Already trained |
| **Task in this project** | Not used | Generates natural language from ML output |
| **How accessed** | `sklearn` / `tensorflow` | REST API (`google-generativeai`) |

**When would a custom neural network be appropriate here?**  
If the system collected data from 10,000+ students annually, a custom neural network could learn subtle cross-feature interactions (e.g. a student who rates maths AND drama both high). Random Forest would struggle with this edge case; a deep network would handle it naturally.

---

## 5. The Hybrid System — ML + Rules + Generative AI

This project uses **three different types of AI** working together. This is called a **hybrid AI system**.

```
┌─────────────────────────────────────────────────────────────┐
│                    Student fills survey                      │
└─────────────────────────┬───────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  LAYER 1: Machine Learning (scikit-learn)                   │
│  • RandomForest → predicts cluster + probabilities          │
│  • Ridge regression → predicts satisfaction score           │
│  • Output: cluster name, match %, cluster affinities        │
└─────────────────────────┬───────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  LAYER 2: Rule-Based Logic                                  │
│  • Signal weights → ranks subjects within each cluster      │
│  • Relevance threshold → filters low-scoring subjects       │
│  • English compulsory rule → always adds English            │
│  • Output: ordered list of recommended subjects             │
└─────────────────────────┬───────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  LAYER 3: Generative AI (Google Gemini)                     │
│  • Receives ML output as structured prompt context          │
│  • Generates personalised written explanation               │
│  • Output: "Why this fits you" paragraph                    │
└─────────────────────────────────────────────────────────────┘
```

### Why combine all three?

| Approach alone | Limitation |
|----------------|------------|
| Rules only | Can't learn patterns from data; requires expert encoding of every case |
| ML only | Can't explain its reasoning in natural language; needs large datasets for subject-level decisions |
| Generative AI only | Makes up subjects; unreliable without grounding in real data |
| **Hybrid** | Each layer does what it's best at — ML predicts, rules order, Gemini explains |

### How Gemini is grounded in the ML output

Without grounding, Gemini might hallucinate subjects or make up reasons. The prompt explicitly passes the ML results as structured context:

```
### WHAT THE MODEL FOUND ###
Primary strength area: Science (69% match)
NOTE: 69% means roughly 7 in every 10 of this student's interest answers pointed toward Science subjects.
Subject area affinities: Science 69%, TAS 18%

Recommended subjects across ALL areas:
  • Science: Physics, Chemistry
  • Maths: Mathematics Advanced

### STUDENT'S KEY INTERESTS ###
logic and maths (9/10); science experiments and lab work; systems thinking and engineering (8/10)

### YOUR TASK ###
Write a warm, encouraging 4-5 sentence explanation that covers:
1. WHY these subjects suit them — connect their interests to specific subjects.
2. WHY the combination makes sense — how the subject areas complement each other.
3. WHAT KIND OF STUDENT thrives here — paint a picture of the learner type.
4. A FORWARD-LOOKING sentence — one example of where this could lead.
5. Mention that around 7 in every 10 of their answers pointed toward Science — not a score, an interest signal.
Do not invent subjects not listed above. Maximum 120 words.
```

This is called **Retrieval-Augmented Generation (RAG)** in the industry — grounding a generative model's output in verified data so it cannot hallucinate facts.

> **Note on prompt evolution:** The task section now asks for 4–5 sentences covering the *why* behind the recommendation (interest connection, cross-cluster logic, learner type, forward-looking pathway, and match percentage framed as "X in every 10 answers"). The match fraction is pre-computed in Python (`round(match_pct / 10)`) before being passed to Gemini — this prevents the model from miscalculating it.

---

## 6. How Effective is the System?

### What the model does well
- **Clear profiles** (pure STEM, pure arts, pure trades) → correct cluster prediction with high confidence
- **Cross-cluster profiles** → a student who is both Science and TAS oriented (e.g. Physics + Chemistry + Design and Technology + Software Engineering) correctly surfaces subjects from both clusters
- **Subject ranking** → within a correctly predicted cluster, the signal weights reliably surface the most relevant subjects first
- **Relevance filtering** → the 0.55 threshold correctly suppresses low-scoring subjects (e.g. Biology is hidden when Q5=2; Maths Standard is hidden when Q1=9)
- **Guard rails** → English and Maths are always surfaced via hard rules even when the ML probability alone wouldn't rank them in the top 3

### Current limitations
- **207 training records** (mostly synthetic) — enough for 7 clusters, but borderline. Real student data will reveal patterns synthetic records miss.
- **Mixed profiles** — a student genuinely split between Science and TAS relies on whichever cluster the RandomForest leans toward. Ensemble uncertainty is not fully exposed to the user.
- **No per-subject ML** — subject ordering is rules-based, not learned. This is the biggest gap between current and ideal.
- **Fragile boundaries** — adding a single strong signal (e.g. Q8=5 for Visual Arts) can shift cluster probabilities enough to change the top-3 ranking. This is expected ML behaviour, but can surprise users.

### What would make it better
1. **Real student data** (100+ Year 11–12 responses) → better cluster boundaries
2. **Binary columns per subject** → enables per-subject classifiers, removes signal weight dependency
3. **Cross-validation reporting** on the admin dashboard → shows true generalisation accuracy, not just training accuracy
4. **Feedback loop** → after students complete Year 11, record whether they actually enrolled in recommended subjects, and retrain on that signal

---

## 7. Summary for Assessments

> **"The HSC Subject Recommendation Engine uses supervised learning — specifically a Random Forest classifier trained on labelled student profiles, where cluster labels were programmatically derived (weak supervision) from subject selection data. A Ridge regression model predicts satisfaction scores. Subject ordering within clusters is handled by a rule-based signal weighting system, which acts as a placeholder until enough per-subject training examples exist. A Gemini generative AI layer produces personalised written explanations grounded in the ML output, preventing hallucination. This three-layer hybrid architecture — ML prediction, rule-based ordering, and generative explanation — reflects how real-world AI systems are built: using the right tool for each sub-task rather than a single model for everything."**
