# 🎓 HSC Subject Recommendation Engine

> A machine learning web application that recommends Year 11–12 HSC subject combinations based on a student's personal interest profile — built with Flask, scikit-learn, and Google Gemini AI.

---

## 📸 Overview

**🌐 Live Demo:** [https://hscmatch.pythonanywhere.com/](https://hscmatch.pythonanywhere.com/) — Try it now!

Students complete a 12-question interest survey. The system uses a trained **Random Forest classifier** to predict their strongest subject cluster, surfaces relevant subjects across multiple areas using signal-weighted ranking, and generates a **personalised AI explanation** via Google Gemini.

```
Student Survey (Q1–Q12)
        ↓
Random Forest Classifier  →  Cluster prediction + confidence %
Ridge Regression          →  Satisfaction score → Match %
Rule-based signal weights →  Subject ranking within clusters
Google Gemini (LLM)       →  "Why this fits you" explanation
```

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🧠 **ML Cluster Prediction** | RandomForestClassifier trained on real + synthetic HSC student profiles |
| 📊 **Match Percentage** | Blends Ridge regression satisfaction score (80%) with classifier confidence (20%) |
| 🔀 **Cross-Cluster Recommendations** | Shows subjects from multiple relevant clusters, not just the primary |
| 🎯 **Relevance Filtering** | Minimum relevance threshold (0.55) suppresses poorly-matched subjects |
| 💬 **AI Explanation** | Google Gemini generates a personalised explanation grounded in the ML output |
| 📈 **Feature Visualisation** | Bar chart of normalised interest profile shown on results page |
| 🔁 **Answer Persistence** | Survey answers saved to session — "Adjust My Answers" navigates back pre-populated |
| 🛡️ **Admin Dashboard** | Password-protected retraining, decision tree visualisation, feature importance chart |
| ✏️ **Training Data Collection** | Teachers can submit real student profiles via `/survey/train` to improve the model |

---

## 🗂️ Project Structure

```
12SE_HSC_subject_selection/
├── app.py                  # Flask routes and application entry point
├── ml_engine.py            # ML pipeline: train, predict, signal weights, charts
├── data_handler.py         # Subject→cluster mapping, CSV I/O, label derivation
├── gemini_client.py        # Google Gemini API integration
├── config.py               # API keys and secrets (NOT committed)
├── training_data.csv       # Student profiles used for ML training (207 records)
├── models/                 # Serialised model artefacts (auto-generated)
│   ├── scaler.pkl
│   ├── clf.pkl
│   ├── poly.pkl
│   └── reg.pkl
├── static/
│   ├── css/custom.css
│   └── charts/             # Admin dashboard chart PNGs (auto-generated)
├── templates/
│   ├── index.html          # Landing page
│   ├── survey_predict.html # Student interest survey
│   ├── results.html        # Recommendation results
│   ├── survey_train.html   # Teacher training data submission
│   └── admin.html          # Admin dashboard
├── SPEC.md                 # Product specification
├── ML.md                   # Educational guide to ML models used
└── AI.md                   # Gemini API integration patterns
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- pip

### 1. Clone the repository
```bash
git clone https://github.com/Richard-Johnson-Anglican-College/12SE_HSC_subject_selection.git
cd 12SE_HSC_subject_selection
```

### 2. Install dependencies
```bash
pip install flask scikit-learn pandas numpy matplotlib google-generativeai
```

### 3. Create `config.py`
```python
# config.py  — DO NOT COMMIT THIS FILE
GEMINI_API_KEY  = "your-gemini-api-key"
ADMIN_PASSWORD  = "your-admin-password"
SECRET_KEY      = "your-secret-key"
```
> Get a free Gemini API key at [aistudio.google.com](https://aistudio.google.com/app/apikey)

### 4. Run the app
```bash
python app.py
```

### 5. Train the model
Navigate to `http://localhost:5000/admin` → click **Retrain Model**.  
The model must be trained before predictions can be made.

---

## 🧪 How to Use

### Student Flow
1. Visit `http://localhost:5000`
2. Click **Start Survey** → answer 12 questions about your interests
3. View your **recommended subject clusters and subjects**
4. Click **Generate AI Analysis** for a personalised Gemini explanation
5. Use **Adjust My Answers** to tweak responses and re-predict

### Teacher Flow
1. Visit `/survey/train`
2. Enter a student's Q1–Q12 answers + their actual subject selections + satisfaction score
3. Submit → record appended to `training_data.csv`
4. Go to `/admin` → **Retrain Model** to incorporate new data

---

## 🤖 Machine Learning Architecture

### Cluster Classification
- **Algorithm:** `RandomForestClassifier` (100 estimators)
- **Features:** Q1–Q12 normalised to [0,1] via `MinMaxScaler`
- **Labels:** Derived from majority subject cluster per student record
- **Clusters:** English, HSIE, Maths, PDHPE, Science, TAS, Visual and Performing Arts

### Satisfaction Regression
- **Algorithm:** `PolynomialFeatures(degree=2)` + `Ridge` regression
- **Output:** Predicted satisfaction 1–10, clipped to floor of 5

### Match Percentage Formula
```
match_pct = (satisfaction / 10) × 80  +  confidence × 20
```

### Subject Ranking
Subjects within each cluster are ranked by a relevance score computed from weighted survey signals:
```python
# Example: Software Engineering prioritises coding, then maths, then systems thinking
"Software Engineering": [("q7", 0.70), ("q1", 0.25), ("q12", 0.15)]
```

### Guard Rails
| Rule | Trigger | Action |
|------|---------|--------|
| Relevance threshold | Any subject | Hidden if score < 0.55 |
| English compulsory | Always | Force-added based on Q2 score |
| Maths gate | Q1 ≥ 7 | Top Maths subject always shown |

---

## 📊 Subject Clusters

| Cluster | Subjects |
|---------|----------|
| **Science** | Biology, Chemistry, Physics |
| **Maths** | Mathematics Advanced, Extension 1, Extension 2, Standard |
| **TAS** | Software Engineering, Design and Technology, Engineering Studies, Food Technology, Hospitality, Industrial Technology Multimedia, Industrial Technology Timber |
| **HSIE** | Ancient History, Biblical Studies, Business Studies, Commerce, Community and Family Studies, Economics, Geography, Legal Studies, School of Languages, Society and Culture |
| **Visual and Performing Arts** | Drama, Music, Visual Arts |
| **PDHPE** | Health and Movement Science |
| **English** | English Advanced, Extension, Standard |

---

## 🔬 Survey Questions

| Q | Scale | Topic |
|---|-------|-------|
| Q1 | 1–10 | Logic puzzles and maths |
| Q2 | 1–5 | Writing and literature |
| Q3 | Yes/No | Hands-on building and making |
| Q4 | 1–10 | Performance and creative expression |
| Q5 | 1–5 | Biology and the human body |
| Q6 | Yes/No | Business and commerce |
| Q7 | 1–10 | Coding and technology |
| Q8 | 1–5 | Visual arts and design |
| Q9 | 1–10 | Society, history, and law |
| Q10 | Yes/No | Science experiments and lab work |
| Q11 | 1–5 | Culinary and hospitality |
| Q12 | 1–10 | Systems thinking and engineering |

---

## 🔐 Security Notes

- `config.py` is in `.gitignore` — **never commit API keys**
- Admin route is protected by password (set in `config.py`)
- Session data is encrypted using `SECRET_KEY`

---

## 📚 Documentation

| File | Contents |
|------|----------|
| [`SPEC.md`](SPEC.md) | Full product specification |
| [`ML.md`](ML.md) | Educational guide: supervised learning, decision trees, neural networks, hybrid AI |
| [`AI.md`](AI.md) | Gemini API integration patterns and prompt engineering guide |

---

## 🏫 Built by

**Richard Johnson Anglican College — Year 12 Software Engineering**  
HSC Major Project · 2025–2026
