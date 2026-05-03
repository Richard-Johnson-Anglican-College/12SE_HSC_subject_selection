# Product Specification: HSC Subject Recommendation Engine

## 1. Overview and Objectives
**Product Name:** HSC Subject Recommendation Engine
**Description:** A Flask-based web application designed to help Year 7–10 students explore potential Stage 6 (HSC) subjects by leveraging data-driven insights from current Year 11–12 students.
**Educational Purpose:** This serves as a live, "white-box" project for Year 11 and 12 Software Engineering students. It demonstrates the Software Development Life Cycle (SDLC), data-driven decision-making, and the integration of machine learning within a beginner-friendly architecture.

## 2. Technology Stack
* **Backend Framework:** Python with Flask.
* **Frontend Styling:** HTML5 with **Pico CSS** (minimal, class-light semantic styling) paired with the **Inter** variable font from Google Fonts. 
    * A custom token-driven CSS architecture (`custom.css`) enforces a premium **card-based layout** and strict typographic hierarchy to minimise student cognitive load.
    * The interface is permanently locked to **Dark Mode** (`data-theme="dark"`) utilizing bespoke shadow and border tokens for optimal contrast.
    * **FontAwesome** icons are used throughout the UI to improve scannability and visual engagement.
* **Data Storage:** Flat **CSV files** (`training_data.csv`) using `pandas` (avoids SQL complexity).
* **Machine Learning:** `scikit-learn` (Decision Trees, KNN, Polynomial Regression), `numpy`.
* **Data Visualisation:** `matplotlib` (for generating visual match reports).
* **Generative AI:** **Gemini API** (to synthesise scores into personalised natural language narratives).
* **Deployment:** PythonAnywhere.

## 3. User Experience & Application Flow

### Phase A: The Entry Portal (Index Page)
Users choose their path:
1.  **"I am in Year 7–10" (Inference Path):** Users provide interests to receive subject predictions.
2.  **"I am in Year 11–12" (Training Path):** Users provide interests and their actual chosen subjects to train the model.

### Phase B: The Interest Questionnaire
Both paths complete the same 12-question interest survey to ensure consistent Features ($X$).

### Phase C: Dual-Path Processing
* **The Year 11–12 Path (Data Collection):** Interests and actual subject choices are appended to `training_data.csv`.
* **The Year 7–10 Path (Subject Prediction):** Backend trains models on current data, identifies subject buckets via classification, and predicts match percentages via regression. Results are displayed via a Matplotlib visual and a Gemini-generated summary. Users can view a ranked list of specific subjects within their predicted cluster, clicking them to open detailed modal descriptions.

## 4. Machine Learning & Architecture
The application uses a hybrid approach:
* **Data Preprocessing:** Before training or inference, `ml_engine.py` must apply a scaling algorithm (e.g., `MinMaxScaler` or `StandardScaler`). This normalises the mixed survey scales (1-10, 1-5, binary) so distance-based algorithms (like KNN) evaluate features equally.
* **Classification (Decision Trees / KNN):** Identifies the **Subject Cluster** (e.g., STEM, Creative Arts) based on interest patterns.
* **Regression (Polynomial Regression):** Predicts a continuous **Match Percentage** (0–100%) by analysing correlations between interest intensity and historical satisfaction. *Constraint: To prevent wild overfitting on initial small student datasets, polynomial degrees must be capped at Degree 2.*
* **NLP Synthesis (Gemini API):** Converts raw ML outputs into a student-friendly explanation of why a subject fits their profile.

## 5. System Design & SDLC Artifacts (Curriculum Rules)
* **Data Flow Diagrams (DFDs):** All diagrams must strictly label flows as **Payloads/Information** (e.g., `survey_results`, `match_rankings`). Actions or verbs are strictly prohibited on flow arrows.
* **Code Modularity:** Code is split into `app.py` (routing), `ml_engine.py` (ML logic), and `data_handler.py` (CSV I/O). 
    * *Duty Definition:* `data_handler.py` is explicitly responsible for mapping. It must contain a mapping dictionary to convert specific user inputs (e.g., "Biology") into the required Target Labels (e.g., "STEM") before passing data to the ML engine.

## 6. The Questionnaire (Feature Set)
1.  **Logic Puzzle (1–10):** Interest in abstract math and logic.
2.  **Narrative Thread (1–5):** Enjoyment of writing and literature.
3.  **Creator’s Tool (Binary):** Interest in physical building/making.
4.  **Spotlight Factor (1–10):** Comfort with live performance.
5.  **Human Machine (1–5):** Fascination with biology and health.
6.  **Entrepreneurial Spirit (Binary):** Interest in business and markets.
7.  **Digital Architect (1–10):** Interest in coding and tech mechanics.
8.  **Visual Eye (1–5):** Preference for visual design and arts.
9.  **Social Observer (1–10):** Interest in social issues and law.
10. **Lab Experiment (Binary):** Enjoyment of the scientific method.
11. **Culinary Interest (1–5):** Interest in food tech and hospitality.
12. **Systems Thinker (1–10):** Focus on processes and systematic solutions.

## 7. Subject Clusters (Target Labels)
The model classifies recommendations into these refined categories:

* **HSIE:** Ancient History, Biblical Studies, Business Studies, Commerce, Community and Family Studies, Economics, Geography, Legal Studies, School of Languages, Society and Culture.
* **TAS:** Design and Technology, Engineering Studies, Food Technology, Hospitality, Industrial Technology Multimedia, Industrial Technology Timber.
* **Visual and Performing Arts:** Drama, Music, Visual Arts.
* **English:** English Advanced, English Extension, English Standard.
* **Maths:** Mathematics Advanced, Mathematics Extension 1, Mathematics Extension 2, Mathematics Standard, Software Engineering.
* **Science:** Biology, Physics.
* **PDHPE:** Health and Movement Science.

## 8. Data Model (CSV Schema)
`student_id, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, target_subject_1, target_subject_2, target_subject_3, target_subject_4, target_subject_5, target_subject_6, target_subject_7, satisfaction_score`

*Note on Targets:* The CSV stores the *raw* user input (e.g., `target_subject_1` = "Visual Arts"). The system dynamically maps these to the broader Subject Clusters (e.g., "Creative Arts") during runtime training via `data_handler.py`. `target_subject_7` is explicitly marked as optional to accommodate students taking 1-unit courses, Extension 2 subjects, or VET courses.

*Solving the "Cold Start" Problem:* Because the Inference Path requires a trained model, `training_data.csv` must be deployed with a **pre-seeded dataset** of approximately 30-50 synthetic student profiles. This ensures Year 7-10 students can receive predictions on Day 1 before Year 11-12s have populated the system.

## 9. Admin & Teacher Dashboard
A password-protected module for model maintenance and pedagogical explainability.

### A. Model Retraining
* **Retrain Button:** Triggers a script to reload the current `training_data.csv`, re-perform the 80/20 train-test split, and overwrite the serialised model files (`.pkl` or `.joblib`).
* **Live Logs:** Displays the accuracy score, precision, and recall of the newly trained model to the students.

### B. Machine Learning Visualisations (White-Box Analysis)
* **Decision Tree Diagram:** A visual export (using `sklearn.tree.plot_tree`) showing the logic gates of the model (e.g., "If Q7 > 8, then Predict STEM").
* **Feature Importance Chart:** A Matplotlib bar chart showing which survey questions (features) are most influential in the model's decision-making process.
* **Regression Curves:** Visual representation of polynomial relationships between specific interests and match percentages.

## 10. Future Iterations
* Prerequisite flags for specific Stage 6 subjects.
* Cohort popularity heatmaps.