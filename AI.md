AI Capabilities Guide
This document outlines exactly how AI is integrated into RJ_BLP_1 so the same pattern can be replicated in new Flask (or similar Python) projects.

1. Overview
The app uses Google's Generative AI (Gemini family) via the google-generativeai Python SDK. A user submits a form, the server constructs a structured prompt, sends it to the model, and parses a JSON response back into the UI.

Flow:

User fills an HTML form (/form).
Browser POSTs data to /generate.
Server builds a prompt from a template + form fields.
Server calls model.generate_content(prompt).
Server strips code-fences, parses JSON, normalises fields, and returns it as jsonify(data).
Frontend JS renders the result.
2. Requirements
Python packages (requirements.txt)
Flask
google-generativeai
Install:

pip install -r requirements.txt
API key
Create a config.py file in the project root (and add it to .gitignore):

# config.py
GEMINI_API_KEY = "YOUR_API_KEY_HERE"
Get a key at https://aistudio.google.com/app/apikey.

.gitignore entries
config.py
__pycache__/
*.pyc
.venv/
3. Model
Currently configured in @/c:\Users\rzamora\py\Education & Teaching\RJ_BLP_1\app.py:105:

model = genai.GenerativeModel('gemma-3n-e4b-it')
Use gemma-3n-e4b-it for text-based work in new projects.

4. SDK Configuration Pattern
import google.generativeai as genai

try:
    import config
    genai.configure(api_key=config.GEMINI_API_KEY)
except (ImportError, AttributeError):
    print("WARNING: GEMINI_API_KEY not found. AI generation will fail.")
This lets the app boot even without a key (useful for UI dev), but logs a clear warning.

5. Prompt Engineering Pattern
The prompt follows a four-section structured template: ROLE, CONTEXT, TASK, and INPUT. This is a reusable pattern for any structured-output task.

5.1 Skeleton (generic, reusable)
### ROLE ###
You are an expert <domain> assistant. Your audience is <audience>. Your goal is to <goal>.

### CONTEXT ###
<Domain knowledge / taxonomy / rules the model must use.
List options, categories, definitions explicitly.>

### TASK ###
Analyze the following <input> provided by a <user-role>. Execute the following steps:

1. **Analyze:** Read the input fields.
2. **Select / Decide:** <constraints on what may be chosen>.
3. **Craft Output:** <tone, perspective, language, length>.
4. **Format Output:** Provide your response as a single, clean JSON object.
   Do not include any text or markdown before or after the JSON.
   The JSON must have these exact keys: `key1`, `key2`, `key3`.

### INPUT ###
* **Field A:** {field_a}
* **Field B:** {field_b}
5.2 Why this works
ROLE anchors persona and audience.
CONTEXT prevents hallucinated categories by enumerating allowed values.
TASK uses a numbered checklist — models follow these reliably.
Constraints in TASK (tone, spelling, perspective, JSON-only) reduce post-processing.
Exact JSON keys specified makes parsing deterministic.
5.3 Conditional instructions (user-driven branching)
The project injects a different selection rule depending on whether the user pre-selected a focus:

if focus_muscle:
    selection_instruction = f"You MUST select '{focus_muscle}'. Choose a sub-item ONLY from those listed under it."
else:
    selection_instruction = "Choose the ONE most relevant item..."
Then {selection_instruction} is interpolated into step 2 of the TASK section. Reuse this trick whenever the UI has optional constraints.

5.4 Feeding ML Model Output into the Prompt
When the "input" to the prompt is the result of a scikit-learn model, serialise the relevant outputs as plain text and interpolate them into the INPUT section.

from sklearn.tree import DecisionTreeRegressor

def build_ml_prompt(model, feature_names, X_row, prediction, metrics):
    # Pull out interpretable bits from the fitted sklearn model
    importances = dict(zip(feature_names, model.feature_importances_))
    top_features = sorted(importances.items(), key=lambda kv: -kv[1])[:3]
    top_features_str = ", ".join(f"{name} ({imp:.2f})" for name, imp in top_features)
    inputs_str = ", ".join(f"{n}={v}" for n, v in zip(feature_names, X_row))

    return f"""### ROLE ###
You are a data-analysis assistant explaining a machine learning prediction
to a non-technical user in plain English.

### CONTEXT ###
The model is a scikit-learn {type(model).__name__}.
Model quality metrics: R2={metrics['r2']:.3f}, MAE={metrics['mae']:.3f}.
Top 3 most important features (with importance scores): {top_features_str}.

### TASK ###
1. Explain what the prediction means for this specific input.
2. Mention which features most influenced the prediction.
3. Caveat the result using the R2/MAE so the user knows how much to trust it.
4. Format Output: JSON with exact keys `headline`, `explanation`, `confidence_note`.

### INPUT ###
* **Feature values:** {inputs_str}
* **Predicted value:** {prediction:.3f}
"""
Then in your Flask endpoint, call build_ml_prompt(...) instead of the form-based construct_prompt, pass it to model.generate_content(...), strip the fence, parse JSON, and render the three keys in your HTML template.

6. Calling the Model
prompt = construct_prompt(request.form)
model = genai.GenerativeModel('gemma-3n-e4b-it')
response = model.generate_content(prompt)
7. Parsing JSON Output
The model returns JSON wrapped in a ```json ... ``` fence. Strip the fence and parse:

json_response_text = response.text.strip().lstrip('```json').rstrip('```')
data = json.loads(json_response_text)
8. Endpoint Pattern (Flask)
@app.route('/generate', methods=['POST'])
def generate():
    try:
        prompt = construct_prompt(request.form)
        model = genai.GenerativeModel('gemma-3n-e4b-it')
        response = model.generate_content(prompt)
        json_response_text = response.text.strip().lstrip('```json').rstrip('```')
        data = json.loads(json_response_text)
        # Normalise output (capitalisation, defaults, etc.)
        return jsonify(data)
    except Exception as e:
        print(f"AI error: {e}")
        return jsonify({"error": "Generation failed."}), 500
9. Replication Checklist for a New Project
Create venv & install:
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install Flask google-generativeai
pip freeze > requirements.txt
Add config.py with GEMINI_API_KEY and add it to .gitignore.
Configure SDK at app startup with a try/except (see §4).
Write a construct_prompt(form_data) function using the ROLE / CONTEXT / TASK / INPUT skeleton (§5.1). List any taxonomies/allowed values explicitly.
Pick a model — use gemma-3n-e4b-it.
Add a POST endpoint that builds the prompt, calls the model, strips the ```json ``` fence, parses JSON, and returns jsonify.
Frontend: simple fetch('/generate', {method:'POST', body: new FormData(form)}) then render the JSON keys.
Error handling: wrap in try/except, log server-side, return a generic JSON error to the client.
10. Security Notes
Never commit config.py — keep it in .gitignore.
Validate and sanitise form input before interpolating into prompts.