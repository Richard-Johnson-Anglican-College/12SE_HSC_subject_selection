"""
HSC Subject Recommendation Engine
app.py - Flask application entry point and route handler
"""

from flask import Flask, render_template, request, redirect, url_for, session, flash, Response, jsonify
from functools import wraps
import os
import data_handler as dh
import ml_engine as ml
import gemini_client as gemini

app = Flask(__name__)

try:
    from config import SECRET_KEY
    app.secret_key = SECRET_KEY
except (ImportError, AttributeError):
    app.secret_key = os.urandom(24)
    print("WARNING: SECRET_KEY not set in config.py — sessions will not persist across restarts.")


def _require_admin(f):
    """HTTP Basic Auth decorator for admin routes."""
    @wraps(f)
    def decorated(*args, **kwargs):
        try:
            from config import ADMIN_PASSWORD
        except (ImportError, AttributeError):
            ADMIN_PASSWORD = "admin"
        auth = request.authorization
        if not auth or auth.password != ADMIN_PASSWORD:
            return Response(
                'Admin access required.',
                401,
                {'WWW-Authenticate': 'Basic realm="Admin Dashboard"'}
            )
        return f(*args, **kwargs)
    return decorated

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    """Entry portal — student chooses their path."""
    return render_template('index.html')


@app.route('/survey/predict', methods=['GET', 'POST'])
def survey_predict():
    """Year 7-10 interest questionnaire (inference path)."""
    if request.method == 'POST':
        # Collect all 12 survey answers from the form
        answers = {f'q{i}': request.form.get(f'q{i}', 0) for i in range(1, 13)}
        session['survey_answers'] = answers
        return redirect(url_for('results'))
    return render_template('survey_predict.html', answers=session.get('survey_answers', {}))


@app.route('/survey/train', methods=['GET', 'POST'])
def survey_train():
    """Year 11-12 training data submission (training path)."""
    if request.method == 'POST':
        answers = {f'q{i}': request.form.get(f'q{i}', 0) for i in range(1, 13)}
        target_subjects = request.form.getlist('target_subjects')
        satisfaction = int(request.form.get('satisfaction_score', 5))

        if len(target_subjects) < 5:
            flash('Please select at least 5 subjects.', 'error')
            return redirect(url_for('survey_train'))

        english_subjects = {'English Advanced', 'English Extension', 'English Standard'}
        if not any(s in english_subjects for s in target_subjects):
            flash('English is compulsory — please select English Advanced, Extension, or Standard.', 'error')
            return redirect(url_for('survey_train'))

        student_id = dh.append_training_row(answers, target_subjects, satisfaction)
        flash(f'Thank you! Your data has been saved (ID: {student_id}).', 'success')
        return redirect(url_for('index'))
    return render_template('survey_train.html')


@app.route('/results')
def results():
    """Display predicted subject cluster and match percentage."""
    answers = session.get('survey_answers', None)
    if not answers:
        return redirect(url_for('survey_predict'))

    if not ml.is_trained():
        flash('The model has not been trained yet. Ask your teacher to visit the Admin page.', 'error')
        return redirect(url_for('survey_predict'))

    prediction = ml.predict(answers)

    return render_template(
        'results.html',
        cluster=prediction['cluster'],
        match_pct=prediction['match_pct'],
        subjects=prediction['subjects'],
        cluster_scores=prediction.get('cluster_scores', []),
        chart_url=prediction.get('chart_b64'),
    )


@app.route('/results/summary', methods=['POST'])
def results_summary():
    """Async endpoint — returns Gemini summary as JSON when the student requests it."""
    answers = session.get('survey_answers')
    if not answers or not ml.is_trained():
        return jsonify({'summary': None}), 400

    prediction = ml.predict(answers)
    summary = gemini.generate_summary(
        cluster=prediction['cluster'],
        match_pct=prediction['match_pct'],
        subjects=prediction['subjects'],
        answers=answers,
        cluster_scores=prediction.get('cluster_scores'),
    )
    return jsonify({'summary': summary})


@app.route('/admin')
@_require_admin
def admin():
    """Teacher/admin dashboard for model retraining and visualisations."""
    info = ml.get_model_info()
    model_stats = {
        'version':  info.get('version', '—'),
        'records':  dh.get_record_count(),
        'accuracy': info.get('accuracy'),
        'r2':       info.get('r2'),
    }
    charts = ml.get_chart_urls()
    return render_template(
        'admin.html',
        stats=model_stats,
        training_log=info.get('log', ''),
        **charts
    )


@app.route('/admin/retrain', methods=['POST'])
@_require_admin
def admin_retrain():
    """Trigger ML pipeline retraining."""
    try:
        info = ml.retrain()
        flash(f"Model v{info['version']} trained successfully — Accuracy: {info['accuracy']:.2%}, R²: {info['r2']:.4f}", 'success')
    except Exception as e:
        flash(f'Training failed: {e}', 'error')
    return redirect(url_for('admin'))


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
