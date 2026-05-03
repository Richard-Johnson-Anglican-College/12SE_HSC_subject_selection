"""
HSC Subject Recommendation Engine
app.py - Flask application entry point and route handler
"""

from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
import data_handler as dh
import ml_engine as ml
import gemini_client as gemini

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Required for session usage

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
    return render_template('survey_predict.html')


@app.route('/survey/train', methods=['GET', 'POST'])
def survey_train():
    """Year 11-12 training data submission (training path)."""
    if request.method == 'POST':
        answers = {f'q{i}': request.form.get(f'q{i}', 0) for i in range(1, 13)}
        target_subjects = request.form.getlist('target_subjects')
        satisfaction = int(request.form.get('satisfaction_score', 5))

        if len(target_subjects) < 6:
            flash('Please select at least 6 subjects.', 'error')
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

    gemini_summary = gemini.generate_summary(
        cluster=prediction['cluster'],
        match_pct=prediction['match_pct'],
        subjects=prediction['subjects'],
        answers=answers,
    )

    return render_template(
        'results.html',
        cluster=prediction['cluster'],
        match_pct=prediction['match_pct'],
        subjects=prediction['subjects'],
        gemini_summary=gemini_summary,
        chart_url=prediction.get('chart_b64'),
    )


@app.route('/admin')
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
