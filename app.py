import os
import pickle
from flask import Flask, render_template_string, request, flash, redirect, url_for

# entry point for the application
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'movie_review_analysis_secret_key_2026')

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Movie Review Sentiment Analysis</title>
    <style>
        :root {
            --bg: radial-gradient(circle at top, #111111 0%, #0b0b0f 45%, #050607 100%);
            --card: rgba(18, 18, 24, 0.96);
            --text: #f4ecd8;
            --muted: #b99f5d;
            --accent: #d9b24b;
            --accent-dark: #ad8f33;
            --border: rgba(255, 255, 255, 0.08);
            --result-border: rgba(217, 178, 75, 0.4);
            --result-bg: rgba(217, 178, 75, 0.08);
            --result-text: #f6e5b8;
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 24px;
        }
        .container {
            width: 100%;
            max-width: 760px;
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 28px;
            box-shadow: 0 32px 80px rgba(0, 0, 0, 0.35);
            overflow: hidden;
        }
        .hero {
            padding: 38px 38px 24px;
            text-align: center;
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.05), transparent 80%);
        }
        .hero h1 { margin: 0 0 8px; font-size: 2.3rem; letter-spacing: -0.03em; }
        .hero p { margin: 0; color: var(--muted); line-height: 1.5; font-size: 0.98rem; }
        form { padding: 0 34px 34px; display: flex; flex-direction: column; gap: 16px; }
        textarea {
            width: 100%;
            min-height: 180px;
            padding: 16px;
            border: 1px solid #cbd5e1;
            border-radius: 14px;
            resize: vertical;
            font-size: 1rem;
            line-height: 1.6;
            color: #111;
        }
        textarea:focus { outline: none; border-color: var(--accent); box-shadow: 0 0 0 4px rgba(217, 178, 75, 0.25); }
        button {
            align-self: flex-start;
            border: none;
            border-radius: 999px;
            background: linear-gradient(90deg, var(--accent), var(--accent-dark));
            color: #111111;
            padding: 14px 28px;
            font-size: 1rem;
            font-weight: 700;
            cursor: pointer;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            box-shadow: 0 16px 32px rgba(173, 143, 51, 0.28);
        }
        button:hover { transform: translateY(-2px); box-shadow: 0 18px 36px rgba(173, 143, 51, 0.32); }
        .result {
            margin: 0 38px 34px;
            padding: 18px 22px;
            border-radius: 16px;
            border: 1px solid var(--result-border);
            background: var(--result-bg);
            color: var(--result-text);
            font-weight: 700;
            min-height: 56px;
            display: flex;
            align-items: center;
        }
        .result.positive { border-color: #22c55e; color: #4ade80; background: rgba(34, 197, 94, 0.1); }
        .result.negative { border-color: #ef4444; color: #f87171; background: rgba(239, 68, 68, 0.1); }
        .result:empty { display: none; }
        @media (max-width: 640px) {
            .hero { padding: 24px 20px 14px; }
            form { padding: 0 20px 24px; }
            .result { margin: 0 20px 24px; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="hero">
            <h1>Movie Review Sentiment Analysis</h1>
            <p>Paste a movie review and let the model predict whether it leans positive or negative.</p>
        </div>
        <form action="/predict" method="post">
            {% with review_messages = get_flashed_messages(category_filter=['review']) %}
                <textarea name="review" placeholder="Enter your movie review here...">{{ review_messages[0] if review_messages else '' }}</textarea>
            {% endwith %}
            <button type="submit">Predict Sentiment</button>
        </form>
        {% with prediction_messages = get_flashed_messages(category_filter=['prediction']) %}
            {% if prediction_messages %}
                <div class="result {% if 'Positive' in prediction_messages[0] %}positive{% elif 'Negative' in prediction_messages[0] %}negative{% endif %}">
                    {{ prediction_messages[0] }}
                </div>
            {% endif %}
        {% endwith %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def welcome():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form.get('review', '').strip()
    if not review:
        flash("Please enter a review to analyze.", 'prediction')
        return redirect(url_for('welcome'))

    # Convert text to numerical features using TF-IDF
    review_vector = vectorizer.transform([review])
    prediction = model.predict(review_vector)[0]

    result = "Positive Review 😊" if prediction == 1 else "Negative Review 😞"

    flash(result, 'prediction')
    flash(review, 'review')
    return redirect(url_for('welcome'))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
