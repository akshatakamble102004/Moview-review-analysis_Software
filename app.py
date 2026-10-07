import os
import pickle
from flask import Flask, render_template, request, flash, redirect, url_for

# entry point for the application
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'movie_review_analysis_secret_key_2026')

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/', methods=['GET'])
def welcome():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form.get('review', '').strip()
    if not review:
        flash("Please enter a review to analyze.", 'prediction')
        return redirect(url_for('welcome'))

    # Convert text to numerical features using TF-IDF
    review_vector = vectorizer.transform([review])
    prediction = model.predict(review_vector)[0]

    if prediction == 1:
        result = "Positive Review"
    else:
        result = "Negative Review"

    flash(result, 'prediction')
    flash(review, 'review')
    return redirect(url_for('welcome'))


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
