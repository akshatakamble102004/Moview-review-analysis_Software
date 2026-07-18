# url rounting

from flask import Flask, render_template, request, jsonify
import pickle
# entry point for the application
app = Flask(__name__)
app.secret_key = 'replace_this_with_a_random_secret'
# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/', methods=['GET'])
def welcome():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']

    # Convert text to numerical features using TF-IDF
    review_vector = vectorizer.transform([review])
    prediction = model.predict(review_vector)[0]

    if prediction == 1:
        result = "Positive Review"
    else:
        result = "Negative Review"

    from flask import flash, redirect, url_for
    flash(result, 'prediction')
    flash(review, 'review')
    return redirect(url_for('welcome'))


if __name__ == '__main__':
    app.run(debug=True)

