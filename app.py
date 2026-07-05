from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load trained model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


@app.route("/")
def home():
    return render_template(
        "index.html",
        prediction=None,
        confidence=None,
        message=""
    )


@app.route("/predict", methods=["POST"])
def predict():

    message = request.form["message"]

    vector = vectorizer.transform([message])

    prediction = model.predict(vector)[0]

    probability = model.predict_proba(vector)

    confidence = round(max(probability[0]) * 100, 2)

    if prediction == 1:
        result = "🚨 SPAM EMAIL"
        color = "#ff4d4d"
    else:
        result = "✅ NOT SPAM"
        color = "#00c853"

    return render_template(
        "index.html",
        prediction=result,
        confidence=confidence,
        message=message,
        color=color
    )


if __name__ == "__main__":
    app.run(debug=True)