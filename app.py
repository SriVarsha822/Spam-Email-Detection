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
        spam_probability=None,
        ham_probability=None,
        message="",
        color="#000000"
    )



@app.route("/predict", methods=["POST"])
def predict():

    message = request.form["message"]

    vector = vectorizer.transform([message])

    prediction = model.predict(vector)[0]

    probability = model.predict_proba(vector)[0]

    spam_probability = round(probability[1] * 100, 2)

    ham_probability = round(probability[0] * 100, 2)

    confidence = max(spam_probability, ham_probability)

    

    

    if prediction == 1:
        result = "🚨 SPAM EMAIL"
        color = "#ff4d4d"
        reason = "This email contains suspicious promotional words."

    else:
        result = "✅ NOT SPAM"
        color = "#00c853"
        reason = "This email looks like a genuine email."

    return render_template(
    "index.html",
    prediction=result,
    confidence=confidence,
    spam_probability=spam_probability,
    ham_probability=ham_probability,
    message=message,
    color=color,
    reason=reason
)


if __name__ == "__main__":
    app.run(debug=True)