from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import logging
import os

logging.basicConfig(level=logging.INFO)

# Config
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/phishing_detector.pkl")
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), "../data/preprocessed_data.pkl")

# Load model and vectorizer once
if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    raise FileNotFoundError("Model or vectorizer file not found. Train the model first!")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    _, _, vectorizer = pickle.load(f)

# Flask app
app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def home():
    """Serve the interactive HTML page."""
    return render_template("index.html")

@app.route('/health', methods=['GET'])
def health():
    """Check if the server is running."""
    return jsonify({"status": "running"})

@app.route('/predict', methods=['POST'])
def predict():
    """Predict if an email is phishing."""
    data = request.json
    email = data.get("email") if data else None

    if not email:
        return jsonify({"error": "Email text is required"}), 400

    try:
        email_vector = vectorizer.transform([email])
        prediction = model.predict(email_vector)
        result = "Phishing" if prediction[0] == 1 else "Not Phishing"
        logging.info(f"Email: {email} | Prediction: {result}")
        return jsonify({"prediction": result})
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({"error": "Prediction failed"}), 500

if __name__ == "__main__":
    logging.info("Starting Phishing Detector API...")
    app.run(debug=True)
