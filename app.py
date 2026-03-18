from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")

# Home route (for browser testing)
@app.route("/")
def home():
    return "MLOps API is running 🚀"

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]
    prediction = model.predict([data])
    return jsonify({"prediction": int(prediction[0])})

# Run app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)