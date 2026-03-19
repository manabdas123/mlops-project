from flask import Flask, request, jsonify,render_template
import joblib

app = Flask(__name__,template_folder="templates")

# Load trained model
model = joblib.load("model.pkl")

# Home route (for browser testing)
@app.route("/")
def home():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]
    prediction = model.predict([data])
    return jsonify({"prediction": int(prediction[0])})

# Run app
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)