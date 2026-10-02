from flask import Flask, request, jsonify
import pickle
import pandas as pd
import logging
from pathlib import Path

app = Flask(__name__)

Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    filename="logs/model_api.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load trained machine learning model
with open("models/customer_churn_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

FEATURES = [
    "age_normalized",
    "monthly_spend_normalized",
    "tenure_months_normalized",
    "support_calls_normalized",
    "contract_encoded",
    "spend_per_month",
    "high_support_usage"
]


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "Customer Churn Prediction API",
        "status": "running",
        "version": "1.0"
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": True
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = request.get_json()

        values = {
            feature: input_data[feature]
            for feature in FEATURES
        }

        dataframe = pd.DataFrame([values])

        prediction = int(model.predict(dataframe)[0])
        probability = float(
            model.predict_proba(dataframe)[0][prediction]
        )

        result = {
            "prediction": prediction,
            "prediction_label":
                "Churn" if prediction == 1 else "No Churn",
            "probability": round(probability, 4),
            "model_version": "1.0"
        }

        logging.info(
            "Prediction generated successfully: %s",
            result
        )

        return jsonify(result)

    except Exception as error:
        logging.error("Prediction error: %s", error)

        return jsonify({
            "error": str(error)
        }), 400


if __name__ == "__main__":
    print("\nCUSTOMER CHURN MODEL API")
    print("------------------------")
    print("Model loaded successfully.")
    print("API running at http://127.0.0.1:5001")
    print("Health endpoint: http://127.0.0.1:5001/health")

    app.run(
        host="127.0.0.1",
        port=5001,
        debug=False
    )
