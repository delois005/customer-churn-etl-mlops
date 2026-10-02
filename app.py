from flask import Flask, request, jsonify
import pickle
import pandas as pd
import logging
from pathlib import Path
from flask import Response, g
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import time

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



# Prometheus monitoring metrics
REQUEST_COUNT = Counter(
    "flask_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "http_status"]
)

REQUEST_LATENCY = Histogram(
    "flask_http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["endpoint"]
)

PREDICTION_COUNT = Counter(
    "model_predictions_total",
    "Total model predictions",
    ["prediction"]
)

@app.before_request
def start_request_timer():
    g.start_time = time.time()

@app.after_request
def record_request_metrics(response):
    endpoint = request.path
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=endpoint,
        http_status=response.status_code
    ).inc()

    if hasattr(g, "start_time"):
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(
            time.time() - g.start_time
        )

    return response

@app.route("/metrics", methods=["GET"])
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

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

        PREDICTION_COUNT.labels(prediction=result["prediction_label"]).inc()
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
        host="0.0.0.0",
        port=5000,
        debug=False
    )
