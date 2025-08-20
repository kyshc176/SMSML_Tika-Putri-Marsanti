import logging
from flask import Flask, request, jsonify, Response
import time
import psutil
import random
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

# ===== DEFINISI METRIK =====
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP Requests')
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP Request Latency')
REQUEST_SUCCESS = Counter('http_request_success_total', 'Total Successful Requests')
# ... (sisa metrik Anda)

@app.route('/metrics', methods=['GET'])
def metrics():
    # ... (isi fungsi metrics tetap sama)
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    REQUEST_COUNT.inc()
    logging.info("Received request on /predict")

    try:
        # === SIMULASI MODEL PREDICTION ===
        time.sleep(random.uniform(0.05, 0.2))
        simulated_prediction = {"prediction": [random.choice([0, 1])]}
        # ===============================

        REQUEST_SUCCESS.inc()
        duration = time.time() - start_time
        REQUEST_LATENCY.observe(duration)

        logging.info(f"Simulated prediction successfully. Result: {simulated_prediction}")
        return jsonify(simulated_prediction), 200
    except Exception as e:
        logging.error(f"An error occurred during simulation: {e}")
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    # Menjalankan di 127.0.0.1 karena berjalan di host
    app.run(host='127.0.0.1', port=8001)
