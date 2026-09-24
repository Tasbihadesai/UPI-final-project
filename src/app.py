"""
app.py
Simple Flask API to serve predictions locally.
POST /predict  - accepts JSON transaction and returns prediction
"""

from flask import Flask, request, jsonify
from predict import predict_transaction
import traceback

app = Flask(__name__)

@app.route('/')
def health():
    return jsonify({"status": "ok", "service": "UPI-fraud-local"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        payload = request.get_json()
        if payload is None:
            return jsonify({"error": "no json body"}), 400
        # single transaction or list
        if isinstance(payload, list):
            results = [predict_transaction(tx) for tx in payload]
            return jsonify(results)
        else:
            res = predict_transaction(payload)
            return jsonify(res)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # dev server - for local testing
    app.run(host='0.0.0.0', port=8000, debug=False)
