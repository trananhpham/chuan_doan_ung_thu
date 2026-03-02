from flask import Flask, request, jsonify, send_from_directory
import os
import logging
from services.predict_service import load_models, predict_ultrasound, predict_biopsy

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Paths: Frontend folder is relative to this app.py (backend/)
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))

# Initialize App — No CORS needed since frontend is served from same Flask server
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path='')

# Load machine learning models at server startup
load_models()

# ---------------------------------------------------------------
# Serve Frontend Pages
# ---------------------------------------------------------------
@app.route('/')
def index():
    """Serve the main web page"""
    return send_from_directory(FRONTEND_DIR, 'index.html')

# ---------------------------------------------------------------
# AI Prediction Endpoints
# ---------------------------------------------------------------
@app.route('/predict/ultrasound', methods=['POST'])
def handle_ultrasound():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    logger.info(f"Received ultrasound image: {file.filename}")
    result = predict_ultrasound(file.read())
    if "error" in result:
        return jsonify(result), 500
    return jsonify(result), 200

@app.route('/predict/biopsy', methods=['POST'])
def handle_biopsy():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    logger.info(f"Received biopsy image: {file.filename}")
    result = predict_biopsy(file.read())
    if "error" in result:
        return jsonify(result), 500
    return jsonify(result), 200

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "running", "message": "API Server is active"}), 200

if __name__ == '__main__':
    logger.info("=========================================")
    logger.info(" Breast Cancer Diagnosis System Running")
    logger.info(" Mở trình duyệt và vào: http://localhost:5000")
    logger.info("=========================================")
    app.run(host='0.0.0.0', port=5000, debug=False)

