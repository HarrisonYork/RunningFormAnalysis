import os
import torch
import logging
from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from dotenv import load_dotenv

from form_analyzer import FormAnalyzer1DCNN, process_results_api

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

# YOLO model
model_path = os.environ.get("YOLO_MODEL_PATH", ".")
logging.info(f"Loading YOLO model (pose estimation)")
try:
    model = YOLO(model_path, task='pose').to(device)
except Exception as e:
    logging.error(f"Failed to load model: {e}")

# CNN model
cnn_model_path = os.environ.get("CNN_MODEL_PATH", ".")
cnn_model = FormAnalyzer1DCNN(num_features=51, num_classes=4, kernel_size=7).to(device)
logging.info(f"Loading CNN (form analysis)")
try:
    cnn_model.load_state_dict(torch.load(cnn_model_path, map_location=device))
except Exception as e:
    logging.error(f"Failed to load CNN model: {e}")
cnn_model.eval()

# file organization
UPLOAD_FOLDER = '../uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
OUTPUT_FOLDER = '../runs/pose/user_submissions/predict/'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app)

@app.route('/api/pose_estimate', methods=['POST'])
def pose_estimate():
    if 'video' not in request.files:
        logging.warning("No video part in the request")
        return jsonify({"error": "No video file provided in the request."}), 400
        
    file = request.files['video']
    
    if file.filename == '':
        logging.warning("Empty filename submitted")
        return jsonify({"error": "No selected file."}), 400

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        output_filename = f"{filename.rsplit('.', 1)[0]}.mp4"
        output_filepath = os.path.join(OUTPUT_FOLDER, output_filename)
        if os.path.exists(output_filepath):
            os.remove(output_filepath)
        
        logging.info(f"Successfully saved {filename} to {filepath}. Starting YOLO inference...")

        results = model(source=filepath, save=True, stream=True, conf=0.5, project="user_submissions", exist_ok=True)
        logging.info("Pose results complete! Processing results...")
        
        keypoint_tensor = process_results_api(results)

        cnn_input = keypoint_tensor.unsqueeze(0).permute(0, 2, 1).to(device)

        with torch.no_grad():
            logging.info("Results processed successfully. Analyzing form...")
            logits = cnn_model(cnn_input)
            
            # convert raw logits into 0.0 -> 1.0 probabilities
            probabilities = torch.sigmoid(logits).squeeze(0) 
            
        # Map probabilities to percentages for the React frontend
        confidences = {
            "heel_strike": round(probabilities[0].item() * 100, 2),
            "lean_forward": round(probabilities[1].item() * 100, 2),
            "arms_tight": round(probabilities[2].item() * 100, 2),
            "arms_loose": round(probabilities[3].item() * 100, 2)
        }
        logging.info(f"Model predictions: {confidences}")

        if os.path.exists(filepath):
            os.remove(filepath)

        return jsonify({
            "status": "success",
            "confidences": confidences,
            "video_url": f"/api/videos/{output_filename}"
        }), 200

    except Exception as e:
        logging.error(f"Error during video processing: {str(e)}")
        return jsonify({"error": "An internal server error occurred during analysis."}), 500


@app.route('/api/videos/<filename>', methods=['GET'])
def get_video(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, mimetype='video/mp4')
