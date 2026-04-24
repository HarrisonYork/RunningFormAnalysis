import os
import torch
import logging
import subprocess
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from dotenv import load_dotenv
from form_analyzer import process_results


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
model_path = os.environ.get("MODEL_PATH", ".")
if model_path.startswith("runs"):
    model_path = os.path.join(model_path, "weights/best.pt")

logging.info(f"Loading YOLO model from: {model_path}")
try:
    model = YOLO(model_path)
except Exception as e:
    logging.error(f"Failed to load model: {e}")

app = Flask(__name__)
CORS(app)

# path to save user upload (unedited video)
UPLOAD_FOLDER = 'tmp_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# path to video with pose overlay
OUTPUT_FOLDER = 'runs/pose/user_submissions/predict/'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

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
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        output_filename = f"{filename.rsplit('.', 1)[0]}.mp4"
        output_filepath = os.path.join(OUTPUT_FOLDER, output_filename)
        if os.path.exists(output_filepath):
            os.remove(output_filepath)
        
        logging.info(f"Successfully saved {filename} to {filepath}. Starting YOLO inference...")

        results = model(source=filepath, save=True, stream=True, conf=0.5, project="user_submissions", exist_ok=True)

        print("Pose results complete! Processing results...")
        
        process_results(results, filename)

        # Clean up the original uploaded file to save space
        if os.path.exists(filepath):
            os.remove(filepath)

        # Send the actual video file back to the React frontend
        return send_file(output_filepath, mimetype='video/mp4')

    except Exception as e:
        logging.error(f"Error during video processing: {str(e)}")
        return jsonify({"error": "An internal server error occurred during analysis."}), 500

# if __name__ == '__main__':
#     app.run(debug=True, port=8000)


# performance notes:
# 793 frames in 58 s using small model
# same video done in 38.46 s using nano 