import os
import logging
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
base_path = os.environ.get("MODEL_PATH", ".")
model_path = os.path.join(base_path, "weights/best.pt")

logging.info(f"Loading YOLO model from: {model_path}")
try:
    model = YOLO(model_path)
except Exception as e:
    logging.error(f"Failed to load model: {e}")

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/api/hello', methods=['GET'])
def hello_world():
    return jsonify({"message": "Hello, World!"})

@app.route('/api/pose_estimate', methods=['POST'])
def pose_estimate():
    if 'video' not in request.files:
        logging.warning("No video part in the request")
        return jsonify({"error": "No video file provided in the request."}), 400
        
    file = request.files['video']
    
    if file.filename == '':
        logging.warning("Empty filename submitted")
        return jsonify({"error": "No selected file."}), 400

    print("Processing video...")
    return
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        logging.info(f"Successfully saved {filename} to {filepath}. Starting YOLO inference...")

        results = model(source=filepath, save=True, conf=0.5, project='user_submissions')

        keypoints_arr = []
        for frame_result in results:
            if frame_result.keypoints is not None:
                kp_data = frame_result.keypoints.data.cpu().numpy().tolist()
                keypoints_arr.append(kp_data)

        logging.info(f"Analysis complete for {filename}. Processed {len(keypoints_arr)} frames with keypoints.")

        # clean up the uploaded raw file to save space
        if os.path.exists(filepath):
            os.remove(filepath)

        return jsonify({
            "status": "success",
            "message": f"Successfully analyzed {filename}",
            "frames_analyzed": len(keypoints_arr),
            # "keypoints": keypoints_arr # Uncomment if you want to send the massive array back to the frontend
        }), 200

    except Exception as e:
        logging.error(f"Error during video processing: {str(e)}")
        return jsonify({"error": "An internal server error occurred during analysis."}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8000)