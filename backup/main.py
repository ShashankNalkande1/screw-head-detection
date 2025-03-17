from flask import Flask, request, render_template, jsonify
import requests
import os
import cv2
import numpy as np
from collections import Counter
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

API_URL = "https://detect.roboflow.com"
API_KEY = "SO5eiEqZLkdsnAp3rokj"
MODEL_ID = "full-screw-head-detection/2"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Send to Roboflow API
    with open(filepath, "rb") as img_file:
        response = requests.post(
            f"{API_URL}/{MODEL_ID}?api_key={API_KEY}",
            files={"file": img_file}
        )
    
    result = response.json()
    
    # Count occurrences of each class
    class_counts = Counter(pred["class"] for pred in result.get("predictions", []))
    
    # Load image and overlay detections
    image = cv2.imread(filepath)
    for pred in result.get("predictions", []):
        x, y, w, h = int(pred["x"]), int(pred["y"]), int(pred["width"]), int(pred["height"])
        class_name = pred["class"]
        confidence = pred["confidence"]
        
        # Draw circle around detected screws
        cv2.circle(image, (x, y), max(w, h) // 2, (0, 255, 0), 5)
        # cv2.putText(image, f"{class_name} ({confidence*100:.1f}%)", (x - 20, y - 10), 
                    # cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    overlay_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"overlay_{filename}")
    cv2.imwrite(overlay_filepath, image)
    
    return jsonify({'image_url': overlay_filepath, 'result': result, 'class_counts': class_counts})

if __name__ == '__main__':
    app.run(debug=True, port=5001)