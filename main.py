import os
import cv2
import json
import requests
import numpy as np
from flask import Flask, request, render_template, send_file, jsonify
from PIL import Image, ImageDraw, ImageFont
import time

# Flask App Setup
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Roboflow API Configuration
ROBOFLOW_API_KEY = "SO5eiEqZLkdsnAp3rokj"
ROBOFLOW_API_URL = "https://detect.roboflow.com/full-screw-head-detection/2"

def create_table_image(predictions, screw_counts):
    """ Creates an image displaying the detected objects in a stacked table format. """
    # Set table dimensions
    table_width = 400  # Fixed width for the table
    row_height = 30  # Height of each row
    padding = 20  # Padding around the text

    # Calculate table height based on the number of rows
    num_rows = len(predictions) * 6 + len(screw_counts) + 2  # 6 rows per prediction + screw count rows + headers
    table_height = num_rows * row_height + 2 * padding

    # Create a new image with white background
    table_img = Image.new("RGB", (table_width, table_height), "white")
    draw = ImageDraw.Draw(table_img)

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()

    # Draw table headers
    y_offset = padding
    draw.text((padding, y_offset), "Detection Results", fill="black", font=font)
    y_offset += row_height

    # Draw detection data for each prediction
    for pred in predictions:
        draw.text((padding, y_offset), f"Class: {pred['class']}", fill="black", font=font)
        y_offset += row_height
        draw.text((padding, y_offset), f"Confidence: {pred['confidence'] * 100:.2f}%", fill="black", font=font)
        y_offset += row_height
        draw.text((padding, y_offset), f"X: {pred['x']}", fill="black", font=font)
        y_offset += row_height
        draw.text((padding, y_offset), f"Y: {pred['y']}", fill="black", font=font)
        y_offset += row_height
        draw.text((padding, y_offset), f"Width: {pred['width']}", fill="black", font=font)
        y_offset += row_height
        draw.text((padding, y_offset), f"Height: {pred['height']}", fill="black", font=font)
        y_offset += row_height

    # Draw screw count
    draw.text((padding, y_offset), "Screw Count:", fill="black", font=font)
    y_offset += row_height
    for key, value in screw_counts.items():
        draw.text((padding, y_offset), f"{key}: {value}", fill="black", font=font)
        y_offset += row_height

    # Save the table image with epoch time in the filename
    epoch_time = int(time.time())  # Get current epoch time
    table_path = os.path.join(PROCESSED_FOLDER, f"table_{epoch_time}.png")
    table_img.save(table_path)
    return table_path

def merge_images(image_path, table_image_path, final_image_path, padding=20):
    """
    Merges the detection image with the table image, adding a specified amount of white space between them.
    """
    img1 = cv2.imread(image_path)
    img2 = cv2.imread(table_image_path)

    # Get dimensions
    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]

    # Determine the final image dimensions
    final_width = max(width1, width2)
    final_height = height1 + height2 + padding  # Add padding between the images

    # Create a new image with white background
    final_img = np.ones((final_height, final_width, 3), dtype=np.uint8) * 255

    # Place the uploaded image at the top
    final_img[:height1, :width1] = img1

    # Place the table image at the bottom, with padding in between
    final_img[height1 + padding:height1 + padding + height2, :width2] = img2

    # Save the final image with epoch time in the filename
    epoch_time = int(time.time())  # Get current epoch time
    final_image_path_with_epoch = os.path.join(PROCESSED_FOLDER, f"final_output_{epoch_time}.png")
    cv2.imwrite(final_image_path_with_epoch, final_img)

    return final_image_path_with_epoch

def process_image(image_path):
    """ Sends the image to Roboflow API and processes the response. """
    with open(image_path, "rb") as img_file:
        response = requests.post(
            f"{ROBOFLOW_API_URL}?api_key={ROBOFLOW_API_KEY}",
            files={"file": img_file}
        )

    data = response.json()

    if "predictions" not in data:
        print("Error: No predictions received from Roboflow API")
        return None

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return None

    screw_counts = {}
    for pred in data["predictions"]:
        x, y, w, h = int(pred["x"]), int(pred["y"]), int(pred["width"]), int(pred["height"])
        class_name = pred["class"]

        # Draw bounding circles
        cv2.circle(image, (x, y), max(w, h) // 2, (0, 255, 0), 2)

        # Count screws
        screw_counts[class_name] = screw_counts.get(class_name, 0) + 1

    # Save processed image with epoch time in the filename
    epoch_time = int(time.time())  # Get current epoch time
    processed_image_path = os.path.join(PROCESSED_FOLDER, f"processed_{epoch_time}.png")
    cv2.imwrite(processed_image_path, image)

    # Generate table image
    table_image_path = create_table_image(data["predictions"], screw_counts)

    # Merge images with custom padding
    final_image_path = merge_images(processed_image_path, table_image_path, "final_output.png", padding=30)

    return final_image_path

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded file with epoch time in the filename
    epoch_time = int(time.time())  # Get current epoch time
    filename = f"{os.path.splitext(file.filename)[0]}_{epoch_time}{os.path.splitext(file.filename)[1]}"
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(image_path)

    processed_image_path = process_image(image_path)
    if processed_image_path is None:
        return jsonify({"error": "Error processing image"}), 500

    return jsonify({
        "processed_image": processed_image_path,
        "download_link": f"/download?filename={os.path.basename(processed_image_path)}"
    })

@app.route("/download", methods=["GET"])
def download():
    file_path = os.path.join(PROCESSED_FOLDER, request.args.get("filename"))
    return send_file(file_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True, port=5001)