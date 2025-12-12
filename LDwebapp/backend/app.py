# backend/app.py
import os
import io
import uuid
import csv
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory

# Prefer TensorFlow's Keras, fall back to standalone Keras if TensorFlow isn't installed
try:
    from tensorflow.keras.models import load_model
except Exception:
    from keras.models import load_model

from utils import segment_cells, draw_bounding_boxes_and_labels

# CONFIG
MODEL_PATH = "leukemia_model.keras"   # place your model file here
STATIC_DIR = "static"
SAVED_DIR = os.path.join(STATIC_DIR, "saved")
CSV_PATH = os.path.join(SAVED_DIR, "predictions_log.csv")
ALLOWED_EXT = {"jpg","jpeg","png","bmp"}

os.makedirs(SAVED_DIR, exist_ok=True)

# Load model ONCE
model = load_model(MODEL_PATH, compile=False)


app = Flask(__name__, static_folder=STATIC_DIR)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".",1)[1].lower() in ALLOWED_EXT

# in backend/app.py - replace preprocess_cell with this
def preprocess_cell(cell):
    """
    Convert crop (BGR) -> RGB, resize to model input and normalize to [0,1].
    Returns shape (1,224,224,3) float32.
    """
    # Convert BGR (OpenCV) to RGB (model expects RGB)
    cell_rgb = cv2.cvtColor(cell, cv2.COLOR_BGR2RGB)
    # Resize to model input
    cell_resized = cv2.resize(cell_rgb, (224, 224))
    # Normalize the same way you trained (scale to 0-1)
    cell_resized = cell_resized.astype("float32") / 255.0
    # Add batch dimension
    cell_input = np.expand_dims(cell_resized, axis=0)
    return cell_input


def save_image_bgr(img_bgr, fname):
    path = os.path.join(SAVED_DIR, fname)
    cv2.imwrite(path, img_bgr)
    return path

def log_predictions(csv_path, row):
    header = ["timestamp","upload_filename","saved_annotated","cell_index","x","y","w","h","pred_label","prob_leukemia","prob_normal"]
    newfile = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if newfile:
            writer.writerow(header)
        writer.writerow(row)

@app.route("/predict", methods=["POST"])
def predict():
    # accepts multiple files via key 'files'
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error":"No files provided"}), 400
    if len(files) > 5:
        return jsonify({"error":"Max 5 files allowed"}), 400

    all_results = []
    for file_obj in files:
        filename = file_obj.filename or f"upload_{uuid.uuid4().hex}.png"
        if not allowed_file(filename):
            return jsonify({"error":"File type not allowed"}), 400

        # read into OpenCV BGR image
        file_bytes = np.frombuffer(file_obj.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error":"Could not decode image"}), 400

        uid = uuid.uuid4().hex
        base_save_name = f"input_{uid}.png"
        # Save original uploaded image
        cv2.imwrite(os.path.join(SAVED_DIR, base_save_name), img)

        # Segmentation
        detections = segment_cells(img)

        boxes_with_labels = []
        per_image_results = []
        for idx, (x,y,w,h,crop) in enumerate(detections):
            cell_input = preprocess_cell(crop)
            preds = model.predict(cell_input)[0]  # [prob_leukemia, prob_normal]
            prob_leukemia, prob_normal = float(preds[0]), float(preds[1])
            label = "leukemia" if prob_leukemia > prob_normal else "normal"

            boxes_with_labels.append({
                "x": int(x), "y": int(y), "w": int(w), "h": int(h),
                "label": label, "prob": prob_leukemia if label=="leukemia" else prob_normal
            })

            # Save crop image as well
            crop_name = f"crop_{uid}_{idx}.png"
            cv2.imwrite(os.path.join(SAVED_DIR, crop_name), crop)

            # Log per cell result to CSV
            log_row = [
                datetime.utcnow().isoformat(),
                filename,
                base_save_name,
                idx, x, y, w, h,
                label,
                prob_leukemia, prob_normal
            ]
            log_predictions(CSV_PATH, log_row)

            per_image_results.append({
                "cell_index": idx,
                "x": int(x), "y": int(y), "w": int(w), "h": int(h),
                "label": label,
                "prob_leukemia": prob_leukemia,
                "prob_normal": prob_normal,
                "crop_url": f"/static/saved/crop_{uid}_{idx}.png"
            })

        # draw annotated image with boxes+labels
        annotated = draw_bounding_boxes_and_labels(img, boxes_with_labels)
        annotated_name = f"annotated_{uid}.png"
        save_image_bgr(annotated, annotated_name)

        all_results.append({
            "original_filename": filename,
            "annotated_url": f"/static/saved/{annotated_name}",
            "input_url": f"/static/saved/{base_save_name}",
            "cells": per_image_results
        })

    return jsonify({"results": all_results})

# Optionally serve the CSV for download / admin
@app.route("/logs")
def download_logs():
    if os.path.exists(CSV_PATH):
        return send_from_directory(SAVED_DIR, "predictions_log.csv", as_attachment=True)
    return jsonify({"error":"No logs yet"}), 404

if __name__ == "__main__":
    # NOTE: For production use a WSGI server and disable debug
    app.run(host="0.0.0.0", port=5000, debug=True)
