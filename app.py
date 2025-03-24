from io import BytesIO

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file
from PIL import Image
from ultralytics import YOLO

from clean.clean import clean_image
from inference.inference_yolo_models import yolo_predict

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def web_clean_image() -> str:
    """Clean PII from an image."""
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    image = Image.open(file.stream)
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    model = YOLO("phi_models/best.pt")

    bboxes = yolo_predict(model, [file.stream])

    image = clean_image(image, bboxes)

    # Convert the processed image back to a format that can be sent in the response
    _, buffer = cv2.imencode(".png", image)
    image_bytes = BytesIO(buffer).getvalue()

    return send_file(BytesIO(image_bytes), mimetype="image/png")

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080)
