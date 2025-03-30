"""Flask app to remove PII from x-ray images."""


from flask import Flask, Response, jsonify, request
from ultralytics import YOLO

from inference.inference_yolo_models import yolo_predict

app = Flask(__name__)

# Load the model once the app starts
model = YOLO("phi_models/best.pt")

@app.route("/predict", methods=["POST"])
def web_clean_image() -> Response | tuple[Response, int]:
    """Clean PII from an image."""
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]

    bboxes = yolo_predict(model, [file.stream], device="cpu")
    all_bboxes = [([bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]])\
                   for bbox in bboxes]
    all_bboxes = [[round(x, 2) for x in bbox if x > 0] for bbox in all_bboxes]

    # Return bounding boxes as JSON
    return jsonify({"bboxes": all_bboxes}), 200

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080)
