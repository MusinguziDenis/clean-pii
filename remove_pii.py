from pathlib import Path
from clean.clean import clean_image
from inference.inference_yolo_models import yolo_predict
from ultralytics import YOLO
from glob import glob


yolon_model = YOLO("phi_models/best.pt")
image_paths = glob("yolo/images/*.png")
results = yolo_predict(yolon_model, [image_paths[0]])
# Clean the first image
output_dir = "results"
clean_image(image_paths[0], results, output_dir)