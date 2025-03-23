
from pathlib import Path
from glob import glob

import cv2
import pandas as pd
import tqdm
from torch import nn
from ultralytics import YOLO


def yolo_predict(model:nn.Module, img_paths:list) -> list[dict]:
    """Predicts bounding boxes on images using a YOLO model.

    Args:
        model (nn.Module): YOLO model
        img_paths (List): List of image paths to predict on
    Returns:
        list[dict]: List of dictionaries containing predictions

    """
    tta_preds = []

    for image_dir in tqdm.tqdm(img_paths, desc="predicting with YOLO"):
        image_id = image_dir.split("/")[-1]
        image = cv2.imread(image_dir)

        predictions = model(
            image, conf=0.4, device="cpu", verbose=False, augment=False,
            )

        final_predictions = []
        for r in predictions:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()

                class_name = model.names[cls]

                xmin, ymin, xmax, ymax = xyxy

                final_predictions.append(
                    {
                        "Image_ID": image_id,
                        "class": class_name,
                        "confidence": conf,
                        "ymin": ymin,
                        "xmin": xmin,
                        "ymax": ymax,
                        "xmax": xmax,
                    },
                )

            if len(boxes) == 0:
                final_predictions.append(
                    {
                        "Image_ID": image_id,
                        "class": "NEG",
                        "confidence": 0,
                        "ymin": 0,
                        "xmin": 0,
                        "ymax": 0,
                        "xmax": 0,
                    },
                )

        tta_preds += final_predictions

    return tta_preds