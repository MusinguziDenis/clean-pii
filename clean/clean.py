# The file imeplements functionality to clean PII from X-ray images given the
# path to the image and the bounding box coordinates of the PII.
# The function clean_pii_from_image takes the following arguments:

from pathlib import Path

import cv2


def clean_image(image_path: str, boxes: list[dict[str, int]], output_dir: str) -> None:
    """Clean PII from an image.

    Args:
        image_path (str): Path to the image
        boxes (List[Dict[str, int]]): List of bounding boxes
        output_dir (str): Output directory

    """
    image = cv2.imread(image_path)
    image_id = Path(image_path).name

    for box in boxes:
        x1, y1, x2, y2 = int(box["xmin"]), int(box["ymin"]), int(box["xmax"]), int(box["ymax"])
        image[y1:y2, x1:x2] = 0

    cv2.imwrite(Path(output_dir)/image_id, image)


if __name__ == "__main__":
    image_path = "images/0b4fc675-Ssemakula_Bashir.png"
    boxes = [{"xmin": 922, "ymin": 3222, "xmax": 1640, "ymax": 3324},
             {"ymin": 445,"xmin":0, "ymax":532, "xmax":513},
             {"ymin": 537,"xmin":2003, "ymax":646, "xmax":2552},
             {"ymin": 3177,"xmin":919, "ymax":3326, "xmax":1648}]
    output_dir = "results"

    clean_image(image_path, boxes, output_dir)
