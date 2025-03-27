"""clean PII from X-ray images.

Fills the bounding boxes with white pixels.
"""


from PIL import Image


def clean_image(image: Image.Image, boxes: list[dict[str, int]]) -> None:
    """Clean PII from an image.

    Args:
        image (Image): Numpy array of the image
        boxes (List[Dict[str, int]]): List of bounding boxes

    """
    for box in boxes:
        x1, y1, x2, y2 = int(box["xmin"]), int(box["ymin"]), int(box["xmax"]), int(box["ymax"])
        image[y1:y2, x1:x2] = 255

    return image
