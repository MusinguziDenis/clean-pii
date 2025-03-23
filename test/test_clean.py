from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from clean.clean import clean_image


@patch("cv2.imread")
@patch("cv2.imwrite")
def test_clean_image(mock_imwrite: MagicMock, mock_imread:MagicMock) -> None:
    """Test the clean_image function."""
    mock_imread.return_value = MagicMock()

    image_path = "images/0b4fc675-Ssemakula_Bashir.png"
    boxes = [{"xmin": 922, "ymin": 3222, "xmax": 1640, "ymax": 3324},
             {"ymin": 445, "xmin": 0, "ymax": 532, "xmax": 513},
             {"ymin": 537, "xmin": 2003, "ymax": 646, "xmax": 2552},
             {"ymin": 3177, "xmin": 919, "ymax": 3326, "xmax": 1648}]
    output_dir = "results"

    # Call the function
    clean_image(image_path, boxes, output_dir)

    # Check if cv2.imread was called with the correct path
    mock_imread.assert_called_once_with(image_path)

    # Check if cv2.imwrite was called with the correct path and image
    expected_output_path = Path(output_dir) / Path(image_path).with_suffix(".png").name
    mock_imwrite.assert_called_once_with(expected_output_path, mock_imread.return_value)

if __name__ == "__main__":
    pytest.main()
