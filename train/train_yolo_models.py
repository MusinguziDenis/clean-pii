from pathlib import Path

import yaml
from torch import nn
from ultralytics import YOLO


def get_trained_yolo_models(config_files:list[str], dataset_path:str, device:str="cuda") -> list[nn.Module]:
    """Train YOLO models using the specified configuration files and dataset."""
    trained_yolo_models = []
    for config_file in config_files:
        with Path.open(config_file) as f:
            config_dict = dict(yaml.safe_load(f))

        model = YOLO(config_dict.pop("model"))

        model.train(data=dataset_path, device=device, **config_dict)

        trained_yolo_models.append(model)

    return trained_yolo_models


if __name__ == "__main__":
    config_files = ["yolo11m_config.yaml"]
    dataset_path = "yolo2/dataset.yaml"
    trained_models = get_trained_yolo_models(config_files, dataset_path)
