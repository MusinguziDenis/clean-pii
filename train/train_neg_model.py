"""Train negative model."""


import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18
from torchvision.transforms import Resize, ToTensor
from tqdm import tqdm


class NegativeDataset(Dataset):
    """Create a PyTorch dataset for the negative samples."""

    def __init__(self, image_ids:list, img_dir:str, labels:list)->None:
        """Initialize the dataset.

        Args:
            image_ids (list): List of image IDs
            img_dir (str): Path to image directory
            labels (list): List of labels

        """
        self.image_ids = image_ids
        self.labels = labels
        self.img_dir = img_dir
        self.transforms = Resize((224, 224)), ToTensor()

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.image_ids)

    def __getitem__(self, idx:int) -> tuple[Image.Image, int]:
        """Get an item from the dataset."""
        img_path = f"{self.img_dir}/{self.image_ids[idx]}"
        image = Image.open(img_path)
        for transform in self.transforms:
            image = transform(image)
        if self.labels is not None:
            return image, self.labels[idx]
        return image, self.image_ids[idx]


def train_model(img_dir:str, train_csv:str, epochs:int) ->nn.Module:
    """Train a model to classify negative samples.

    Args:
        img_dir (str): Path to image directory
        train_csv (str): Path to training CSV
        epochs (int): Number of epochs to train
    Returns:
        nn.Module: Trained model

    """
    # Load data
    train_df = pd.read_csv(train_csv)

    train_negs = train_df[train_df["Finding Label"]
                           == "NEG"]["Image Index"].unique().tolist()

    train_ids = train_df["Image Index"].unique().tolist()

    train, val = train_test_split(train_ids, test_size=0.2, random_state=42)

    y_train = [1 if i in train_negs else 0 for i in train]
    y_val = [1 if i in train_negs else 0 for i in val]

    # Create datasets
    train_dataset = NegativeDataset(train, img_dir, y_train)
    val_dataset = NegativeDataset(val, img_dir, y_val)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Load the pre-trained model
    model: nn.Module = resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Set up training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        model.train()
        train_loss = 0.0
        train_correct = 0
        for images, labels in tqdm(train_loader, desc="Training", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()

        train_loss /= len(train_loader)
        train_accuracy = train_correct / len(train_dataset)
        print(f"Train Loss: {train_loss:.4f},\
               Train Accuracy: {train_accuracy:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for images, labels in tqdm(
                val_loader,
                desc="Validating",
                leave=False,
                ):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = val_correct / len(val_dataset)
        print(
            f"Validation Loss: {val_loss:.4f},\
                  Validation Accuracy: {val_accuracy:.4f}",
        )

    return model


if __name__ == "__main__":
    model = train_model("data/images", "data/negative_samples.csv", epochs=10)
