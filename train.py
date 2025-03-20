"""
# @ Author: Meet Patel
# @ Create Time: 2025-02-14 08:15:58
# @ Modified by: Meet Patel
# @ Modified time: 2025-03-20 22:10:55
# @ Description:
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as T
from datasets import load_dataset
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm


class CatDogDataset(torch.utils.data.Dataset):
    # Load dataset from Hugging Face
    dataset = load_dataset("microsoft/cats_vs_dogs")
    # Split dataset into train and validation sets
    train_size = int(0.8 * len(dataset["train"]))
    val_size = len(dataset["train"]) - train_size
    train_dataset, val_dataset = random_split(dataset["train"], [train_size, val_size])

    def __init__(self, is_train=True):
        if is_train:
            self.ds = self.__class__.train_dataset
            self.transform = self.train_transforms()
        else:
            self.ds = self.__class__.val_dataset
            self.transform = self.train_transforms()

    def train_transforms(self):
        transform = transforms.Compose(
            [
                T.Resize((224, 224)),
                T.Lambda(
                    lambda img: img.convert("RGB") if img.mode != "RGB" else img
                ),  # Convert grayscale to RGB
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # ImageNet normalization
            ]
        )
        return transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        example = self.ds[idx]
        img = example["image"]
        example.pop("image")
        example["images"] = self.transform(img)
        example["labels"] = torch.tensor(example["labels"], dtype=torch.long)
        return example


class ResNetClassifier(nn.Module):
    def __init__(self, ckpt_path=None):
        super(ResNetClassifier, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        self.loss_fn = nn.CrossEntropyLoss()

        if ckpt_path is not None:
            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError(f"File not present at: {ckpt_path}.")
            self.model.load_state_dict(torch.load(ckpt_path))

    def forward(self, images, labels=None):
        logits = self.model(images)
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return (loss, logits) if loss is not None else logits


class Trainer:
    def __init__(
        self,
        model,
        train_dataset,
        test_dataset,
        batch_size,
        num_workers,
        num_epochs,
        lr,
        device,
        model_name,
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_epochs = num_epochs
        self.lr = lr
        self.device = device
        self.model_name = model_name

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, num_workers=num_workers
        )
        self.model = model
        self.model = self.model.to(self.device)

    def fit(self):
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # Training loop
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
            for batch in tqdm(self.train_loader):
                images, labels = batch["images"].to(self.device), batch["labels"].to(
                    self.device
                )
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
            print(
                f"Epoch {epoch+1}/{self.num_epochs}, Loss: {total_loss/len(self.train_loader)}, Accuracy: {correct/total*100:.2f}%"
            )

            # Test loop
            self.model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in tqdm(self.test_loader):
                    images, labels = batch["images"].to(self.device), batch[
                        "labels"
                    ].to(self.device)
                    outputs = self.model(images)
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(labels).sum().item()
                    total += labels.size(0)
            print(f"Test Accuracy: {correct/total*100:.2f}%")

        # Save Model
        torch.save(
            self.model.state_dict(),
            f"{self.model_name.replace('.', '_').replace('/', '_')}.pth",
        )

    def test(self):
        # Test loop
        self.model.to(self.device)
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in tqdm(self.test_loader):
                images, labels = batch["images"].to(self.device), batch["labels"].to(
                    self.device
                )
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
        acc = correct / total * 100
        print(f"Test Accuracy: {acc:.2f}%")
        return acc


if __name__ == "__main__":
    # # Hyperparameters
    BATCH_SIZE = 32
    LR = 1e-3
    EPOCHS = 3
    NUM_WORKERS = 4  # Adjust based on CPU cores
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_NAME = "resnet18_cats_dogs.pth"

    train_dataset = CatDogDataset(is_train=True)
    test_dataset = CatDogDataset(is_train=False)
    model = ResNetClassifier(None).to(DEVICE)

    tr = Trainer(
        model,
        train_dataset,
        test_dataset,
        BATCH_SIZE,
        NUM_WORKERS,
        EPOCHS,
        LR,
        DEVICE,
        MODEL_NAME,
    )
    tr.fit()
    tr.test()
