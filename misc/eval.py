"""
# @ Author: Meet Patel
# @ Create Time: 1970-02-15 11:26:43
# @ Modified by: Meet Patel
# @ Modified time: 2025-03-20 22:14:09
# @ Description:
"""

import torch
import torch.nn.utils.prune as prune
import torchvision.models as models
import torchvision.transforms as transforms
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from misc.label_helper import map_imagenet_class_id


def prune_resnet50(amount=0.2):
    model = models.resnet50(pretrained=True)
    model.eval()  # Set to evaluation mode

    # Prune all convolutional layers
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name="weight", amount=amount)

    return model


def evaluate_model(model, dataloader, device):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images, labels = batch["image"].to(device), batch["label"].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


# Load ImageNet dataset from Hugging Face
def get_imagenet_dataloader(imagenet_preproc, batch_size=32):
    # Only load test set which contains 5000 images
    dataset = load_dataset(
        "timm/mini-imagenet",
        data_files={"test": "data/test-0000*-of-00002.parquet"},
        split="test",
        verification_mode="no_checks",
    )
    transform = transforms.Compose(
        [
            transforms.Lambda(
                lambda img: img.convert("RGB") if img.mode != "RGB" else img
            ),  # Convert grayscale to RGB
            transforms.Lambda(lambda img: imagenet_preproc(img)),
        ]
    )
    dataset.set_transform(
        lambda examples: {
            "image": [transform(img) for img in examples["image"]],
            "label": [
                torch.tensor(map_imagenet_class_id(idx, dataset))
                for idx in examples["label"]
            ],
        }
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)


# Example usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Original model
weights = models.ResNet50_Weights.DEFAULT
preprocess = weights.transforms()
dataloader = get_imagenet_dataloader(preprocess)
original_model = models.resnet50(weights=weights)
original_acc = evaluate_model(original_model, dataloader, device)
print(f"Accuracy before pruning: {original_acc:.2f}%")


# Pruned model
pruned_model = prune_resnet50(amount=0.5)  # Prune 30% of the weights in Conv2d layers
pruned_acc = evaluate_model(pruned_model, dataloader, device)
print(f"Accuracy after pruning: {pruned_acc:.2f}%")

# To remove pruning reparameterization and store permanently
for name, module in pruned_model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        prune.remove(module, "weight")
