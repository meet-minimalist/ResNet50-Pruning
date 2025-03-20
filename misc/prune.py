"""
# @ Author: Meet Patel
# @ Create Time: 2025-02-15 11:20:04
# @ Modified by: Meet Patel
# @ Modified time: 2025-03-20 22:13:10
# @ Description:
"""

import copy
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision.models as models
import torchvision.transforms as transforms
from datasets import load_dataset
from torch.nn.utils import prune
from torch.utils.data import DataLoader
from tqdm import tqdm

import helper
from misc.label_helper import map_imagenet_class_id

# Example usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def update_remaining_channels(name, module, pruned_channels, in_channel=True):
    mask = module.weight_mask
    channel = "input" if in_channel else "output"
    weight_shape = len(mask.shape)
    if in_channel:
        if weight_shape == 4:
            # out_channel, in_channel, k_height, k_width --> Conv layer
            dim = [0, 2, 3]
        else:
            # out_channel, in_channel --> FC layer
            dim = [0]
        remaining_out_channels = torch.sum(mask, dim=dim) > 0
    else:
        if weight_shape == 4:
            # out_channel, in_channel, k_height, k_width --> Conv layer
            dim = [1, 2, 3]
        else:
            # out_channel, in_channel --> FC layer
            dim = [1]
        remaining_out_channels = torch.sum(mask, dim=dim) > 0
    pruned_channels[name][channel] = remaining_out_channels


def prune_weights(name, module, prune_ratio, pruned_channels, in_channel=True):
    dim = 1 if in_channel else 0
    prune.ln_structured(module, name="weight", amount=prune_ratio, n=1, dim=dim)
    update_remaining_channels(name, module, pruned_channels, in_channel)


def apply_structured_pruning(base_model, prune_ratio):
    pruned_model = copy.deepcopy(base_model)

    # Apply structured pruning and store pruned channels
    pruned_channels = defaultdict(dict)  # Dictionary to store pruned channels per layer
    module_dict = {}

    for name, module in pruned_model.named_modules():
        if (
            name == "conv1"
            or name.startswith("layer1")
            or name.startswith("layer2")
            or name.startswith("layer3")
        ):
            continue
        module_dict[name] = module
        if isinstance(module, nn.Conv2d):
            # Update out channel
            prune_weights(name, module, prune_ratio, pruned_channels, False)

            if (
                name != "conv1"
                and name != "layer4.0.conv1"
                and name != "layer4.0.downsample.0"
            ):
                # Update in channel
                prune_weights(name, module, prune_ratio, pruned_channels, True)

        elif isinstance(module, nn.Linear):
            if name == "fc":
                # Update in channel
                prune_weights(name, module, prune_ratio, pruned_channels, True)

    existing_modules = list(pruned_model.named_modules())
    for name, module in tqdm(existing_modules):
        if (
            name == "conv1"
            or name == "bn1"
            or name.startswith("layer1")
            or name.startswith("layer2")
            or name.startswith("layer3")
        ):
            continue
        if name in pruned_channels and isinstance(module, nn.Conv2d):
            channel_dict = pruned_channels[name]
            if "input" in channel_dict:
                new_in_channels = (
                    channel_dict["input"].sum().item()
                )  # Count remaining filters
            else:
                new_in_channels = module.in_channels

            if "output" in channel_dict:
                new_out_channels = (
                    channel_dict["output"].sum().item()
                )  # Count remaining filters
            else:
                new_out_channels = module.out_channels

            new_conv2d = nn.Conv2d(
                in_channels=new_in_channels,
                out_channels=new_out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=module.bias,
                padding_mode=module.padding_mode,
            )
            if "input" in channel_dict:
                new_conv2d.weight.data = module.weight.data[
                    :, channel_dict["input"], :, :
                ].clone()

            if "output" in channel_dict:
                if "input" in channel_dict:
                    new_conv2d.weight.data = new_conv2d.weight.data[
                        channel_dict["output"]
                    ].clone()
                    if module.bias is not None:
                        new_conv2d.bias.data = new_conv2d.bias.data[
                            channel_dict["output"]
                        ].clone()
                else:
                    new_conv2d.weight.data = module.weight.data[
                        channel_dict["output"]
                    ].clone()
                    if module.bias is not None:
                        new_conv2d.bias.data = module.bias.data[
                            channel_dict["output"]
                        ].clone()

            if name.startswith("layer"):
                splits = name.split(".")
                layer_idx, sub_layer_idx, conv_name = splits[:3]
                if len(splits) == 4:
                    downsample_idx = splits[3]
                    par_module = getattr(
                        getattr(getattr(pruned_model, layer_idx), sub_layer_idx),
                        conv_name,
                    )
                    setattr(par_module, downsample_idx, new_conv2d)
                else:
                    par_module = getattr(
                        getattr(pruned_model, layer_idx), sub_layer_idx
                    )
                    setattr(par_module, conv_name, new_conv2d)
            else:
                setattr(pruned_model, name, new_conv2d)

        if isinstance(module, nn.BatchNorm2d):
            if "downsample" in name:
                conv_name = name.replace("downsample.1", "downsample.0")
            else:
                conv_name = name.replace("bn", "conv")

            channel_dict = pruned_channels[conv_name]
            if "output" in channel_dict:
                new_out_channels = (
                    channel_dict["output"].sum().item()
                )  # Count remaining filters

                new_bn = nn.BatchNorm2d(
                    num_features=new_out_channels,
                    eps=module.eps,
                    momentum=module.momentum,
                    affine=module.affine,
                )

                new_bn.running_mean.data = module.running_mean.data[
                    channel_dict["output"]
                ].clone()
                new_bn.running_var.data = module.running_var.data[
                    channel_dict["output"]
                ].clone()
                if module.affine:
                    new_bn.weight.data = module.weight.data[
                        channel_dict["output"]
                    ].clone()
                    new_bn.bias.data = module.bias.data[channel_dict["output"]].clone()

                if name.startswith("layer"):
                    splits = name.split(".")
                    layer_idx, sub_layer_idx, bn_name = splits[:3]
                    if len(splits) == 4:
                        downsample_idx = splits[3]
                        par_module = getattr(
                            getattr(getattr(pruned_model, layer_idx), sub_layer_idx),
                            bn_name,
                        )
                        setattr(par_module, downsample_idx, new_bn)
                    else:
                        par_module = getattr(
                            getattr(pruned_model, layer_idx), sub_layer_idx
                        )
                        setattr(par_module, bn_name, new_bn)
                else:
                    setattr(pruned_model, name, new_bn)

        if isinstance(module, nn.Linear):
            channel_dict = pruned_channels[name]
            if "input" in channel_dict:
                new_in_channels = (
                    channel_dict["input"].sum().item()
                )  # Count remaining filters
            else:
                new_in_channels = module.in_features

            if "output" in channel_dict:
                new_out_channels = (
                    channel_dict["output"].sum().item()
                )  # Count remaining filters
            else:
                new_out_channels = module.out_features

            new_linear = nn.Linear(
                in_features=new_in_channels,
                out_features=new_out_channels,
                bias=module.bias is not None,
            )

            if "input" in channel_dict:
                new_linear.weight.data = module.weight.data[
                    :, channel_dict["input"]
                ].clone()

            if "output" in channel_dict:
                if "input" in channel_dict:
                    new_linear.weight.data = new_linear.weight.data[
                        channel_dict["output"]
                    ].clone()
                    if module.bias is not None:
                        new_linear.bias.data = new_linear.bias.data[
                            channel_dict["output"]
                        ].clone()
                else:
                    new_linear.weight.data = module.weight.data[
                        channel_dict["output"]
                    ].clone()
                    if module.bias is not None:
                        new_linear.bias.data = module.bias.data[
                            channel_dict["output"]
                        ].clone()

            if name.startswith("layer"):
                splits = name.split(".")
                layer_idx, sub_layer_idx, linear_name = splits
                par_module = getattr(getattr(pruned_model, layer_idx), sub_layer_idx)
                setattr(par_module, linear_name, new_linear)
            else:
                setattr(pruned_model, name, new_linear)

    sparsity = helper.compute_structured_sparsity(base_model, pruned_model)
    print(f"%age params after pruning: {sparsity*100:.2f}")
    return pruned_model


model = models.resnet50(pretrained=True)
model.eval()

pruned_model = apply_structured_pruning(model, prune_ratio=0.1)
pruned_model.eval()
import pdb

pdb.set_trace()
dummy_input = torch.randn(1, 3, 224, 224).to(torch.float32)
pruned_model(dummy_input)

torch.onnx.export(
    pruned_model,
    (dummy_input,),
    "./pruned_resnet50.onnx",
    input_name=["inputs"],
    output_names=["logits"],
    dynamic_axes={"inputs": {0: "batch"}, "logits": {0: "batch"}},
)


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


# Original model
weights = models.ResNet50_Weights.DEFAULT
preprocess = weights.transforms()
dataloader = get_imagenet_dataloader(preprocess)

original_acc = evaluate_model(model, dataloader, device)
print(f"Accuracy before pruning: {original_acc:.2f}%")

# Pruned model
pruned_acc = evaluate_model(pruned_model, dataloader, device)
print(f"Accuracy after pruning: {pruned_acc:.2f}%")
