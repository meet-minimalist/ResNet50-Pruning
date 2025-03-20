"""
# @ Author: Meet Patel
# @ Create Time: 2025-02-15 14:16:08
# @ Modified by: Meet Patel
# @ Modified time: 2025-03-20 22:11:55
# @ Description:
"""

import copy
from collections import defaultdict

import torch
import torch.nn as nn
from torch.nn.utils import prune
from tqdm import tqdm

import helper
from train import CatDogDataset, ResNetClassifier, Trainer


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

    for name, module in pruned_model.model.named_modules():
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

    existing_modules = list(pruned_model.model.named_modules())
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
                        getattr(getattr(pruned_model.model, layer_idx), sub_layer_idx),
                        conv_name,
                    )
                    setattr(par_module, downsample_idx, new_conv2d)
                else:
                    par_module = getattr(
                        getattr(pruned_model.model, layer_idx), sub_layer_idx
                    )
                    setattr(par_module, conv_name, new_conv2d)
            else:
                setattr(pruned_model.model, name, new_conv2d)

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
                            getattr(
                                getattr(pruned_model.model, layer_idx), sub_layer_idx
                            ),
                            bn_name,
                        )
                        setattr(par_module, downsample_idx, new_bn)
                    else:
                        par_module = getattr(
                            getattr(pruned_model.model, layer_idx), sub_layer_idx
                        )
                        setattr(par_module, bn_name, new_bn)
                else:
                    setattr(pruned_model.model, name, new_bn)

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
                par_module = getattr(
                    getattr(pruned_model.model, layer_idx), sub_layer_idx
                )
                setattr(par_module, linear_name, new_linear)
            else:
                setattr(pruned_model.model, name, new_linear)

    sparsity, base_param_cnt, prune_param_cnt = helper.compute_structured_sparsity(
        base_model, pruned_model
    )
    return pruned_model, sparsity * 100, base_param_cnt, prune_param_cnt


if __name__ == "__main__":
    # # Hyperparameters
    BATCH_SIZE = 32
    LR = 1e-3
    EPOCHS = 3
    NUM_WORKERS = 4  # Adjust based on CPU cores
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_NAME = "resnet18_cats_dogs"

    train_dataset = CatDogDataset(is_train=True)
    test_dataset = CatDogDataset(is_train=False)
    base_model = ResNetClassifier(None).to(DEVICE)

    pruning_iters = 3
    prune_ratio = 0.1

    base_ckpt = f"{MODEL_NAME}.pth"
    base_model.load_state_dict(torch.load(base_ckpt))
    base_trainer = Trainer(
        base_model,
        train_dataset,
        test_dataset,
        BATCH_SIZE,
        NUM_WORKERS,
        EPOCHS,
        LR,
        DEVICE,
        MODEL_NAME,
    )
    # base_trainer.fit()
    base_acc = base_trainer.test()
    print(f"Accuracy before pruning: {base_acc:.2f}%")

    model = base_model
    for i in tqdm(range(pruning_iters)):
        print(f"[Step-{i+1}] Pruning the model. Prune ratio={prune_ratio}")
        model = model.to("cpu")
        pruned_model, prune_percentage, base_param_cnt, prune_param_cnt = (
            apply_structured_pruning(model, prune_ratio=prune_ratio)
        )
        print(f"[Step-{i+1}] Base model params: {base_param_cnt:,}")
        print(f"[Step-{i+1}] Pruned model params: {prune_param_cnt:,}")
        print(f"[Step-{i+1}] %age params after pruning: {prune_percentage:.2f} %")

        pruned_model.to(DEVICE)
        dummy_input = torch.randn(1, 3, 224, 224).to(torch.float32).to(DEVICE)
        pruned_model(dummy_input)

        # Measure accuracy
        PRUNED_MODEL_NAME = f"{MODEL_NAME}_pruned_iter_{i+1}"
        prune_trainer = Trainer(
            pruned_model,
            train_dataset,
            test_dataset,
            BATCH_SIZE,
            NUM_WORKERS,
            1,
            LR,
            DEVICE,
            PRUNED_MODEL_NAME,
        )
        prune_acc = prune_trainer.test()
        print(f"[Step-{i+1}] Accuracy after pruning: {prune_acc:.2f}%")

        # Recovery training
        prune_trainer.fit()
        prune_acc = prune_trainer.test()
        print(f"[Step-{i+1}] Accuracy after re-training: {prune_acc:.2f}%")

        model = pruned_model
