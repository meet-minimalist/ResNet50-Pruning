import torch
from torch.nn.utils import prune
from collections import defaultdict
import helper
import copy
import torch.nn as nn
import torch
import torch.nn.utils.prune as prune
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm 


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
    prune.ln_structured(module, name='weight', amount=prune_ratio, n=1, dim=dim)
    update_remaining_channels(name, module, pruned_channels, in_channel)

def apply_structured_pruning(base_model, prune_ratio):
    pruned_model = copy.deepcopy(base_model)
    
    # flat_modules = {}
    # Apply structured pruning and store pruned channels
    pruned_channels = defaultdict(dict)     # Dictionary to store pruned channels per layer
    module_dict = {}

    for name, module in pruned_model.named_modules():
        module_dict[name] = module
        # if len(list(module.children())) == 0:
        #     # flat_modules[name] = module
        if isinstance(module, nn.Conv2d):
            # Update out channel
            prune_weights(name, module, prune_ratio, pruned_channels, False)
            
            if name != "conv1":
                # Update in channel
                prune_weights(name, module, prune_ratio, pruned_channels, True)
                           
        elif isinstance(module, nn.Linear):
            if name == "fc":
                # Update in channel
                prune_weights(name, module, prune_ratio, pruned_channels, True)

                
                
    
    # total_layers = 4
    # sub_layer_reps = [3, 4, 6, 3]
    # for i in tqdm(range(1, total_layers + 1)):
    #     for j in range(0, sub_layer_reps[i-1]):
    #         # Update in channel
    #         prune_weights(f"layer{i}.{j}.conv1", flat_modules[f"layer{i}.{j}.conv1"], prune_ratio, pruned_channels, True)
    #         # Update out channel
    #         prune_weights(f"layer{i}.{j}.conv1", flat_modules[f"layer{i}.{j}.conv1"], prune_ratio, pruned_channels, False)
            
    #         # Update in channel
    #         prune_weights(f"layer{i}.{j}.conv2", flat_modules[f"layer{i}.{j}.conv2"], prune_ratio, pruned_channels, True)
    #         # Update out channel
    #         prune_weights(f"layer{i}.{j}.conv2", flat_modules[f"layer{i}.{j}.conv2"], prune_ratio, pruned_channels, False)
    
    #         # Update in channel
    #         prune_weights(f"layer{i}.{j}.conv3", flat_modules[f"layer{i}.{j}.conv3"], prune_ratio, pruned_channels, True)
    #         # Update out channel
    #         prune_weights(f"layer{i}.{j}.conv3", flat_modules[f"layer{i}.{j}.conv3"], prune_ratio, pruned_channels, False)

    #         if j == 0:
    #             # Update in channel
    #             prune_weights(f"layer{i}.{j}.downsample.0", flat_modules[f"layer{i}.{j}.downsample.0"], prune_ratio, pruned_channels, True)
    #             # Update out channel
    #             prune_weights(f"layer{i}.{j}.downsample.0", flat_modules[f"layer{i}.{j}.downsample.0"], prune_ratio, pruned_channels, False)
            
    # # Update in channel
    # prune_weights("fc", flat_modules["fc"], prune_ratio, pruned_channels, True)

    # module_dict = {}
    # for name, module in pruned_model.named_modules():
    #     module_dict[name] = module
        
    existing_modules = list(pruned_model.named_modules())
    for name, module in tqdm(existing_modules):
        if name in pruned_channels and isinstance(module, nn.Conv2d):
            channel_dict = pruned_channels[name]
            if "input" in channel_dict:
                new_in_channels = channel_dict["input"].sum().item()    # Count remaining filters
            else:
                new_in_channels = module.in_channels
                
            if "output" in channel_dict:
                new_out_channels = channel_dict["output"].sum().item()  # Count remaining filters
            else:
                new_out_channels = module.out_channels
            
            module.in_channels = new_in_channels
            module.out_channels = new_out_channels
            
            if "output" in channel_dict:
                module.weight.data = module.weight.data[channel_dict["output"]].clone()
                if module.bias is not None:
                    module.bias.data = module.bias.data[channel_dict["output"]].clone()
                
            if "input" in channel_dict:
                module.weight.data = module.weight.data[:, channel_dict["input"], :, :].clone()
            
        if isinstance(module, nn.BatchNorm2d):
            conv_name = name.replace("bn", "conv")
            
            channel_dict = pruned_channels[conv_name]
            if "output" in channel_dict:
                new_out_channels = channel_dict["output"].sum().item()  # Count remaining filters
                module.num_features = new_out_channels
            
                module.running_mean.data = module.running_mean.data[channel_dict["output"]].clone()
                module.running_var.data = module.running_var.data[channel_dict["output"]].clone()
                if module.affine:
                    module.weight.data = module.weight.data[channel_dict["output"]].clone()
                    module.bias.data = module.bias.data[channel_dict["output"]].clone()
                
    import pdb; pdb.set_trace()
    sparsity = helper.compute_structured_sparsity(base_model, pruned_model)
    print(sparsity)


model = models.resnet50(pretrained=True)
model.eval()

apply_structured_pruning(model, prune_ratio=0.2)
