"""
# @ Author: Meet Patel
# @ Create Time: 2025-02-15 11:28:41
# @ Modified by: Meet Patel
# @ Modified time: 2025-03-20 22:12:40
# @ Description:
"""

import torch


# Compute overall model sparsity
def compute_global_sparsity(model):
    total_params = 0
    zero_params = 0

    for name, module in model.named_modules():
        if hasattr(module, "weight"):
            total_params += module.weight.nelement()  # Total elements
            zero_params += torch.sum(module.weight == 0).item()  # Zero elements
        elif hasattr(module, "bias") and module.bias is not None:
            total_params += module.bias.nelement()  # Total elements
            zero_params += torch.sum(module.bias == 0).item()  # Zero elements

    sparsity = zero_params / total_params * 100
    return sparsity


def compute_structured_sparsity(base_model, pruned_model):
    def compute_params(model):
        model_params = 0
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.CrossEntropyLoss):
                continue
            if hasattr(module, "weight"):
                model_params += module.weight.nelement()  # Total elements
            elif hasattr(module, "bias") and module.bias is not None:
                model_params += module.bias.nelement()  # Total elements
        return model_params

    base_model_params = compute_params(base_model)
    pruned_model_params = compute_params(pruned_model)

    sparsity = pruned_model_params / base_model_params
    return sparsity, base_model_params, pruned_model_params
