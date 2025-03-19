
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
    base_model_params = 0
    pruned_model_params = 0
    
    for name, module in base_model.named_modules():
        if hasattr(module, "weight"):
            base_model_params += module.weight.nelement()  # Total elements
        elif hasattr(module, "bias") and module.bias is not None:
            base_model_params += module.bias.nelement()  # Total elements

    for name, module in pruned_model.named_modules():
        if hasattr(module, "weight"):
            pruned_model_params += module.weight.nelement()  # Total elements
        elif hasattr(module, "bias") and module.bias is not None:
            pruned_model_params += module.bias.nelement()  # Total elements

    sparsity = pruned_model_params / base_model_params
    return sparsity
    