# ResNet-Pruning
Pytorch pruning of ResNet model. The repo shows code to prune the ResNet18 model trained on Cats v/s Dogs dataset.

1. First we will train the ResNet18 model on Cats v/s Dogs dataset for 3 epochs.
2. Then we will use Pytorch's structured pruning API torch.nn.utils.prune to remove redundant input and/or output channels from the ResNet18 model's last block and FC layer.
3. Due to pruning the accuracy is impacted. To recover the same, we will train the pruned model for 1 epoch on entire dataset. This improves the accuracy.
4. Then we will repeat step 2 and step 3 for total 3 times. This will give us the final pruned model.

| Sr. No. | Model Version | # of params | Test Accuracy | Test Accuracy after training |
|-----------------|-----------------|-----------------|-----------------|-----------------|
| 1 | ResNet18 (Original) | 23,485,568 | 93.64% | N/A |
| 2 | ResNet18 (~11% pruned) | 20,887,302 | 48.91% (random guess) | 91.97% |
| 3 | ResNet18 (~20% pruned) | 18,756,720 | 56.26% (random guess) | 95.19% |
| 4 | ResNet18 (~28% pruned) | 16,990,908 | 63.16% | 91.11% |   


### Run command
```
python prune_resnet.py
```