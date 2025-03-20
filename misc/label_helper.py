"""
# @ Author: Meet Patel
# @ Create Time: 2025-02-15 11:48:36
# @ Modified by: Meet Patel
# @ Modified time: 2025-03-20 22:13:42
# @ Description:
"""

import json

class_name_to_id = {}
with open("imagenet_class_index.json", "r") as f:
    data = json.load(f)
    for cls_id, cls_data in data.items():
        cls_encoded_name = cls_data[0]
        class_name_to_id[cls_encoded_name] = int(cls_id)


def map_imagenet_class_id(mini_idx, dataset):
    mini_class_name = dataset.features["label"].names[mini_idx]
    imagenet1k_class_id = class_name_to_id[mini_class_name]
    return imagenet1k_class_id
