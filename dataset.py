import os
import json
import torch
from torch.utils.data import Dataset
import data_utils.transforms as T
from scipy.stats import truncnorm
from PIL import Image
import numpy as np
from .torchvision_datasets import CocoDetection
from util.misc import nested_tensor_from_tensor_list

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

class PoETDataset(CocoDetection):
    """
    通用6D位姿数据集，支持COCO风格/自定义标注，抖动、相机/物体/相对位姿等。
    标注格式要求：每个样本为dict，包含至少 file_name, boxes, labels, relative_pose:position, relative_pose:rotation。
    """
    def __init__(self, images_dir, ann_path, device='cpu', jitter=False, jitter_probability=0.5, std=0.02):
        super(PoETDataset, self).__init__(images_dir, ann_path)
        self.images_dir = images_dir
        self.ann_path = ann_path
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        ])
        self.transform = T.Compose([normalize, ])
        self.device = device
        self.jitter = jitter
        self.jitter_probability = jitter_probability
        self.std = std
        self.anns = load_json(ann_path)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image, target = super(PoETDataset, self).__getitem__(idx)
        img_id = self.ids[idx]
        ann = target
        img_id = torch.tensor([img_id])
        w, h = image.size

        # boxes
        boxes = torch.tensor([obj["bbox"] for obj in ann], dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        labels = torch.tensor([obj["category_id"] for obj in ann], dtype=torch.int64)

        translation = torch.tensor([obj["relative_pose"]['position'] for obj in ann], dtype=torch.float32)
        rotation = torch.tensor([obj["relative_pose"]['rotation'] for obj in ann], dtype=torch.float32)

        # 抖动/jitter
        jitter_boxes = boxes.clone()
        if self.jitter and boxes.shape[0] > 0:
            if np.random.rand() < self.jitter_probability:
                jitter_boxes += torch.randn_like(jitter_boxes) * 0.005

        # 其它可选字段
        target = {
            'boxes': boxes,
            'labels': labels,
            "image_id": img_id,
            'relative_position': translation,
            'relative_rotation': rotation,
            'jitter_boxes': jitter_boxes
        }
        if 'area' in ann:
            target['area'] = torch.tensor(ann['area'], dtype=torch.float32)
        if 'iscrowd' in ann:
            target['iscrowd'] = torch.tensor(ann['iscrowd'], dtype=torch.uint8)
        if 'intrinsics' in ann:
            target['intrinsics'] = torch.tensor(ann['intrinsics'], dtype=torch.float32)

        image, target = self.transform(image, target)
        
        return image, target

    def collate_fn(self, batch):
        batch = list(zip(*batch))
        batch[0] = nested_tensor_from_tensor_list(batch[0])
        if batch[1][0] is None:
            # Targets are None
            batch[1] = None
        return tuple(batch)
