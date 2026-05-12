"""
MotivAid Blood Loss — Dataset Loader
======================================
PyTorch Dataset for loading blood-stained images with masks and labels.
Used by both train_seg.py and train_reg.py.

Usage: Imported by training scripts.
"""

import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

SURFACE_MAP = {
    'bowl': 0,
    'container': 1,
    'pad': 2,
    'pampers': 3,
    'drape': 4,
    'floor': 5,
    'cloth': 6,
    'bedsheet': 7,
    'towel': 8,
    'gauze': 9,
    'sheet': 10,
    'floor-and-cloth': 11,
    'cloth-and-floor': 12,
    'pad-and-container': 13,
    'pad-and-floor': 14,
    'other': 15
}


def get_transforms(mode='train'):
    """Get augmentation pipeline for training or validation."""
    if mode == 'train':
        return A.Compose([
            A.RandomResizedCrop(size=(256, 256), scale=(0.7, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=25, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.35, contrast_limit=0.25, p=0.8),
            A.HueSaturationValue(
                hue_shift_limit=12, sat_shift_limit=25, val_shift_limit=20, p=0.7),
            A.CLAHE(clip_limit=4.0, p=0.5),
            A.GaussianBlur(blur_limit=(1, 3), p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(height=256, width=256),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])


class BloodLossDataset(Dataset):
    """
    Dataset for blood loss estimation.
    
    Args:
        image_dir: Path to image directory
        mask_dir: Path to mask directory
        labels_file: Path to labels JSON file
        mode: 'train' or 'val'
    """

    def __init__(self, image_dir, mask_dir, labels_file, mode='train'):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.mode = mode
        self.transforms = get_transforms(mode)

        # Load labels
        with open(labels_file, 'r') as f:
            all_labels = json.load(f)

        # Determine which split to use based on directory name
        split_name = os.path.basename(image_dir)
        if split_name in all_labels:
            self.labels = all_labels[split_name]
        else:
            # Flatten all labels
            self.labels = {}
            for split_data in all_labels.values():
                if isinstance(split_data, dict):
                    self.labels.update(split_data)

        # Only include images that exist and have labels
        self.samples = []
        for fname, info in self.labels.items():
            img_path = os.path.join(image_dir, fname)
            if os.path.exists(img_path):
                self.samples.append((fname, info))

        print(f"  Dataset [{mode}]: {len(self.samples)} samples from {image_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, info = self.samples[idx]

        # Load image
        img_path = os.path.join(self.image_dir, fname)
        img = np.array(Image.open(img_path).convert('RGB'))

        # Load mask if available
        mask_name = os.path.splitext(fname)[0] + '_mask.png'
        mask_path = os.path.join(self.mask_dir, mask_name)
        if os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path).convert('L'))
            mask = (mask > 127).astype(np.float32)
        else:
            # If no mask, use a dummy full mask
            mask = np.ones((img.shape[0], img.shape[1]), dtype=np.float32)

        # Apply augmentations
        augmented = self.transforms(image=img, mask=mask)
        img_tensor = augmented['image']  # (3, H, W)
        mask_tensor = augmented['mask'].unsqueeze(0)  # (1, H, W)

        # Volume label (log-transform for regression)
        volume_ml = info['volume_ml']
        log_volume = np.log(max(volume_ml, 1.0))  # log(0) protection

        # Surface type one-hot
        surface = info.get('surface_type', 'other')
        surface_idx = SURFACE_MAP.get(surface, 15)  # Default to 'other' index
        surface_onehot = torch.zeros(16)  # Updated to 16 surface types
        surface_onehot[surface_idx] = 1.0

        # Extras: distance (cm) / 100, light (0=day, 1=led, 2=dim), clot (0 or 1)
        dist = info.get('distance_cm', 40.0) / 100.0
        lighting_str = info.get('lighting', 'daylight')
        light_val = 0.0
        if lighting_str == 'led': light_val = 1.0
        elif lighting_str == 'dim': light_val = 2.0
        clot_val = 1.0 if info.get('has_clot', False) else 0.0
        extras = torch.tensor([dist, light_val, clot_val], dtype=torch.float32)

        return {
            'image': img_tensor,
            'mask': mask_tensor,
            'volume_ml': torch.tensor(volume_ml, dtype=torch.float32),
            'log_volume': torch.tensor(log_volume, dtype=torch.float32),
            'surface_onehot': surface_onehot,
            'surface_idx': surface_idx,
            'extras': extras,
            'filename': fname
        }
