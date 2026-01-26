# src/datasets.py
import os
import cv2
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class MRISegDataset(Dataset):
    """
    Custom Dataset class for LGG Brain MRI segmentation.
    Automatically handles nested folder structure like:
    data/lgg-mri-segmentation/kaggle_3m/<patient_id>/*.tif
    """

    def __init__(self, root_dir, transform=None, use_augment=True):

        """
        Args:
            root_dir (str): Path to dataset root (e.g., "data/lgg-mri-segmentation/kaggle_3m")
            transform (callable, optional): Optional transform to apply to images and masks
        """
        self.root_dir = root_dir
        self.transform = transform

        # ✅ Data augmentation pipeline
        self.augment = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(15),
        ])

        # ✅ Get all image paths recursively
        all_tif_files = sorted(glob.glob(os.path.join(root_dir, "**", "*.tif"), recursive=True))

        # ✅ Separate MRI and mask files
        self.image_paths = [p for p in all_tif_files if "_mask" not in p and "_seg" not in p]
        self.mask_paths = [p for p in all_tif_files if "_mask" in p or "_seg" in p]

        print(f"✅ Found {len(self.image_paths)} MRI images and {len(self.mask_paths)} masks")

        # ✅ Pair images and masks correctly
        paired_images, paired_masks = [], []
        for img_path in self.image_paths:
            mask_path = self._find_mask(img_path)
            if mask_path is not None:
                paired_images.append(img_path)
                paired_masks.append(mask_path)

        self.image_paths = paired_images
        self.mask_paths = paired_masks
        print(f"🎯 Using {len(self.image_paths)} image-mask pairs")

    # -------------------------------------------------------------------------
    def _find_mask(self, img_path):
        """Find the correct mask file for a given MRI image path."""
        base = os.path.splitext(os.path.basename(img_path))[0]  # e.g. TCGA_CS_4941_19960909_1
        folder = os.path.dirname(img_path)

        # Expected mask name pattern
        mask_name = base + "_mask.tif"
        mask_path = os.path.join(folder, mask_name)

        # ✅ 1️⃣ Exact match
        if os.path.exists(mask_path):
            return mask_path

        # ✅ 2️⃣ Fallback: check similar files
        for f in os.listdir(folder):
            f_lower = f.lower()
            if (("_mask" in f_lower or "_seg" in f_lower) and base in f):
                return os.path.join(folder, f)

        # ✅ 3️⃣ Fallback by patient prefix
        patient_prefix = "_".join(base.split("_")[:-1])
        for f in os.listdir(folder):
            f_lower = f.lower()
            if (("_mask" in f_lower or "_seg" in f_lower) and patient_prefix in f):
                return os.path.join(folder, f)

        # ❌ No mask found
        return None
    # -------------------------------------------------------------------------

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # ✅ Read grayscale MRI and mask
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            raise ValueError(f"Error reading files:\nImage: {img_path}\nMask: {mask_path}")

        # ✅ Normalize to [0,1]
        img = img.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)

        # ✅ Add channel dimension (C, H, W)
        img = np.expand_dims(img, axis=0)
        mask = np.expand_dims(mask, axis=0)

        # ✅ Convert to tensors
        img_tensor = torch.from_numpy(img)
        mask_tensor = torch.from_numpy(mask)

        # ✅ Apply same augmentation to both
        if self.augment:
            seed = np.random.randint(0, 9999)
            torch.manual_seed(seed)
            img_tensor = self.augment(img_tensor)
            torch.manual_seed(seed)
            mask_tensor = self.augment(mask_tensor)

        # ✅ Apply any optional transform
        if self.transform:
            img_tensor, mask_tensor = self.transform(img_tensor, mask_tensor)

        return {"image": img_tensor, "mask": mask_tensor}
