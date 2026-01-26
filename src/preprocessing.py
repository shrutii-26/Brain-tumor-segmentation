# src/preprocessing.py
import cv2
import numpy as np
import albumentations as A

class Preprocessor:
    def __init__(self, use_clahe=True, use_zscore=True, use_augment=True):
        self.use_clahe = use_clahe
        self.use_zscore = use_zscore
        self.use_augment = use_augment

        # Define Albumentations augmentations
        if use_augment:
            self.augment = A.Compose([
                A.ElasticTransform(alpha=50, sigma=5, alpha_affine=5, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.RandomBrightnessContrast(p=0.2),
            ])
        else:
            self.augment = None

    def apply(self, img, mask=None):
        # --- 1️⃣ CLAHE
        if self.use_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img = clahe.apply(img)

        # --- 2️⃣ Z-score normalization
        if self.use_zscore:
            img = img.astype(np.float32)
            img = (img - np.mean(img)) / (np.std(img) + 1e-8)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        else:
            img = img.astype(np.float32) / 255.0

        # --- 3️⃣ Augmentation (if enabled)
        if self.augment and mask is not None:
            augmented = self.augment(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        return img, mask
