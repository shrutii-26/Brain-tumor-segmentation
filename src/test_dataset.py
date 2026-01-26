# from dataset import MRISegDataset

# dataset = MRISegDataset(root_dir="data/lgg-mri-segmentation/kaggle_3m")

# print(f"Total pairs found: {len(dataset)}")

# sample = dataset[0]
# print(f"Image shape: {sample['image'].shape}")
# print(f"Mask shape: {sample['mask'].shape}")







# src/test_dataset.py
import torch
import matplotlib.pyplot as plt
from dataset import MRISegDataset

# ======================================================
# CONFIGURATION
# ======================================================
DATA_PATH = "data/lgg-mri-segmentation/kaggle_3m"

# ======================================================
# 1️⃣ LOAD DATASET
# ======================================================
print("🧠 Loading dataset with augmentations enabled...\n")
dataset = MRISegDataset(root_dir=DATA_PATH, use_augment=True)

print(f"✅ Total image-mask pairs found: {len(dataset)}")

# ======================================================
# 2️⃣ LOAD A SAMPLE
# ======================================================
sample = dataset[0]
img_tensor = sample["image"]
mask_tensor = sample["mask"]

print(f"Image shape: {img_tensor.shape}")
print(f"Mask shape:  {mask_tensor.shape}")
print(f"Image dtype: {img_tensor.dtype}, Mask dtype: {mask_tensor.dtype}")

# ======================================================
# 3️⃣ CONVERT TO NUMPY FOR DISPLAY
# ======================================================
img_np = img_tensor.squeeze().numpy()
mask_np = mask_tensor.squeeze().numpy()

# ======================================================
# 4️⃣ VISUALIZE IMAGE + MASK
# ======================================================
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(img_np, cmap="gray")
plt.title("Preprocessed MRI Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(mask_np, cmap="gray")
plt.title("Segmentation Mask")
plt.axis("off")

plt.suptitle("Sample from MRISegDataset (with preprocessing + augmentation)", fontsize=14)
plt.tight_layout()
plt.show()
