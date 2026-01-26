import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

from dataset import MRISegDataset
from model import UNet

# -------------------------------
# 1. CONFIGURATION
# -------------------------------
DATA_PATH = "data/lgg-mri-segmentation/kaggle_3m"
MODEL_PATH = "checkpoints/best_unet.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4

print("Using device:", DEVICE)

# -------------------------------
# 2. METRIC FUNCTIONS
# -------------------------------

def dice_score(pred, target, smooth=1e-6):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def iou_score(pred, target, smooth=1e-6):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def pixel_accuracy(pred, mask):
    pred = (pred > 0.5).float()
    return (pred == mask).float().mean()


# -------------------------------
# 3. LOAD DATASET
# -------------------------------
full_dataset = MRISegDataset(DATA_PATH)
val_size = int(0.2 * len(full_dataset))
train_size = len(full_dataset) - val_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Validation samples: {len(val_dataset)}")


# -------------------------------
# 4. LOAD TRAINED MODEL
# -------------------------------
model = UNet(n_channels=1, n_classes=1).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("✔ Model loaded successfully!")


# -------------------------------
# 5. EVALUATE MODEL
# -------------------------------
dice_scores = []
iou_scores = []
accuracies = []
val_losses = []

bce = torch.nn.BCELoss()

with torch.no_grad():
    for batch in val_loader:
        imgs = batch["image"].to(DEVICE, dtype=torch.float32)
        masks = batch["mask"].to(DEVICE, dtype=torch.float32)

        preds = model(imgs)

        # Loss
        loss = bce(preds, masks).item()
        val_losses.append(loss)

        # Metrics
        for p, m in zip(preds, masks):
            dice_scores.append(dice_score(p, m).item())
            iou_scores.append(iou_score(p, m).item())
            accuracies.append(pixel_accuracy(p, m).item())


print("\n-----------------------------")
print("📊 EVALUATION RESULTS")
print("-----------------------------")
print(f"Average Dice Score       : {np.mean(dice_scores):.4f}")
print(f"Average IoU              : {np.mean(iou_scores):.4f}")
print(f"Average Pixel Accuracy   : {np.mean(accuracies):.4f}")
print(f"Average Validation Loss  : {np.mean(val_losses):.4f}")
print("-----------------------------\n")

# -------------------------------
# 6. PLOT EVALUATION GRAPHS
# -------------------------------

plt.figure(figsize=(10,5))
plt.plot(val_losses, label="Validation Loss", marker="o")
plt.title("Validation Loss Trend")
plt.xlabel("Batch Index")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.savefig("eval_val_loss.png")
plt.show()


plt.figure(figsize=(10,5))
plt.plot(dice_scores, label="Dice Score")
plt.plot(iou_scores, label="IoU Score")
plt.plot(accuracies, label="Pixel Accuracy")
plt.title("Evaluation Metrics per Batch")
plt.xlabel("Batch Index")
plt.ylabel("Score")
plt.grid(True)
plt.legend()
plt.savefig("eval_metrics_curve.png")
plt.show()

print("✔ Saved graphs:")
print(" - eval_val_loss.png")
print(" - eval_metrics_curve.png")
