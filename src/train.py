"""
U-Net Training Script (GPU Optimized for Google Colab)
-------------------------------------------------------

✔ Trains U-Net on LGG Brain Tumor Segmentation Dataset
✔ Uses GPU automatically (if available)
✔ Saves:
    - Ground Truth mask (GT)
    - Predicted mask (Pred)
    - Loss curve
✔ Supports custom image inference after training
✔ Best model saved at: checkpoints/best_unet.pth
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset import MRISegDataset
from model import UNet

# -------------------- CONFIG --------------------
DATA_PATH = "data/lgg-mri-segmentation/kaggle_3m"
SAVE_DIR = "checkpoints"
RESULTS_DIR = "results_training"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
EPOCHS = 60
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.2

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("💻 Device in use:", DEVICE)


# -------------------- LOSS FUNCTIONS --------------------
def dice_loss(pred, target, smooth=1e-6):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    dice = (2.0 * intersection + smooth) / (
        pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth
    )
    return 1 - dice.mean()


# -------------------- LOAD DATASET --------------------
print("\n🧠 Loading dataset...")
full_dataset = MRISegDataset(root_dir=DATA_PATH, use_augment=True)

train_size = int((1 - VAL_SPLIT) * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1)

print(f"📦 Total Samples: {len(full_dataset)}")
print(f"📘 Training Samples: {len(train_dataset)}")
print(f"📗 Validation Samples: {len(val_dataset)}")

# -------------------- MODEL INIT --------------------
model = UNet(n_channels=1, n_classes=1).to(DEVICE)
bce_loss = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

best_val = float("inf")
train_losses = []
val_losses = []

# -------------------- TRAINING LOOP --------------------
print("\n🚀 Starting Training...\n")

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0

    progress = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]", leave=False)

    for batch in progress:
        imgs = batch["image"].to(DEVICE, dtype=torch.float32)
        masks = batch["mask"].to(DEVICE, dtype=torch.float32)

        preds = model(imgs)

        loss_bce = bce_loss(preds, masks)
        loss_dice = dice_loss(preds, masks)
        loss = 0.3 * loss_bce + 0.7 * loss_dice  # BETTER WEIGHTING

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        progress.set_postfix({"loss": loss.item()})

    avg_train_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # ---------------- VALIDATION ----------------
    model.eval()
    val_loss_total = 0

    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["image"].to(DEVICE, dtype=torch.float32)
            masks = batch["mask"].to(DEVICE, dtype=torch.float32)

            preds = model(imgs)

            val_loss = dice_loss(preds, masks)
            val_loss_total += val_loss.item()

    avg_val_loss = val_loss_total / len(val_loader)
    val_losses.append(avg_val_loss)

    print(
        f"📘 Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
    )

    # SAVE BEST MODEL
    if avg_val_loss < best_val:
        best_val = avg_val_loss
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_unet.pth"))
        print("🔥 Saved improved model at checkpoints/best_unet.pth")

    # SAVE SAMPLE OUTPUT
    if epoch % 5 == 0:
        gt = masks[0][0].cpu().numpy()
        pr = preds[0][0].cpu().numpy()

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(imgs[0][0].cpu(), cmap="gray")
        plt.title("MRI")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(gt, cmap="gray")
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(pr > 0.5, cmap="gray")
        plt.title("Prediction")
        plt.axis("off")

        plt.savefig(f"{RESULTS_DIR}/epoch_{epoch}_sample.png")
        plt.close()

# ---------------- PLOT LOSS CURVE ----------------
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig(f"{RESULTS_DIR}/loss_curve.png")
plt.close()

print("\n🎉 Training Completed!")


# ------------------------------------------------------------
# ----------- CUSTOM IMAGE INFERENCE FUNCTION ----------------
# ------------------------------------------------------------
def predict_custom_image(image_path):
    import cv2

    model.eval()

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (256, 256))
    tensor = (
        torch.from_numpy(img_resized / 255.0)
        .unsqueeze(0)
        .unsqueeze(0)
        .float()
        .to(DEVICE)
    )

    with torch.no_grad():
        pred = model(tensor)
        pred = (torch.sigmoid(pred).squeeze().cpu().numpy() > 0.5).astype("uint8")

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img_resized, cmap="gray")
    plt.title("Custom MRI Input")

    plt.subplot(1, 2, 2)
    plt.imshow(pred, cmap="gray")
    plt.title("Predicted Mask")
    plt.show()


print("\n💡 To test any custom image:")
print("➡ Run: predict_custom_image('your_image_path.tif')")
