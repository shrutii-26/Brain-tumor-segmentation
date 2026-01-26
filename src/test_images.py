import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import UNet

# --------------------------------------------------------
# PATHS
# --------------------------------------------------------
MODEL_PATH = r"D:\image project\checkpoints\best_unet.pth"
TEST_IMAGE_DIR = r"D:\image project\test_images"
OUTPUT_DIR = r"D:\image project\output_masks"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------------
# DEVICE
# --------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", device)


# --------------------------------------------------------
# PREPROCESSING (same as training)
# --------------------------------------------------------
def preprocess_image(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.resize(img, (256, 256))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


# --------------------------------------------------------
# LOAD MODEL
# --------------------------------------------------------
model = UNet(n_channels=1, n_classes=1).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print("Model loaded successfully.\n")

# --------------------------------------------------------
# FIND TEST IMAGES
# --------------------------------------------------------
image_files = [
    f
    for f in os.listdir(TEST_IMAGE_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif"))
    and "_mask" not in f.lower()
]

print(f"Found {len(image_files)} test images.\n")

# --------------------------------------------------------
# RUN PREDICTIONS
# --------------------------------------------------------
for img_name in image_files:

    img_path = os.path.join(TEST_IMAGE_DIR, img_name)

    # Load MRI
    mri = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if mri is None:
        continue

    # Check for original ground-truth mask
    base = os.path.splitext(img_name)[0]
    mask_path = os.path.join(TEST_IMAGE_DIR, base + "_mask.tif")
    original_mask = None

    if os.path.exists(mask_path):
        original_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        original_mask = cv2.resize(original_mask, (256, 256))
        original_mask = (original_mask > 127).astype(np.uint8)
    else:
        print(f"⚠ No ground-truth mask found for: {img_name}")

    # Preprocess input
    input_img = preprocess_image(mri)
    input_tensor = torch.from_numpy(input_img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(input_tensor).squeeze().cpu().numpy()

    pred_mask = (pred > 0.35).astype(np.uint8)

    # Save predicted mask
    save_path = os.path.join(OUTPUT_DIR, f"{base}_pred_mask.png")
    cv2.imwrite(save_path, pred_mask * 255)

    # ----------------------------------------------------
    # SHOW MRI + TRUE MASK + PRED MASK
    # ----------------------------------------------------
    plt.figure(figsize=(15, 5))

    # MRI
    plt.subplot(1, 3, 1)
    plt.title("MRI Image")
    plt.imshow(cv2.resize(mri, (256, 256)), cmap="gray")
    plt.axis("off")

    # Ground Truth Mask (if available)
    plt.subplot(1, 3, 2)
    plt.title("Ground Truth Mask")
    if original_mask is not None:
        plt.imshow(original_mask, cmap="gray")
    else:
        plt.text(0.3, 0.4, "No Mask Found", fontsize=14)
    plt.axis("off")

    # Predicted Mask
    plt.subplot(1, 3, 3)
    plt.title("Predicted Mask")
    plt.imshow(pred_mask, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    print(f"✔ Processed: {img_name}")

print("\n🎉 All predictions completed!")
