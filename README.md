# BrainSeg

**Brain Tumor Segmentation Pipeline — Upload an MRI scan, get an automated tumor segmentation mask.**
Built with U-Net · PyTorch · Albumentations · OpenCV

## What It Does

1. Load MRI scans with their corresponding tumor masks
2. Preprocess and augment the data automatically
3. U-Net trains on the prepared dataset with combined BCE + Dice loss
4. Evaluate segmentation quality using Dice, IoU, and Accuracy
5. Predict tumor masks on new MRI images

## Pipeline

```
MRI Image Input
        ↓
Dataset Loader         → auto image-mask pairing, tensor conversion
        ↓
Preprocessing          → normalization, binarization, augmentation
        ↓
U-Net (PyTorch)        → encoder-decoder with skip connections
        ↓
Loss (BCE + Dice)      → pixel-wise + overlap-based optimization
        ↓
Output                 → segmented tumor mask + evaluation metrics
```

## Key Features

| Feature                 | Detail                                                    |
| ----------------------- | --------------------------------------------------------- |
| Auto image–mask pairing | Detects and matches MRI scans with masks                  |
| Preprocessing           | Normalization + mask binarization                         |
| Augmentation            | Flips, rotations, elastic transforms, brightness/contrast |
| Architecture            | U-Net with sigmoid output                                 |
| Loss Function           | 0.3 × BCE + 0.7 × Dice                                    |
| Evaluation Metrics      | Dice Score, IoU, Pixel Accuracy                           |
| Best Model Saving       | Auto-saves to `checkpoints/best_unet.pth`                 |
| Loss Visualization      | Training + validation curve plotting                      |

## Tech Stack

| Component        | Technology                    |
| ---------------- | ----------------------------- |
| Deep Learning    | PyTorch                       |
| Architecture     | U-Net (custom implementation) |
| Augmentation     | Albumentations                |
| Image Processing | OpenCV + PIL                  |
| Dataset          | LGG Brain MRI Segmentation    |
| Loss Functions   | BCE + Dice (combined)         |

## Project Structure

```
project/
├── model.py
├── dataset.py
├── preprocessing.py
├── train.py
├── evaluate.py
├── test_images.py
├── test_dataset.py
├── plot_loss_from_txt.py
├── checkpoints/
└── results_training/
```

## Run Locally

```bash
git clone https://github.com/your-username/BrainSeg
cd BrainSeg
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision opencv-python matplotlib albumentations numpy tqdm scikit-image
```

**Train**

```bash
python train.py
```

**Evaluate**

```bash
python evaluate.py
```

**Predict on new MRI**

```bash
python test_images.py
```

**Test dataset integrity**

```bash
python test_dataset.py
```

## Training Configuration

| Hyperparameter  | Value     |
| --------------- | --------- |
| Epochs          | 60        |
| Batch Size      | 4         |
| Learning Rate   | 1e-4      |
| Optimizer       | Adam      |
| Train/Val Split | 80% / 20% |

## Architecture — U-Net

```
Input MRI
    ↓
Encoder        → Conv → Conv → MaxPool (×4)
    ↓
Bottleneck     → high-level feature extraction
    ↓
Decoder        → Upsample + Skip Connections (×4)
    ↓
Output Layer   → 1-channel mask via Sigmoid
```

## Future Improvements

- 3D U-Net for volumetric MRI segmentation
- Tumor classification stage on top of segmentation
- Web deployment via Flask or FastAPI
- Transformer-based models (UNetR, SegFormer)

---

_Built for educational and research purposes only. Not intended for clinical diagnosis._
