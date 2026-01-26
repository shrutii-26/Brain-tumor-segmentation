
#Brain Tumor Segmentation Using U-Net

A complete deep-learning pipeline for brain tumor segmentation using MRI images and the U-Net architecture.
This project performs dataset loading, preprocessing, training, evaluation, and prediction using PyTorch.

🔍 Project Overview

This project aims to create an automated system that detects and segments brain tumors from MRI scans.
It uses a U-Net convolutional neural network, ideal for medical image segmentation because of its symmetric encoder–decoder structure and spatial skip connections.

📁 Project Structure
project/
│── dataset.py
│── preprocessing.py
│── train.py
│── evaluate.py
│── test_images.py
│── test_dataset.py
│── model.py
│── plot_loss_from_txt.py
│── checkpoints/
│── results_training/
│── README.md

📌 Key Features

✔ Automatic image–mask pairing
✔ Preprocessing: normalization, thresholding
✔ Data augmentation using flips, rotations, elastic transforms
✔ U-Net (PyTorch) architecture
✔ Combined BCE + Dice loss
✔ Evaluation metrics: Dice, IoU, Accuracy
✔ Testing script for predictions
✔ Loss curve visualization

🚀 Pipeline

Dataset Loading

Preprocessing & Augmentation

Training U-Net

Validation & Saving Best Model

Evaluation on Test Data

Predicting Masks for New MRI Images

📦 1. Dataset

The project uses the LGG Brain MRI Segmentation Dataset.

Each MRI slice includes:

Grayscale .tif MRI image

Tumor mask (_mask.tif or _seg.tif)

dataset.py automatically:

Detects all MRI files

Matches each scan with its correct mask

Converts to tensors

Normalizes images

Applies Torch augmentations

🧪 2. Preprocessing

Implemented in preprocessing.py & dataset.py.

Includes:

✔ Image Normalization

Normalize pixel intensities → [0, 1]

✔ Mask Binarization

Convert segmentation mask → 0 or 1

✔ Augmentation

Horizontal flip

Vertical flip

Random rotation

Elastic transform

Brightness/contrast adjustment

These prevent overfitting and improve generalization.

🧠 3. Model — U-Net

Defined in model.py.

Architecture Overview

Encoder: Downsampling with Conv → Conv → MaxPool

Bottleneck: High-level features extracted

Decoder: Upsampling + Skip Connections

Output: 1-channel segmented tumor mask

Final activation:
Sigmoid → outputs probabilities between 0 and 1

🎯 4. Loss Functions

Defined in train.py.

Binary Cross Entropy (BCE) Loss

Good for pixel-wise classification.

Dice Loss

Measures overlap between predicted and true tumor region.

✔ Combined Loss Used:
Loss = 0.3 * BCE + 0.7 * Dice


Dice gets higher weight since tumor pixels are much fewer than background pixels.

📈 5. Training

The training script (train.py) handles:

✔ Hyperparameters

Epochs: 60

Batch size: 4

Learning rate: 1e-4

Optimizer: Adam

✔ Validation

Splits dataset:

80% training
20% validation

✔ Best Model Saving

Automatically saves:

checkpoints/best_unet.pth

✔ Outputs

Loss curves

Predicted masks saved every 5 epochs

📊 6. Evaluation

Using evaluate.py.

Metrics computed:

Dice Score

IoU (Jaccard Index)

Pixel Accuracy

Validation BCE Loss

Saved graphs:

eval_val_loss.png
eval_metrics_curve.png

🔍 7. Test on New MRI Images

Run:

python test_images.py


This script:

Loads trained model

Preprocesses MRI

Predicts mask

Saves output mask

Displays MRI, Ground Truth, Predicted Mask

📉 8. Plot Training Loss from Log File

plot_loss_from_txt.py extracts training + validation loss from a .txt file and plots them.

Useful for:

Reports

Performance comparison

▶️ How to Run the Project
Train
python train.py

Evaluate
python evaluate.py

Predict on new MRI
python test_images.py

Test dataset integrity
python test_dataset.py

🧩 Installation

Install dependencies:

pip install torch torchvision opencv-python matplotlib albumentations numpy tqdm scikit-image python-pptx

📌 Future Improvements

Use 3D U-Net for volumetric MRI segmentation

Add tumor classification stage

Deploy as a web application (Flask / FastAPI)

Apply transformer-based segmentation models (UNetR, SegFormer)


📜 License

This work is for educational and research purposes only.
Not intended for clinical diagnosis.