# Medical Polyp Segmentation using UNet++ (PyTorch)

## Overview

Early detection of colorectal polyps is crucial for preventing colorectal cancer. Manual examination of colonoscopy frames is time-consuming and prone to human error.

This project implements a **UNet++ deep learning model** in PyTorch to **segment polyps in colonoscopy images** using publicly available datasets such as **Kvasir-SEG** and **CVC-ClinicDB**.

The system generates **segmentation masks** highlighting polyps, enabling automated analysis to assist medical professionals.

## Repository Structure

```
📁 Medical-Polyp-Segmentation
│── model.py                # UNet++ architecture
│── dataset.py              # PolypDataset class
│── train.py                # Training script
│── predict.py              # Single image prediction
│── metrics.py              # Dice, IoU metrics
│── plot_curves.py          # Training curve plotting
│── README.md               # Project documentation
│── /data                   # Dataset folder
│     ├── images            # Colonoscopy images
│     └── masks             # Segmentation masks
│── requirements.txt        # Required Python packages
```

## Project Outcomes
* Understand polyp segmentation challenges
* Learn **IoU** and **Dice Score** metrics
* Apply **data augmentation** using PyTorch + Albumentations
* Implement **VGG and UNet++ architectures** in PyTorch
* Train segmentation models on medical images
* Visualize input images, masks, and predictions

## Dataset
We use **real polyp datasets**:

| Dataset      | Images | Masks | Download Link                                                  |
| ------------ | ------ | ----- | -------------------------------------------------------------- |
| Kvasir-SEG   | 1000   | 1000  | [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/)           |

> **Folder structure for training:**

```
dataset/
  images/
    img_0001.jpg
    img_0002.jpg
    ...
  masks/
    img_0001_mask.png
    img_0002_mask.png
    ...
```

## Tech Stack
* **Python 3.8+**
* **PyTorch** (Deep Learning)
* **Albumentations** (Data Augmentation)
* **OpenCV** (Image Processing)
* **NumPy, Pandas**
* **Matplotlib / Seaborn** (Visualization)

## Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/medical-polyp-segmentation.git
cd medical-polyp-segmentation
pip install -r requirements.txt
```

## Training the Model

```bash
python train.py
```

* Model checkpoints will be saved as `unetpp_epoch{n}.pth`
* Training logs will include **Loss**, **IoU**, and **Dice scores**

## Making Predictions

```bash
python predict.py --image data/images/img_0001.jpg --weights unetpp_epoch50.pth
```

* Output mask will be saved as `prediction_mask.png`
* Use the predicted mask to visualize polyp segmentation

## Evaluation Metrics

* **Dice Coefficient**: Measures overlap between predicted mask and ground truth
* **IoU (Intersection over Union)**: Standard segmentation metric

> Both metrics handle binary masks and threshold predictions at 0.5.

## Data Augmentation

Implemented with Albumentations:

* Horizontal & vertical flips
* Random brightness/contrast
* Affine transformations (scale, rotate, translate)
* Resize and tensor conversion

## UNet++ Architecture
* Encoder with **VGG blocks**
* Nested skip connections
* Dense convolutional blocks in decoder
* Reduces semantic gap between encoder and decoder
* Handles small or low-contrast polyps

## Visualization
* **Input image**
* **Ground truth mask**
* **Predicted mask**

> Training curve plots: `plot_curves.py` generates Loss, IoU, Dice curves for monitoring.

## Future Enhancements
* Attention UNet / UNet++
* Transformer-based segmentation (Swin-UNet)
* Multi-class polyp classification (benign vs malignant)
* Real-time deployment (FastAPI + React)
* Explainable AI for medical validation

## References
1. Jha, D. et al., *Kvasir-SEG: A segmented polyp dataset*, 2020
2. Bernal, J. et al., *CVC-ClinicDB dataset*, 2015
3. Medical image segmentation literature: UNet, UNet++ papers

## License
MIT License
