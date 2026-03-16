# vgg-unet-polyp-segmentation-research

## VGG-UNet Hybrid Architecture for Polyp Segmentation

> **Research Status: 🔬 In Progress**  
> Investigating the effectiveness of a VGG16 encoder fused with a U-Net decoder for colorectal polyp segmentation on the CVC-ClinicDB dataset.

---

## 📌 Research Hypothesis

Standard U-Net uses a randomly initialized or lightly pretrained encoder. This research investigates whether replacing the encoder with a **pretrained VGG16 backbone** — known for strong feature extraction capabilities — yields measurable improvements in polyp segmentation accuracy, particularly for boundary delineation and small polyp detection.

**Core Question:**  
*Does transfer learning via a pretrained VGG16 encoder improve segmentation performance over a standard U-Net baseline on the CVC-ClinicDB polyp dataset?*

---

## 🎯 Objectives

- [ ] Implement a standard U-Net baseline
- [ ] Implement VGG16-UNet hybrid (pretrained encoder + custom decoder)
- [ ] Train and evaluate both models on CVC-ClinicDB
- [ ] Compare performance using Dice coefficient, IoU, and F1-score
- [ ] Analyze failure cases — where does VGG-UNet underperform?
- [ ] Document findings regardless of outcome (positive or negative results)
- [ ] Publish preprint on arXiv

---

## 📂 Repository Structure

```
vgg-unet-polyp-segmentation-research/
│
├── README.md
├── requirements.txt
├── config.yaml                    # All hyperparameters in one place
│
├── data/
│   ├── download_dataset.py        # Script to fetch CVC-ClinicDB from Kaggle
│   ├── dataset.py                 # PyTorch Dataset class
│   └── augmentations.py           # Albumentations pipeline
│
├── models/
│   ├── unet_baseline.py           # Standard U-Net (baseline)
│   ├── vgg_encoder.py             # Pretrained VGG16 encoder blocks
│   ├── unet_decoder.py            # Decoder with skip connections
│   └── vgg_unet.py                # Combined VGG-UNet model
│
├── training/
│   ├── train.py                   # Main training loop
│   ├── loss.py                    # Combined Dice + BCE loss
│   └── metrics.py                 # IoU, Dice, F1 score
│
├── evaluation/
│   ├── evaluate.py                # Run evaluation on test set
│   └── visualize.py               # Prediction overlays and heatmaps
│
├── notebooks/
│   └── experiments.ipynb          # Kaggle/Colab exploration notebook
│
├── results/
│   ├── figures/                   # Output plots and visualizations
│   └── metrics.csv                # Logged experiment results
│
└── paper/
    └── draft.md                   # Research write-up (in progress)
```

---

## 📊 Dataset

**CVC-ClinicDB** — Colonoscopy polyp segmentation dataset

| Property | Value |
|---|---|
| Total Images | 612 |
| Image Resolution | 384 × 288 pixels |
| Mask Type | Binary (polyp vs background) |
| Source | Hospital Clinic of Barcelona, Spain |
| Train / Val / Test Split | 490 / 61 / 61 |

**Download:**
```bash
# Via Kaggle API
kaggle datasets download -d balraj98/cvcclinicdb
unzip cvcclinicdb.zip -d data/raw/
```

**Citation:**  
Bernal, J., et al. (2015). WM-DOVA maps for accurate polyp highlighting in colonoscopy: Validation vs. saliency maps from physicians. *Computerized Medical Imaging and Graphics*, 43, 99–111.

---

## 🏗️ Model Architecture

### Baseline: Standard U-Net
- Randomly initialized encoder
- Standard skip connections
- Trained from scratch

### Proposed: VGG16-UNet Hybrid

```
Input Image (384×288×3)
       │
┌──────▼──────────────────────────────┐
│         VGG16 Encoder               │
│  (Pretrained on ImageNet)           │
│                                     │
│  Block 1: Conv→Conv (64 ch)   ──────┼──► skip1
│  Block 2: Conv→Conv (128 ch)  ──────┼──► skip2
│  Block 3: Conv→Conv (256 ch)  ──────┼──► skip3
│  Block 4: Conv→Conv (512 ch)  ──────┼──► skip4
│  Block 5: Conv→Conv (512 ch)  ──────┼──► skip5
└──────────────────────────────┬──────┘
                               │
                    ┌──────────▼──────────┐
                    │     Bottleneck       │
                    │   (1024 channels)    │
                    └──────────┬──────────┘
                               │
┌──────────────────────────────▼──────┐
│         U-Net Decoder               │
│                                     │
│  UpConv + skip5 → Dec4 (512 ch)     │
│  UpConv + skip4 → Dec3 (256 ch)     │
│  UpConv + skip3 → Dec2 (128 ch)     │
│  UpConv + skip2 → Dec1 (64 ch)      │
└──────────────────────────────┬──────┘
                               │
                    ┌──────────▼──────────┐
                    │   Output (1×384×288) │
                    │   Sigmoid activation │
                    └─────────────────────┘
```

**Key design decisions:**
- VGG16 encoder initialized with ImageNet pretrained weights
- Encoder weights fine-tuned during training (not frozen)
- Skip connections from each VGG block to corresponding decoder layer
- Decoder uses transposed convolutions for upsampling
- Combined Dice + BCE loss for handling class imbalance

---

## ⚙️ Setup & Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/vgg-unet-polyp-segmentation-research.git
cd vgg-unet-polyp-segmentation-research

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**requirements.txt**
```
torch>=2.0.0
torchvision>=0.15.0
albumentations>=1.3.0
opencv-python>=4.7.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
scikit-learn>=1.2.0
tqdm>=4.65.0
pyyaml>=6.0
kaggle>=1.5.0
```

---

## 🚀 Quick Start

```bash
# 1. Download dataset
python data/download_dataset.py

# 2. Train baseline U-Net
python training/train.py --model unet --config config.yaml

# 3. Train VGG-UNet
python training/train.py --model vgg_unet --config config.yaml

# 4. Evaluate both models
python evaluation/evaluate.py --model unet --checkpoint results/unet_best.pth
python evaluation/evaluate.py --model vgg_unet --checkpoint results/vgg_unet_best.pth

# 5. Visualize predictions
python evaluation/visualize.py --model vgg_unet --checkpoint results/vgg_unet_best.pth
```

---

## 📈 Results (Updated as experiments run)

| Model | Dice ↑ | IoU ↑ | F1 ↑ | Params |
|---|---|---|---|---|
| Standard U-Net (baseline) | TBD | TBD | TBD | ~31M |
| VGG16-UNet (proposed) | TBD | TBD | TBD | ~45M |

*Results will be updated as experiments complete.*

**Current SOTA on CVC-ClinicDB for reference:**

| Model | Dice | IoU |
|---|---|---|
| EENet (2024) | 0.9316 | 0.8817 |
| FAENet (2025) | 0.9330 | 0.8830 |
| DCATNet (2025) | 0.9444 | — |

---

## 🔬 Experiment Log

| Date | Experiment | Notes |
|---|---|---|
| — | Repo initialized | Literature review in progress |

---

## 📖 Background & Related Work

### Why Polyp Segmentation?
Colorectal cancer is one of the leading causes of cancer-related deaths worldwide. Early and accurate detection of polyps during colonoscopy is critical for prevention. Automated segmentation reduces dependence on manual annotation and clinician fatigue.

### Why VGG as Encoder?
U-Net was originally designed with a symmetric encoder-decoder structure trained from scratch. However, using a pretrained VGG16 backbone as the encoder introduces rich ImageNet features, potentially improving convergence speed and segmentation quality — especially on small datasets like CVC-ClinicDB (612 images).

### Key Related Papers
- Ronneberger et al. (2015) — *U-Net: Convolutional Networks for Biomedical Image Segmentation*
- Simonyan & Zisserman (2014) — *Very Deep Convolutional Networks (VGGNet)*
- Jha et al. (2019) — *ResUNet++: An Advanced Architecture for Medical Image Segmentation*
- Fan et al. (2020) — *PraNet: Parallel Reverse Attention Network for Polyp Segmentation*

---

## 🗺️ Research Roadmap

- [x] Literature review
- [x] Repository setup
- [ ] Dataset preprocessing pipeline
- [ ] Baseline U-Net implementation and training
- [ ] VGG-UNet implementation
- [ ] Hyperparameter tuning
- [ ] Results analysis and visualization
- [ ] Failure case analysis
- [ ] Paper write-up
- [ ] arXiv preprint submission

---

## 📝 Notes on Methodology

This research follows an honest experimental approach — results will be reported regardless of outcome. Negative results (VGG-UNet not outperforming baseline) are equally valid contributions to the field as they help other researchers avoid redundant paths.

---

## 👤 Author

**Rishabh Shukla**  
M.S. Computer Science, University of Alabama at Birmingham (2025)  
[LinkedIn](https://linkedin.com) · [GitHub](https://github.com)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

*This research is independent and not affiliated with any institution.*
