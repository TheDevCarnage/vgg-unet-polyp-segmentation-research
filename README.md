# VGG-UNet Hybrid Architecture for Polyp Segmentation

> **Research Status: ✅ Complete — Paper submitted to arXiv**
> Systematic evaluation of a VGG16 encoder fused with a U-Net decoder for colorectal polyp segmentation on the CVC-ClinicDB dataset.

---

## 📄 Paper

**VGG-UNet Hybrid Architecture for Colorectal Polyp Segmentation: A Systematic Evaluation on CVC-ClinicDB**
Rishabh Shukla, Shriyansh Singh — arXiv preprint, 2025
> arXiv link will be added upon publication

---

## 📌 Research Hypothesis

Standard U-Net uses a randomly initialized encoder, which can be suboptimal on small medical imaging datasets. This research investigates whether replacing the encoder with a **pretrained VGG16 backbone** yields measurable improvements in polyp segmentation accuracy, particularly for boundary delineation and small polyp detection.

**Core Question:**
*Does transfer learning via a pretrained VGG16 encoder improve segmentation performance over a standard U-Net baseline on the CVC-ClinicDB polyp dataset?*

**Answer: Yes — significantly.** VGG16-UNet achieves Val Dice 0.9297 vs. 0.7743 for standard U-Net — a **+15.54 point improvement**.

---

## 📈 Results

### Validation Results — All 8 Experiments

| Model / Configuration | Epoch | Val Dice | Val IoU | Precision | Recall |
|---|---|---|---|---|---|
| UNet Baseline (BCE-Dice) | 44 | 0.7743 | 0.6444 | 0.8917 | 0.6917 |
| **VGG16-UNet (BCE-Dice) ★** | **41** | **0.9297** | **0.8698** | **0.9419** | **0.9189** |
| VGG16-UNet (Dice only) | 50 | 0.9200 | 0.8558 | 0.9416 | 0.9015 |
| VGG16-UNet (Focal) | 43 | 0.9237 | 0.8608 | 0.9384 | 0.9111 |
| VGG16-UNet (Frozen Encoder) | 48 | 0.9104 | 0.8370 | 0.9184 | 0.9044 |
| VGG16-UNet (Minimal Aug) | 19 | 0.9054 | 0.8304 | 0.9303 | 0.8844 |
| VGG16-UNet (Heavy Aug) | 42 | 0.9176 | 0.8505 | 0.9218 | 0.9144 |
| VGG19-UNet (BCE-Dice) | 36 | 0.9181 | 0.8507 | 0.9457 | 0.8938 |

★ Primary proposed model

### Test Results (Held-Out Set)

| Model / Configuration | Test Dice | Test IoU | Precision | Recall |
|---|---|---|---|---|
| UNet Baseline (BCE-Dice) | 0.7962 | 0.6685 | 0.8268 | 0.7924 |
| **VGG16-UNet (BCE-Dice) ★** | **0.9156** | **0.8526** | **0.8970** | **0.9449** |
| VGG16-UNet (Dice only) † | 0.9200 | 0.8550 | 0.9065 | 0.9383 |
| VGG16-UNet (Focal) | 0.8892 | 0.8144 | 0.8679 | 0.9317 |
| VGG16-UNet (Frozen Encoder) | 0.9144 | 0.8469 | 0.9318 | 0.9016 |
| VGG16-UNet (Minimal Aug) | 0.8961 | 0.8158 | 0.9036 | 0.8937 |
| VGG16-UNet (Heavy Aug) | 0.8935 | 0.8167 | 0.9018 | 0.8986 |
| VGG19-UNet (BCE-Dice) | 0.9159 | 0.8494 | 0.9135 | 0.9221 |

† Best test-set Dice (0.9200, zero val→test degradation)

### Comparison with State-of-the-Art on CVC-ClinicDB

| Method | Val Dice | Val IoU | Year |
|---|---|---|---|
| U-Net baseline | 0.7743 | 0.6444 | 2015 |
| ResUNet++ | 0.7900 | 0.7900 | 2019 |
| PraNet | 0.8990 | — | 2020 |
| **VGG16-UNet (ours) ★** | **0.9297** | **0.8698** | **2025** |
| EENet | 0.9316 | 0.8817 | 2024 |
| FAENet | 0.9330 | 0.8830 | 2025 |
| DCATNet | 0.9444 | — | 2025 |

---

## 🏗️ Model Architecture

### Baseline: Standard U-Net
- Randomly initialized encoder
- 4-level downsampling: 64 → 128 → 256 → 512 channels
- Standard skip connections, ~31M parameters

### Proposed: VGG16-UNet Hybrid

```
Input Image (256×256×3)
       │
┌──────▼──────────────────────────────┐
│         VGG16 Encoder               │
│  (Pretrained on ImageNet)           │
│                                     │
│  Block 1: feats[0:4]  → 64ch  ──────┼──► skip1
│  Block 2: feats[5:9]  → 128ch ──────┼──► skip2
│  Block 3: feats[10:16]→ 256ch ──────┼──► skip3
│  Block 4: feats[17:23]→ 512ch ──────┼──► skip4
│  Block 5: feats[24:30]→ 512ch ──────┼──► skip5
└──────────────────────────────┬──────┘
                               │ MaxPool
                    ┌──────────▼──────────┐
                    │     Bottleneck       │
                    │   (1024 channels)    │
                    └──────────┬──────────┘
                               │
┌──────────────────────────────▼──────┐
│         U-Net Decoder               │
│                                     │
│  UpConv(1024→512) + skip5 → 512ch   │
│  UpConv(512→256)  + skip4 → 256ch   │
│  UpConv(256→128)  + skip3 → 128ch   │
│  UpConv(128→64)   + skip2 → 64ch    │
│  UpConv(64→32)    + skip1 → 32ch    │
└──────────────────────────────┬──────┘
                               │
                    ┌──────────▼──────────┐
                    │  Conv1×1 + Sigmoid   │
                    │  Output: (1×256×256) │
                    └─────────────────────┘
```

**Key design decisions:**
- VGG16 encoder initialized with ImageNet pretrained weights
- Differential learning rates: encoder 1e-5, decoder 1e-4
- Encoder fine-tuned during training (not frozen by default)
- Skip connections from each VGG block to corresponding decoder stage
- Transposed convolutions for upsampling
- Combined Dice + BCE loss for class imbalance

---

## 📊 Dataset

**CVC-ClinicDB** — Colonoscopy polyp segmentation dataset

| Property | Value |
|---|---|
| Total Images | 612 |
| Original Resolution | 384 × 288 pixels |
| Resized To | 256 × 256 pixels |
| Mask Type | Binary (polyp vs background) |
| Source | Hospital Clinic of Barcelona, Spain |
| Train / Val / Test Split | 489 / 61 / 62 (80/10/10, seed=42) |

**Download:**
```bash
kaggle datasets download -d balraj98/cvcclinicdb
unzip cvcclinicdb.zip -d data/raw/
```

**Citation:**
Bernal, J., et al. (2015). WM-DOVA maps for accurate polyp highlighting in colonoscopy. *Computerized Medical Imaging and Graphics*, 43, 99–111.

---

## ⚙️ Setup & Installation

This project uses **Pipenv** for virtual environment and dependency management.

```bash
# Clone the repository
git clone https://github.com/yourusername/vgg-unet-polyp-segmentation-research.git
cd vgg-unet-polyp-segmentation-research

# Install pipenv if not already installed
pip install pipenv

# Install dependencies and activate virtual environment
pipenv install
pipenv shell
```

Alternatively, using pip directly:
```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start

```bash
# 1. Download and verify dataset
python -m data.download_dataset

# 2. Split into train/val/test
python -m data.split

# 3. Verify data pipeline
python -m data.verify

# 4. Train baseline U-Net
python run_training.py --model unet --loss bce_dice --aug standard

# 5. Train proposed VGG16-UNet
python run_training.py --model vgg_unet --loss bce_dice --aug standard

# 6. Evaluate on held-out test set
python evaluation/evaluate.py --model unet --loss bce_dice
python evaluation/evaluate.py --model vgg_unet --loss bce_dice

# 7. Print results summary across all experiments
python results_summary.py
```

### Training CLI Arguments

| Argument | Options | Description |
|---|---|---|
| `--model` | `unet`, `vgg_unet`, `vgg_unet_frozen`, `vgg_unet_vgg19` | Model architecture |
| `--loss` | `bce_dice`, `dice`, `focal` | Loss function |
| `--aug` | `standard`, `minimal`, `heavy` | Data augmentation strategy |

---

## 📂 Repository Structure

```
vgg-unet-polyp-segmentation-research/
│
├── README.md
├── requirements.txt
├── Pipfile                        # Pipenv dependency file
├── constants.py                   # All paths and hyperparameters
├── run_training.py                # Training entry point (CLI)
├── results_summary.py             # Print all experiment results
│
├── data/
│   ├── download_dataset.py        # Download CVC-ClinicDB from Kaggle
│   ├── dataset.py                 # PyTorch Dataset class (TIF format support)
│   ├── augmentation.py            # Albumentations transform pipelines
│   ├── split.py                   # Reproducible train/val/test split
│   └── verify.py                  # Data pipeline sanity check
│
├── models/
│   ├── unet_baseline.py           # Standard U-Net (randomly initialized)
│   └── vgg_unet.py                # VGG16/VGG19-UNet hybrid (configurable)
│
├── training/
│   ├── train.py                   # Training loop with checkpointing & logging
│   ├── loss.py                    # Loss registry: BCE, Dice, BCE-Dice, Focal
│   └── metrics.py                 # Dice, IoU, Precision, Recall
│
├── evaluation/
│   ├── evaluate.py                # Full test set evaluation
│   └── visualize.py               # Prediction grids and training curves
│
└── results/
    ├── checkpoints/               # Best model weights per experiment (.pth)
    ├── figures/                   # Training curves and prediction visualizations
    └── *_history.csv              # Per-epoch training logs per experiment
```

---

## 🔬 Experiment Log

| # | Model | Loss | Encoder | Augmentation | Val Dice | Test Dice |
|---|---|---|---|---|---|---|
| 1 | UNet Baseline | BCE-Dice | Random | Standard | 0.7743 | 0.7962 |
| 2 | VGG16-UNet ★ | BCE-Dice | Pretrained | Standard | **0.9297** | 0.9156 |
| 3 | VGG16-UNet | Dice only | Pretrained | Standard | 0.9200 | **0.9200** |
| 4 | VGG16-UNet | Focal | Pretrained | Standard | 0.9237 | 0.8892 |
| 5 | VGG16-UNet | BCE-Dice | Frozen | Standard | 0.9104 | 0.9144 |
| 6 | VGG16-UNet | BCE-Dice | Pretrained | Minimal | 0.9054 | 0.8961 |
| 7 | VGG16-UNet | BCE-Dice | Pretrained | Heavy | 0.9176 | 0.8935 |
| 8 | VGG19-UNet | BCE-Dice | Pretrained | Standard | 0.9181 | 0.9159 |

---

## 💡 Key Findings

1. **Pretrained encoder dramatically outperforms random initialization** — +15.54 Val Dice and +11.94 Test Dice points over U-Net baseline

2. **BCE-Dice loss achieves best validation Dice** (0.9297); **Dice-only achieves best test generalization** (0.9200 → 0.9200, zero val→test degradation)

3. **Standard augmentation is optimal** — minimal augmentation causes rapid overfitting (best epoch: 19); heavy augmentation introduces unrealistic transformations that hurt performance

4. **Frozen encoder is competitive** — only 0.019 Dice gap vs fine-tuned, suggesting pretrained VGG features generalize well without modification

5. **VGG16 preferred over VGG19** — nearly identical test performance (0.9156 vs 0.9159) with ~20% fewer parameters

---

## 📝 Methodology Notes

This research follows an honest experimental approach. Each experiment varies exactly one component while holding all others constant, enabling direct attribution of performance differences. Results are reported for both validation and held-out test sets.

---

## 👥 Authors

**Rishabh Shukla**
M.S. Computer Science, University of Alabama at Birmingham (2025)
[LinkedIn](https://linkedin.com) · [GitHub](https://github.com)

**Shriyansh Singh**
Northeastern University

---

## 📄 Citation

```bibtex
@misc{shukla2025vggunet,
  title   = {VGG-UNet Hybrid Architecture for Colorectal Polyp Segmentation:
             A Systematic Evaluation on CVC-ClinicDB},
  author  = {Shukla, Rishabh and Singh, Shriyansh},
  year    = {2025},
  note    = {arXiv preprint}
}
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
