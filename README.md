# Lymphoma WSI Classifier

A deep learning pipeline for classifying Whole Slide Images (WSIs) of B-cell lymphomas using PyTorch and ResNet18. Built in Google Colab using publicly available data from the GDC (Genomic Data Commons) portal.

---

## Project Goal

Train a binary image classifier to distinguish between:
- **Burkitt Lymphoma** — characterised by a "starry sky" histological pattern
- **Diffuse Large B-Cell Lymphoma (DLBCL)** — the most common aggressive B-cell lymphoma

This is Phase 1 of a two-phase project. Phase 2 will add a Reed-Sternberg cell detector for Hodgkin Lymphoma classification.

---

## Data Sources

All data is publicly available via the [GDC Data Portal](https://portal.gdc.cancer.gov).

| Class | Project | Slides | Patches | Stain |
|---|---|---|---|---|
| Burkitt Lymphoma | CGCI-BLGSP | 20 | 3,278 | H&E |
| DLBCL | TCGA-DLBC | 20 | 4,217 | H&E |
| **Total** | | **40** | **7,495** | |

### Key decisions made along the way

- **TCIA was ruled out** — no Hodgkin Lymphoma WSI collections exist on TCIA. The TCGA-DLBC collection on TCIA contains only radiology (CT/MR), not pathology slides. WSI data lives on GDC.
- **CGCI-HTMCP-DLBCL was mostly IHC** — out of 20 downloaded slides, 17 were immunohistochemistry stains (CD3, Ki67, BCL6, TP53, BCL2, EBER, etc.) not H&E. These were filtered out to avoid stain-type leakage.
- **TCGA-DLBC naming convention** — TCGA slides do not include "-HE." in their filenames unlike CGCI slides. They use codes like DX (diagnostic), BS (bottom slide), TS (top slide) — but all are H&E.
- **Slides were capped at 500MB each** to stay within Colab free tier disk limits (~100GB total).

---

## Environment

- Google Colab (free tier)
- GPU: Tesla T4 (15.6GB VRAM)
- Disk: ~100GB (slides + patches stored on Google Drive)

### Dependencies

```
openslide-python
openslide-tools (apt)
torch
torchvision
tqdm
scikit-learn
seaborn
matplotlib
pandas
requests
tqdm
```

---

## Project Structure

```
lymphoma_classifier/
├── slides/
│   ├── burkitt/          # 20 SVS whole slide images
│   └── dlbcl/            # 40 SVS files (20 H&E used, 20 IHC filtered out)
├── patches/
│   ├── burkitt/          # 3,278 PNG patches (256×256)
│   └── dlbcl/            # 4,217 PNG patches (256×256)
├── manifest.csv          # Burkitt download manifest
├── manifest_dlbcl_new.csv # DLBCL H&E download manifest
└── best_model.pth        # Saved best model weights
```

---

## Pipeline

### Step 1 — Query GDC API

Used the GDC REST API (`https://api.gdc.cancer.gov/files`) to search for SVS slide images filtered by:
- `cases.disease_type = Mature B-Cell Lymphomas`
- `data_type = Slide Image`
- `data_format = SVS`
- `cases.diagnoses.primary_diagnosis` = Burkitt or DLBCL

### Step 2 — Build Download Manifest

Created a balanced manifest of 20 slides per class, capped at 500MB per file. Saved as CSV to Google Drive for resume safety across Colab sessions.

### Step 3 — Download Slides

Streamed SVS files from GDC in 8KB chunks directly to Google Drive. Resume-safe — already downloaded files are skipped automatically.

### Step 4 — Tile WSIs into Patches

Used `openslide` to extract 256×256 patches at **level 1** (4× downsampled from full resolution). Key decisions:

- **Level 1 chosen** because level 0 downsamples jump from 1× to 4× (no 2× level exists in these slides)
- **Tissue filter applied** — patches where >70% of pixels are near-white (>220 grayscale) are discarded as background
- **200 patches per slide maximum** to keep dataset size manageable
- Patch filenames encode slide name + row + column (zero-padded to 4 digits) for traceability

### Step 5 — Train ResNet18

- Loaded pretrained ResNet18 (ImageNet weights)
- Replaced final fully connected layer: `Linear(512 → 2)` for binary classification
- Training augmentations: random horizontal/vertical flips, 90° rotation, colour jitter
- Normalisation: ImageNet mean/std (required since model was pretrained on ImageNet)
- Optimiser: Adam, lr=1e-4
- LR scheduler: StepLR (decay by 0.5 every 3 epochs)
- Loss: CrossEntropyLoss
- 10 epochs, batch size 32
- Best model saved automatically to Google Drive

### Step 6 — Evaluate

- Confusion matrix on held-out test set
- Classification report (precision, recall, F1 per class)
- Training/validation loss and accuracy curves

---

## Data Splits

| Split | Patches | Percentage |
|---|---|---|
| Train | 5,246 | 70% |
| Val | 1,124 | 15% |
| Test | 1,125 | 15% |

> **Note:** Splits are currently done randomly at patch level, not slide level. This means patches from the same slide can appear in both train and test sets (data leakage). A future improvement is to split at the slide level to get a more honest evaluation.

---

## Known Limitations

- **Small dataset** — 40 slides total is modest for a WSI classifier. Performance may improve significantly with more slides.
- **Patch-level leakage** — see note above on data splits.
- **Single stain type** — all patches are H&E. Model has not been tested on IHC or other stain types.
- **No slide-level aggregation yet** — the model predicts at patch level. A proper WSI classifier aggregates patch predictions (e.g. majority vote or attention pooling) to produce a slide-level label.

---

## Phase 2 — Reed-Sternberg Cell Detector (Planned)

Phase 2 will train a separate model to detect Reed-Sternberg cells, the hallmark of Hodgkin Lymphoma. Since no Hodgkin Lymphoma WSIs exist in public repositories, this will use pre-annotated patch datasets specifically curated for RS cell detection.

---

## Resuming in a New Colab Session

```python
from google.colab import drive
drive.mount('/content/drive')

BASE_DIR = "/content/drive/MyDrive/lymphoma_classifier"

!pip install openslide-python -q
!apt-get install -y openslide-tools -q

import os
for label in ["burkitt", "dlbcl"]:
    folder = os.path.join(BASE_DIR, "patches", label)
    count = len([f for f in os.listdir(folder) if f.endswith(".png")])
    print(f"{label}: {count} patches")
```

---

## References

- [GDC Data Portal](https://portal.gdc.cancer.gov)
- [CGCI-BLGSP Collection](https://portal.gdc.cancer.gov/projects/CGCI-BLGSP)
- [TCGA-DLBC Collection](https://portal.gdc.cancer.gov/projects/TCGA-DLBC)
- [OpenSlide Python](https://openslide.org/api/python/)
- [PyTorch ResNet](https://pytorch.org/vision/stable/models/resnet.html)
