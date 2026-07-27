<div align="center">

# 🧠 NeuroVision AI

### Brain MRI Tumor Detection using Deep Learning & Explainable AI

An end-to-end deep learning application that classifies brain MRI scans into four categories — **Glioma, Meningioma, Pituitary Tumor, and No Tumor** — using an **EfficientNet-B0** classifier, visualizes model decisions with **Grad-CAM**, and generates downloadable **PDF diagnostic reports**, all through a single-page **Streamlit** dashboard.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EfficientNet--B0-orange.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-ImageProcessing-5C3EE8.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()

</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Why This Project](#-why-this-project)
- [Features](#-features)
- [Demo](#️-demo)
- [System Architecture](#️-system-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#️-installation)
- [Usage](#-usage)
- [How It Works](#-how-it-works)
- [Model Details](#-model-details)
- [Explainable AI — Grad-CAM](#-explainable-ai--grad-cam)
- [PDF Report Generation](#-pdf-report-generation)
- [Results](#-results)
- [Roadmap](#️-roadmap-version-2)
- [Challenges & Learnings](#-challenges--learnings)
- [Author](#-author)
- [License](#-license)

---

## 📌 Overview

NeuroVision AI is a computer-aided diagnosis (CAD) tool built to assist in the preliminary screening of brain MRI scans. A user uploads one or more MRI images through a web dashboard, and the system returns:

- The **predicted tumor class** with a confidence score
- A **full probability breakdown** across all four classes (Glioma, Meningioma, Pituitary, No Tumor)
- A **Grad-CAM heatmap** highlighting the exact region of the scan the model focused on to reach its decision — making the "black box" model interpretable
- A **downloadable PDF report** summarizing the scan, prediction, confidence, and visual explanation — ready to save, print, or share

The project was built to go beyond a typical "train a CNN in a notebook and report accuracy" exercise. It covers the full pipeline: data preprocessing → model training → inference → explainability → a usable interface → exportable output — the same shape as a real deployed ML product.

> ⚠️ **Disclaimer:** This project is for educational and research purposes only. It is **not** a certified medical diagnostic tool and must never be used as a substitute for professional medical advice, diagnosis, or treatment.

---

## 🎯 Why This Project

Brain tumors are among the most critical conditions to detect early, and MRI is the primary imaging modality used for diagnosis. Manual analysis of MRI scans is time-consuming and depends heavily on radiologist expertise. This project explores how deep learning can:

- Act as a **fast, consistent second opinion** during preliminary screening
- Reduce the "black box" problem in medical AI through **Grad-CAM explainability**, so predictions aren't just a label — they come with visual justification
- Package a trained model into something a **non-technical user could actually use**, not just a script a developer runs locally

---

## ✨ Features

| Category | Details |
|---|---|
| 🧠 **Core AI** | EfficientNet-B0 classifier (transfer learning) · 4-class prediction · confidence + full probability scores |
| 🔥 **Explainable AI** | Grad-CAM heatmap generation · original vs. heatmap vs. AI-focus overlay comparison |
| 🌐 **Dashboard** | Single-page Streamlit UI · multi-image batch upload · clean dark theme · progress indicators |
| 📄 **Reporting** | Auto-generated, per-scan PDF report with filename, prediction, confidence, probability table, and Grad-CAM visualization |
| 🗂️ **Architecture** | Fully modular codebase — prediction, explainability, and reporting are decoupled into separate modules |
| ⚡ **Batch Processing** | Upload and analyze multiple MRI scans in a single session, each with its own independent result card and report |

---

## 🖥️ Demo

<div align="center">

<i>Add a screenshot or short GIF of the dashboard in action here:</i>

`assets/demo_screenshot.png`

**Suggested shots to include:**
- The upload screen
- A result card showing prediction + confidence + Grad-CAM overlay
- The generated PDF report

</div>

---

## 🏗️ System Architecture

```
                ┌─────────────────────┐
                │   Streamlit UI       │
                │   (app.py)           │
                └──────────┬───────────┘
                           │  MRI image(s)
                           ▼
                ┌─────────────────────┐
                │   src/predict.py     │
                │   EfficientNet-B0    │──► Prediction + Confidence
                │   Inference Engine   │      + Probabilities
                └──────────┬───────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │   src/gradcam.py     │
                │   Grad-CAM Engine    │──► Heatmap + Overlay
                └──────────┬───────────┘      + Comparison Image
                           │
                           ▼
                ┌─────────────────────┐
                │  src/pdf_report.py   │
                │  ReportLab Generator │──► Downloadable PDF Report
                └─────────────────────┘
```

---

## 🏗️ Tech Stack

**Model & Training**
- PyTorch
- EfficientNet-B0 (transfer learning, `torchvision` base)

**Explainability**
- `pytorch-grad-cam`

**Frontend / Dashboard**
- Streamlit

**Reporting**
- ReportLab (PDF generation)

**Image Processing**
- OpenCV
- Pillow (PIL)
- NumPy
- Matplotlib (comparison figure generation)

---

## 📁 Project Structure

```
NeuroVision-AI/
├── app.py                     # Single-page Streamlit dashboard
├── requirements.txt           # Python dependencies
├── src/
│   ├── predict.py             # Model loading + inference logic
│   ├── gradcam.py             # Grad-CAM heatmap generation
│   ├── pdf_report.py          # PDF report generation (ReportLab)
│   ├── models/
│   │   └── efficientnet.py    # EfficientNet-B0 classifier definition
│   └── data/
│       ├── transforms.py      # Image preprocessing/transforms
│       └── dataloader.py      # Dataset loading + CLASS_NAMES
├── configs/
│   └── config.py              # Device, image size, and other configs
├── models/
│   └── best_model.pth         # Trained model weights
├── reports/
│   └── gradcam/                # Generated Grad-CAM outputs per scan
│       └── <scan_name>/
│           ├── original.png
│           ├── heatmap.png
│           ├── overlay.png
│           └── comparison.png
├── assets/                    # Screenshots / demo media for README
└── README.md
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.10 or higher
- pip
- (Optional but recommended) a virtual environment tool

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/DivyomSrivastava/NeuroVision-AI.git
cd NeuroVision-AI

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place your trained model weights
# Ensure best_model.pth is inside the models/ directory
```

**`requirements.txt`**
```
torch
torchvision
streamlit
opencv-python
pillow
numpy
matplotlib
grad-cam
reportlab
```

> Generate the exact pinned versions from your environment with `pip freeze > requirements.txt` before pushing, so anyone cloning the repo gets a working setup on the first try.

---

## 🚀 Usage

### Run the dashboard
```bash
streamlit run app.py
```

Then open the local URL Streamlit prints in your terminal (typically `http://localhost:8501`), and:

1. Upload one or more MRI images (JPG/PNG)
2. Click **🧠 Analyze MRI Scans**
3. Review the prediction, confidence, and Grad-CAM visualization for each scan
4. Click **📄 Download PDF Report** to save an individual report per scan

### Run inference from the terminal
```bash
python -m src.predict
```
Enter the path to an MRI image when prompted, and the terminal will print the predicted class, confidence, and full probability breakdown.

### Generate a Grad-CAM visualization standalone
```bash
python -m src.gradcam
```

---

## 🔍 How It Works

1. **Upload** — The user uploads one or more MRI scans through the Streamlit interface.
2. **Preprocessing** — Each image is converted to RGB and passed through the same `test_transform` pipeline used during training, ensuring consistency between training and inference.
3. **Inference** — The EfficientNet-B0 model outputs raw logits, which are converted to probabilities via softmax. The highest-probability class becomes the prediction, paired with its confidence percentage.
4. **Explainability** — Grad-CAM computes gradients flowing into the final convolutional block to produce a heatmap of the image regions most influential in the prediction. This heatmap is overlaid on the original scan.
5. **Reporting** — All results (filename, prediction, confidence, probabilities, and the Grad-CAM overlay) are compiled into a structured PDF report using ReportLab, generated on the fly and offered as a direct download — nothing is written to disk beyond the Grad-CAM images.

---

## 🧬 Model Details

| Property | Value |
|---|---|
| Architecture | EfficientNet-B0 (transfer learning) |
| Classes | Glioma, Meningioma, Pituitary, No Tumor |
| Input | MRI scan (JPG/PNG) |
| Output | Predicted class, confidence %, per-class probabilities |
| Explainability | Grad-CAM (final convolutional block, `model.features[8]`) |

> **To fill in:** dataset source and size, train/val/test split, number of training epochs, augmentation strategy, and final accuracy / precision / recall / F1-score per class. This is the single most important section for technical reviewers and recruiters — a model card with real numbers signals rigor.

---

## 🔥 Explainable AI — Grad-CAM

Rather than treating the model as a black box, NeuroVision AI generates four artifacts per scan:

| Artifact | Description |
|---|---|
| **Original** | The unmodified input MRI scan |
| **Heatmap** | Raw Grad-CAM activation map (JET colormap) |
| **Overlay** | Heatmap blended over the original scan, showing exactly where the model "looked" |
| **Comparison** | A single side-by-side figure of all three, saved for documentation/reporting |

This matters clinically and technically: a correct prediction backed by activation in an irrelevant region of the scan is a red flag worth catching — and Grad-CAM makes that visible instead of hidden inside the model.

---

## 📄 PDF Report Generation

Each analyzed MRI gets its own auto-generated PDF report containing:

```
NeuroVision AI — Brain MRI Analysis Report
--------------------------------------------
Filename     : BT-MRI Test GL (4).jpg
Prediction   : Glioma
Confidence   : 96.42%
--------------------------------------------
Class Probabilities
Glioma        96.42%
Meningioma     2.13%
Pituitary      0.98%
No Tumor       0.47%
--------------------------------------------
Explainable AI: Original MRI + Grad-CAM Overlay
--------------------------------------------
Generated by NeuroVision AI
```

Reports are generated entirely in memory (`io.BytesIO`) and served through Streamlit's `download_button` — no report files are persisted on the server.

---

## 📊 Results

> **To fill in before publishing:** overall test accuracy, confusion matrix, and per-class precision/recall/F1. A simple table or an embedded confusion-matrix image here turns this from "a demo" into "a validated model" in the eyes of anyone reviewing your GitHub.

| Metric | Value |
|---|---|
| Test Accuracy | `TBD` |
| Precision (macro avg) | `TBD` |
| Recall (macro avg) | `TBD` |
| F1-score (macro avg) | `TBD` |

---

## 🗺️ Roadmap (Version 2)

- [ ] Fine-tune EfficientNet-B0 for improved accuracy (target: 94–97%)
- [ ] Expand and improve data augmentation strategy
- [ ] DICOM file format support
- [ ] Clinical-style UI redesign
- [ ] Multi-scan comparison/history view
- [ ] Public deployment (Streamlit Cloud / HuggingFace Spaces)

---

## 🧩 Challenges & Learnings

> **Optional but valuable section** — a few sentences on real problems you solved (e.g. debugging a mismatch between training and inference transforms that caused wrong predictions in the dashboard, or restructuring from a multi-page app into a single-page flow for better UX) show up well to both recruiters and anyone reading the commit history. Consider adding 2–3 bullets here from your own experience building this.

---

## 👤 Author

**Divyom Srivastava**
B.Tech CSE, Pranveer Singh Institute of Technology (PSIT), Kanpur

- GitHub: [github.com/DivyomSrivastava](https://github.com/DivyomSrivastava)
- LinkedIn: [linkedin.com/in/divyom-srivastava-260b95342](https://linkedin.com/in/divyom-srivastava-260b95342)

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">
<i>If you found this project useful or interesting, consider giving it a ⭐ on GitHub.</i>
</div>
