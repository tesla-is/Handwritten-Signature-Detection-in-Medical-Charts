# Signature Detection on PDFs (YOLOv8)

Detect and localize signatures in PDF documents using a YOLOv8 model.  
This project converts each PDF page to an image, applies light contrast enhancement, runs detection, saves annotated pages, and produces analytics at both PDF-level and page-level.

## Features

- **PDF → Image rendering** with PyMuPDF at configurable DPI.
- **Preprocessing** with CLAHE to boost faint pen strokes.
- **YOLOv8 inference** (Ultralytics) with configurable `conf`, `iou`, `device`.
- **Annotation** via Supervision (red boxes; no labels).
- **Analytics outputs**:
  - `pdf_level_summary.csv`: pages count and which pages contain signatures.
  - `pdf_page_level_analytics.csv`: per-page metrics (confidence, bbox, density, position category, etc.).
- **Parallel page processing** for speed.

---

## Quick Start

### 1) Prerequisites

- Python 3.9+ (Linux/Mac/Windows)
- (Optional) CUDA-enabled GPU for faster inference
- Access to the gated Hugging Face model **`tech4humans/yolov8s-signature-detector`** (requires `HF_TOKEN`)

### 2) Install

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
