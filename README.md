# Signature Detection on PDFs (YOLOv8)

Detect and localize handwritten (ink and digital) signatures in PDF documents using a YOLOv8 model.  
This project converts each PDF page to an image, enhances contrast, runs a trained signature detector, saves annotated pages for all detections, and generates **structured analytics** at both the PDF level and the page level.

---

## ✨ Highlights

- **End‑to‑end pipeline**: PDF → high‑DPI images → contrast enhancement (CLAHE) → YOLOv8 inference → annotation → analytics CSVs  
- **Batch & multithreaded**: Processes many PDFs and pages efficiently  
- **Detailed analytics**: Confidence, bounding boxes, area/density metrics, position categories, page counts with signatures  
- **Portable**: Works in plain Python; includes display helpers for notebook/Databricks environments  
- **Secure by design**: Uses environment variables for secrets (Hugging Face token)

---

## 🧠 What it does 

Signature verification is a frequent requirement in document workflows (intake, claims, contracting, audits). This project automates the **detection** (not semantic verification) of signature regions on PDF pages. It:

1. **Loads PDFs** from an input directory and **renders each page** into a high‑resolution RGB image using **PyMuPDF** (DPI configurable).
2. Applies **CLAHE** (contrast‑limited adaptive histogram equalization) to gently enhance faint pen strokes so thin ink lines are more detectable against noisy backgrounds.
3. Runs a **YOLOv8** model specialized for signature detection. Thresholds for confidence (`conf`) and NMS IOU (`iou`) are configurable, and it supports GPU or CPU execution.
4. **Annotates** only the pages that contain at least one detection and writes them as `PNG` images (red bounding boxes without labels).
5. Computes **page‑level metrics** such as top detection confidence, bounding box coordinates/area/aspect ratio, a **signature density** ratio (total detected area over page area), a **position category** (center/quadrants/edges), and the number of signatures per page.
6. Aggregates a **PDF‑level summary** listing the total pages, the number of pages with signatures, and the 1‑based page indices where signatures were detected.
7. Exports two CSVs to your output folder:
   - `pdf_level_summary.csv`
   - `pdf_page_level_analytics.csv`

The pipeline is optimized for **practical throughput** with multithreading per page. For very small or faint signatures, increasing DPI (e.g., 400–600) and maintaining a larger inference image size (e.g., 1280) can boost recall.

---

## 📦 Repository Structure

```
.
├── signature_detection.py           # Main pipeline script (rename from your current file if desired)
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── .gitignore
├── CHANGELOG.md                    
├── CONTRIBUTING.md                 
├── CODE_OF_CONDUCT.md               
├── SECURITY.md                      
└── LICENSE                          
```

---

## 🛠️ Installation

### 1) Python environment

```
pip install -r requirements.txt
```

### 2) GPU (optional)
If you have CUDA, install a CUDA‑enabled PyTorch build per your environment. Ultralytics/YOLOv8 will leverage it automatically when `DEVICE="0"` (or another CUDA device index).

---

## ⚙️ Configuration

Use **environment variables** (recommended) so paths and secrets are never hard‑coded:

```bash
export INPUT_PDF_DIR="/path/to/pdfs"             # required
export OUTPUT_ROOT="./output"                    # defaults to ./output if not set
export HF_TOKEN="***your_hf_access_token***"     # required (gated model access)
export DEVICE="0"                                # "cpu" or CUDA index like "0"
export DPI="600"                                 # 300–600 recommended
export CONF_THRESHOLD="0.25"
export IOU_THRESHOLD="0.5"
export MAX_WORKERS="8"                           # 4–16 depending on CPU cores/IO
```

---

## 🚀 Quick Start

> Rename your script to `signature_detection.py` (optional). If you keep a different name, adjust the command below.

```bash
python signature_detection.py
```

Outputs:

- Annotated pages (only where signatures were detected):
  ```
  $OUTPUT_ROOT/<pdf_stem>/annotated/page_XXX.png
  ```
- CSVs:
  ```
  $OUTPUT_ROOT/pdf_level_summary.csv
  $OUTPUT_ROOT/pdf_page_level_analytics.csv
  ```

---

## 📊 Output Details

### 1) `pdf_level_summary.csv`
- `pdf_name` — PDF file stem (no extension)
- `no_of_pages` — total page count
- `no_of_pages_with_signatures` — how many pages had ≥1 detection
- `pages_with_signatures` — list of 1‑based page indices with detections

### 2) `pdf_page_level_analytics.csv`
- `pdf_name`
- `page_number` (1‑based)
- `Signature_detection` (boolean)
- `Confidence_Score` (float; highest confidence on the page)
- `bbox` (`[x1, y1, x2, y2]` for the highest‑confidence detection)
- `bbox_area` (pixels²)
- `normalized_confidence` (0–100)
- `bbox_aspect_ratio` (`w/h` for the top detection)
- `page_signature_density` (sum of all detected areas ÷ page area)
- `num_signatures_per_page` (count)
- `signature_position_category` ∈ `{center, top_left, top_right, bottom_left, bottom_right, top_edge, bottom_edge, left_edge, right_edge, none}`

---

## 🧩 How it works (pipeline & internals)

1. **PDF rendering**  
   Uses **PyMuPDF** to rasterize each page:
   - `DPI` controls the zoom factor (`zoom = DPI / 72`). Higher DPI yields more pixels and helps detect small signatures, with a trade‑off in memory and time.

2. **Preprocessing**  
   Applies **CLAHE** on the L channel in LAB space to gently boost local contrast (useful for faint ink strokes), then converts back to BGR.

3. **YOLOv8 inference**  
   - Loads a **YOLOv8s** signature detector from a **gated Hugging Face** repo:
     ```
     repo_id: tech4humans/yolov8s-signature-detector
     filename: yolov8s.pt
     ```
   - You must supply `HF_TOKEN` (via env/secret).  
   - Runs prediction with configurable `conf`, `iou`, and `device`.

4. **Annotation**  
   - Only pages with detections are saved to disk.
   - Red bounding boxes (no labels) are drawn using the **Supervision** library.

5. **Analytics**  
   - Aggregates page‑level and PDF‑level metrics (see Output Details).

6. **Parallelization**  
   - Each page is processed in a thread (`ThreadPoolExecutor`).
   - Tune `MAX_WORKERS` per your CPU, IO, and memory profile.

---

## ✅ Practical settings & tips

- **Recall on tiny signatures**:  
  Increase `DPI` to 400–600 and keep YOLO input size ≥1280 (already set).
- **Performance**:  
  Start with `MAX_WORKERS=4–8`. Increasing beyond core count yields diminishing returns due to I/O and GIL on some steps.
- **Memory constraints**:  
  Rendering at high DPI for long PDFs can be memory‑intensive. If you hit OOM:
  - Reduce `DPI` (e.g., 300–450)
  - Process a limited set of pages at a time (future: streaming refactor)
- **GPU vs CPU**:  
  Use `DEVICE="0"` for GPU 0; use `"cpu"` when a GPU isn’t available.

---

## 🧪 Example (pseudo session)

```bash
export INPUT_PDF_DIR="./samples"
export OUTPUT_ROOT="./output"
export HF_TOKEN="***"
export DPI="600"
export CONF_THRESHOLD="0.25"
export IOU_THRESHOLD="0.5"
export DEVICE="0"
export MAX_WORKERS="8"

python signature_detection.py
```

After completion:
```
./output/
  ├── pdf_level_summary.csv
  ├── pdf_page_level_analytics.csv
  ├── Contract_123/
  │   └── annotated/
  │       ├── page_001.png
  │       └── page_004.png
  └── Agreement_ACME/
      └── annotated/
          └── page_007.png
```

---

## 🔐 Security & Compliance

- **Secrets**: Never commit tokens or API keys. Use environment variables or your organization’s secret management (e.g., Vault, Databricks Secrets).
- **PHI/PII**: If PDFs may contain PHI/PII, follow your org’s data handling, access, and retention policies. Annotated page images can capture sensitive content; store outputs securely and delete when no longer needed.
- **Model licensing**: The model is fetched from a **gated** Hugging Face repository. Ensure each user has access and accepts the upstream license/usage terms. This repo **does not** redistribute model weights.

---

## 🧰 Troubleshooting

- **`HF_TOKEN not set` / cannot download weights**  
  Ensure the token is exported and has access to the gated repo.
- **Zero detections**  
  - Try `DPI=600`, lower `CONF_THRESHOLD` to `0.2`, ensure the page isn’t too low‑resolution.
- **OOM when rasterizing**  
  - Lower `DPI`, or process fewer pages at once (future: streaming approach).
- **Slow performance**  
  - Use a GPU (`DEVICE="0"`), try `MAX_WORKERS=8–16`, and run from fast storage (local SSD).

---

## 🧭 Roadmap

- CLI wrapper via `argparse` (replace env‑only configuration)
- Streamed page processing to reduce peak memory
- Optional **redaction** exports (blur or mask signature regions)
- Unit tests for metrics, position classification, and IO
- Dockerfile for reproducible runs and CI
- Configurable visual styles (labels, colors, thickness)
- Structured logs and progress reporting

---

## 🧱 Known limitations

- **Detection ≠ authentication**: This model detects signature‑like regions; it does not verify authenticity or match identities.
- **Model generalization**: Performance depends on document quality (scan DPI, noise, skew) and signature appearance (ink color, thickness).

---

## 📄 Requirements

See `requirements.txt`. Typical stack:

```text
ultralytics>=8.0.0
huggingface_hub>=0.20.0
pymupdf>=1.23.0
pillow>=10.0.0
reportlab>=4.0.0
opencv-python>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
supervision>=0.17.0
tqdm>=4.66.0
```

---

## 🙏 Acknowledgments

- **Ultralytics YOLOv8**
- **PyMuPDF** for PDF rasterization
- **Supervision** for fast, clean annotations
- **Hugging Face Hub** for model distribution

---

## 🧾 Citation 

If you publish a report or internal paper citing this tool, you can use:

```bibtex
@software{signature_detection_yolov8,
  title = {Signature Detection on PDFs (YOLOv8)},
  author = {Rahul Pathak},
  year = {2026},
  url = {https://github.com/tesla-is/Handwritten-Signature-Detection-in-Medical-Charts},
  note = {PDF-to-image pipeline with CLAHE preprocessing, YOLOv8 inference, and analytics export}
}
```
