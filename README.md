# PAWS:Physical Altercation and Weapon Surveillance System

PAWS is an automated surveillance application designed to detect physical altercations and lethal weapons in video feeds. By utilizing computer vision and deep learning models, the system provides alerts for security monitoring, enhancing safety and response times in various environments.

## Features

- Detection of physical violence and altercations using VideoMAE transformer model
- Identification of weapons (knives and guns) within a video stream through hierarchical person and object detection
- A user-friendly interface built with CustomTkinter for video playback and event logging
- GPU acceleration support via CUDA for high-performance frame processing
- Automated evidence capture with timestamps for security auditing

## System Architecture

The project is structured into modular components to handle specific detection tasks:

1. **Main UI (`main_ui.py`)**: The central application hub that manages the graphical interface, video file ingestion, and coordinates between detection modules.
2. **Human Gate (`human_gate.py`)**: Lightweight human detector that runs on every frame. If no humans are detected, all heavy models are skipped entirely to save GPU resources.
3. **Weapon Module (`weapon_module.py`)**: Employs a two-stage detection process. It first receives human bounding boxes from the Human Gate and then runs knife and gun detection on those specific cropped regions in a single GPU batch.
4. **Altercation Module (`altercation_module.py`)**: Uses a VideoMAE video transformer with a MobileNetV2 CNN pre-filter and ROI-focused optical flow to detect sustained physical violence while suppressing false positives like hugging.

---

## Setup Instructions

### 1. Clone the Repository

```
git clone https://github.com/YanKylee/PAWS-Physical-Altercation-Weapon-Surveillance-System.git
```

```
cd PAWS-Physical-Altercation-Weapon-Surveillance-System
```

### 2. Prerequisites

**Python**
This project requires [Python 3.9 or later](https://www.python.org/downloads/). It is strongly recommended to use a virtual environment.

```
python -m venv venv
venv\Scripts\activate
```

**CUDA (Recommended)**
For optimal performance, an NVIDIA GPU with [CUDA 11.8 or later](https://developer.nvidia.com/cuda-downloads) is recommended.

Install the correct PyTorch version for your hardware from the [Official PyTorch website](https://pytorch.org/get-started/locally/). Example:
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Python Dependencies**

Install all required packages:
```
pip install ultralytics opencv-python customtkinter pillow pygame transformers
```

---

### 3. Models

Due to GitHub's file size limits, the VideoMAE model file is **not included** in this repository and must be downloaded separately and placed into the `models/` folder.

#### VideoMAE Violence Classifier (`models/videomae-violence-local/`)
This is the core violence detection model (VideoMAEForVideoClassification). It must be downloaded and placed inside the `models/videomae-violence-local/` directory. The folder must contain these three files:
- `model.safetensors`
- `config.json`
- `preprocessor_config.json`

> 📥 **Download link:** [https://drive.google.com/drive/folders/1wiuMf-xOhmmsg9hlOkcnG53-DRUTx1zK?usp=sharing]()

**Included models** (already in the repository, no download needed):
- `models/cctv_gun_detector.pt` — Gun detection
- `models/yolov8s.pt` — Human detection
- `models/knife_detector_adamw.pt` — Knife detection
- `models/mobilenet_violence_prefilter.pt` — Violence pre-filter (CNN)

After downloading, your `models/` folder should look like this:
```
models/
├── videomae-violence-local/
│   ├── config.json
│   ├── model.safetensors
│   └── preprocessor_config.json
├── yolov8s.pt
├── knife_detector_adamw.pt
├── mobilenet_violence_prefilter.pt
└── cctv_gun_detector.pt
```

---

### 4. Run the Application

```
python main_ui.py
```

Once launched:
1. Use the **Insert Video File** button to load a CCTV or recorded video clip.
2. The system will automatically begin analyzing the feed.
3. Monitor the **System Logs** panel on the right for real-time detection alerts.
4. Switch to the **Evidence History** tab to review captured snapshots of detected events.

---

## Credits & Acknowledgements

PAWS was built on top of several outstanding open-source models and frameworks. The following pre-trained models were used but are **not owned by this project** — full credit goes to their original authors.

| Model | Used For | Original Source |
|-------|----------|-----------------|
| **VideoMAE** (Violence/NonViolence classifier) | Physical altercation detection | [cliffer1/videomae-base-finetuned-kinetics-violence-nonviolence-tuned](https://huggingface.co/cliffer1/videomae-base-finetuned-kinetics-violence-nonviolence-tuned) on Hugging Face |
| **Knife Detector** | Knife detection | [MehmetAliKOYLU/Knife-Detector](https://github.com/MehmetAliKOYLU/Knife-Detector) on GitHub — model weights re-trained with AdamW optimizer for this project |
| **YOLOv8s** (`yolov8s.pt`) | Human / person detection | [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) |
| **MobileNetV2** (pre-filter) | Lightweight violence pre-screening | [PyTorch torchvision](https://pytorch.org/vision/stable/models/mobilenetv2.html) — fine-tuned by this project |

> The custom gun detection model (`cctv_gun_detector.pt`) and the MobileNetV2 pre-filter (`mobilenet_violence_prefilter.pt`) were trained specifically for this project using publicly available datasets.

---

## License

This project is **dual-licensed** under the **MIT License** and the **GNU General Public License v3.0 (GPL-3.0)**. You may choose either license when using or distributing this software.

| License | File | When to use |
|---------|------|-------------|
| **MIT** | [LICENSE](./LICENSE) | Permissive use — you can use, modify, and distribute freely with minimal restrictions |
| **GPL-3.0** | [LICENSE-GPL](./LICENSE-GPL) | Copyleft use — any derivative work must also be distributed under GPL-3.0 |

**Third-party dependency licenses:**
- **Ultralytics YOLOv8** → [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)
- **Hugging Face Transformers** → Apache 2.0
- **PyTorch** → BSD License

> Please review the third-party licenses above before any commercial use.


---

## Disclaimer

This system is a research and development project designed to assist in surveillance monitoring. The accuracy of detections is dependent on lighting conditions, video quality, and model training data. It is not intended to replace human security personnel.
