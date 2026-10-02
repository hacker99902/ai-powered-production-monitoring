# Production Manager – AI-Powered Production Monitoring

An AI-powered computer vision module for smart factory production monitoring. The Production Manager uses YOLO-based object detection to identify automotive components in production videos and supports automated tracking and counting to help monitor manufacturing activities.

## Project Overview

Manual monitoring of manufacturing lines can be time-consuming and may lead to inaccurate product counts. This module aims to automate component detection and production monitoring using computer vision.

The system processes recorded videos to detect automotive components, identify their classes, and provide annotated video output for visual inspection.

## Objectives

* Detect automotive components in production videos.
* Identify different component types using a trained YOLO model.
* Support object tracking and automated counting.
* Process recorded production-line videos.
* Generate annotated video output for monitoring and analysis.
* Provide a foundation for production activity monitoring.

## Features

* **Object Detection:** Detects automotive components using a trained YOLO model.
* **Multi-Class Classification:** Identifies four different automotive component classes.
* **Video Processing:** Runs inference on recorded videos.
* **Object Tracking:** Includes tracking functionality to maintain object identities across frames.
* **Production Counting:** Includes counting functionality to support monitoring of detected objects.
* **Visual Output:** Generates annotated videos with detection boxes and class labels.

Counting performance depends on reliable detection and tracking, and should be validated against manually counted videos.

## Supported Object Classes

The current trained model detects four automotive components.

| Class ID | Class Name |
| -------- | ---------- |
| 0        | brake_pad  |
| 1        | bearing    |
| 2        | spark_plug |
| 3        | gear       |

## Model Details

| Parameter               | Details                           |
| ----------------------- | --------------------------------- |
| Model                   | YOLO                              |
| Training run            | `production_v4_synthetic_spark-2` |
| Task                    | Object detection                  |
| Number of classes       | 4                                 |
| Framework               | Ultralytics                       |
| Deep learning framework | PyTorch                           |
| Model weights           | `best.pt`                         |

**Model path:**

```text
runs/detect/runs/detect/production_v4_synthetic_spark-2/weights/best.pt
```

### Validation Metrics

The model's recorded validation metrics are:

| Metric    |  Score |
| --------- | -----: |
| Precision | 88.32% |
| Recall    | 76.81% |
| mAP50     | 84.74% |
| mAP50-95  | 63.59% |

These scores are from the model's recorded validation results. Performance on actual conveyor-belt footage may differ depending on lighting, camera angle, object overlap, speed, and background.

## Technology Stack

| Technology       | Purpose                    |
| ---------------- | -------------------------- |
| Python           | Main programming language  |
| Ultralytics YOLO | Object detection           |
| PyTorch          | Deep learning framework    |
| OpenCV           | Video and image processing |
| ByteTrack        | Object tracking            |
| NumPy            | Numerical processing       |
| Git and GitHub   | Version control            |

## Project Structure

```text
smart-factory-cv/
│
├── cv/
│   └── production/
│       ├── __init__.py
│       ├── counter.py
│       ├── detector.py
│       ├── line_detector.py
│       ├── occupancy.py
│       ├── production_pipeline.py
│       └── tracker.py
│
├── data/
│   └── videos/
│       └── test/
│           └── new_video.mp4
│
├── runs/
│   └── detect/
│       └── runs/
│           └── detect/
│               └── production_v4_synthetic_spark-2/
│                   └── weights/
│                       └── best.pt
│
├── tests/
├── requirements.txt
├── app.py
├── .gitignore
└── README.MD
```

The structure above highlights the production monitoring code and relevant model and video files. Other project files may exist in the repository.

## Installation

### Prerequisites

* Python 3.12
* Git
* pip
* NVIDIA GPU with a compatible CUDA-enabled PyTorch installation for GPU inference (optional)

### 1. Clone the Repository

```powershell
git clone https://github.com/hacker99902/ai-powered-production-monitoring.git
cd ai-powered-production-monitoring
```

### 2. Create a Virtual Environment

For Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If using an NVIDIA GPU, ensure that the installed PyTorch and TorchVision versions are compatible with your CUDA environment.

### 4. Verify the Model

Check that the trained model weights are available:

```powershell
Test-Path "runs/detect/runs/detect/production_v4_synthetic_spark-2/weights/best.pt"
```

The output should be `True` if the model file is present.

## Running Object Detection

A sample test video is included in the repository:

```text
data/videos/test/new_video.mp4
```

Run the trained model on this video from the project root:

```powershell
yolo detect predict model="runs/detect/runs/detect/production_v4_synthetic_spark-2/weights/best.pt" source="data/videos/test/new_video.mp4" device=0
```

For CPU inference, use:

```powershell
yolo detect predict model="runs/detect/runs/detect/production_v4_synthetic_spark-2/weights/best.pt" source="data/videos/test/new_video.mp4" device=cpu
```

The annotated prediction output is saved by Ultralytics under `runs/detect/predict/` or a numbered `predict` folder if previous runs exist.

### Test with Another Video

You can replace the sample video path with another compatible video:

```powershell
yolo detect predict model="runs/detect/runs/detect/production_v4_synthetic_spark-2/weights/best.pt" source="path/to/your/video.mp4" device=0
```

## Object Tracking and Counting

The production module includes separate code for tracking and counting detected objects:

* `detector.py` – detection functionality
* `tracker.py` – object tracking functionality
* `counter.py` – counting functionality
* `line_detector.py` – production-line detection functionality
* `production_pipeline.py` – production monitoring pipeline
* `occupancy.py` – occupancy-related processing

These components are intended to work together to support production monitoring. Their exact behavior depends on the implementation and configuration in the source code.

## Evaluation

The model is evaluated using common object detection metrics:

* **Precision:** Proportion of predicted detections that are correct.
* **Recall:** Proportion of actual objects that are detected.
* **mAP50:** Mean average precision at an IoU threshold of 0.50.
* **mAP50-95:** Mean average precision across IoU thresholds from 0.50 to 0.95.

For production counting, additional evaluation should include:

* Missed objects
* False detections
* Incorrect class predictions
* Object tracking ID switches
* Difference between automated counts and manually verified counts

A separate, manually labelled conveyor-belt test set is recommended for assessing performance under real production conditions.

## Current Scope and Limitations

* The current trained model detects four automotive component classes.
* The model has recorded validation results, but its performance on different real-world production lines requires further testing.
* Counting reliability depends on accurate detection and tracking.
* Changes in lighting, object orientation, speed, occlusion, and camera placement can affect detection.
* Training data and the dataset YAML are needed for retraining, but not for inference when the trained weights are available.

## Future Improvements

* Improve detection performance on real conveyor-belt footage.
* Expand the manually verified production-line test dataset.
* Improve tracking stability for moving and overlapping components.
* Reduce false detections and missed objects.
* Validate counting accuracy across longer production videos.
* Add production summaries and downloadable reports.
* Support live camera streams.

## Author's Module

**Role:** Production Manager

**Project:** Multi-Role Computer Vision AI for Smart Factory Monitoring

**Focus:** Automotive component detection, object tracking, production counting, and video-based production monitoring.

## License

Add a license to the repository if the project is intended for reuse or distribution.
