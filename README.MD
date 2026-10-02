# Multi-Role Computer Vision AI for Smart Factory Monitoring

An AI-powered computer vision platform designed to support smart factory operations through automated monitoring of production activities, product quality, and workplace safety.

The system is designed around multiple operational roles, with a centralized orchestration engine to route inputs to the appropriate computer vision tasks and organize their results.

## Project Overview

Traditional factory monitoring often depends on manual inspection and observation. This can make it difficult to monitor production continuously, detect defects consistently, and track operational activities efficiently.

This project aims to provide a unified computer vision platform that processes images, recorded videos, and potentially live camera streams to support different factory monitoring roles.

### Key Objectives

* Automate the detection and monitoring of automotive components.
* Support production counting and tracking.
* Provide a modular structure for safety, quality, and production monitoring.
* Coordinate computer vision tasks through a centralized orchestration engine.
* Generate structured results to support factory monitoring and decision-making.

## System Architecture

```text
          Image / Video / Camera
                    |
                    v
              Input Manager
                    |
                    v
           Role / Task Router
                    |
                    v
        Model Manager & Scheduler
                    |
          +---------+---------+
          |         |         |
          v         v         v
        Safety    Quality   Production
        Officer   Inspector  Manager
          |         |         |
          v         v         v
        PPE and   Defect    Object Detection
        Safety    Detection  Tracking
        Monitoring           Counting
          |         |         |
          +---------+---------+
                    |
                    v
        Unified Event and Result
               Normalizer
                    |
                    v
          Database / Evidence
                    |
                    v
          Factory Monitoring
               Dashboard
```

The architecture separates input handling, task routing, model inference, and result processing to support modular development and future expansion.

## Core Modules

### 1. Safety Officer

Designed to support workplace safety monitoring through computer vision.

Potential functions include:

* Personal protective equipment (PPE) detection.
* Safety-zone monitoring.
* Tracking people in monitored areas.
* Identifying potential safety violations.

### 2. Quality Inspector

Designed to assist in inspecting products and identifying visible defects.

Potential functions include:

* Product and component detection.
* Region-of-interest inspection.
* Visual anomaly and defect detection.
* Recording inspection results for review.

### 3. Production Manager

The production monitoring module uses object detection and video processing to identify automotive components and support production tracking.

The current trained production model detects four component classes:

| Class ID | Object     |
| -------- | ---------- |
| 0        | brake_pad  |
| 1        | bearing    |
| 2        | spark_plug |
| 3        | gear       |

The production module is intended to support:

* Component detection on production videos.
* Object tracking and counting.
* Production activity monitoring.
* Annotated video generation.
* Structured detection results for further processing.

### 4. Orchestration Engine

The orchestration engine coordinates the monitoring workflow by managing:

* Input and task routing.
* Model selection and activation.
* Inference scheduling.
* Result normalization.
* Duplicate-event handling.
* Communication between monitoring modules.

The architecture is modular, allowing individual monitoring capabilities to be developed and integrated independently.

## Technology Stack

| Category                   | Technologies              |
| -------------------------- | ------------------------- |
| Programming Language       | Python                    |
| Computer Vision            | OpenCV                    |
| Object Detection           | Ultralytics YOLO          |
| Deep Learning              | PyTorch                   |
| Object Tracking            | ByteTrack                 |
| Backend / Application      | Python, Streamlit         |
| Frontend Architecture      | React, Vite, Tailwind CSS |
| API / Service Architecture | FastAPI                   |
| Version Control            | Git, GitHub               |

Specific dependencies and installed versions are maintained in `requirements.txt`.

## Production Detection Model

The current production model is a four-class YOLO object detection model trained to detect automotive components.

**Model:** `production_v4_synthetic_spark-2`

**Weights location:**

```text
runs/detect/runs/detect/production_v4_synthetic_spark-2/weights/best.pt
```

### Validation Metrics

| Metric    |  Score |
| --------- | -----: |
| Precision | 88.32% |
| Recall    | 76.81% |
| mAP50     | 84.74% |
| mAP50-95  | 63.59% |

These metrics are from the recorded validation results for this training run. Performance may differ on unseen videos and actual production-line conditions.

## Repository Structure

```text
smart-factory-cv/
│
├── app.py
├── common/
├── configs/
├── cv/
│   └── production/
│       ├── counter.py
│       ├── detector.py
│       ├── line_detector.py
│       ├── occupancy.py
│       ├── production_pipeline.py
│       └── tracker.py
│
├── data/
│   ├── dataset/
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
├── scripts/
├── tests/
├── weights/
├── requirements.txt
├── README.MD
└── .gitignore
```

The repository may evolve as additional monitoring roles and supporting services are integrated.

## Installation and Setup

### Prerequisites

* Python 3.12
* Git
* NVIDIA GPU with a compatible CUDA-enabled PyTorch installation for GPU inference (optional)
* Windows, Linux, or another supported Python environment

### 1. Clone the Repository

```bash
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

Ensure PyTorch and TorchVision are installed in compatible versions for your system. For NVIDIA GPU inference, install a CUDA-compatible PyTorch build supported by your GPU and driver.

### 4. Verify the Model

From the repository root, check that the trained model exists:

```powershell
Test-Path "runs/detect/runs/detect/production_v4_synthetic_spark-2/weights/best.pt"
```

The command should return `True` if the model weights are present.

## Running the Production Monitoring Application

If the application entry point is configured with Streamlit, launch it from the project root:

```powershell
streamlit run app.py
```

Open the local address displayed in the terminal to access the application.

## Testing the Trained YOLO Model

A sample test video is included in the repository:

```text
data/videos/test/new_video.mp4
```

Run object detection on the sample video:

```powershell
yolo detect predict model="runs/detect/runs/detect/production_v4_synthetic_spark-2/weights/best.pt" source="data/videos/test/new_video.mp4" device=0
```

For CPU inference, use:

```powershell
yolo detect predict model="runs/detect/runs/detect/production_v4_synthetic_spark-2/weights/best.pt" source="data/videos/test/new_video.mp4" device=cpu
```

The annotated prediction video is saved in an Ultralytics output directory, typically under:

```text
runs/detect/predict/
```

If predictions have already been run, Ultralytics may create a numbered directory such as `predict2`.

### Test with Another Video

To test a different video, replace the source path:

```powershell
yolo detect predict model="runs/detect/runs/detect/production_v4_synthetic_spark-2/weights/best.pt" source="path/to/your/video.mp4" device=0
```

## Model Inference Notes

* The trained `.pt` weights file is required for inference.
* The original training dataset is not required to run predictions.
* GPU inference requires a compatible CUDA-enabled PyTorch setup.
* CPU inference can be used if a compatible GPU environment is not available.
* Detection results should be evaluated on representative, manually verified videos before being used for production counting.

## Model Training

The trained model weights are provided for inference. To retrain or fine-tune the model, the corresponding training dataset, dataset configuration YAML, training scripts, and compatible dependencies are required.

Training example:

```powershell
yolo detect train model=yolo26n.pt data="path/to/data.yaml" epochs=20 imgsz=640 device=0
```

Adjust the dataset configuration and training parameters according to the dataset and available hardware.

## Evaluation

Model evaluation should consider more than validation metrics alone.

Recommended evaluation measures include:

* Precision and recall.
* mAP50 and mAP50-95.
* Missed detections.
* False detections.
* Incorrect class predictions.
* Object counting error on manually annotated test videos.

Testing should use separate videos for training, validation, and testing to reduce data leakage.

## Current Scope

The current production model supports detection of four automotive component classes:

* Brake pads
* Bearings
* Spark plugs
* Gears

The wider project architecture is designed to support safety monitoring, quality inspection, production monitoring, and orchestration. The actual availability of each capability depends on its implementation and integration in the current codebase.

## Future Enhancements

* Improve model performance on real conveyor-belt footage.
* Integrate verified production-line datasets.
* Improve object tracking and counting reliability.
* Expand safety and quality inspection capabilities.
* Integrate monitoring results into a unified dashboard.
* Add live camera stream processing.
* Improve event logging and reporting.

## Contributing

Contributions and improvements are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Make and test your changes.
4. Commit your changes with a descriptive message.
5. Open a pull request.

## License

Add the appropriate license information here if and when a license is selected for the project.
