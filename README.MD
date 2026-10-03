# 🏭 AI-Powered Production Monitoring System

A computer vision-based production monitoring system designed to automate product detection, tracking, and counting in manufacturing environments. The application uses a trained **Ultralytics YOLO model**, OpenCV, and a **Streamlit dashboard** to process multiple recorded camera feeds and display production activity.

> **Current implementation:** The dashboard simulates seven camera feeds using recorded MP4 videos. These are not live IP-camera connections.

---

## 📌 Project Overview

Manual product counting on manufacturing lines can be time-consuming and prone to human error. This project explores how computer vision can assist production monitoring by detecting automotive components in video, maintaining object identities across frames, and counting objects as they cross a virtual line.

The system is organized into a detection and monitoring pipeline, with a Streamlit interface for viewing annotated frames and per-camera counts.

## ✨ Key Features

- **YOLO object detection:** Detects trained automotive components in video frames.
- **Multi-camera dashboard:** Displays seven configured video feeds in a single Streamlit interface.
- **Object tracking:** Assigns track IDs and associates detections between frames using an IoU-based tracker.
- **Line-crossing count:** Counts a tracked object when it crosses a configured virtual counting line, while avoiding recounting the same track.
- **Throughput calculation:** Calculates rolling production rates in products per minute.
- **Annotated frames:** Displays detection bounding boxes, class labels, track IDs, confidence values, and production statistics.
- **Session controls:** Start monitoring, stop processing, and reset a session from the dashboard.
- **Performance monitoring components:** Includes utilities for observing processing FPS, CPU and RAM use, and available GPU memory information.
- **CPU/GPU selection:** Uses CUDA device 0 when CUDA is available; otherwise, runs on CPU.

## 🧠 Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application and pipeline logic |
| Ultralytics YOLO | Automotive component detection |
| PyTorch | Model inference and device support |
| OpenCV | Video decoding, frame processing, and annotation |
| Streamlit | Interactive monitoring dashboard |
| NumPy | Numerical and array operations |
| Pandas | Data handling dependency |
| psutil | CPU and memory monitoring |
| Git & GitHub | Version control |

## 🏗️ System Architecture

```text
              Recorded MP4 Camera Feeds
                 (7 configured feeds)
                          |
                          v
               Video Reader Threads
                          |
                          v
                Per-Camera Frame Queues
                          |
                          v
               Shared YOLO Detector
                          |
                          v
                Object Detections
              (boxes, class, confidence)
                          |
                          v
                Per-Camera Tracker
                          |
                          v
              Track IDs and Trajectories
                          |
                          v
               Production Pipeline
               /        |         \
              v         v          v
        Line Crossing  Throughput  Frame
          Counting     Calculation Annotation
               \        |         /
                v       v        v
                Shared Runtime State
                          |
                          v
                Streamlit Dashboard
            (frames, counts, session status)
```

Each camera has its own tracking and counting state. The application uses a shared detector to run inference across the configured video streams. Frame queues are bounded; when a queue fills, the corresponding reader waits instead of intentionally discarding frames. Processing can therefore take longer than the source video duration when inference is slower than video playback.

## 📁 Project Structure

```text
ai-powered-production-monitoring/
│
├── app.py
├── streamlit_app.py
├── requirements.txt
├── README.MD
├── .gitignore
│
├── cv/
│   └── production/
│       ├── __init__.py
│       ├── counter.py
│       ├── detector.py
│       ├── line_detector.py
│       ├── occupancy.py
│       ├── performance_monitor.py
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
│       ├── models/
│       │   └── production_7class_car_v1-2/
│       │       └── weights/
│       │           └── best.pt
│       └── runs/
│           └── detect/
│               └── production_v4_synthetic_spark-2/
│                   └── weights/
│                       └── best.pt
│
├── scripts/
│   ├── audit_dataset_v2.py
│   ├── audit_new_datasets.py
│   ├── audit_spark_plugs.py
│   ├── balance_dataset.py
│   ├── build_dataset_v2.py
│   ├── clean_brake_pad_dataset.py
│   ├── download_roboflow.py
│   ├── extract_conveyor_frames.py
│   ├── extract_correction_frames.py
│   ├── inspect_dataset_classes.py
│   ├── merge_generated_spark_plugs.py
│   ├── merge_spark_correction.py
│   ├── prepare_dataset.py
│   ├── prepare_spark_correction.py
│   ├── visualize_dataset.py
│   └── visualize_dataset_v2.py
│
└── tests/
    ├── check_splits.py
    ├── count_new_video.py
    ├── test_counter.py
    ├── test_counting_video.py
    ├── test_dwell_time.py
    ├── test_event_engine.py
    ├── test_occupancy.py
    ├── test_occupancy_video.py
    ├── test_scheduler.py
    └── verify_dataset.py
```

The repository also contains model and dataset preparation scripts. Some scripts and tests may support experiments or components that are not directly exposed in the current Streamlit dashboard.

## 🔍 Detection and Monitoring Pipeline

1. **Read video frames:** Each configured MP4 is opened and read by a camera reader thread.
2. **Queue frames:** Frames are placed into a bounded queue for that camera.
3. **Detect components:** The YOLO model predicts object bounding boxes, classes, and confidence scores.
4. **Track objects:** The tracker associates detections across frames using class-aware Intersection over Union (IoU) matching and maintains track IDs.
5. **Initialize a counting line:** The production pipeline configures a horizontal or vertical virtual line based on camera settings.
6. **Count crossings:** When a tracked object's center crosses the virtual line, it is counted once for that track.
7. **Calculate throughput:** Recent crossing events are used to calculate a rolling products-per-minute rate.
8. **Display results:** The Streamlit dashboard updates the camera tiles, counts, and session status.

The repository also includes an occupancy counter for objects within a configured region of interest (ROI), and an automatic line detector based on object trajectories. These are separate reusable components and may require integration/configuration for a particular workflow.

## 🧩 Configured Camera Feeds

The current `app.py` configures these seven feeds:

| Dashboard feed | Expected video path | Line orientation | Target class |
|---|---|---|---|
| Door Camera | `data/videos/test/door.mp4` | Vertical | `car_door` |
| Bonnet Camera | `data/videos/test/hood.mp4` | Horizontal | `car_bonnet` |
| Car Camera | `data/videos/test/car.mp4` | Vertical | `car` |
| Gear Camera | `data/videos/test/Gears.mp4` | Horizontal | `gear` |
| Spark Plug Camera | `data/videos/test/Spark_plugs.mp4` | Horizontal | `spark_plug` |
| Brake Pad Camera | `data/videos/test/Brake_pads.mp4` | Horizontal | `brake_pad` |
| Bearing Camera | `data/videos/test/bearing.mp4` | Horizontal | `bearing` |

**Important:** The uploaded repository archive includes `data/videos/test/new_video.mp4`, but the dashboard configuration references the seven filenames above. Add the corresponding video files or update the camera paths in `app.py` to match the videos you have. The app checks that the configured videos and model weights exist before starting.

## ⚙️ Model Details

The dashboard currently loads the weights at:

```text
runs/detect/models/production_7class_car_v1-2/weights/best.pt
```

A separate model artifact is also present at:

```text
runs/detect/runs/detect/production_v4_synthetic_spark-2/weights/best.pt
```

The Streamlit dashboard uses the first path, as configured by `MODEL_PATH` in `app.py`. Confirm that the model's trained class names match the classes expected by the configured camera feeds. Model performance depends on training data and the real production environment; validate the model on representative footage before relying on its counts.

## 💻 Requirements

- Python 3.12 (the project has been used with Python 3.12)
- pip
- Git
- Sufficient RAM for video processing
- NVIDIA GPU with compatible CUDA-enabled PyTorch and drivers for GPU inference (optional)
- The trained YOLO weights and all configured video files

The application can use CPU inference if CUDA is unavailable, although processing speed may be lower.

## 🚀 Installation

### 1. Clone the repository

```powershell
git clone https://github.com/hacker99902/ai-powered-production-monitoring.git
cd ai-powered-production-monitoring
```

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation scripts, follow your machine's approved Python environment setup method. On systems where activation is unavailable, you can invoke the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe --version
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The current `requirements.txt` contains:

```text
streamlit
ultralytics==8.4.167
opencv-python
numpy==1.26.4
pandas
```

The app also imports `torch` and `psutil` directly. Install compatible versions of PyTorch and `psutil` in your environment if they are not already installed. For NVIDIA GPU support, select a PyTorch build compatible with your CUDA setup using the official PyTorch installation guidance.

### 4. Verify required files

Check that the model file exists:

```powershell
Test-Path "runs/detect/models/production_7class_car_v1-2/weights/best.pt"
```

Check the configured video files:

```powershell
Test-Path "data/videos/test/door.mp4"
Test-Path "data/videos/test/hood.mp4"
Test-Path "data/videos/test/car.mp4"
Test-Path "data/videos/test/Gears.mp4"
Test-Path "data/videos/test/Spark_plugs.mp4"
Test-Path "data/videos/test/Brake_pads.mp4"
Test-Path "data/videos/test/bearing.mp4"
```

Each command should return `True` for the dashboard to start with the current configuration. If a file is missing, add it or edit the corresponding path in `app.py`.

## ▶️ Run the Streamlit Dashboard

From the project root, with the virtual environment active:

```powershell
python -m streamlit run streamlit_app.py
```

Then open the local URL printed in the terminal, commonly:

```text
http://localhost:8501
```

In the dashboard:

1. Select **Start Live Monitoring** in the sidebar.
2. View the seven configured camera tiles and their product counts.
3. Monitor the session status and total counters.
4. Select **Stop monitoring** to stop processing.
5. Select **Reset session** when processing has stopped to clear the dashboard state.

The feeds are recorded MP4 simulations, not physical camera streams.

### Alternative: Run YOLO on one video

To run a basic prediction on the sample video using the separate four-class model:

```powershell
yolo detect predict model="runs/detect/runs/detect/production_v4_synthetic_spark-2/weights/best.pt" source="data/videos/test/new_video.mp4" device=cpu
```

For a CUDA-enabled setup, replace `device=cpu` with `device=0`.

Ultralytics saves annotated predictions under a `runs/detect/predict/` directory (or a numbered equivalent if that folder already exists). This standalone command performs detection; it does not run the full multi-camera Streamlit monitoring workflow.

## 🧪 Tests and Dataset Utilities

The `tests/` directory contains scripts related to counter behavior, video counting, occupancy, dwell time, event processing, scheduling, and dataset checks. Run a test from the project root, for example:

```powershell
python -m pytest
```

If `pytest` is not installed, install it in your environment first:

```powershell
python -m pip install pytest
```

Some files in `tests/` are standalone scripts rather than pytest-style tests, so they may need to be executed individually. Check each script's contents and required input files before running it.

The `scripts/` directory contains dataset auditing, preparation, balancing, frame extraction, merging, and visualization utilities. Review each script's paths and configuration before use.

## 🛠️ Configuration

The main runtime configuration is near the top of `app.py`:

| Setting | Current value | Description |
|---|---:|---|
| `MODEL_PATH` | `runs/detect/models/production_7class_car_v1-2/weights/best.pt` | YOLO weights used by the dashboard |
| `CONFIDENCE` | `0.20` | Detection confidence threshold |
| `IMAGE_SIZE` | `512` | YOLO inference image size |
| `DEVICE` | CUDA `0` if available, otherwise CPU | Inference device selection |
| `QUEUE_SIZE` | `5` | Maximum queued frames per camera |

Camera names, video paths, target classes, and line orientations are configured in the `CAMERAS` list in `app.py`. Update these values to suit the videos and model classes used in your environment.

## ⚠️ Troubleshooting

### `Application Control policy has blocked this file`

Some Windows systems enforce application execution policies that can block `streamlit.exe` or other executables. First try launching Streamlit through the active Python environment:

```powershell
python -m streamlit run streamlit_app.py
```

If Windows still blocks execution, check the applicable Windows security or organization-managed Application Control logs and request approval from the system administrator. Do not disable organizational security controls or rename executables to bypass them.

### `No module named streamlit` (or another package)

Confirm that the intended virtual environment is active and that the package is installed in that environment:

```powershell
python -m pip show streamlit
python -m pip install -r requirements.txt
```

If package installation is restricted by your device's policy, use an approved package installation process.

### YOLO model file not found

Confirm the path in `MODEL_PATH` in `app.py` and ensure the corresponding `best.pt` file exists at that location.

### Video file not found or cannot be opened

Ensure that all seven configured video files exist at the paths listed above and that OpenCV can read their formats. If using different clips, update the paths in `CAMERAS`.

### CUDA or GPU-related errors

Check the installed PyTorch build, NVIDIA driver, and CUDA compatibility. To test with CPU, ensure `DEVICE` is set to `"cpu"` in `app.py`.

### Dashboard processing is slow

Inference speed depends on the hardware, video resolution, model size, and number of feeds. CPU processing may be slower than the source video frame rate. The current bounded queues apply backpressure, so the videos may take longer to process than their playback duration.

## 📊 Evaluation

Object detection metrics such as precision, recall, and mean Average Precision (mAP) can be used to evaluate the model when labelled validation data is available.

For production monitoring, also validate:

- Missed and false detections
- Class prediction errors
- Tracking ID stability and ID switches
- Count differences compared with manually verified totals
- Throughput estimates against known production rates
- Performance under different lighting, camera angles, occlusion, and object speeds

Validate with representative, manually reviewed production footage before using the system for operational decisions.

## 🔭 Limitations and Future Work

### Current scope and limitations

- The dashboard uses recorded video files as camera-feed simulations.
- The seven-camera workflow depends on the configured video files and model weights being present.
- Detection and counting accuracy depend on model quality, video conditions, and tracking stability.
- Real-time performance is hardware-dependent; processing recorded files does not guarantee real-time operation with physical cameras.
- The repository includes additional modules that may not be fully connected to the current dashboard workflow.

### Potential improvements

- Integrate physical IP cameras or RTSP streams.
- Validate and improve the model using diverse real-world production footage.
- Improve tracking robustness for overlapping or fast-moving objects.
- Add configurable counting-line and region-of-interest controls.
- Store production summaries and export reports.
- Add alerts for abnormal production rates or detected defects.
- Expand automated tests and document reproducible model evaluation.

## 👨‍💻 Project Information

**Project:** AI-Powered Production Monitoring System  
**Domain:** Computer Vision / Smart Manufacturing  
**Focus:** Automotive component detection, multi-camera video monitoring, object tracking, line-crossing counts, and throughput calculation  
**Interface:** Streamlit

## 📄 License

No license is specified in this repository at present. Add a license file if you intend to define permissions for use, modification, and distribution.
