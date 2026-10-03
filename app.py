from pathlib import Path
import queue
import threading
import time

import cv2
import psutil
import torch

from cv.production.detector import ProductDetector
from cv.production.production_pipeline import ProductionPipeline


# --------------------------------------------------
# 1. PATHS AND SETTINGS
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR / "runs" / "detect" / "models"
    / "production_7class_car_v1-2"
    / "weights" / "best.pt"
)

CAMERAS = [
    {"name": "DOOR CAMERA", "path": BASE_DIR / "data/videos/test/door.mp4",
     "orientation": "vertical", "target_class": "car_door"},
    {"name": "BONNET CAMERA", "path": BASE_DIR / "data/videos/test/hood.mp4",
     "orientation": "horizontal", "target_class": "car_bonnet"},
    {"name": "CAR CAMERA", "path": BASE_DIR / "data/videos/test/car.mp4",
     "orientation": "vertical", "target_class": "car"},
    {"name": "GEAR CAMERA", "path": BASE_DIR / "data/videos/test/Gears.mp4",
     "orientation": "horizontal", "target_class": "gear"},
    {"name": "SPARK PLUG CAMERA", "path": BASE_DIR / "data/videos/test/Spark_plugs.mp4",
     "orientation": "horizontal", "target_class": "spark_plug"},
    {"name": "BRAKE PAD CAMERA", "path": BASE_DIR / "data/videos/test/Brake_pads.mp4",
     "orientation": "horizontal", "target_class": "brake_pad"},
    {"name": "BEARING CAMERA", "path": BASE_DIR / "data/videos/test/bearing.mp4",
     "orientation": "horizontal", "target_class": "bearing"},
]

CONFIDENCE = 0.20
IMAGE_SIZE = 512
DEVICE = 0 if torch.cuda.is_available() else "cpu"
QUEUE_SIZE = 5
TILE_WIDTH = 640
TILE_HEIGHT = 360


# --------------------------------------------------
# 2. SHARED RUNTIME STATE
# --------------------------------------------------
def new_runtime():
    return {
        "lock": threading.Lock(),
        "stop_event": threading.Event(),
        "thread": None,
        "running": False,
        "finished": False,
        "error": None,
        "frames": [None] * len(CAMERAS),
        "counts": [0] * len(CAMERAS),
        "read": [0] * len(CAMERAS),
        "processed": [0] * len(CAMERAS),
        "queues": [0] * len(CAMERAS),
        "fps": [0.0] * len(CAMERAS),
        "cpu": 0.0,
        "ram": 0.0,
        "gpu": None,
        "vram": None,
        "started_at": None,
    }


def gpu_stats():
    if not torch.cuda.is_available():
        return None, None
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=1, check=True
        )
        values = result.stdout.strip().split(",")
        return int(values[0].strip()), int(values[1].strip())
    except Exception:
        return None, None


# --------------------------------------------------
# 3. CAMERA WORKER
# --------------------------------------------------
def run_cameras(runtime):
    caps = []
    reader_threads = []
    queues = [queue.Queue(maxsize=QUEUE_SIZE) for _ in CAMERAS]
    reader_finished = [False] * len(CAMERAS)
    read_counts = [0] * len(CAMERAS)
    processed_counts = [0] * len(CAMERAS)
    frame_times = [[] for _ in CAMERAS]
    processing_seconds = [0.0] * len(CAMERAS)
    fps_values = [30.0] * len(CAMERAS)

    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"YOLO model not found: {MODEL_PATH}")
        for cam in CAMERAS:
            if not cam["path"].exists():
                raise FileNotFoundError(f'Video not found: {cam["path"]}')

        for i, cam in enumerate(CAMERAS):
            cap = cv2.VideoCapture(str(cam["path"]))
            if not cap.isOpened():
                raise RuntimeError(f'Could not open {cam["name"]}: {cam["path"]}')
            fps = cap.get(cv2.CAP_PROP_FPS)
            fps_values[i] = fps if fps and fps > 0 else 30.0
            caps.append(cap)

        detector = ProductDetector(
            model_path=str(MODEL_PATH),
            conf=CONFIDENCE,
            imgsz=IMAGE_SIZE,
            device=DEVICE,
        )
        pipelines = [
            ProductionPipeline(
                model_path=str(MODEL_PATH),
                conf=CONFIDENCE,
                imgsz=IMAGE_SIZE,
                device=DEVICE,
                window_seconds=60,
                fps=fps_values[i],
                line_orientation=CAMERAS[i]["orientation"],
                line_position=0.5,
                detector=detector,
            )
            for i in range(len(CAMERAS))
        ]

        with runtime["lock"]:
            runtime["running"] = True
            runtime["started_at"] = time.monotonic()

        def reader(camera_index):
            cap = caps[camera_index]
            fps = fps_values[camera_index]
            frame_index = 0
            pace_start = time.perf_counter()
            try:
                while not runtime["stop_event"].is_set():
                    ok, frame = cap.read()
                    if not ok:
                        break

                    # Simulate the source video's frame rate. A full queue
                    # blocks the reader; it does not discard the frame.
                    target_time = pace_start + frame_index / fps
                    delay = target_time - time.perf_counter()
                    if delay > 0:
                        runtime["stop_event"].wait(delay)
                    if runtime["stop_event"].is_set():
                        break

                    while not runtime["stop_event"].is_set():
                        try:
                            queues[camera_index].put(
                                (frame_index, frame), timeout=0.1
                            )
                            break
                        except queue.Full:
                            continue
                    if runtime["stop_event"].is_set():
                        break

                    frame_index += 1
                    read_counts[camera_index] = frame_index
                    with runtime["lock"]:
                        runtime["read"][camera_index] = frame_index
            finally:
                reader_finished[camera_index] = True

        for i in range(len(CAMERAS)):
            t = threading.Thread(target=reader, args=(i,), daemon=True)
            t.start()
            reader_threads.append(t)

        last_report = 0.0
        while not runtime["stop_event"].is_set():
            processed_something = False

            for i, cam in enumerate(CAMERAS):
                try:
                    frame_index, frame = queues[i].get_nowait()
                except queue.Empty:
                    continue

                processed_something = True
                start = time.perf_counter()
                timestamp = frame_index / fps_values[i]
                annotated, counts, rates, newly_counted = (
                    pipelines[i].process_frame(frame, timestamp)
                )
                processing_seconds[i] += time.perf_counter() - start
                processed_counts[i] += 1

                target_count = counts.get(cam["target_class"], 0)
                now = time.monotonic()
                frame_times[i].append(now)
                frame_times[i] = [
                    t for t in frame_times[i] if now - t <= 5.0
                ]
                fps = len(frame_times[i]) / 5.0

                # Copy to RGB so Streamlit can render it safely.
                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                with runtime["lock"]:
                    runtime["frames"][i] = rgb
                    runtime["counts"][i] = target_count
                    runtime["processed"][i] = processed_counts[i]
                    runtime["fps"][i] = fps

                queues[i].task_done()

            now = time.monotonic()
            if now - last_report >= 1.0:
                cpu = psutil.cpu_percent(interval=None)
                ram = psutil.virtual_memory().percent
                gpu, vram = gpu_stats()
                with runtime["lock"]:
                    runtime["queues"] = [q.qsize() for q in queues]
                    runtime["cpu"] = cpu
                    runtime["ram"] = ram
                    runtime["gpu"] = gpu
                    runtime["vram"] = vram
                last_report = now

            if all(reader_finished) and all(q.empty() for q in queues):
                break
            if not processed_something:
                time.sleep(0.003)

    except Exception as exc:
        with runtime["lock"]:
            runtime["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        runtime["stop_event"].set()
        for t in reader_threads:
            t.join(timeout=2)
        for cap in caps:
            cap.release()
        with runtime["lock"]:
            runtime["running"] = False
            runtime["finished"] = True

# This module contains the camera runtime. Run the UI with: streamlit run streamlit.py
