
from pathlib import Path
import time

import cv2
import streamlit as st

from cv.production.production_pipeline import ProductionPipeline


# 1. Project paths
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "runs/detect/runs/detect/production_v4_synthetic_spark-2/weights/best.pt"
)

VIDEO_PATH = BASE_DIR / "data/videos/test/new_video.mp4"

CLASS_NAMES = ["brake_pad", "bearing", "spark_plug", "gear"]

# Dashboard settings
DISPLAY_WIDTH = 480
UI_UPDATE_EVERY = 1


# 2. Streamlit page setup
st.set_page_config(
    page_title="Smart Factory Monitor",
    page_icon="🏭",
    layout="wide"
)

st.title("🏭 Smart Factory Production Monitor")
st.caption("Production Manager Dashboard | Conveyor Line 01")

st.divider()


# 3. Production overview
st.subheader("Production Overview")

count_columns = st.columns(4)
count_placeholders = {}

for column, class_name in zip(count_columns, CLASS_NAMES):
    with column:
        st.caption(class_name.replace("_", " ").title())
        count_placeholders[class_name] = st.empty()
        count_placeholders[class_name].metric(
            label="Count",
            value=0
        )

rate_columns = st.columns(4)
rate_placeholders = {}

for column, class_name in zip(rate_columns, CLASS_NAMES):
    with column:
        st.caption(class_name.replace("_", " ").title())
        rate_placeholders[class_name] = st.empty()
        rate_placeholders[class_name].metric(
            label="Products / min",
            value="0.0"
        )

total_placeholder = st.empty()
total_placeholder.metric("Total Products Counted", 0)

st.divider()


# 4. Compact camera feed
st.subheader("Conveyor Belt Camera Feed")

video_placeholder = st.empty()
progress_placeholder = st.empty()
status_placeholder = st.empty()


# 5. Start monitoring
if st.button("▶ Start Production Monitoring", type="primary"):

    if not MODEL_PATH.exists():
        st.error(f"Model not found: {MODEL_PATH}")
        st.stop()

    if not VIDEO_PATH.exists():
        st.error(f"Video not found: {VIDEO_PATH}")
        st.stop()

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        st.error("Unable to open the test video.")
        st.stop()

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = 1.0 / fps

    pipeline = ProductionPipeline(
        model_path=str(MODEL_PATH),
        conf=0.20,
        imgsz=1280,
        device=0,
        window_seconds=60,
        fps=fps
    )

    frame_number = 0
    status_placeholder.info("Monitoring in progress...")

    try:
        while True:
            # Start timing this frame
            frame_start_time = time.perf_counter()

            success, frame = cap.read()

            if not success:
                break

            # Timestamp based on the source video's timeline
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            # Process every frame for tracking and counting
            annotated_frame, counts, rates, newly_counted = (
                pipeline.process_frame(frame, timestamp)
            )

            frame_number += 1

            # Resize only the displayed frame
            if frame_number % UI_UPDATE_EVERY == 0:
                height, width = annotated_frame.shape[:2]

                display_height = max(
                    1,
                    int(height * DISPLAY_WIDTH / width)
                )

                small_frame = cv2.resize(
                    annotated_frame,
                    (DISPLAY_WIDTH, display_height),
                    interpolation=cv2.INTER_AREA
                )

                rgb_frame = cv2.cvtColor(
                    small_frame,
                    cv2.COLOR_BGR2RGB
                )

                video_placeholder.image(
                    rgb_frame,
                    channels="RGB",
                    width=DISPLAY_WIDTH
                )

                # Update live product counts and rates
                for class_name in CLASS_NAMES:
                    count = counts.get(class_name, 0)
                    rate = rates.get(class_name, 0.0)

                    count_placeholders[class_name].metric(
                        label="Count",
                        value=count
                    )

                    rate_placeholders[class_name].metric(
                        label="Products / min",
                        value=f"{rate:.1f}"
                    )

                total_placeholder.metric(
                    "Total Products Counted",
                    sum(counts.values())
                )

            # Update progress
            if frame_number % 10 == 0 or frame_number == total_frames:
                if total_frames > 0:
                    progress_placeholder.progress(
                        min(frame_number / total_frames, 1.0),
                        text=(
                            f"Processing frame {frame_number} "
                            f"of {total_frames}"
                        )
                    )

            # Pace playback to the original video's FPS
            elapsed = time.perf_counter() - frame_start_time
            remaining_time = frame_interval - elapsed

            if remaining_time > 0:
                time.sleep(remaining_time)

        # 6. Final production counts
        final_counts = pipeline.get_counts()
        final_total = sum(final_counts.values())

        for class_name in CLASS_NAMES:
            count_placeholders[class_name].metric(
                label="Count",
                value=final_counts.get(class_name, 0)
            )

        total_placeholder.metric(
            "Total Products Counted",
            final_total
        )

        progress_placeholder.progress(
            1.0,
            text="Processing complete"
        )

        status_placeholder.success("Monitoring finished.")

        st.divider()
        st.subheader("Final Production Counts")

        final_columns = st.columns(4)

        for column, class_name in zip(final_columns, CLASS_NAMES):
            with column:
                st.metric(
                    class_name.replace("_", " ").title(),
                    final_counts.get(class_name, 0)
                )

        st.metric("Total Products", final_total)

    except Exception as error:
        status_placeholder.error(f"Processing failed: {error}")

    finally:
        cap.release()