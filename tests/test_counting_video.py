import cv2
from pathlib import Path
from ultralytics import YOLO

from cv.production.counter import LineCounter
from cv.production.line_detector import AutomaticLineDetector


# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_DIR = Path("data/videos/test")
MODEL_PATH = "runs/detect/runs/detect/production_v4_synthetic_spark-2/weights/best.pt"
CONFIDENCE = 0.20
IMAGE_SIZE = 1280
DEVICE = 0

# Number of frames used to automatically detect the conveyor flow
CALIBRATION_FRAMES = 20

# How many previous positions to keep for drawing trajectories
TRAIL_LENGTH = 30


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_video_path():
    """
    Find the first MP4 video inside data/videos/test.
    """

    videos = sorted(VIDEO_DIR.glob("*.mp4"))

    if not videos:
        raise FileNotFoundError(
            f"No MP4 video found inside: {VIDEO_DIR}"
        )

    print("\nAvailable video:")
    for video in videos:
        print(f"  - {video}")

    return videos[0]


def convert_results_to_tracks(result):
    """
    Convert YOLO tracking results into the format expected by
    LineCounter and AutomaticLineDetector.
    """

    tracks = []

    if result.boxes is None:
        return tracks

    boxes = result.boxes

    if boxes.id is None:
        return tracks

    xyxy = boxes.xyxy.cpu().numpy()
    classes = boxes.cls.cpu().numpy()
    confidences = boxes.conf.cpu().numpy()
    track_ids = boxes.id.cpu().numpy().astype(int)

    class_names = result.names

    for bbox, cls_id, confidence, track_id in zip(
        xyxy,
        classes,
        confidences,
        track_ids
    ):

        x1, y1, x2, y2 = bbox

        class_name = class_names[int(cls_id)]

        tracks.append({
            "track_id": int(track_id),
            "class_name": class_name,
            "bbox": (
                int(x1),
                int(y1),
                int(x2),
                int(y2)
            ),
            "confidence": float(confidence)
        })

    return tracks


def get_center(bbox):
    """
    Calculate center point of a bounding box.
    """

    x1, y1, x2, y2 = bbox

    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)

    return center_x, center_y


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    # --------------------------------------------------------
    # STEP 1: Find video
    # --------------------------------------------------------

    video_path = get_video_path()

    print("\n==============================================")
    print("AUTOMATIC COUNTING LINE TEST")
    print("==============================================")

    print(f"Video : {video_path}")
    print(f"Model : {MODEL_PATH}")

    # --------------------------------------------------------
    # STEP 2: Open video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("\nVideo information:")
    print(f"  Width        : {width}")
    print(f"  Height       : {height}")
    print(f"  FPS          : {fps}")
    print(f"  Total frames : {total_frames}")

    # --------------------------------------------------------
    # STEP 3: Load YOLO model for calibration
    # --------------------------------------------------------

    print("\nLoading YOLO model for calibration...")

    calibration_model = YOLO(MODEL_PATH)

    # --------------------------------------------------------
    # STEP 4: Create AutomaticLineDetector
    # --------------------------------------------------------

    line_detector = AutomaticLineDetector(
        min_track_length=10,
        min_movement=20,
        percentile=50
    )

    print("\n==============================================")
    print("STARTING AUTOMATIC LINE CALIBRATION")
    print("==============================================")

    print(
        f"Collecting trajectories from first "
        f"{CALIBRATION_FRAMES} frames..."
    )

    # --------------------------------------------------------
    # STEP 5: Calibration
    # --------------------------------------------------------

    calibration_frame_count = 0

    while calibration_frame_count < CALIBRATION_FRAMES:

        ret, frame = cap.read()

        if not ret:
            print("\nVideo ended during calibration.")
            break

        calibration_frame_count += 1

        # Run YOLO + ByteTrack
        results = calibration_model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=CONFIDENCE,
            imgsz=IMAGE_SIZE,
            device=DEVICE,
            verbose=False
        )

        if not results:
            continue

        result = results[0]

        tracks = convert_results_to_tracks(result)

        # Give tracks to automatic line detector
        line_detector.update(tracks)

        # ----------------------------------------------------
        # Show calibration visualization
        # ----------------------------------------------------

        calibration_display = frame.copy()

        # Draw detected tracks
        for track in tracks:

            x1, y1, x2, y2 = track["bbox"]

            track_id = track["track_id"]
            class_name = track["class_name"]
            confidence = track["confidence"]

            center_x, center_y = get_center(track["bbox"])

            # Bounding box
            cv2.rectangle(
                calibration_display,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Center
            cv2.circle(
                calibration_display,
                (center_x, center_y),
                5,
                (0, 0, 255),
                -1
            )

            # Label
            label = (
                f"ID:{track_id} "
                f"{class_name} "
                f"{confidence:.2f}"
            )

            cv2.putText(
                calibration_display,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        # Calibration text
        cv2.putText(
            calibration_display,
            f"CALIBRATION: "
            f"{calibration_frame_count}/{CALIBRATION_FRAMES}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )

        cv2.putText(
            calibration_display,
            "Collecting product movement...",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.imshow(
            "Automatic Line Calibration",
            calibration_display
        )

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            print("\nCalibration cancelled by user.")
            cap.release()
            cv2.destroyAllWindows()
            return

    # --------------------------------------------------------
    # STEP 6: Detect automatic line
    # --------------------------------------------------------

    print("\n==============================================")
    print("DETECTING AUTOMATIC COUNTING LINE")
    print("==============================================")

    detected_line = line_detector.detect_line(
        frame_width=width,
        frame_height=height
    )

    if detected_line is None:

        print("\nERROR:")
        print("Could not automatically detect a counting line.")

        print("\nDetector status:")
        print(line_detector.get_status())

        cap.release()
        cv2.destroyAllWindows()

        return

    line_start = detected_line["start"]
    line_end = detected_line["end"]

    print("\nAutomatic line detected!")

    print(f"  Start : {line_start}")
    print(f"  End   : {line_end}")

    print("\nDetector status:")
    print(line_detector.get_status())

    # --------------------------------------------------------
    # STEP 7: Lock the line
    # --------------------------------------------------------

    line_detector.lock()

    print("\nCounting line LOCKED.")

    print(
        "\nThe line will NOT move during counting."
    )

    # --------------------------------------------------------
    # STEP 8: Show detected line before counting
    # --------------------------------------------------------

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    ret, first_frame = cap.read()

    if ret:

        preview = first_frame.copy()

        cv2.line(
            preview,
            line_start,
            line_end,
            (0, 255, 255),
            4
        )

        cv2.circle(
            preview,
            line_start,
            8,
            (255, 0, 0),
            -1
        )

        cv2.circle(
            preview,
            line_end,
            8,
            (255, 0, 0),
            -1
        )

        cv2.putText(
            preview,
            "AUTOMATIC COUNTING LINE",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )

        cv2.putText(
            preview,
            "Press any key to start counting | ESC to exit",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "Detected Counting Line",
            preview
        )

        key = cv2.waitKey(0) & 0xFF

        try:
            cv2.destroyWindow("Detected Counting Line")
        except cv2.error:
            pass

        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            return

    # --------------------------------------------------------
    # STEP 9: Reset video
    # --------------------------------------------------------

    cap.release()

    # IMPORTANT:
    # Create a fresh YOLO model so ByteTrack does not retain
    # calibration tracking state.

    print("\nStarting fresh tracking session...")

    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            "Could not reopen video."
        )

    # --------------------------------------------------------
    # STEP 10: Create LineCounter
    # --------------------------------------------------------

    counter = LineCounter(
        line_start=line_start,
        line_end=line_end,
        direction="any"
    )

    # --------------------------------------------------------
    # STEP 11: Track trajectories for visualization
    # --------------------------------------------------------

    trajectories = {}

    frame_number = 0

    print("\n==============================================")
    print("STARTING AUTOMATIC COUNTING")
    print("==============================================")

    print("\nPress ESC to stop.\n")

    # --------------------------------------------------------
    # STEP 12: Process complete video
    # --------------------------------------------------------

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        # ----------------------------------------------------
        # YOLO + ByteTrack
        # ----------------------------------------------------

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=CONFIDENCE,
            imgsz=IMAGE_SIZE,
            device=DEVICE,
            verbose=False
        )

        if not results:
            continue

        result = results[0]

        tracks = convert_results_to_tracks(result)

        # ----------------------------------------------------
        # Update LineCounter
        # ----------------------------------------------------

        counter.update(tracks)

        # ----------------------------------------------------
        # Draw counting line
        # ----------------------------------------------------

        display = frame.copy()

        cv2.line(
            display,
            line_start,
            line_end,
            (0, 255, 255),
            4
        )

        # ----------------------------------------------------
        # Draw tracks
        # ----------------------------------------------------

        for track in tracks:

            track_id = track["track_id"]
            class_name = track["class_name"]
            confidence = track["confidence"]

            x1, y1, x2, y2 = track["bbox"]

            center_x, center_y = get_center(
                track["bbox"]
            )

            # -----------------------------------------------
            # Save trajectory
            # -----------------------------------------------

            if track_id not in trajectories:
                trajectories[track_id] = []

            trajectories[track_id].append(
                (center_x, center_y)
            )

            # Keep only last N points
            trajectories[track_id] = trajectories[
                track_id
            ][-TRAIL_LENGTH:]

            # -----------------------------------------------
            # Bounding box
            # -----------------------------------------------

            cv2.rectangle(
                display,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # -----------------------------------------------
            # Center
            # -----------------------------------------------

            cv2.circle(
                display,
                (center_x, center_y),
                5,
                (0, 0, 255),
                -1
            )

            # -----------------------------------------------
            # Trajectory
            # -----------------------------------------------

            points = trajectories[track_id]

            for i in range(1, len(points)):

                cv2.line(
                    display,
                    points[i - 1],
                    points[i],
                    (255, 0, 255),
                    2
                )

            # -----------------------------------------------
            # Label
            # -----------------------------------------------

            label = (
                f"ID:{track_id} "
                f"{class_name} "
                f"{confidence:.2f}"
            )

            cv2.putText(
                display,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        # ----------------------------------------------------
        # Display counts
        # ----------------------------------------------------

        counts = counter.get_counts()

        cv2.putText(
            display,
            f"Frame: {frame_number}/{total_frames}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        y_position = 70

        for class_name in [
            "brake_pad",
            "bearing",
            "spark_plug",
            "gear"
        ]:

            count = counts.get(
                class_name,
                0
            )

            text = (
                f"{class_name}: {count}"
            )

            cv2.putText(
                display,
                text,
                (20, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            y_position += 30

        # ----------------------------------------------------
        # Line label
        # ----------------------------------------------------

        cv2.putText(
            display,
            "AUTOMATIC COUNTING LINE",
            (20, height - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        # ----------------------------------------------------
        # Show frame
        # ----------------------------------------------------

        cv2.imshow(
            "Production Counting - Automatic Line",
            display
        )

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

    # --------------------------------------------------------
    # STEP 13: Final results
    # --------------------------------------------------------

    cap.release()
    cv2.destroyAllWindows()

    final_counts = counter.get_counts()

    print("\n==============================================")
    print("FINAL COUNTS")
    print("==============================================")

    for class_name in [
        "brake_pad",
        "bearing",
        "spark_plug",
        "gear"
    ]:

        print(
            f"{class_name:12s}: "
            f"{final_counts.get(class_name, 0)}"
        )

    print("==============================================")

    print("\nAutomatic line:")
    print(f"Start: {line_start}")
    print(f"End  : {line_end}")

    print("\nTest completed.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()