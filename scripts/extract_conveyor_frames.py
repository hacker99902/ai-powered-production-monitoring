import cv2
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_PATH = Path(
    "data/videos/test/new.mp4"
)

OUTPUT_DIR = Path(
    "data/dataset/raw/conveyor_spark_plugs"
)

# Extract this many frames
NUM_FRAMES = 30


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {VIDEO_PATH}"
        )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    print("\n======================================")
    print("CONVEYOR FRAME EXTRACTION")
    print("======================================")

    print(f"Video        : {VIDEO_PATH}")
    print(f"Resolution   : {width} x {height}")
    print(f"FPS          : {fps}")
    print(f"Total frames : {total_frames}")
    print(f"Extracting   : {NUM_FRAMES} frames")

    # Spread frames across the entire video
    frame_indices = [
        int(i * (total_frames - 1) / (NUM_FRAMES - 1))
        for i in range(NUM_FRAMES)
    ]

    saved = 0

    for index, frame_number in enumerate(
        frame_indices
    ):

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_number
        )

        ret, frame = cap.read()

        if not ret:
            print(
                f"Could not read frame {frame_number}"
            )
            continue

        output_path = (
            OUTPUT_DIR /
            f"conveyor_{index + 1:03d}.jpg"
        )

        cv2.imwrite(
            str(output_path),
            frame
        )

        saved += 1

        print(
            f"[{saved:02d}/{NUM_FRAMES}] "
            f"Frame {frame_number} -> "
            f"{output_path.name}"
        )

    cap.release()

    print("\n======================================")
    print("EXTRACTION COMPLETE")
    print("======================================")

    print(f"Saved: {saved} frames")
    print(f"Location: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()